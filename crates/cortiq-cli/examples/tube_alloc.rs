//! The FFN width budget, allocated where it is cheap.
//!
//! Layers are not equally tolerant of a narrow FFN (measured: the first
//! half of Qwen3.5-0.8B loses <2% PPL at half width, the last layer
//! loses 16%), so a uniform width spends the budget in the worst place.
//! This measures each layer's own loss-vs-width curve and then solves
//! the allocation: give the next 32 neurons to whichever layer pays the
//! most for them, until the budget is gone.
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_alloc -- \
//!       model.cmf mass.mass text.txt out.json [budget] [tokens] [grid]
use cortiq_core::mask::{MaskPriority, TaskMask};
use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let mass_p = a.next().expect("mass file");
    let text_p = a.next().expect("text file");
    let out_p = a.next().expect("out.json");
    let budget: f64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let ntok: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    let grid: Vec<f64> = a
        .next()
        .unwrap_or_else(|| "0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
    let (nl, inter) = (m.arch().num_layers, m.arch().intermediate_size);
    let heads = m.arch().num_attention_heads;
    let b = std::fs::read(&mass_p)?;
    let mut mass = vec![vec![0f64; inter]; nl];
    let mut order = vec![Vec::new(); nl];
    for li in 0..nl {
        for n in 0..inter {
            let o = 8 + (li * inter + n) * 4;
            mass[li][n] = f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
        }
        let mut ord: Vec<usize> = (0..inter).collect();
        ord.sort_by(|&x, &y| mass[li][y].total_cmp(&mass[li][x]));
        order[li] = ord;
    }
    let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;
    let mut ids = p.tokenizer.encode(&std::fs::read_to_string(&text_p)?);
    ids.truncate(ntok);
    let mut hb = vec![0u8; heads.div_ceil(8)];
    for h in 0..heads {
        hb[h / 8] |= 1 << (h % 8);
    }
    let mk = |target: usize, keep_n: usize| -> TaskMask {
        let mut ffn_masks = Vec::with_capacity(nl);
        for li in 0..nl {
            let mut bits = vec![0u8; inter.div_ceil(8)];
            let k = if li == target { keep_n } else { inter };
            for &n in order[li].iter().take(k) {
                bits[n / 8] |= 1 << (n % 8);
            }
            ffn_masks.push(bits);
        }
        TaskMask {
            task_id: 1,
            name: "probe".into(),
            description: None,
            sparsity: 0.0,
            quality: None,
            ffn_masks,
            head_masks: vec![hb.clone(); nl],
            layer_gates: vec![true; nl],
            expert_masks: Vec::new(),
            parent: None,
            priority: MaskPriority::Normal,
            has_hot_pack: false,
        }
    };
    let nll = |p: &mut Pipeline, mask: Option<&TaskMask>| -> f64 {
        let (mut s, mut c) = (0f64, 0usize);
        for ch in ids.chunks(256) {
            if ch.len() < 2 {
                continue;
            }
            let (l, k) = p.nll_ids_masked(ch, 0, mask);
            s += l;
            c += k;
        }
        s / c.max(1) as f64
    };
    let base = nll(&mut p, None);
    println!("dense NLL {base:.5} (PPL {:.3}) over {} tokens", base.exp(), ids.len());
    // cost[l][g] — the NLL this layer adds at grid width g.
    let mut cost = vec![vec![0f64; grid.len()]; nl];
    for li in 0..nl {
        let mut line = format!("L{li:02}");
        for (gi, &w) in grid.iter().enumerate() {
            let keep_n = ((inter as f64 * w).round() as usize / 32 * 32).clamp(32, inter);
            let d = if keep_n >= inter {
                0.0
            } else {
                (nll(&mut p, Some(&mk(li, keep_n))) - base).max(0.0)
            };
            cost[li][gi] = d;
            line += &format!(" {:.4}", d);
        }
        println!("{line}");
    }
    // Greedy marginal allocation: everyone starts at the smallest width,
    // then the widening that buys the most NLL per neuron wins, until
    // the budget is spent.
    let mut at = vec![0usize; nl];
    let target: f64 = budget * nl as f64;
    let mut spent: f64 = grid[0] * nl as f64;
    loop {
        let mut best: Option<(f64, usize)> = None;
        for li in 0..nl {
            if at[li] + 1 >= grid.len() {
                continue;
            }
            let dw = grid[at[li] + 1] - grid[at[li]];
            if spent + dw > target + 1e-9 {
                continue;
            }
            let gain = (cost[li][at[li]] - cost[li][at[li] + 1]) / dw;
            if best.is_none_or(|(g, _)| gain > g) {
                best = Some((gain, li));
            }
        }
        let Some((gain, li)) = best else { break };
        if gain <= 0.0 && spent >= target - 1e-9 {
            break;
        }
        spent += grid[at[li] + 1] - grid[at[li]];
        at[li] += 1;
    }
    let widths: Vec<usize> = (0..nl)
        .map(|li| ((inter as f64 * grid[at[li]]).round() as usize / 32 * 32).clamp(32, inter))
        .collect();
    let mean: f64 = widths.iter().sum::<usize>() as f64 / (nl * inter) as f64;
    println!("allocated widths (mean {:.1}% of {inter}): {widths:?}", mean * 100.0);
    let pred: f64 = (0..nl).map(|li| cost[li][at[li]]).sum();
    println!("predicted NLL cost if additive: {pred:.4} → PPL {:.3}", (base + pred).exp());
    std::fs::write(&out_p, serde_json::to_string(&widths)?)?;
    // And the honest joint measurement of exactly that allocation.
    let mut ffn_masks = Vec::with_capacity(nl);
    for li in 0..nl {
        let mut bits = vec![0u8; inter.div_ceil(8)];
        for &n in order[li].iter().take(widths[li]) {
            bits[n / 8] |= 1 << (n % 8);
        }
        ffn_masks.push(bits);
    }
    let joint = TaskMask {
        task_id: 1,
        name: "alloc".into(),
        description: None,
        sparsity: 1.0 - mean as f32,
        quality: None,
        ffn_masks,
        head_masks: vec![hb.clone(); nl],
        layer_gates: vec![true; nl],
        expert_masks: Vec::new(),
        parent: None,
        priority: MaskPriority::Normal,
        has_hot_pack: false,
    };
    let j = nll(&mut p, Some(&joint));
    println!("measured joint: NLL {j:.5} → PPL {:.3} ({:+.1}% vs dense)", j.exp(), (j.exp() / base.exp() - 1.0) * 100.0);
    // The uniform allocation of the same budget, for the comparison.
    let uw = ((inter as f64 * mean).round() as usize / 32 * 32).clamp(32, inter);
    let mut ffn_masks = Vec::with_capacity(nl);
    for li in 0..nl {
        let mut bits = vec![0u8; inter.div_ceil(8)];
        for &n in order[li].iter().take(uw) {
            bits[n / 8] |= 1 << (n % 8);
        }
        ffn_masks.push(bits);
    }
    let unif = TaskMask { name: "uniform".into(), ffn_masks, ..joint.clone() };
    let u = nll(&mut p, Some(&unif));
    println!("uniform {uw}/{inter}: NLL {u:.5} → PPL {:.3} ({:+.1}% vs dense)", u.exp(), (u.exp() / base.exp() - 1.0) * 100.0);
    Ok(())
}
