//! Per-layer pruning sensitivity: prune ONE layer's FFN to `width` and
//! score — the profile a tube plan should spend its budget against
//! (a uniform width is only right if every layer is equally tolerant,
//! and no model's layers are).
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_sens -- \
//!       model.cmf mass.mass text.txt [width] [tokens]
use cortiq_core::CmfModel;
use cortiq_core::mask::{MaskPriority, TaskMask};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let mass_p = a.next().expect("mass file");
    let text_p = a.next().expect("text file");
    let width: f32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let ntok: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
    // Layers per probe: 1 = the per-layer profile, N = a depth profile in
    // groups (what a 64-layer model can afford to measure).
    let group: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(1);

    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
    let (nl, inter) = (m.arch().num_layers, m.arch().intermediate_size);
    let heads = m.arch().num_attention_heads;
    let b = std::fs::read(&mass_p)?;
    let mut mass = vec![vec![0f64; inter]; nl];
    for (li, row) in mass.iter_mut().enumerate() {
        for (n, v) in row.iter_mut().enumerate() {
            let o = 8 + (li * inter + n) * 4;
            *v = f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
        }
    }
    let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;
    let mut ids = p.tokenizer.encode(&std::fs::read_to_string(&text_p)?);
    ids.truncate(ntok);
    let score = |p: &mut Pipeline, mask: Option<&TaskMask>| -> f64 {
        let (mut nll, mut cnt) = (0f64, 0usize);
        for c in ids.chunks(256) {
            if c.len() < 2 {
                break;
            }
            let (l, k) = p.nll_ids_masked(c, 0, mask);
            nll += l;
            cnt += k;
        }
        (nll / cnt.max(1) as f64).exp()
    };
    let keep_n = ((inter as f32 * width).ceil() as usize).clamp(1, inter);
    let mut hb = vec![0u8; heads.div_ceil(8)];
    for h in 0..heads {
        hb[h / 8] |= 1 << (h % 8);
    }
    let dense = score(&mut p, None);
    println!(
        "dense PPL {dense:.4} | pruning one layer at a time to {:.0}%",
        width * 100.0
    );
    for g0 in (0..nl).step_by(group) {
        let g1 = (g0 + group).min(nl);
        let mut ffn_masks = Vec::with_capacity(nl);
        for li in 0..nl {
            let mut bits = vec![0xffu8; inter.div_ceil(8)];
            if li >= g0 && li < g1 {
                bits.iter_mut().for_each(|x| *x = 0);
                let mut order: Vec<usize> = (0..inter).collect();
                order.sort_by(|&x, &y| mass[li][y].total_cmp(&mass[li][x]));
                for &n in order.iter().take(keep_n) {
                    bits[n / 8] |= 1 << (n % 8);
                }
            } else {
                // clear the tail bits past `inter` — phantom neurons
                for n in inter..inter.div_ceil(8) * 8 {
                    bits[n / 8] &= !(1 << (n % 8));
                }
            }
            ffn_masks.push(bits);
        }
        let mask = TaskMask {
            task_id: 1,
            name: format!("L{g0}..{g1}"),
            description: None,
            sparsity: (1.0 - width) / nl as f32,
            quality: None,
            ffn_masks,
            head_masks: vec![hb.clone(); nl],
            layer_gates: vec![true; nl],
            expert_masks: Vec::new(),
            parent: None,
            priority: MaskPriority::Normal,
            has_hot_pack: false,
        };
        let s = score(&mut p, Some(&mask));
        println!(
            "L{g0:02}..{g1:02}  PPL {s:8.4}   Δ {:+7.2}%",
            (s / dense - 1.0) * 100.0
        );
    }
    Ok(())
}
