//! The tube quality gate: is a PER-TASK narrow FFN better than a GLOBAL
//! narrow FFN of the same width?
//!
//!   CMF_GPU=0 cargo run --release -p cortiq-cli --example tube_gate -- \
//!       model.cmf mass_dir eval_dir 1.0,0.75,0.5,0.35,0.25 [tokens] [cross]
//!
//! `mass_dir` holds `<task>.mass` from `tube_probe` (the calibration
//! prompts); `eval_dir` holds `<task>.txt` — HELD-OUT text of the same
//! task. For every width the gate scores each task's held-out text three
//! ways: dense, with that task's own top-w mask, and with the global
//! top-w mask (mass summed over all tasks — the task-blind width prune
//! every NVG-style compressor does). A per-task win over global at equal
//! width is the whole claim.
//!
//! With `cross` as the last argument it also prints the full task×mask
//! PPL matrix at the first width — the specialization evidence.
use cortiq_core::CmfModel;
use cortiq_core::mask::{MaskPriority, TaskMask};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::sync::Arc;

fn load_mass(dir: &str) -> Vec<(String, Vec<Vec<f64>>)> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir).unwrap().flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("mass") {
            continue;
        }
        let b = std::fs::read(&p).unwrap();
        let nl = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let inter = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        let mut rows = Vec::with_capacity(nl);
        for l in 0..nl {
            let mut row = Vec::with_capacity(inter);
            for i in 0..inter {
                let o = 8 + (l * inter + i) * 4;
                row.push(f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64);
            }
            rows.push(row);
        }
        out.push((p.file_stem().unwrap().to_str().unwrap().to_string(), rows));
    }
    out
}

/// Top-`keep` neurons per layer by mass → a first-class task mask.
fn mask_from_mass(name: &str, mass: &[Vec<f64>], keep: f32, heads: usize) -> TaskMask {
    let inter = mass[0].len();
    let keep_n = ((inter as f32 * keep).ceil() as usize).clamp(1, inter);
    let mut ffn_masks = Vec::with_capacity(mass.len());
    for row in mass {
        let mut order: Vec<usize> = (0..inter).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        let mut bits = vec![0u8; inter.div_ceil(8)];
        for &n in order.iter().take(keep_n) {
            bits[n / 8] |= 1 << (n % 8);
        }
        ffn_masks.push(bits);
    }
    let mut hb = vec![0u8; heads.div_ceil(8)];
    for h in 0..heads {
        hb[h / 8] |= 1 << (h % 8);
    }
    TaskMask {
        task_id: 1,
        name: name.to_string(),
        description: None,
        sparsity: 1.0 - keep_n as f32 / inter as f32,
        quality: None,
        ffn_masks,
        head_masks: vec![hb; mass.len()],
        layer_gates: vec![true; mass.len()],
        expert_masks: Vec::new(),
        parent: None,
        priority: MaskPriority::Normal,
        has_hot_pack: false,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let model_p = a.next().expect("model.cmf");
    let mass_d = a.next().expect("mass dir");
    let eval_d = a.next().expect("eval dir");
    let widths: Vec<f32> = a
        .next()
        .unwrap_or_else(|| "1.0,0.75,0.5,0.35,0.25".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let ntok: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let cross = a.next().is_some();

    let m = Arc::new(CmfModel::open_sharded(&model_p)?);
    let heads = m.arch().num_attention_heads;
    let mut p = Pipeline::from_model(&m, SamplerConfig::default())?;

    let mut mass = load_mass(&mass_d);
    // Wanda's structured form: dropping neuron n costs `a_n · W_down[:,n]`,
    // so weight the activation mass by the column norm when one is given.
    if let Ok(p) = std::env::var("TUBE_COLNORM") {
        let b = std::fs::read(&p)?;
        let inter = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        for (_, rows) in mass.iter_mut() {
            for (l, row) in rows.iter_mut().enumerate() {
                for (n, v) in row.iter_mut().enumerate() {
                    let o = 8 + (l * inter + n) * 4;
                    *v *= f32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as f64;
                }
            }
        }
        eprintln!("colnorm weighting from {p}");
    }
    // The task-blind baseline: one width prune for everyone.
    let mut global: Vec<Vec<f64>> = mass[0].1.clone();
    for (_, rows) in mass.iter().skip(1) {
        for (g, r) in global.iter_mut().zip(rows) {
            for (a, b) in g.iter_mut().zip(r) {
                *a += *b;
            }
        }
    }

    // Held-out ids per task.
    let mut evals: Vec<(String, Vec<u32>)> = Vec::new();
    for (task, _) in &mass {
        let path = format!("{eval_d}/{task}.txt");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("no eval text for {task} ({path})");
            continue;
        };
        let mut ids = p.tokenizer.encode(&text);
        ids.truncate(ntok);
        evals.push((task.clone(), ids));
    }

    let score = |p: &mut Pipeline, ids: &[u32], mask: Option<&TaskMask>| -> f64 {
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

    println!(
        "{:16} {:>9} {:>9} {:>9}  {:>7}",
        "task", "dense", "own", "global", "own/gl"
    );
    for &w in &widths {
        println!(
            "── width {:.0}% ───────────────────────────────────",
            w * 100.0
        );
        let gmask = mask_from_mass("global", &global, w, heads);
        let (mut so, mut sg, mut sd) = (0f64, 0f64, 0f64);
        for (task, ids) in &evals {
            let own = mass.iter().find(|(t, _)| t == task).unwrap();
            let omask = mask_from_mass(task, &own.1, w, heads);
            let dense = score(&mut p, ids, None);
            let po = score(&mut p, ids, Some(&omask));
            let pg = score(&mut p, ids, Some(&gmask));
            println!("{task:16} {dense:9.3} {po:9.3} {pg:9.3}  {:6.3}", po / pg);
            sd += dense;
            so += po;
            sg += pg;
        }
        let n = evals.len() as f64;
        println!(
            "{:16} {:9.3} {:9.3} {:9.3}  {:6.3}",
            "MEAN",
            sd / n,
            so / n,
            sg / n,
            so / sg
        );
    }

    if cross {
        let w = widths[0];
        println!(
            "\ncross matrix at width {:.0}% (rows = text, cols = mask):",
            w * 100.0
        );
        let masks: Vec<(String, TaskMask)> = mass
            .iter()
            .map(|(t, r)| (t.clone(), mask_from_mass(t, r, w, heads)))
            .collect();
        print!("{:16}", "");
        for (t, _) in &masks {
            print!("{:>9}", &t[..t.len().min(8)]);
        }
        println!();
        for (task, ids) in &evals {
            print!("{task:16}");
            for (_, mk) in &masks {
                print!("{:9.3}", score(&mut p, ids, Some(mk)));
            }
            println!();
        }
    }
    Ok(())
}
