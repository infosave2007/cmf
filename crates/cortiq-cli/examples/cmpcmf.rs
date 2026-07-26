//! Numeric tensor diff between two .cmf files sharing tensor names —
//! the bring-up knife for "which mapped tensor is wrong": quant noise
//! shows as ~1e-2 relative, a layout/convention bug as O(1).
//!
//!     cargo run --release -p cortiq-cli --example cmpcmf -- a.cmf b.cmf [filter]

use cortiq_core::CmfModel;

fn tensor_f32(m: &CmfModel, name: &str) -> Option<Vec<f32>> {
    let e = m.tensor(name)?;
    let mut out = vec![0f32; e.shape.iter().product()];
    cortiq_core::quant::dequant_tensor(e, m.entry_bytes(e), &mut out).ok()?;
    Some(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: cmpcmf a.cmf b.cmf [name-filter]");
        std::process::exit(2);
    }
    let a = CmfModel::open(&args[1]).expect("open a");
    let b = CmfModel::open(&args[2]).expect("open b");
    let filter = args.get(3).cloned().unwrap_or_default();
    let mut rows: Vec<(f64, f64, String)> = Vec::new();
    for e in &a.tensors {
        if !filter.is_empty() && !e.name.contains(&filter) {
            continue;
        }
        let Some(va) = tensor_f32(&a, &e.name) else {
            continue;
        };
        let Some(vb) = tensor_f32(&b, &e.name) else {
            println!("{:<58} ONLY IN A", e.name);
            continue;
        };
        if va.len() != vb.len() {
            println!("{:<58} SHAPE MISMATCH {} vs {}", e.name, va.len(), vb.len());
            continue;
        }
        // Cosine similarity + relative L2 — layout bugs kill cosine.
        let (mut dot, mut na, mut nb, mut d2) = (0f64, 0f64, 0f64, 0f64);
        for (&x, &y) in va.iter().zip(&vb) {
            let (x, y) = (x as f64, y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            d2 += (x - y) * (x - y);
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
        let rel = (d2 / na.max(1e-30)).sqrt();
        rows.push((cos, rel, e.name.clone()));
    }
    rows.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    println!("{:<58} {:>8} {:>10}", "tensor (worst cosine first)", "cos", "rel-l2");
    for (cos, rel, name) in rows.iter().take(40) {
        println!("{name:<58} {cos:>8.4} {rel:>10.4}");
    }
    println!("... {} tensors compared", rows.len());
}
