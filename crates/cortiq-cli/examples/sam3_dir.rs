use cortiq_core::CmfModel;

fn main() {
    let path = std::env::args().nth(1).expect("cmf path");
    let prefix = std::env::args().nth(2).unwrap_or_default();
    let m = CmfModel::open(&path).expect("open");
    let mut names: Vec<String> = m
        .tensors
        .iter()
        .map(|t| format!("{} {:?} {:?}", t.name, t.dtype, t.shape))
        .collect();
    names.sort();
    for n in names.iter().filter(|n| n.starts_with(&prefix)) {
        println!("{n}");
    }
    println!("--- total {} tensors", m.tensors.len());
    for t in m.tensors.iter().filter(|t| t.name.starts_with("cfg.")) {
        let b = m.entry_bytes(t);
        let s = String::from_utf8_lossy(b);
        println!("=== {} ===", s);
    }
    if let (Some(name), Some(dst)) = (std::env::args().nth(3), std::env::args().nth(4)) {
        use cortiq_core::quant::dequant_tensor;
        let e = m.tensor(&name).unwrap_or_else(|| panic!("no tensor {name}"));
        let bytes = m.entry_bytes(e);
        let mut vals = vec![0f32; e.n_elems()];
        dequant_tensor(e, bytes, &mut vals).unwrap();
        let n = vals.len();
        let mut out = Vec::with_capacity(n * 4);
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&dst, out).unwrap();
        eprintln!("dumped {name} -> {dst} ({n} elems)");
    }
}