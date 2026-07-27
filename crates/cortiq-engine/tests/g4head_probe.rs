#[test]
fn lm_head_matvec_vs_matmat_b32() {
    let p = "/Users/oleg/Documents/cortiq-bot/models/g4moe-q4t.cmf";
    if !std::path::Path::new(p).exists() { return; }
    let m = std::sync::Arc::new(cortiq_core::CmfModel::open(p).unwrap());
    let pl = cortiq_engine::Pipeline::from_model(&m, cortiq_engine::SamplerConfig::default()).unwrap();
    let lm = &pl.weights.lm_head;
    let rows = lm.rows();
    let hs = pl.hidden_size;
    let b = 32usize;
    let mut hh = vec![0.0f32; b * hs];
    for bi in 0..b {
        for i in 0..hs {
            hh[bi * hs + i] = (((bi * hs + i) as f32 * 12.9898).sin() * 43758.547).fract() - 0.5;
        }
    }
    let mut gemm = vec![0.0f32; b * rows];
    lm.matmat(&hh, b, &mut gemm, None);
    let mut nbad = 0usize;
    let mut worst = (0usize, 0usize, 0.0f32, 0.0f32, 0.0f32);
    for bi in 0..b {
        let mut mv = vec![0.0f32; rows];
        lm.matvec(&hh[bi * hs..(bi + 1) * hs], &mut mv, None);
        for r in 0..rows {
            let g = gemm[bi * rows + r];
            let d = (mv[r] - g).abs();
            let rel = d / mv[r].abs().max(1e-2);
            if rel > 0.01 { nbad += 1; }
            if rel > worst.4 { worst = (bi, r, mv[r], g, rel); }
        }
        if bi == 0 {
            eprintln!("row0 sample: mv {:.4} gemm {:.4}", mv[0], gemm[0]);
        }
    }
    eprintln!("bad(rel>1%): {nbad} of {}; worst bi {} row {} mv {:.4} gemm {:.4} rel {:.4}",
        b * rows, worst.0, worst.1, worst.2, worst.3, worst.4);
}
