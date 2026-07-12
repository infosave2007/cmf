//! Nyström kernel golden parity: the f32 streaming kernel vs the
//! validated fp64 matrix math (nystrom_ppl3_06b.py head_out, transposed
//! to streaming semantics).  Fixture: tools/gen_nystrom_golden.py.

use cortiq_engine::nystrom::NystromState;

fn rows(v: &serde_json::Value) -> Vec<Vec<f32>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

fn flat(rows: &[Vec<f32>], range: std::ops::Range<usize>) -> Vec<f32> {
    rows[range].iter().flatten().copied().collect()
}

/// Run the kernel over the golden decode range against the fixture
/// output set named `out_key`, with `sink` permanent exact keys.
fn run_golden(sink: usize, out_key: &str) {
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("data/nystrom_golden.json")).unwrap();
    let g = |k: &str| fx[k].as_u64().unwrap() as usize;
    let (d, dv, m, w, t, p) = (g("d"), g("dv"), g("m"), g("w"), g("t"), g("p"));
    let q = rows(&fx["q"]);
    let k = rows(&fx["k"]);
    let v = rows(&fx["v"]);
    let expect = rows(&fx[out_key]);

    let mut st = NystromState::new(m, w, sink);
    st.prefill(&flat(&q, 0..p), &flat(&k, 0..p), &flat(&v, 0..p), p, d, dv);

    let mut out = vec![0.0f32; dv];
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut cnt = 0usize;
    for i in p..t {
        st.step(&q[i], &k[i], &v[i], &mut out);
        for (a, b) in out.iter().zip(&expect[i]) {
            let e = (*a as f64 - *b as f64).abs();
            max_abs = max_abs.max(e);
            sum_abs += e;
            cnt += 1;
        }
    }
    let mean_abs = sum_abs / cnt as f64;
    println!("nystrom parity (sink={sink}): max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
    assert!(max_abs < 1e-3, "max abs error {max_abs:.3e} >= 1e-3");
    assert!(mean_abs < 1e-4, "mean abs error {mean_abs:.3e} >= 1e-4");
}

#[test]
fn nystrom_matches_python_golden() {
    // Default operating mode: sink tokens on (spec §5b).
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("data/nystrom_golden.json")).unwrap();
    let sink = fx["sink"].as_u64().unwrap() as usize;
    assert!(sink > 0, "fixture must carry a nonzero sink count");
    run_golden(sink, "out");
}

#[test]
fn nystrom_sink0_matches_legacy_golden() {
    // Regression: sink = 0 must reproduce the original sink-free
    // kernel against the sink-free golden outputs.
    run_golden(0, "out_sink0");
}

/// Deterministic pseudo-random f32 in [-1, 1] (LCG) — the short test
/// needs no fixture.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Plain causal softmax over all positions, f64 — the exact-only
/// reference.
fn softmax_row(q: &[f32], ks: &[Vec<f32>], vs: &[Vec<f32>], upto: usize, dv: usize) -> Vec<f32> {
    let d = q.len();
    let scale = 1.0 / (d as f64).sqrt();
    let lg: Vec<f64> = (0..=upto)
        .map(|j| {
            q.iter()
                .zip(&ks[j])
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>()
                * scale
        })
        .collect();
    let c = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ws: Vec<f64> = lg.iter().map(|x| (x - c).exp()).collect();
    let den: f64 = ws.iter().sum();
    let mut out = vec![0.0f32; dv];
    for (j, wj) in ws.iter().enumerate() {
        for (o, &x) in out.iter_mut().zip(&vs[j]) {
            *o += (wj / den * x as f64) as f32;
        }
    }
    out
}

#[test]
fn nystrom_short_prompt_is_exact_softmax() {
    // t = 20 < w = 32 → exact-only mode: no skeleton, and the window
    // still covers every position, so the kernel must equal plain
    // causal softmax attention (sink setting is irrelevant here — every
    // key is already permanent-exact).
    let (d, dv, m, w, sink, t, p) =
        (16usize, 8usize, 4usize, 32usize, 4usize, 20usize, 12usize);
    let mut rng = Lcg(20260712);
    let q: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(d)).collect();
    let k: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(d)).collect();
    let v: Vec<Vec<f32>> = (0..t).map(|_| rng.vec(dv)).collect();

    let mut st = NystromState::new(m, w, sink);
    let fl = |rows: &[Vec<f32>], hi: usize| -> Vec<f32> {
        rows[..hi].iter().flatten().copied().collect()
    };
    st.prefill(&fl(&q, p), &fl(&k, p), &fl(&v, p), p, d, dv);

    let mut out = vec![0.0f32; dv];
    let mut max_abs = 0.0f32;
    for i in p..t {
        st.step(&q[i], &k[i], &v[i], &mut out);
        let expect = softmax_row(&q[i], &k, &v, i, dv);
        for (a, b) in out.iter().zip(&expect) {
            max_abs = max_abs.max((a - b).abs());
        }
    }
    println!("nystrom exact-only parity: max_abs={max_abs:.3e}");
    assert!(max_abs < 1e-5, "exact-only max abs error {max_abs:.3e} >= 1e-5");
}
