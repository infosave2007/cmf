//! Resident wgpu graph ↔ per-op parity on a grown file (spec §2): the
//! graph's resonance kernels apply the growth shell exactly as the host
//! `Resonance::scores` does (`−∞` outside the shell, record word 30 + the
//! `−∞` sentinel), for both shell modes — so the graph routes to the
//! same experts as the host and its logits match the per-op path within
//! the resident tolerance, while the two shell modes visibly differ.
//!
//! Opt-in: runs only under `CMF_GPU=wgpu` (a Vulkan device); the wgpu
//! context is process-global, so this binary holds one test. The CPU
//! contracts live in `growth_records.rs`.

#![cfg(feature = "gpu")]

#[path = "common/embryo_synth.rs"]
mod embryo_synth;
#[path = "common/growth_synth.rs"]
mod growth_synth;

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::{FfnKind, set_growth_shell};
use cortiq_engine::{Pipeline, SamplerConfig};
use growth_synth::{E0, GROWN_LAYERS, geom, tempdir, write_growth_pair};
use std::sync::Arc;

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn logits_walk(p: &mut Pipeline, ids: &[u32], steps: usize) -> Vec<Vec<f32>> {
    p.reset_session();
    let mut out = vec![p.forward_ids(ids, None).expect("forward_ids")];
    for s in 0..steps {
        let t = argmax(out.last().unwrap());
        out.push(p.decode_step_logits(t, ids.len() + s));
    }
    out
}

fn max_diff(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert_eq!(x.len(), y.len());
            x.iter()
                .zip(y)
                .map(|(p, q)| {
                    let d = (p - q).abs();
                    assert!(d.is_finite(), "non-finite logit: {p} vs {q}");
                    d
                })
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max)
}

fn grown_wins(p: &Pipeline, layer: usize) -> u64 {
    match &p.weights.layers[layer].ffn {
        FfnKind::Moe(m) => {
            let st = m.stats.borrow();
            st.iter().skip(E0).sum()
        }
        _ => panic!("layer {layer} is not MoE"),
    }
}

#[test]
fn resident_graph_matches_per_op_on_a_grown_file_in_both_shell_modes() {
    if std::env::var("CMF_GPU").as_deref() != Ok("wgpu") {
        eprintln!("growth graph parity skipped: set CMF_GPU=wgpu (a Vulkan device)");
        return;
    }
    // SAFETY: before any pipeline of this process exists.
    unsafe {
        std::env::set_var("CMF_GPU_PROBE", "0");
        std::env::set_var("CMF_GPU_WGPU_GRAPH", "1");
        std::env::set_var("CMF_EMBRYO_RESIDENT", "parallel");
        std::env::remove_var("CMF_GROWTH");
        std::env::remove_var("CMF_GROWTH_SHELL");
    }
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();
    let dir = tempdir("graph");
    let (f0, f1) = write_growth_pair(&dir, "active");
    let ids = embryo_synth::synth_ids(24, 11, geom().vocab);
    let steps = 6;
    let tol = std::env::var("CMF_EMBRYO_PARITY_TOL")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(2e-3);
    let cfg = SamplerConfig::default;
    let m1 = Arc::new(CmfModel::open(&f1).expect("open F1"));

    let mut per_mode = Vec::new();
    for shell_on in [true, false] {
        set_growth_shell(Some(shell_on));
        // The graph is packed at the first forward with the shell as it
        // is switched NOW (word 30).
        let mut graph = Pipeline::from_model(&m1, cfg()).expect("graph pipeline");
        let g = logits_walk(&mut graph, &ids, steps);
        assert!(
            !graph.graph_refused(),
            "shell {shell_on}: the resident graph refused the grown file (RUST_LOG=warn shows why)"
        );
        assert_eq!(
            graph.device_sequence_position(),
            Some(ids.len() + steps),
            "shell {shell_on}: the device does not hold the sequence — the graph did not run"
        );
        let mut perop = Pipeline::from_model(&m1, cfg()).expect("per-op pipeline");
        perop.mark_graph_refused();
        let o = logits_walk(&mut perop, &ids, steps);
        let d = max_diff(&g, &o);
        let wins: u64 = GROWN_LAYERS.iter().map(|&l| grown_wins(&perop, l)).sum();
        eprintln!("growth graph parity: shell {shell_on}: graph vs per-op max|Δ|={d:.3e} (tol {tol:.1e}), per-op grown wins {wins}");
        assert!(d <= tol, "shell {shell_on}: graph diverged from per-op: {d} > {tol}");
        if shell_on {
            assert_eq!(wins, 0, "shell on: a grown expert won on the host");
        } else {
            assert!(wins > 0, "shell off: the magnet never won on the host");
        }
        per_mode.push((g, o));
    }
    set_growth_shell(None);

    // The two modes route differently, and the graph shows it: the
    // shell-on graph is F0's forward, the shell-off graph is not.
    let (on_graph, _) = &per_mode[0];
    let (off_graph, off_perop) = &per_mode[1];
    let split = max_diff(on_graph, off_perop);
    assert!(
        split > 10.0 * tol,
        "the shell made no visible difference on the graph: max|Δ| on-vs-off = {split:.3e}"
    );
    let m0 = Arc::new(CmfModel::open(&f0).expect("open F0"));
    let mut p0 = Pipeline::from_model(&m0, cfg()).expect("F0 pipeline");
    p0.mark_graph_refused();
    let f0_logits = logits_walk(&mut p0, &ids, steps);
    let d0 = max_diff(on_graph, &f0_logits);
    assert!(d0 <= tol, "shell on: the graph is not F0's forward: {d0} > {tol}");
    let d1 = max_diff(off_graph, &f0_logits);
    assert!(d1 > 10.0 * tol, "shell off: the graph still looks like F0: {d1:.3e}");
    eprintln!("growth graph parity: on-graph vs F0 {d0:.3e}, off-graph vs F0 {d1:.3e}");
    std::fs::remove_dir_all(dir).ok();
}
