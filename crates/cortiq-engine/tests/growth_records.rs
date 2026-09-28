//! Growth records (`kind = "expert_append"`, spec §2) in the runtime:
//!
//! * the loader mounts a record's experts behind the trunk of every layer
//!   the record declares, in header order, under `CMF_GROWTH` = `active`
//!   (unset) | `all` | `off`; `Resonance.shell` carries `+inf` on trunk
//!   rows and the record's finite `desc.shell` on grown rows; the grown
//!   expert's index is the one its tensor name declares (the chain rule);
//! * the per-op forward of the grown file on tokens outside every shell
//!   is bit-identical to the plain genome's (G2 by construction); with the
//!   shell off (`CMF_GROWTH_SHELL=off`, or the in-process switch
//!   `growth-eval --shell` uses) the grown "magnet" expert wins and the
//!   logits move; `CMF_GROWTH=off` is F0's forward.
//!
//! CPU (`CMF_GPU=0`, set in-process before the first pipeline); every
//! test takes `ENV_LOCK` (the loader reads `CMF_GROWTH`, the router the
//! process-wide shell switch).

#[path = "common/embryo_synth.rs"]
mod embryo_synth;
#[path = "common/growth_synth.rs"]
mod growth_synth;

use cortiq_core::CmfModel;
use cortiq_engine::loader::{GrowthMode, growth_mode, mounted_growth_records};
use cortiq_engine::pipeline::{FfnKind, GrownExpert, MoeFfn, growth_shell_enabled, set_growth_shell};
use cortiq_engine::{Pipeline, SamplerConfig};
use growth_synth::{
    COUNT, E0, FAR_SHELL, GROWN_LAYERS, MAGNET_SHELL, RANK, RECORD_ID, append_record, geom,
    tempdir, write_growth_pair,
};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_guard() -> MutexGuard<'static, ()> {
    let g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: under the lock, before any pipeline of the test is built;
    // every test of this binary asks for the CPU path.
    unsafe {
        std::env::set_var("CMF_GPU", "0");
        std::env::remove_var("CMF_EMBRYO_RESIDENT");
        std::env::remove_var("CMF_GROWTH_SHELL");
    }
    set_growth(None);
    set_growth_shell(None);
    g
}

/// `CMF_GROWTH` for the next loader run (None = unset = `active`).
fn set_growth(mode: Option<&str>) {
    // SAFETY: under ENV_LOCK, before the pipeline that reads it is built.
    unsafe {
        match mode {
            Some(m) => std::env::set_var("CMF_GROWTH", m),
            None => std::env::remove_var("CMF_GROWTH"),
        }
    }
}

fn open(path: &Path) -> Arc<CmfModel> {
    Arc::new(CmfModel::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display())))
}

fn pipeline(model: &Arc<CmfModel>) -> Pipeline {
    Pipeline::from_model(model, SamplerConfig::default()).expect("pipeline")
}

fn moe(p: &Pipeline, layer: usize) -> &MoeFfn {
    match &p.weights.layers[layer].ffn {
        FfnKind::Moe(m) => m,
        _ => panic!("layer {layer} is not an MoE layer"),
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Per-position logits: the prefill of `ids`, then `steps` greedy tokens
/// — the walk `dump-logits` / `growth-eval` record.
fn logits_walk(p: &mut Pipeline, ids: &[u32], steps: usize) -> Vec<Vec<f32>> {
    p.reset_session();
    let mut out = vec![p.forward_ids(ids, None).expect("forward_ids")];
    for s in 0..steps {
        let t = argmax(out.last().unwrap());
        out.push(p.decode_step_logits(t, ids.len() + s));
    }
    out
}

fn bits(v: &[Vec<f32>]) -> Vec<Vec<u32>> {
    v.iter()
        .map(|row| row.iter().map(|x| x.to_bits()).collect())
        .collect()
}

/// Routing counters of `layer`, padded to the expert count.
fn stats(p: &Pipeline, layer: usize) -> Vec<u64> {
    let m = moe(p, layer);
    let mut st = m.stats.borrow().clone();
    st.resize(m.experts.len(), 0);
    st
}

fn grown_wins(p: &Pipeline, layer: usize) -> u64 {
    let m = moe(p, layer);
    stats(p, layer)[m.experts.len() - m.grown.len()..].iter().sum()
}

fn assert_plain_layer(p: &Pipeline, layer: usize) {
    let m = moe(p, layer);
    assert_eq!(m.experts.len(), E0, "layer {layer}: experts");
    assert!(m.grown.is_empty(), "layer {layer}: grown {:?}", m.grown);
    let r = m.resonance.as_ref().expect("resonance");
    assert_eq!(r.shell, vec![f32::INFINITY; E0], "layer {layer}: shell");
    assert!(!r.has_shell());
    assert_eq!(r.bias.len(), E0);
    assert_eq!(r.k, RANK);
}

// ───────────────────────── loader ─────────────────────────

#[test]
fn loader_mounts_expert_append_per_declared_layer_under_each_growth_mode() {
    let _g = env_guard();
    let dir = tempdir("mount");
    let (f0, f1) = write_growth_pair(&dir, "active");
    let h = geom().hidden;

    // F0: every layer plain.
    let p0 = pipeline(&open(&f0));
    for l in 0..geom().layers {
        assert_plain_layer(&p0, l);
    }
    drop(p0);

    // F1, default mode (`active`): layer 0 untouched, the grown layers
    // carry E0 + COUNT experts with the record's descriptors behind the
    // trunk's, the shells +inf on the trunk and the record's on the tail.
    let m1 = open(&f1);
    assert_eq!(growth_mode(), GrowthMode::Active);
    let mounted = mounted_growth_records(&m1.header);
    assert_eq!(mounted.len(), 1);
    assert_eq!(mounted[0].0, 0);
    assert_eq!(mounted[0].1.id, RECORD_ID);
    assert_eq!(m1.arch().moe.as_ref().unwrap().num_experts, E0, "arch keeps E0");
    let p1 = pipeline(&m1);
    assert_plain_layer(&p1, 0);
    for l in GROWN_LAYERS {
        let m = moe(&p1, l);
        assert_eq!(m.experts.len(), E0 + COUNT, "layer {l}: experts");
        assert_eq!(
            m.grown,
            (0..COUNT)
                .map(|k| GrownExpert {
                    record: RECORD_ID.into(),
                    record_index: 0,
                    layer: l,
                    expert: E0 + k,
                })
                .collect::<Vec<_>>(),
            "layer {l}: the declared indices follow the chain rule"
        );
        let r = m.resonance.as_ref().expect("resonance");
        assert_eq!(r.k, RANK);
        assert_eq!(r.shell, vec![f32::INFINITY, f32::INFINITY, FAR_SHELL, MAGNET_SHELL]);
        assert!(r.has_shell());
        assert_eq!(r.bias, vec![0.0; E0 + COUNT]);
        assert_eq!(r.mu.len(), (E0 + COUNT) * h);
        assert_eq!(r.u.len(), (E0 + COUNT) * RANK * h);
        // The magnet copies the trunk expert 0's centre; the far expert
        // sits at 100·1 with an empty subspace.
        assert_eq!(&r.mu[(E0 + 1) * h..(E0 + 2) * h], &r.mu[..h]);
        assert!(r.mu[E0 * h..(E0 + 1) * h].iter().all(|&x| x == 100.0));
        assert!(r.u[E0 * RANK * h..(E0 + 1) * RANK * h].iter().all(|&x| x == 0.0));
        assert_eq!(r.effective_shell(m.experts.len()), r.shell);
        assert_eq!(m.top_k, 1);
        assert_eq!(m.experts[E0].gate_proj.rows(), m.experts[0].gate_proj.rows());
        assert_eq!(m.experts[E0].down_proj.cols(), m.experts[0].down_proj.cols());
    }
    drop(p1);

    // A second record ("spices", quarantine) on layer 2 only: its index
    // is E0 + COUNT there (the chain), whatever its status. The default
    // mode leaves it out; `all` mounts it behind herbs; `off` mounts none.
    let f2 = dir.join("f2.cmf");
    std::fs::copy(&f1, &f2).unwrap();
    append_record(&f2, "spices", &[2], 1, "quarantine");
    let m2 = open(&f2);
    assert_eq!(m2.header.skills.len(), 2);
    assert!(
        m2.tensor(&cortiq_core::knowledge::expert_append_tensor_name(
            "spices",
            2,
            E0 + COUNT,
            cortiq_core::knowledge::expert_leaf::MU
        ))
        .is_some()
    );
    let p2 = pipeline(&m2);
    assert_eq!(moe(&p2, 1).experts.len(), E0 + COUNT);
    assert_eq!(moe(&p2, 2).experts.len(), E0 + COUNT, "quarantine is not mounted by default");
    assert_eq!(moe(&p2, 2).grown.len(), COUNT);
    drop(p2);

    set_growth(Some("all"));
    assert_eq!(growth_mode(), GrowthMode::All);
    assert_eq!(mounted_growth_records(&m2.header).len(), 2);
    let p2 = pipeline(&m2);
    assert_plain_layer(&p2, 0);
    assert_eq!(moe(&p2, 1).experts.len(), E0 + COUNT);
    let m = moe(&p2, 2);
    assert_eq!(m.experts.len(), E0 + COUNT + 1);
    assert_eq!(
        m.grown[COUNT],
        GrownExpert {
            record: "spices".into(),
            record_index: 1,
            layer: 2,
            expert: E0 + COUNT,
        }
    );
    let r = m.resonance.as_ref().unwrap();
    assert_eq!(
        r.shell,
        vec![f32::INFINITY, f32::INFINITY, FAR_SHELL, MAGNET_SHELL, FAR_SHELL]
    );
    assert_eq!(r.mu.len(), (E0 + COUNT + 1) * h);
    assert_eq!(r.u.len(), (E0 + COUNT + 1) * RANK * h);
    drop(p2);

    set_growth(Some("off"));
    assert_eq!(growth_mode(), GrowthMode::Off);
    assert!(mounted_growth_records(&m2.header).is_empty());
    let p2 = pipeline(&m2);
    for l in 0..geom().layers {
        assert_plain_layer(&p2, l);
    }
    drop(p2);
    for (v, want) in [
        ("0", GrowthMode::Off),
        ("none", GrowthMode::Off),
        ("ALL", GrowthMode::All),
        ("active", GrowthMode::Active),
        ("", GrowthMode::Active),
    ] {
        set_growth(Some(v));
        assert_eq!(growth_mode(), want, "CMF_GROWTH={v:?}");
    }
    set_growth(None);
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── forward ─────────────────────────

/// Outside every shell the grown file's per-op logits are F0's bit for
/// bit and no grown expert wins; with the shell off the magnet takes
/// every token the trunk expert 0 took and the logits move; `CMF_GROWTH=off`
/// is F0 again. `CMF_GROWTH_SHELL=off` (the environment) and the
/// in-process switch agree.
#[test]
fn per_op_logits_outside_every_shell_equal_f0_and_the_shell_switch_moves_them() {
    let _g = env_guard();
    let dir = tempdir("forward");
    let (f0, f1) = write_growth_pair(&dir, "active");
    let vocab = geom().vocab;
    let ids = embryo_synth::synth_ids(24, 11, vocab);
    let steps = 6;
    let positions = (ids.len() + steps) as u64;

    let m0 = open(&f0);
    let mut p0 = pipeline(&m0);
    let reference = logits_walk(&mut p0, &ids, steps);
    assert_eq!(reference.len(), steps + 1);
    assert!(reference.iter().flatten().all(|x| x.is_finite()));
    let trunk0_wins: Vec<u64> = GROWN_LAYERS.iter().map(|&l| stats(&p0, l)[0]).collect();
    for (l, w) in GROWN_LAYERS.iter().zip(&trunk0_wins) {
        assert_eq!(stats(&p0, *l).iter().sum::<u64>(), positions, "layer {l}: routed tokens");
        eprintln!("F0 layer {l}: expert 0 won {w} of {positions} tokens");
    }
    assert!(
        trunk0_wins.iter().any(|&w| w > 0),
        "the fixture needs the trunk expert 0 to win somewhere: {trunk0_wins:?}"
    );
    drop(p0);

    // Shell on (the default): invisible growth.
    let m1 = open(&f1);
    assert!(growth_shell_enabled());
    let mut p1 = pipeline(&m1);
    let on = logits_walk(&mut p1, &ids, steps);
    assert_eq!(bits(&on), bits(&reference), "shell on: F1 == F0 bit for bit");
    for l in GROWN_LAYERS {
        let st = stats(&p1, l);
        assert_eq!(st.iter().sum::<u64>(), positions);
        assert_eq!(grown_wins(&p1, l), 0, "layer {l}: grown wins with the shell on: {st:?}");
    }
    drop(p1);

    // Shell off (the in-process switch `growth-eval --shell off` uses):
    // the far expert still never wins, the magnet takes every token the
    // trunk expert 0 took (and possibly more), the logits change.
    set_growth_shell(Some(false));
    assert!(!growth_shell_enabled());
    let mut p1 = pipeline(&m1);
    let off = logits_walk(&mut p1, &ids, steps);
    let mut total_grown = 0;
    for (l, &w0) in GROWN_LAYERS.iter().zip(&trunk0_wins) {
        let st = stats(&p1, *l);
        eprintln!("F1 shell off layer {l}: routing {st:?}");
        assert_eq!(st.iter().sum::<u64>(), positions);
        assert_eq!(st[E0], 0, "layer {l}: the far expert won");
        assert!(st[E0 + 1] >= w0, "layer {l}: magnet {} < trunk expert 0's F0 wins {w0}", st[E0 + 1]);
        total_grown += grown_wins(&p1, *l);
    }
    assert!(total_grown > 0);
    // The first grown layer's routing changed at the first position where
    // the magnet won, so the F1 forward is no longer F0's.
    assert_ne!(bits(&off), bits(&reference), "shell off: the magnet changes the forward");
    let moved = off
        .iter()
        .zip(&reference)
        .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max))
        .fold(0.0f32, f32::max);
    assert!(moved.is_finite() && moved > 0.0);
    eprintln!("shell off: max|Δ| vs F0 = {moved:.3e}, grown wins {total_grown}");
    drop(p1);

    // The environment switch says the same as the in-process one.
    set_growth_shell(None);
    // SAFETY: under ENV_LOCK, before the pipeline that reads it.
    unsafe { std::env::set_var("CMF_GROWTH_SHELL", "off") };
    assert!(!growth_shell_enabled());
    let mut p1 = pipeline(&m1);
    let off_env = logits_walk(&mut p1, &ids, steps);
    assert_eq!(bits(&off_env), bits(&off));
    drop(p1);
    unsafe { std::env::remove_var("CMF_GROWTH_SHELL") };
    set_growth_shell(None);
    assert!(growth_shell_enabled());

    // Shell back on through the environment: F0 again.
    let mut p1 = pipeline(&m1);
    assert_eq!(bits(&logits_walk(&mut p1, &ids, steps)), bits(&reference));
    drop(p1);

    // CMF_GROWTH=off: the plain genome's forward on the grown file.
    set_growth(Some("off"));
    let mut p1 = pipeline(&m1);
    for l in 0..geom().layers {
        assert_plain_layer(&p1, l);
    }
    assert_eq!(bits(&logits_walk(&mut p1, &ids, steps)), bits(&reference));
    set_growth(None);
    std::fs::remove_dir_all(dir).ok();
}
