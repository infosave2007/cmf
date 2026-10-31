//! `grow-trunk` preserves the function: a GDN + bounded-anchor genome with
//! experts, grown wider (hidden 64 → 128, inter 128 → 256) and twice as deep,
//! gives the same per-position NLL as its source. Then one training step on
//! the grown genome moves the new channels off zero (none of them is dead).
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::metal::ctx;
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, Mixer, init_params};
use cortiq_embryo::ops::lcg_vec;
use cortiq_embryo::train::Checkpoint;
use cortiq_embryo::trunk::{TrunkGrowth, depth_map, grow_trunk};

fn cfg() -> EmbryoCfg {
    let mut c = EmbryoCfg::tiny();
    c.experts = 4;
    c.anchor_window = 16;
    c.anchor_sink = 2;
    c.mixer = Mixer::Gdn;
    c.gdn_heads = 2;
    c.gdn_dk = 32;
    c.gdn_dv = 32;
    c.layers = 4;
    c.anchor_layers = Some(vec![1, 3]);
    c
}

fn tokens(c: &EmbryoCfg, seed: u64, n: usize) -> Vec<u32> {
    lcg_vec(seed, n)
        .iter()
        .map(|x| ((x * 0.5 + 0.5) * c.vocab as f32) as u32 % c.vocab as u32)
        .collect()
}

#[test]
fn depth_map_doubles_every_anchor_block() {
    let mut c = cfg();
    c.layers = 8;
    c.anchor_layers = Some(vec![3, 7]);
    let m = depth_map(&c, true);
    let src: Vec<usize> = m.iter().map(|x| x.0).collect();
    assert_eq!(src, [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]);
    let anchors: Vec<usize> = (0..m.len()).filter(|&l| c.is_anchor(m[l].0)).collect();
    assert_eq!(anchors, [3, 7, 11, 15]);
    assert!(m[4..8].iter().all(|x| x.1) && m[..4].iter().all(|x| !x.1));
}

#[test]
fn grown_trunk_computes_the_same_nll() {
    let Some(_) = ctx() else { return };
    let c0 = cfg();
    let (b, t) = (2usize, 64usize);
    let p0 = init_params(&c0, &Layout::new(&c0), 51);
    let src = EmbryoGpu::new_eval_dropless(c0.clone(), b, t, &p0).expect("source model");
    src.desc_updates.set(false);
    let x = tokens(&c0, 5101, b * t);
    let y = tokens(&c0, 5102, b * t);
    let l0 = src.eval_loss(&x, &y);
    let per0 = src.per_position_loss();
    let extras: Vec<(String, Vec<f32>)> = src.desc_host().into_iter().map(|(n, v)| (n.to_string(), v)).collect();
    let ck = Checkpoint {
        cfg: c0.clone(),
        step: 7,
        params: p0,
        m: None,
        v: None,
        extras,
    };
    for (hidden, inter, double) in [(128, 256, false), (64, 128, true), (128, 256, true)] {
        let g = grow_trunk(
            &ck,
            &TrunkGrowth {
                hidden,
                inter,
                double_depth: double,
                seed: 3,
            },
        )
        .expect("grow");
        assert_eq!(g.params.len(), Layout::new(&g.cfg).total);
        let m = EmbryoGpu::new_eval_dropless(g.cfg.clone(), b, t, &g.params).expect("grown model");
        m.desc_updates.set(false);
        m.set_desc(&g.extras);
        let l1 = m.eval_loss(&x, &y);
        let per1 = m.per_position_loss();
        let worst = per0
            .iter()
            .zip(&per1)
            .map(|(a, b)| ((a - b).abs() / a.abs().max(1e-6)) as f64)
            .fold(0.0f64, f64::max);
        eprintln!(
            "grown hidden {hidden} inter {inter} double {double} ({} layers): loss {l0:.6} → {l1:.6}, \
             per-position rel {worst:.2e}",
            g.cfg.layers
        );
        assert!(worst <= 2e-5, "grown genome computes a different NLL: {worst:e}");
    }
}

#[test]
fn new_channels_wake_up_after_one_step() {
    let Some(_) = ctx() else { return };
    let c0 = cfg();
    let (b, t) = (2usize, 64usize);
    let p0 = init_params(&c0, &Layout::new(&c0), 52);
    let ck = Checkpoint {
        cfg: c0.clone(),
        step: 0,
        params: p0,
        m: None,
        v: None,
        extras: Vec::new(),
    };
    let g = grow_trunk(
        &ck,
        &TrunkGrowth {
            hidden: 128,
            inter: 256,
            double_depth: true,
            seed: 4,
        },
    )
    .expect("grow");
    let lay = Layout::new(&g.cfg);
    let mut m = EmbryoGpu::new(g.cfg.clone(), b, t, &g.params).expect("grown model");
    let x = tokens(&c0, 5201, b * t);
    let y = tokens(&c0, 5202, b * t);
    for _ in 0..2 {
        m.train_step(&x, &y, 1e-3, 0.0, 1.0);
    }
    let p = m.params_host();
    let h0 = c0.hidden;
    // new output rows of an old layer, the output projection of an inserted
    // copy, and the new embedding columns all moved off zero
    let get = |name: &str| {
        let (_, o, n) = lay.names.iter().find(|x| x.0 == name).expect(name);
        p[*o..*o + *n].to_vec()
    };
    let o0 = get("layers.0.gdn.o");
    let new_rows = o0[h0 * (o0.len() / 128)..].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let copy_o = get("layers.2.gdn.o").iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let e = get("embed");
    let new_cols = e.chunks_exact(128).map(|r| r[h0..].iter().map(|v| v.abs()).fold(0.0f32, f32::max)).fold(0.0f32, f32::max);
    eprintln!("after 2 steps: new o rows {new_rows:.2e}, copied layer o {copy_o:.2e}, new embed columns {new_cols:.2e}");
    assert!(new_rows > 0.0 && copy_o > 0.0 && new_cols > 0.0, "a grown part received no gradient");
}
