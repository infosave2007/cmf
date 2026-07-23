//! GatedDeltaNet golden parity: the Rust core vs the validated numpy
//! oracle (vmfcore/gdn_layer.py). Fixture: tests/gen_gdn_golden.py.

use cortiq_engine::linear_core::{GdnCfg, GdnWeights, gdn_forward};
use cortiq_engine::qtensor::QTensor;

fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn gdn_matches_numpy_oracle() {
    let fx: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gdn_golden.json")).unwrap();
    let c = &fx["cfg"];
    let g = |k: &str| c[k].as_u64().unwrap() as usize;
    let cfg = GdnCfg {
        num_v_heads: g("num_v_heads"),
        num_k_heads: g("num_k_heads"),
        key_head_dim: g("key_head_dim"),
        value_head_dim: g("value_head_dim"),
        conv_kernel: g("conv_kernel"),
        hidden_size: g("hidden_size"),
        rms_eps: c["rms_eps"].as_f64().unwrap(),
    };
    let (c_dim, vd, h) = (
        cfg.conv_dim(),
        cfg.num_v_heads * cfg.value_head_dim,
        cfg.hidden_size,
    );
    let wj = &fx["weights"];
    let mat = |k: &str, rows: usize, cols: usize| QTensor::from_f32(f32s(&wj[k]), rows, cols);
    let w = GdnWeights {
        in_proj_qkv: mat("in_proj_qkv", c_dim, h),
        in_proj_z: mat("in_proj_z", vd, h),
        in_proj_a: mat("in_proj_a", cfg.num_v_heads, h),
        in_proj_b: mat("in_proj_b", cfg.num_v_heads, h),
        conv1d: f32s(&wj["conv1d"]),
        a_log: f32s(&wj["A_log"]),
        dt_bias: f32s(&wj["dt_bias"]),
        norm: f32s(&wj["norm"]),
        out_proj: mat("out_proj", h, vd),
    };

    let xs: Vec<Vec<f32>> = fx["x"].as_array().unwrap().iter().map(f32s).collect();
    let es: Vec<Vec<f32>> = fx["expect"].as_array().unwrap().iter().map(f32s).collect();

    let mut state = Vec::new();
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for (t, (x, e)) in xs.iter().zip(&es).enumerate() {
        let o = gdn_forward(x, &w, &cfg, &mut state, None);
        let scale = e.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-9);
        for (a, b) in o.iter().zip(e) {
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / scale);
        }
        assert!(
            max_rel < 1e-3,
            "position {t}: max_rel {max_rel:.2e} (abs {max_abs:.2e})"
        );
    }
    println!("gdn oracle parity: max_rel={max_rel:.2e} max_abs={max_abs:.2e}");
}
