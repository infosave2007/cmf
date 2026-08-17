//! Metal batched q4t GEMM (q4t_mul_mm — tiles decoded in the K loop)
//! vs an exact f64 dequant reference. Half shared-memory tiles inside
//! the GEMM make it tolerance-class — the bound is loose but honest.
//!
//!     cargo test -p cortiq-engine --release --test gpu_q4t_mm -- --nocapture
#![cfg(target_os = "macos")]

use cortiq_core::quant::{GROUP_SIZE, f16_to_f32, f32_to_f16};
use cortiq_core::*;

fn require_metal() -> bool {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    if cortiq_engine::gpu_metal::enabled() {
        true
    } else {
        eprintln!(
            "gpu q4t mm test skipped: {}",
            cortiq_engine::gpu_metal::initialization_error().unwrap_or("Metal disabled")
        );
        false
    }
}

#[test]
fn gpu_q4t_matmat_matches_dequant_reference() {
    if !require_metal() {
        return;
    }
    let (rows, cols, b) = (512usize, 1024usize, 64usize);
    let gpr = cols / GROUP_SIZE;
    const TILE: usize = 18;
    let mut payload = vec![0u8; rows * gpr * TILE];
    for t in 0..rows * gpr {
        let sc = 0.02 + 0.0005 * (t % 64) as f32;
        payload[t * TILE..t * TILE + 2].copy_from_slice(&f32_to_f16(sc).to_le_bytes());
        for k in 0..16 {
            payload[t * TILE + 2 + k] = ((t * 31 + k * 13) % 251) as u8;
        }
    }
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "tiny",
        "hidden_size": cols,
        "intermediate_size": cols * 2,
        "num_layers": 1,
        "num_attention_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "vocab_size": rows,
        "layer_types": ["FullAttention"],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 8,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))
    .unwrap();
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type: QuantType::Q4Block,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    let spec = TensorSpec {
        name: "w".into(),
        dtype: TensorDtype::Q4Tiled,
        shape: vec![rows, cols],
        data: payload.clone(),
    };
    // Tail pad: the no-copy file window rounds DOWN to a page.
    let pad = TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0u8; 8192 * 8],
    };
    let dir = std::env::temp_dir().join(format!("cmf-q4tmm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();

    let x: Vec<f32> = (0..b * cols)
        .map(|i| ((i * 13 + 7) % 97) as f32 / 97.0 - 0.5)
        .collect();
    let mut got = vec![0f32; b * rows];
    assert!(
        cortiq_engine::gpu::q4t_matmat(&model, idx, &x, b, rows, cols, &mut got),
        "gpu q4t_matmat refused"
    );

    // Exact reference: f64 matmul over the dequantized tiles.
    let mut max_rel = 0f64;
    for bi in 0..b {
        for r in 0..rows {
            let mut want = 0f64;
            for g in 0..gpr {
                let t = (r * gpr + g) * TILE;
                let s = f16_to_f32(u16::from_le_bytes([payload[t], payload[t + 1]])) as f64;
                for (k, &bb) in payload[t + 2..t + TILE].iter().enumerate() {
                    let w0 = ((bb & 0x0F) as f64 - 8.0) * s;
                    let w1 = (((bb >> 4) & 0x0F) as f64 - 8.0) * s;
                    want += w0 * x[bi * cols + g * GROUP_SIZE + 2 * k] as f64;
                    want += w1 * x[bi * cols + g * GROUP_SIZE + 2 * k + 1] as f64;
                }
            }
            let d = (got[bi * rows + r] as f64 - want).abs();
            max_rel = max_rel.max(d / want.abs().max(1.0));
        }
    }
    println!("gpu q4t GEMM max rel dev {max_rel:.2e}");
    assert!(max_rel < 2e-2, "gpu q4t GEMM diverged: {max_rel}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// q4t payload: f16 scale + 16 nibble bytes per 32-weight group,
/// deterministic per (seed, tile).
fn q4t_payload(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    const TILE: usize = 18;
    let gpr = cols / GROUP_SIZE;
    let mut p = vec![0u8; rows * gpr * TILE];
    for t in 0..rows * gpr {
        let sc = 0.02 + 0.0004 * ((t * 7 + seed) % 64) as f32;
        p[t * TILE..t * TILE + 2].copy_from_slice(&f32_to_f16(sc).to_le_bytes());
        for k in 0..16 {
            p[t * TILE + 2 + k] = ((t * 31 + k * 13 + seed * 97) % 251) as u8;
        }
    }
    p
}

/// Exact f64 y[b, rows] = X · dequant(W)ᵀ over a q4t payload.
fn ref_matmat(payload: &[u8], x: &[f64], b: usize, rows: usize, cols: usize) -> Vec<f64> {
    const TILE: usize = 18;
    let gpr = cols / GROUP_SIZE;
    let mut y = vec![0f64; b * rows];
    for bi in 0..b {
        for r in 0..rows {
            let mut acc = 0f64;
            for g in 0..gpr {
                let t = (r * gpr + g) * TILE;
                let s = f16_to_f32(u16::from_le_bytes([payload[t], payload[t + 1]])) as f64;
                for (k, &bb) in payload[t + 2..t + TILE].iter().enumerate() {
                    let w0 = ((bb & 0x0F) as f64 - 8.0) * s;
                    let w1 = (((bb >> 4) & 0x0F) as f64 - 8.0) * s;
                    acc += w0 * x[bi * cols + g * GROUP_SIZE + 2 * k];
                    acc += w1 * x[bi * cols + g * GROUP_SIZE + 2 * k + 1];
                }
            }
            y[bi * rows + r] = acc;
        }
    }
    y
}

#[test]
fn gpu_q4t_ffn_matches_dequant_reference() {
    if !require_metal() {
        return;
    }
    let (hidden, inter, b) = (256usize, 512usize, 48usize);
    let p1 = q4t_payload(inter, hidden, 1);
    let p3 = q4t_payload(inter, hidden, 3);
    let p2 = q4t_payload(hidden, inter, 2);
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "tiny",
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_layers": 1,
        "num_attention_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 4,
        "vocab_size": inter,
        "layer_types": ["FullAttention"],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 8,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0,
    }))
    .unwrap();
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type: QuantType::Q4Block,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    let spec = |name: &str, rows: usize, cols: usize, data: &[u8]| TensorSpec {
        name: name.into(),
        dtype: TensorDtype::Q4Tiled,
        shape: vec![rows, cols],
        data: data.to_vec(),
    };
    let pad = TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0u8; 8192 * 8],
    };
    let dir = std::env::temp_dir().join(format!("cmf-q4tffn-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(
        &path,
        &header,
        &[
            spec("w1", inter, hidden, &p1),
            spec("w3", inter, hidden, &p3),
            spec("w2", hidden, inter, &p2),
            pad,
        ],
        None,
        None,
    )
    .unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let (i1, i3, i2) = (
        model.tensor_index("w1").unwrap(),
        model.tensor_index("w3").unwrap(),
        model.tensor_index("w2").unwrap(),
    );

    let x: Vec<f32> = (0..b * hidden)
        .map(|i| ((i * 17 + 5) % 89) as f32 / 89.0 - 0.5)
        .collect();
    let mut got = vec![0f32; b * hidden];
    assert!(
        cortiq_engine::gpu::q4t_ffn(&model, i1, i3, i2, &x, b, hidden, inter, &mut got),
        "gpu q4t_ffn refused"
    );

    let xf: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let g = ref_matmat(&p1, &xf, b, inter, hidden);
    let u = ref_matmat(&p3, &xf, b, inter, hidden);
    let act: Vec<f64> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| gv / (1.0 + (-gv).exp()) * uv)
        .collect();
    let want = ref_matmat(&p2, &act, b, hidden, inter);
    let mut max_rel = 0f64;
    for (i, &w) in want.iter().enumerate() {
        let d = (got[i] as f64 - w).abs();
        max_rel = max_rel.max(d / w.abs().max(1.0));
    }
    println!("gpu q4t FFN max rel dev {max_rel:.2e}");
    assert!(max_rel < 3e-2, "gpu q4t FFN diverged: {max_rel}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn gpu_vae_conv2d_matches_cpu() {
    if !require_metal() {
        return;
    }
    // Odd everything: K-tail (ic·k² % 32 ≠ 0), edge oc/position tiles,
    // borders. k=3 and the 1×1 shortcut case.
    for (ic, oc, h, w, k) in [
        (20usize, 70usize, 13usize, 17usize, 3usize),
        (24, 40, 9, 11, 1),
    ] {
        let mk = |seed: usize, len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 37 + seed * 11 + 3) % 101) as f32 / 101.0 - 0.5)
                .collect()
        };
        let conv = cortiq_engine::vae::Conv2d {
            w: mk(1, oc * ic * k * k),
            b: mk(2, oc),
            oc,
            ic,
            k,
        };
        let x = mk(3, ic * h * w);
        let want = conv.apply(&x, h, w); // small shape → CPU path
        let mut got = vec![0f32; oc * h * w];
        assert!(
            cortiq_engine::gpu::vae_conv2d(&conv.w, &conv.b, &x, ic, oc, h, w, k, &mut got),
            "gpu vae_conv2d refused (k={k})"
        );
        let mut max_rel = 0f64;
        for (g, wv) in got.iter().zip(&want) {
            let d = (*g as f64 - *wv as f64).abs();
            max_rel = max_rel.max(d / (*wv as f64).abs().max(1.0));
        }
        println!("gpu vae conv k={k} max rel dev {max_rel:.2e}");
        assert!(max_rel < 2e-2, "gpu vae conv k={k} diverged: {max_rel}");
    }
}

#[test]
fn gpu_vae_resnet_matches_cpu() {
    if !require_metal() {
        return;
    }
    let (ic, oc, h, w, groups) = (32usize, 64usize, 9usize, 7usize, 8usize);
    let mk = |seed: usize, len: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 41 + seed * 13 + 5) % 89) as f32 / 89.0 - 0.5)
            .collect()
    };
    let conv = |seed: usize, ic: usize, oc: usize, k: usize| cortiq_engine::vae::Conv2d {
        w: mk(seed, oc * ic * k * k),
        b: mk(seed + 1, oc),
        oc,
        ic,
        k,
    };
    let gn = |seed: usize, c: usize| cortiq_engine::vae::GroupNorm {
        g: groups,
        w: mk(seed, c).iter().map(|v| v + 1.0).collect(),
        b: mk(seed + 1, c),
    };
    let (n1, c1) = (gn(10, ic), conv(20, ic, oc, 3));
    let (n2, c2) = (gn(30, oc), conv(40, oc, oc, 3));
    let sc = conv(50, ic, oc, 1);
    let x = mk(60, ic * h * w);

    // CPU reference: norm+silu → conv ×2 → shortcut → add.
    let silu_v = |t: &mut Vec<f32>| {
        for v in t.iter_mut() {
            *v /= 1.0 + (-*v).exp();
        }
    };
    let mut t = x.clone();
    n1.apply(&mut t, h, w);
    silu_v(&mut t);
    let mut t = c1.apply(&t, h, w);
    n2.apply(&mut t, h, w);
    silu_v(&mut t);
    let t = c2.apply(&t, h, w);
    let skip = sc.apply(&x, h, w);
    let want: Vec<f32> = skip.iter().zip(&t).map(|(a, b)| a + b).collect();

    let args = cortiq_engine::gpu::VaeResnetArgs {
        groups,
        ic,
        oc,
        h,
        w,
        n1w: &n1.w,
        n1b: &n1.b,
        c1w: &c1.w,
        c1b: &c1.b,
        c1k: 3,
        n2w: &n2.w,
        n2b: &n2.b,
        c2w: &c2.w,
        c2b: &c2.b,
        c2k: 3,
        shortcut: Some((&sc.w, &sc.b, 1)),
    };
    let mut got = vec![0f32; oc * h * w];
    assert!(
        cortiq_engine::gpu::vae_resnet(&args, &x, &mut got),
        "gpu vae_resnet refused"
    );
    let mut max_rel = 0f64;
    for (g, wv) in got.iter().zip(&want) {
        let d = (*g as f64 - *wv as f64).abs();
        max_rel = max_rel.max(d / (*wv as f64).abs().max(1.0));
    }
    println!("gpu vae resnet max rel dev {max_rel:.2e}");
    assert!(max_rel < 2e-2, "gpu vae resnet diverged: {max_rel}");

    // Upsample+conv fusion vs CPU (nearest 2× then conv).
    let upc = conv(70, ic, ic, 3);
    let mut got_up = vec![0f32; ic * 4 * h * w];
    assert!(
        cortiq_engine::gpu::vae_upsample_conv(&upc.w, &upc.b, &x, ic, ic, h, w, 3, &mut got_up),
        "gpu vae_upsample_conv refused"
    );
    let mut xu = vec![0f32; ic * 4 * h * w];
    for ci in 0..ic {
        for yy in 0..2 * h {
            for xx in 0..2 * w {
                xu[ci * 4 * h * w + yy * 2 * w + xx] = x[ci * h * w + (yy / 2) * w + xx / 2];
            }
        }
    }
    let want_up = upc.apply(&xu, 2 * h, 2 * w);
    let mut max_rel = 0f64;
    for (g, wv) in got_up.iter().zip(&want_up) {
        let d = (*g as f64 - *wv as f64).abs();
        max_rel = max_rel.max(d / (*wv as f64).abs().max(1.0));
    }
    println!("gpu vae upsample+conv max rel dev {max_rel:.2e}");
    assert!(max_rel < 2e-2, "gpu vae upsample+conv diverged: {max_rel}");
}

#[test]
fn gpu_dit_attention_matches_reference() {
    if !require_metal() {
        return;
    }
    // Odd n exercises the edge tiles of both GEMMs; GQA via nkv < nh.
    let (nh, nkv, n, hd) = (4usize, 2usize, 100usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let mk = |seed: usize, len: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 29 + seed * 31 + 7) % 83) as f32 / 83.0 - 0.5)
            .collect()
    };
    let qh = mk(1, nh * n * hd);
    let kh = mk(2, nkv * n * hd);
    let vh = mk(3, nkv * n * hd);
    let mut got = vec![0f32; n * nh * hd];
    assert!(
        cortiq_engine::gpu::dit_attention(&qh, &kh, &vh, nh, nkv, n, hd, scale, &mut got),
        "gpu dit_attention refused"
    );

    let hpk = nh / nkv;
    let mut max_abs = 0f64;
    for h in 0..nh {
        let kv = h / hpk;
        for p in 0..n {
            let q = &qh[(h * n + p) * hd..(h * n + p + 1) * hd];
            let mut sc: Vec<f64> = (0..n)
                .map(|j| {
                    let k = &kh[(kv * n + j) * hd..(kv * n + j + 1) * hd];
                    q.iter()
                        .zip(k)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let mx = sc.iter().cloned().fold(f64::MIN, f64::max);
            let mut den = 0f64;
            for v in sc.iter_mut() {
                *v = (*v - mx).exp();
                den += *v;
            }
            for d in 0..hd {
                let want = (0..n)
                    .map(|j| sc[j] * vh[(kv * n + j) * hd + d] as f64)
                    .sum::<f64>()
                    / den;
                let g = got[(p * nh + h) * hd + d] as f64;
                max_abs = max_abs.max((g - want).abs());
            }
        }
    }
    println!("gpu dit attention max abs dev {max_abs:.2e}");
    assert!(max_abs < 5e-3, "gpu dit attention diverged: {max_abs}");
}
