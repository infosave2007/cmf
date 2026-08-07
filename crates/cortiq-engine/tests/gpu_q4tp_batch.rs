//! The q4tp GEMM on the device against the host, swept over BATCH.
//!
//! The video DiT packs 1 879 tokens where the image one packs 375, and
//! the device arm that was right at the smaller batch returned NaN at
//! the larger — silently, two steps into a render. A GEMM that is
//! correct at one batch and not another is a bounds bug, and the only
//! way to see it is to walk the axis.
//!
//! Needs a device; without one the call declines and the test skips.

#![cfg(feature = "gpu")]

use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::quant::{f32_to_f16, q2tp_sections, q4tp_put_code, q4tp_sections};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use cortiq_core::CmfModel;
use std::sync::Arc;

const GROUP: usize = 32;

/// A minimal but legal `q4tp` payload: one scale for the whole tensor,
/// so every group's rung code is 0 and the ladder's step is zero. The
/// kernel reads exactly the same planes it would for a real weight.
fn encode(vals: &[f32], rows: usize, cols: usize, scale: f32) -> Vec<u8> {
    let gpr = cols / GROUP;
    let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    let inv = 1.0 / scale;
    for r in 0..rows {
        for g in 0..gpr {
            let tile = &vals[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
            let dst = &mut out[(r * gpr + g) * 16..(r * gpr + g + 1) * 16];
            for k in 0..16 {
                let q = |v: f32| ((v * inv).round_ties_even().clamp(-8.0, 7.0) as i8 + 8) as u8;
                dst[k] = (q(tile[k * 2]) & 0x0F) | (q(tile[k * 2 + 1]) << 4);
            }
            q4tp_put_code(&mut out[codes_off + r * stride..codes_off + (r + 1) * stride], g, 0);
        }
        let lo = f32_to_f16(scale.log2());
        out[params_off + r * 4..params_off + r * 4 + 2].copy_from_slice(&lo.to_le_bytes());
        // step 0: every rung is the same scale.
        out[params_off + r * 4 + 2..params_off + r * 4 + 4].copy_from_slice(&0u16.to_le_bytes());
    }
    out
}

/// The same for `q2tp`: 8 bytes a group, four 2-bit codes to a byte
/// LSB-first, and rung 1 (not 0 — that one names the exact zero) so the
/// scale is `2^lo` throughout.
fn encode2(vals: &[f32], rows: usize, cols: usize, scale: f32) -> Vec<u8> {
    let gpr = cols / GROUP;
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    let inv = 1.0 / scale;
    for r in 0..rows {
        for g in 0..gpr {
            let tile = &vals[r * cols + g * GROUP..r * cols + (g + 1) * GROUP];
            let dst = &mut out[(r * gpr + g) * 8..(r * gpr + g + 1) * 8];
            for k in 0..8 {
                let mut byte = 0u8;
                for j in 0..4 {
                    let q = (tile[k * 4 + j] * inv + 1.5).round_ties_even().clamp(0.0, 3.0) as u8;
                    byte |= q << (2 * j);
                }
                dst[k] = byte;
            }
            q4tp_put_code(&mut out[codes_off + r * stride..codes_off + (r + 1) * stride], g, 1);
        }
        let lo = f32_to_f16(scale.log2());
        out[params_off + r * 4..params_off + r * 4 + 2].copy_from_slice(&lo.to_le_bytes());
        out[params_off + r * 4 + 2..params_off + r * 4 + 4].copy_from_slice(&0u16.to_le_bytes());
    }
    out
}

fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

#[test]
fn the_device_gemm_agrees_with_the_host_at_every_batch() {
    // The video DiT's four projections at both batches that matter —
    // 375 rows is a 256x160 pack, 1 879 is 512x288. `fc1` at the larger
    // one writes a 215 MB output, which is where a binding limit or an
    // index width would first bite.
    for (rows, cols) in [
        (21504usize, 5376usize), // attn.qkv_proj
        (5376, 7168),            // attn.out_proj
        (28672, 5376),           // mlp.fc1
        (5376, 14336),           // mlp.fc2
        (4096, 5376),            // a shape nothing uses, as a control
    ] {
        one_shape(rows, cols, false);
    }
}

/// The two-bit plane's own sweep. `mlp.fc1` is the only shape a q2tp
/// file actually puts two bits on, and it is the widest one there is.
#[test]
fn the_two_bit_device_gemm_agrees_with_the_host() {
    for (rows, cols) in [(28672usize, 5376usize), (4096, 5376)] {
        one_shape(rows, cols, true);
    }
}

fn one_shape(rows: usize, cols: usize, two_bit: bool) {
    let scale = 0.01f32;
    let w = noise(rows * cols, 0xfeed_face);
    let payload = if two_bit {
        encode2(&w, rows, cols, scale)
    } else {
        encode(&w, rows, cols, scale)
    };

    // Unique per SHAPE and width, not just per process: the two tests
    // run on separate threads of one process, and a shared path means
    // one truncating the file the other has mapped — SIGBUS, not a
    // wrong number.
    let dir = std::env::temp_dir().join(format!(
        "cmf-tp-batch-{}-{rows}x{cols}-{}",
        std::process::id(),
        if two_bit { "q2" } else { "q4" }
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("t.cmf");
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch: serde_json::from_value::<ModelArch>(serde_json::json!({
            "arch_name": "test", "hidden_size": cols, "intermediate_size": rows,
            "num_layers": 1, "num_attention_heads": 1, "num_kv_heads": 1,
            "head_dim": 1, "vocab_size": 1, "layer_types": ["FullAttention"],
            "rms_norm_eps": 1e-6, "max_position_embeddings": 1,
            "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
        }))
        .unwrap(),
        quant_type: QuantType::Q4Block,
        provenance: None,
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
    };
    CmfModel::write(
        &path,
        &header,
        &[TensorSpec {
            name: "w".into(),
            dtype: if two_bit {
                TensorDtype::Q2TiledP
            } else {
                TensorDtype::Q4TiledP
            },
            shape: vec![rows, cols],
            data: payload,
        }],
        None,
        None,
    )
    .unwrap();
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensors.iter().position(|t| t.name == "w").unwrap();

    // The host reference reads the SAME bytes back, so the comparison
    // is device-versus-host arithmetic and not encoder-versus-kernel.
    let mut wq = vec![0f32; rows * cols];
    cortiq_core::quant::dequant_tensor(&model.tensors[idx], model.entry_bytes(&model.tensors[idx]), &mut wq)
        .unwrap();

    let mut worst = 0f32;
    for &b in &[375usize, 1879] {
        let x = noise(b * cols, 0x1234 + b as u64);
        let mut got = vec![0f32; b * rows];
        let ok = if two_bit {
            cortiq_engine::gpu::q2tp_matmat(&model, idx, &x, b, rows, cols, &mut got)
        } else {
            cortiq_engine::gpu::q4tp_matmat(&model, idx, &x, b, rows, cols, &mut got)
        };
        if !ok {
            eprintln!("b={b}: device declined — skipping");
            continue;
        }
        let mut want = vec![0f32; b * rows];
        cortiq_engine::fcd_ops::gemm_nt(&x, &wq, &mut want, b, cols, rows, None);
        let nans = got.iter().filter(|v| !v.is_finite()).count();
        let d = got
            .iter()
            .zip(&want)
            .fold(0f32, |a, (&p, &q)| a.max((p - q).abs()));
        let mag = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
        println!(
            "[{rows:5} x {cols:5}] b={b:5} {}: max |Δ| {d:.3e} over |ref| {mag:.3e}  \
             out {:.0} MB  non-finite {nans}",
            if two_bit { "q2tp" } else { "q4tp" },
            (b * rows * 4) as f64 / 1e6
        );
        assert_eq!(
            nans, 0,
            "[{rows}x{cols}] b={b}: the device produced {nans} non-finite values"
        );
        worst = worst.max(d / mag);
    }
    std::fs::remove_dir_all(&dir).ok();
    // f32 accumulation in a different order, not a different answer.
    assert!(
        worst < 1e-3,
        "[{rows}x{cols}] device disagrees with the host: {worst:.3e} relative"
    );
}
