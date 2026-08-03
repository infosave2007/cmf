//! The Metal q4tp kernel against the canonical scalar dequant.
//!
//! A wrong GPU kernel is the expensive kind of bug: the model still emits
//! fluent text, so nothing looks broken until someone measures quality. This
//! pins each backend's kernel to `dequant_q4tp`, the format's definition.
//!
//! The two tests select their backend through `CMF_GPU`, which cargo's
//! threads share. Their contexts are independent `OnceLock`s so both do run,
//! but a backend that fails to come up says so on stderr rather than passing
//! quietly — check for "skipped" before trusting a green combined run.

use cortiq_core::format::{CMF_VERSION, CmfHeader, CmfModel, TensorSpec};
use cortiq_core::quant::{
    GROUP_SIZE, dequant_q4tp, f32_to_f16, q4tp_code_stride, q4tp_put_code, q4tp_sections,
};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};

/// Random nibbles plus a per-row ladder whose span varies row to row, so the
/// codes cover the full 0..31 range instead of clustering on one rung.
fn synth(rows: usize, cols: usize) -> Vec<u8> {
    let gpr = cols / GROUP_SIZE;
    let stride = q4tp_code_stride(gpr);
    let (params_off, codes_off, _) = q4tp_sections(rows, cols);
    let mut b = vec![0u8; codes_off + rows * stride];
    for r in 0..rows {
        for g in 0..gpr {
            let t = (r * gpr + g) * 16;
            for k in 0..16 {
                b[t + k] = ((r * 31 + g * 7 + k * 13) % 251) as u8;
            }
        }
        let p = params_off + r * 4;
        b[p..p + 2].copy_from_slice(&f32_to_f16(-6.0 - 0.03 * (r % 17) as f32).to_le_bytes());
        b[p + 2..p + 4].copy_from_slice(&f32_to_f16(0.01 + 0.004 * (r % 11) as f32).to_le_bytes());
        let crow = &mut b[codes_off + r * stride..codes_off + (r + 1) * stride];
        for g in 0..gpr {
            q4tp_put_code(crow, g, (r * 5 + g * 3) % 32);
        }
    }
    b
}

/// `tag` keeps the four backend tests off each other's file: cargo runs them
/// in parallel threads of ONE process, so a shared path had them reading a
/// model another test was mid-write on — which reads exactly like a broken
/// kernel.
fn tiny_model(
    tag: &str,
    rows: usize,
    cols: usize,
    payload: Vec<u8>,
) -> (std::sync::Arc<CmfModel>, usize) {
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "tiny",
        "hidden_size": cols,
        "intermediate_size": rows,
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
    };
    let spec = TensorSpec {
        name: "w".into(),
        dtype: TensorDtype::Q4TiledP,
        shape: vec![rows, cols],
        data: payload,
    };
    // The Metal path maps the file and refuses tensors whose payload would
    // run past the mapped span, so leave slack behind the weight.
    let pad = TensorSpec {
        name: "pad".into(),
        dtype: TensorDtype::F32,
        shape: vec![8192, 2],
        data: vec![0u8; 8192 * 8],
    };
    let dir = std::env::temp_dir().join(format!("cmf-q4tp-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.cmf");
    CmfModel::write(&path, &header, &[spec, pad], None, None).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let idx = model.tensor_index("w").unwrap();
    (model, idx)
}

/// Compare a backend's kernel against the scalar dequant over a whole tensor.
fn check(
    tag: &str,
    run: impl Fn(&std::sync::Arc<CmfModel>, usize, &[f32], usize, usize, &mut [f32]) -> bool,
) {
    let (rows, cols) = (512usize, 1024usize);
    let payload = synth(rows, cols);
    let mut w = vec![0f32; rows * cols];
    dequant_q4tp(&payload, rows, cols, &mut w);
    let (model, idx) = tiny_model(tag, rows, cols, payload);

    let xs: Vec<f32> = (0..cols)
        .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
        .collect();
    let mut got = vec![0f32; rows];
    assert!(
        run(&model, idx, &xs, rows, cols, &mut got),
        "GPU refused a well-formed q4tp tensor"
    );

    for r in 0..rows {
        let want: f32 = (0..cols).map(|c| w[r * cols + c] * xs[c]).sum();
        // Tolerance against the summed magnitude, not the result: these dot
        // products cancel, and GPU/CPU differ only in summation order.
        let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * xs[c]).abs()).sum();
        assert!(
            (got[r] - want).abs() <= 1e-5 * mag,
            "row {r}: GPU {} vs dequant {want}",
            got[r]
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_q4tp_matvec_matches_dequant_reference() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    // NOT a silent skip. A broken MSL kernel makes ctx() fail, Metal falls
    // back to CPU, and `enabled()` goes false — so an early return here turns
    // "the shader does not compile" into a green test. That shipped a dead
    // Metal backend in 0.5.40. On macOS the backend must come up.
    assert!(
        cortiq_engine::gpu_metal::enabled(),
        "Metal did not initialize — check the MSL compile log above"
    );
    check("metal-mv", cortiq_engine::gpu_metal::q4tp_matvec_for_test);
}

/// wgpu covers Vulkan/DX12; `CMF_GPU=wgpu` selects it on macOS too, so this
/// runs locally instead of only on the machines that have no other backend.
#[cfg(feature = "gpu")]
#[test]
fn wgpu_q4tp_matvec_matches_dequant_reference() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    if !cortiq_engine::gpu_wgpu::enabled() {
        eprintln!("skipped: no wgpu adapter");
        return;
    }
    check("wgpu-mv", cortiq_engine::gpu_wgpu::q4tp_matvec_for_test);
}

/// The batched MATVEC — the mv4/mv16 pair with `_p0 > 1`, which is a
/// different kernel from the GEMM above. It carries the whole prefill chunk
/// and every speculative verify, and it shares its inner loop with decode's
/// matvec: a bug in the batch dimension is one row-block boundary away from
/// being a bug in decode. Both row blockings are covered — 16 rows when the
/// shape is narrow, 8 when it is wide — because the batch offset is computed
/// per block and the two compute it separately.
#[cfg(feature = "gpu")]
#[test]
fn wgpu_q4tp_matvec_batch_matches_dequant_reference() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    if !cortiq_engine::gpu_wgpu::enabled() {
        eprintln!("skipped: no wgpu adapter");
        return;
    }
    // cols 512 -> gpr 16, the 16-row kernel; cols 4096 -> gpr 128, the 8-row.
    //
    // The models are built up front and HELD. The device-side weight cache is
    // keyed by the mapping's address, so a model dropped mid-loop frees an
    // address the next mmap can take — and the second shape then reads the
    // first one's weights. That is a test bug that reads exactly like a
    // broken kernel, and it cost a bisect to tell apart.
    let mut built = Vec::new();
    for (rows, cols) in [(256usize, 512usize), (128, 4096)] {
        let payload = synth(rows, cols);
        let mut w = vec![0f32; rows * cols];
        dequant_q4tp(&payload, rows, cols, &mut w);
        let (model, idx) = tiny_model(&format!("wgpu-mvb-{rows}-{cols}"), rows, cols, payload);
        built.push((rows, cols, w, model, idx));
    }
    for (rows, cols, w, model, idx) in &built {
        let (rows, cols, idx) = (*rows, *cols, *idx);
        for b in [1usize, 2, 4] {
            let xs: Vec<f32> = (0..b * cols)
                .map(|i| ((i * 29 + 13) % 103) as f32 / 103.0 - 0.5)
                .collect();
            let mut got = vec![0f32; b * rows];
            assert!(
                cortiq_engine::gpu_wgpu::q4tp_matvec_batch_for_test(
                    &model, idx, &xs, b, rows, cols, &mut got
                ),
                "GPU refused a well-formed batched q4tp matvec ({rows}x{cols}, b={b})"
            );
            // Every batch element against the SAME weight, which is the claim
            // the kernel makes: one read, B uses.
            for t in 0..b {
                let x = &xs[t * cols..(t + 1) * cols];
                for r in 0..rows {
                    let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
                    let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * x[c]).abs()).sum();
                    assert!(
                        (got[t * rows + r] - want).abs() <= 1e-4 * mag,
                        "{rows}x{cols} b={b} token {t} row {r}: GPU {} vs dequant {want}",
                        got[t * rows + r]
                    );
                }
            }
        }
    }
}

/// Same idea for the batched GEMM. Prefill runs through this kernel, so a
/// wrong one corrupts the prompt while decode still looks fine — the failure
/// mode is a model that answers a question it was never asked.
fn check_mm(
    tag: &str,
    run: impl Fn(&std::sync::Arc<CmfModel>, usize, &[f32], usize, usize, usize, &mut [f32]) -> bool,
) {
    check_mm_b(tag, 48, &run);
    // b=1 is the lm_head route: `gpu::q4tp_matvec` is this kernel with a
    // batch of one, and a GEMM that only ever ran at b=48 could carry a
    // tail bug that decode — not prefill — would be the one to hit.
    check_mm_b(tag, 1, &run);
}

fn check_mm_b(
    tag: &str,
    b: usize,
    run: &impl Fn(&std::sync::Arc<CmfModel>, usize, &[f32], usize, usize, usize, &mut [f32]) -> bool,
) {
    let (rows, cols) = (256usize, 512usize);
    let payload = synth(rows, cols);
    let mut w = vec![0f32; rows * cols];
    dequant_q4tp(&payload, rows, cols, &mut w);
    let (model, idx) = tiny_model(tag, rows, cols, payload);

    let xs: Vec<f32> = (0..b * cols)
        .map(|i| ((i * 29 + 13) % 103) as f32 / 103.0 - 0.5)
        .collect();
    let mut got = vec![0f32; b * rows];
    assert!(
        run(&model, idx, &xs, b, rows, cols, &mut got),
        "GPU refused a well-formed q4tp GEMM"
    );

    for t in 0..b {
        let x = &xs[t * cols..(t + 1) * cols];
        for r in 0..rows {
            let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
            let mag: f32 = (0..cols).map(|c| (w[r * cols + c] * x[c]).abs()).sum();
            assert!(
                (got[t * rows + r] - want).abs() <= 1e-4 * mag,
                "batch {t} row {r}: GPU {} vs dequant {want}",
                got[t * rows + r]
            );
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_q4tp_matmat_matches_dequant_reference() {
    unsafe { std::env::set_var("CMF_GPU", "1") };
    // NOT a silent skip. A broken MSL kernel makes ctx() fail, Metal falls
    // back to CPU, and `enabled()` goes false — so an early return here turns
    // "the shader does not compile" into a green test. That shipped a dead
    // Metal backend in 0.5.40. On macOS the backend must come up.
    assert!(
        cortiq_engine::gpu_metal::enabled(),
        "Metal did not initialize — check the MSL compile log above"
    );
    check_mm("metal-mm", cortiq_engine::gpu_metal::q4tp_matmat);
}

#[cfg(feature = "gpu")]
#[test]
fn wgpu_q4tp_matmat_matches_dequant_reference() {
    unsafe { std::env::set_var("CMF_GPU", "wgpu") };
    if !cortiq_engine::gpu_wgpu::enabled() {
        eprintln!("skipped: no wgpu adapter");
        return;
    }
    check_mm("wgpu-mm", cortiq_engine::gpu_wgpu::q4tp_matmat);
}
