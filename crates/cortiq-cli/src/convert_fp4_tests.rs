//! Native FP4 in the converter: the `mxfp4` codec, the `fp4` profile's
//! byte-identical DeepSeek-V4.1 expert path, the NVIDIA ModelOpt NVFP4
//! source decoder, and `--exclude-tensors` (the 1017 fp4 file drops its MTP
//! stack to fit one Hugging Face file). Every source here is a synthetic
//! safetensors file written by the test; nothing is downloaded.

use super::tests::ENV_LOCK;
use super::*;
use cortiq_core::format::CmfModel;
use std::sync::Arc;

type Raw = (String, &'static str, Vec<usize>, Vec<u8>);

fn raw_st(tensors: &[Raw]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let hjson = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (hjson.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&hjson);
    out.extend_from_slice(&data);
    out
}

/// Deterministic pseudo-random bytes.
fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

/// bf16 bytes of small deterministic values (exact in bf16).
fn bf16_bytes(n: usize, seed: u64, amp: f32) -> Vec<u8> {
    bytes(n, seed)
        .iter()
        .flat_map(|&b| {
            let v = (b as f32 / 255.0 - 0.5) * amp;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("cortiq-fp4-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn clear_convert_env() {
    unsafe {
        std::env::remove_var(LOCAL_READY_DIR_ENV);
        std::env::remove_var(SOURCE_SHARD_PRIORITY_ENV);
        std::env::remove_var(CONSUME_SOURCE_SHARDS_ENV);
    }
    set_tensor_quant_overrides(&[]).unwrap();
}

/// A bounded device budget (4 GiB unless the caller set `CMF_GPU_VRAM_MB`,
/// as the V4.1 GPU fixture does) for every test here that runs a forward:
/// the wgpu backend self-selects on a Linux/Windows host with an adapter,
/// and without a budget the V4.1 expert bank sizes itself from the whole
/// card (75 % of a 96 GB card that another process may be holding: an
/// OOM) or from u64::MAX on wgpu-over-Metal. The budget is read once, when
/// the backend comes up: a context another test of this binary already
/// brought up keeps its own, which is why the device test also pins its
/// bank's slot count.
fn bound_device_budget() {
    if std::env::var_os("CMF_GPU_VRAM_MB").is_none() {
        // SAFETY: under ENV_LOCK, like every env write in these tests.
        unsafe { std::env::set_var("CMF_GPU_VRAM_MB", "4096") };
    }
}

/// MoE calls the V4.1 executor's device expert bank served so far (0 in a
/// build without the wgpu backend).
fn bank_calls(p: &cortiq_engine::Pipeline) -> u64 {
    #[cfg(feature = "gpu")]
    {
        p.dsv41.as_ref().map_or(0, |d| d.3.gpu_moe_stats().0)
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = p;
        0
    }
}

// ── codec plumbing ──────────────────────────────────────────────────────

#[test]
fn mxfp4_and_fp4_parse_and_resolve_per_tensor() {
    assert_eq!(parse_quant("mxfp4").unwrap(), Quant::Mxfp4);
    assert_eq!(parse_quant("MXFP4").unwrap(), Quant::Mxfp4);
    assert_eq!(parse_quant("fp4").unwrap(), Quant::Fp4);
    assert_eq!(quant_name(Quant::Mxfp4), "mxfp4");
    assert_eq!(quant_name(Quant::Fp4), "fp4");
    let (pat, q) = parse_tensor_quant("model.layers.*.mlp.experts.*=mxfp4").unwrap();
    assert_eq!(
        (pat.as_str(), q),
        ("model.layers.*.mlp.experts.*", Quant::Mxfp4)
    );
    let e = parse_tensor_quant("x=fp4").unwrap_err().to_string();
    assert!(e.contains("whole-file profile"), "{e}");

    let arch = build_arch(&dsv41_config()).unwrap();
    for name in [
        "model.layers.3.mlp.experts.17.gate_proj.weight",
        "model.layers.3.mlp.experts.17.up_proj.weight",
        "model.layers.3.mlp.experts.17.down_proj.weight",
        "model.layers.3.mlp.shared_expert.down_proj.weight",
        "model.mtp.0.ffn.experts.4.w1.weight",
    ] {
        assert_eq!(
            profile_quant(&arch, Quant::Fp4, name),
            Quant::Mxfp4,
            "{name}"
        );
    }
    for name in [
        "model.layers.3.self_attn.wq_a.weight",
        "model.layers.3.self_attn.wo_b.weight",
        "lm_head.weight",
        "model.embed_tokens.weight",
    ] {
        assert_eq!(
            profile_quant(&arch, Quant::Fp4, name),
            profile_quant(&arch, Quant::Q4TiledP, name),
            "{name}: the rest follows the q4tp profile"
        );
    }
    // The codec itself: OCP quantization of float values, q8_2f fallback on
    // a shape the format does not allow (like every grouped codec).
    let vals: Vec<f32> = (0..4 * 64)
        .map(|i| ((i * 7) % 19) as f32 * 0.03 - 0.25)
        .collect();
    let (dt, data) = quantize_2d(Quant::Mxfp4, &vals, 4, 64);
    assert_eq!(dt, TensorDtype::Mxfp4);
    assert_eq!(data, cortiq_core::mxfp4::quantize(&vals, 4, 64));
    assert_eq!(
        quantize_2d(Quant::Mxfp4, &vals[..4 * 48], 4, 48).0,
        TensorDtype::Q8_2f
    );
}

/// The historical float decoder and the dtype's dequant agree bit for bit on
/// every code under every finite scale byte (including −0 for code 8).
#[test]
fn unpack_mxfp4_equals_core_dequant_bit_for_bit() {
    let (rows, cols) = (255usize, 64usize);
    let codes: Vec<u8> = (0..rows * cols / 2)
        .map(|i| ((i % 16) as u8) | ((((i / 16) + 3) % 16) as u8) << 4)
        .collect();
    let scales: Vec<u8> = (0..rows * 2).map(|i| (i / 2) as u8).collect();
    let floats = unpack_mxfp4(&codes, &scales, rows, cols / 2).unwrap();
    let payload = cortiq_core::mxfp4::repack(&codes, &scales, rows, cols).unwrap();
    let mut core = vec![0f32; rows * cols];
    cortiq_core::mxfp4::dequant(&payload, rows, cols, &mut core);
    for (i, (a, b)) in floats.iter().zip(&core).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs {b}");
    }
}

/// Independent FP8 E4M3 decode for the NVFP4 reference (bias 7, subnormal
/// m·2^-9, 0x7F/0xFF NaN).
fn e4m3_ref(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 15) as i32;
    let m = (b & 7) as f64;
    if e == 15 && m == 7.0 {
        return f64::NAN;
    }
    s * if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

fn e2m1_ref(code: u8) -> f64 {
    let e = ((code >> 1) & 3) as i32;
    let m = (code & 1) as f64;
    let mag = if e == 0 {
        m * 0.5
    } else {
        (1.0 + m / 2.0) * 2f64.powi(e - 1)
    };
    if code & 8 != 0 { -mag } else { mag }
}

/// `w = e2m1 · (e4m3_scale · weight_scale_2)` — ModelOpt/vLLM order, every
/// E4M3 byte as a block scale, both nibble positions, a non-trivial global.
#[test]
fn nvfp4_unpack_matches_the_modelopt_formula() {
    let global = 0.013_671_875f32 * 1.37;
    let finite: Vec<u8> = (0..=255u8).filter(|b| b & 0x7F != 0x7F).collect();
    let rows = finite.len();
    let cols = 16usize;
    let packed = bytes(rows * cols / 2, 5);
    let out = unpack_nvfp4(&packed, &finite, global, rows, cols / 2).unwrap();
    for r in 0..rows {
        let sf = e4m3_ref(finite[r]) as f32;
        assert_eq!(sf, fp8_e4m3_to_f32(finite[r]), "e4m3 byte {:#x}", finite[r]);
        let s = sf * global;
        for c in 0..cols {
            let b = packed[r * cols / 2 + c / 2];
            let code = if c % 2 == 0 { b & 15 } else { b >> 4 };
            let want = (e2m1_ref(code) as f32) * s;
            assert_eq!(
                out[r * cols + c].to_bits(),
                want.to_bits(),
                "row {r} col {c}"
            );
        }
    }
    // NaN block scale and a non-finite global are refused; shapes checked.
    assert!(unpack_nvfp4(&packed[..8], &[0x7F], global, 1, 8).is_err());
    assert!(unpack_nvfp4(&packed[..8], &[0x38], f32::NAN, 1, 8).is_err());
    assert!(unpack_nvfp4(&packed[..8], &[0x38, 0x38], global, 1, 8).is_err());
    assert!(unpack_nvfp4(&packed[..4], &[0x38], global, 1, 4).is_err());
}

#[test]
fn nvfp4_detection_reads_hf_quant_config_and_quantization_config() {
    let d = tmp("nvdetect");
    let plain = serde_json::json!({"model_type": "llama"});
    assert_eq!(nvfp4_source(&d, &plain).unwrap(), None);
    let modelopt = serde_json::json!({"model_type": "llama", "quantization_config": {
        "quant_method": "modelopt", "quant_algo": "NVFP4",
        "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}});
    assert_eq!(nvfp4_source(&d, &modelopt).unwrap(), Some(16));
    let w4a16 =
        serde_json::json!({"quantization_config": {"quant_algo": "W4A16_NVFP4", "group_size": 16}});
    assert_eq!(nvfp4_source(&d, &w4a16).unwrap(), Some(16));
    let bad_group =
        serde_json::json!({"quantization_config": {"quant_algo": "NVFP4", "group_size": 32}});
    assert!(nvfp4_source(&d, &bad_group).is_err());
    let ct = serde_json::json!({"quantization_config": {"quant_method": "compressed-tensors",
        "format": "nvfp4-pack-quantized"}});
    assert!(nvfp4_source(&d, &ct).is_err());
    fs::write(
        d.join("hf_quant_config.json"),
        r#"{"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4",
           "kv_cache_quant_algo": "FP8", "group_size": 16, "exclude_modules": ["lm_head"]}}"#,
    )
    .unwrap();
    assert_eq!(nvfp4_source(&d, &plain).unwrap(), Some(16));
    let _ = fs::remove_dir_all(&d);
}

// ── DeepSeek-V4.1: fp4 profile end to end ───────────────────────────────

const DIM: usize = 64;
const HEADS: usize = 4;
const HD: usize = 64;
const QLORA: usize = 32;
const OGROUPS: usize = 2;
const OLORA: usize = 32;
const MI: usize = 64;
const NEXP: usize = 4;
const VOCAB: usize = 64;
const HC: usize = 2;
const LAYERS: usize = 2;

fn dsv41_config() -> serde_json::Value {
    dsv41_config_mtp(0)
}

/// [`dsv41_config`] declaring `mtp` next-token-prediction layers.
fn dsv41_config_mtp(mtp: usize) -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "tie_word_embeddings": false,
        "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic",
            "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": VOCAB, "hidden_size": DIM, "moe_intermediate_size": MI,
            "num_hidden_layers": LAYERS, "num_attention_heads": HEADS, "num_key_value_heads": 1,
            "head_dim": HD, "qk_rope_head_dim": 16, "q_lora_rank": QLORA,
            "o_lora_rank": OLORA, "o_groups": OGROUPS, "hidden_act": "silu",
            "swiglu_limit": 10.0, "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
            "max_position_embeddings": 4096, "rope_theta": 10000,
            "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32,
                "beta_slow": 1, "original_max_position_embeddings": 1024},
            "n_routed_experts": NEXP, "n_shared_experts": 1, "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
            "norm_topk_prob": true, "routed_scaling_factor": 1.5,
            "sliding_window": 16, "compress_ratios": [0, 0], "compress_rope_theta": 160000,
            "hc_mult": HC, "hc_sinkhorn_iters": 3, "hc_eps": 1e-6,
            "engram_layer_ids": [], "num_nextn_predict_layers": mtp
        }
    })
}

/// I8 FP4 weight `[rows, cols/2]` + F8_E8M0 `.scale` `[rows, cols/32]`.
fn fp4_pair(
    stem: &str,
    rows: usize,
    cols: usize,
    seed: u64,
    out: &mut Vec<Raw>,
) -> (Vec<u8>, Vec<u8>) {
    let codes = bytes(rows * cols / 2, seed);
    let scales: Vec<u8> = bytes(rows * cols / 32, seed + 1000)
        .iter()
        .map(|b| 119 + b % 5)
        .collect();
    out.push((
        format!("{stem}.weight"),
        "I8",
        vec![rows, cols / 2],
        codes.clone(),
    ));
    out.push((
        format!("{stem}.scale"),
        "F8_E8M0",
        vec![rows, cols / 32],
        scales.clone(),
    ));
    (codes, scales)
}

/// F8_E4M3 weight + E8M0 32×32 block scale (the V4.1 skeleton/shared form).
fn fp8_pair(stem: &str, rows: usize, cols: usize, seed: u64, out: &mut Vec<Raw>) {
    // Finite E4M3 bytes only (never the 0x7F/0xFF NaN pattern), small range.
    let w: Vec<u8> = bytes(rows * cols, seed)
        .iter()
        .map(|b| (b & 0x80) | (0x20 + (b & 0x1F)))
        .collect();
    let s: Vec<u8> = bytes(rows.div_ceil(32) * cols.div_ceil(32), seed + 77)
        .iter()
        .map(|b| 122 + b % 3)
        .collect();
    out.push((format!("{stem}.weight"), "F8_E4M3", vec![rows, cols], w));
    out.push((
        format!("{stem}.scale"),
        "F8_E8M0",
        vec![rows.div_ceil(32), cols.div_ceil(32)],
        s,
    ));
}

struct Dsv41Src {
    dir: PathBuf,
    /// Canonical name → (source codes, source scales) of every FP4 matrix.
    fp4: HashMap<String, (Vec<u8>, Vec<u8>)>,
}

fn write_dsv41_source(tag: &str) -> Dsv41Src {
    write_dsv41_source_with(tag, false)
}

/// The 1017 checkpoint's MTP/DSpark stage in its own spelling (`mtp.0.*`:
/// attention, mHC head, `main_proj`, router, FP4 routed and FP8 shared
/// experts, the DSpark Markov and confidence heads). The converter maps it
/// to `model.mtp.0.*`; the V4.1 runtime never executes it.
fn push_dsv41_mtp_stage(t: &mut Vec<Raw>) {
    let p = "mtp.0";
    let bf = |name: &str, shape: Vec<usize>, seed: u64, amp: f32| -> Raw {
        let n = shape.iter().product();
        (
            format!("{p}.{name}"),
            "BF16",
            shape,
            bf16_bytes(n, seed, amp),
        )
    };
    for (name, n, seed) in [
        ("attn_norm.weight", DIM, 901),
        ("ffn_norm.weight", DIM, 902),
        ("main_norm.weight", DIM, 903),
        ("norm.weight", DIM, 904),
    ] {
        t.push(bf(name, vec![n], seed, 0.2));
    }
    t.push(bf("attn.wq_a.weight", vec![QLORA, DIM], 905, 0.4));
    t.push(bf("attn.wkv.weight", vec![HD, DIM], 906, 0.4));
    fp8_pair(&format!("{p}.attn.wo_b"), DIM, OGROUPS * OLORA, 907, t);
    t.push(bf("main_proj.weight", vec![DIM, 2 * DIM], 908, 0.3));
    t.push((
        format!("{p}.hc_head_fn"),
        "F32",
        vec![HC, HC * DIM],
        f32_bytes(&vec![0.01; HC * HC * DIM]),
    ));
    t.push((
        format!("{p}.hc_head_base"),
        "F32",
        vec![HC],
        f32_bytes(&[0.0, 0.1]),
    ));
    t.push((
        format!("{p}.hc_head_scale"),
        "F32",
        vec![1],
        f32_bytes(&[1.0]),
    ));
    t.push(bf("ffn.gate.weight", vec![NEXP, DIM], 909, 0.5));
    t.push((
        format!("{p}.ffn.gate.bias"),
        "F32",
        vec![NEXP],
        f32_bytes(&[0.0, 0.01, -0.01, 0.02]),
    ));
    for e in 0..NEXP {
        for (w, rows, cols) in [("w1", MI, DIM), ("w3", MI, DIM), ("w2", DIM, MI)] {
            fp4_pair(
                &format!("{p}.ffn.experts.{e}.{w}"),
                rows,
                cols,
                950 + 3 * e as u64 + w.len() as u64 * 7 + rows as u64,
                t,
            );
        }
    }
    for (w, rows, cols) in [("w1", MI, DIM), ("w3", MI, DIM), ("w2", DIM, MI)] {
        fp8_pair(
            &format!("{p}.ffn.shared_experts.{w}"),
            rows,
            cols,
            990 + rows as u64,
            t,
        );
    }
    t.push(bf("markov_head.markov_w1.weight", vec![DIM, DIM], 995, 0.3));
    t.push(bf("confidence_head.proj.weight", vec![1, DIM], 996, 0.3));
}

/// [`write_dsv41_source`], optionally with the MTP/DSpark stage (and a
/// config declaring one MTP layer).
fn write_dsv41_source_with(tag: &str, mtp: bool) -> Dsv41Src {
    let dir = tmp(tag);
    fs::write(
        dir.join("config.json"),
        dsv41_config_mtp(usize::from(mtp)).to_string(),
    )
    .unwrap();
    let mut t: Vec<Raw> = Vec::new();
    if mtp {
        push_dsv41_mtp_stage(&mut t);
    }
    let mut fp4 = HashMap::new();
    let mix = (2 + HC) * HC;
    t.push((
        "embed.weight".into(),
        "BF16",
        vec![VOCAB, DIM],
        bf16_bytes(VOCAB * DIM, 1, 0.5),
    ));
    t.push((
        "head.weight".into(),
        "BF16",
        vec![VOCAB, DIM],
        bf16_bytes(VOCAB * DIM, 2, 0.5),
    ));
    t.push((
        "norm.weight".into(),
        "BF16",
        vec![DIM],
        bf16_bytes(DIM, 3, 0.2),
    ));
    for li in 0..LAYERS {
        let p = format!("layers.{li}");
        let s = 100 * (li as u64 + 1);
        let bf = |name: &str, shape: Vec<usize>, seed: u64, amp: f32| -> Raw {
            let n = shape.iter().product();
            (
                format!("{p}.{name}"),
                "BF16",
                shape,
                bf16_bytes(n, seed, amp),
            )
        };
        // RMS norm weights near one.
        let ones = |name: &str, n: usize, seed: u64| -> Raw {
            let v: Vec<u8> = bytes(n, seed)
                .iter()
                .flat_map(|&b| {
                    let x = 1.0 + (b as f32 / 255.0 - 0.5) * 0.25;
                    ((x.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            (format!("{p}.{name}"), "BF16", vec![n], v)
        };
        t.push(ones("attn_norm.weight", DIM, s + 1));
        t.push(ones("ffn_norm.weight", DIM, s + 2));
        t.push(ones("attn.q_norm.weight", QLORA, s + 3));
        t.push(ones("attn.kv_norm.weight", HD, s + 4));
        t.push(bf("attn.wq_a.weight", vec![QLORA, DIM], s + 5, 0.4));
        t.push(bf("attn.wq_b.weight", vec![HEADS * HD, QLORA], s + 6, 0.4));
        t.push(bf("attn.wkv.weight", vec![HD, DIM], s + 7, 0.4));
        t.push(bf(
            "attn.wo_a.weight",
            vec![OGROUPS * OLORA, HEADS * HD / OGROUPS],
            s + 8,
            0.3,
        ));
        t.push(bf(
            "attn.wo_b.weight",
            vec![DIM, OGROUPS * OLORA],
            s + 9,
            0.3,
        ));
        t.push((
            format!("{p}.attn.attn_sink"),
            "F32",
            vec![HEADS],
            f32_bytes(&[0.1, -0.2, 0.0, 0.3]),
        ));
        for (part, seed) in [("attn", s + 10), ("ffn", s + 20)] {
            let fnw: Vec<f32> = bytes(mix * HC * DIM, seed)
                .iter()
                .map(|&b| (b as f32 / 255.0 - 0.5) * 0.02)
                .collect();
            let base: Vec<f32> = (0..mix).map(|i| (i % 3) as f32 * 0.1 - 0.1).collect();
            t.push((
                format!("{p}.hc_{part}_fn"),
                "F32",
                vec![mix, HC * DIM],
                f32_bytes(&fnw),
            ));
            t.push((
                format!("{p}.hc_{part}_base"),
                "F32",
                vec![mix],
                f32_bytes(&base),
            ));
            t.push((
                format!("{p}.hc_{part}_scale"),
                "F32",
                vec![3],
                f32_bytes(&[1.0, 0.5, 0.25]),
            ));
        }
        t.push(bf("ffn.gate.weight", vec![NEXP, DIM], s + 30, 0.5));
        t.push((
            format!("{p}.ffn.gate.bias"),
            "F32",
            vec![NEXP],
            f32_bytes(&[0.0, 0.05, -0.05, 0.02]),
        ));
        for e in 0..NEXP {
            for (w, canon, rows, cols) in [
                ("w1", "gate_proj", MI, DIM),
                ("w3", "up_proj", MI, DIM),
                ("w2", "down_proj", DIM, MI),
            ] {
                let pair = fp4_pair(
                    &format!("{p}.ffn.experts.{e}.{w}"),
                    rows,
                    cols,
                    s + 40 + 3 * e as u64 + w.len() as u64 * 7 + rows as u64,
                    &mut t,
                );
                fp4.insert(
                    format!("model.layers.{li}.mlp.experts.{e}.{canon}.weight"),
                    pair,
                );
            }
        }
        // Layer 0's shared expert is FP4 (re-packed byte for byte); layer
        // 1's is FP8 like the published checkpoint (OCP-quantized to mxfp4
        // so the expert bank keeps one codec).
        for (w, canon, rows, cols) in [
            ("w1", "gate_proj", MI, DIM),
            ("w3", "up_proj", MI, DIM),
            ("w2", "down_proj", DIM, MI),
        ] {
            let stem = format!("{p}.ffn.shared_experts.{w}");
            if li == 0 {
                let pair = fp4_pair(&stem, rows, cols, s + 90 + rows as u64, &mut t);
                fp4.insert(
                    format!("model.layers.{li}.mlp.shared_expert.{canon}.weight"),
                    pair,
                );
            } else {
                fp8_pair(&stem, rows, cols, s + 95 + rows as u64, &mut t);
            }
        }
    }
    fs::write(dir.join("model.safetensors"), raw_st(&t)).unwrap();
    Dsv41Src { dir, fp4 }
}

/// The same model with every `mxfp4` tensor dequantized to F32 — the exact
/// matrices the FP4 kernels must reproduce.
fn dequantized_twin(model: &CmfModel, out: &Path) {
    scaled_twin(model, out, 1.0)
}

/// [`dequantized_twin`] with the expert matrices multiplied by `k`.
fn scaled_twin(model: &CmfModel, out: &Path, k: f32) {
    let specs: Vec<TensorSpec> = model
        .tensors
        .iter()
        .map(|e| {
            let src = model.entry_bytes(e);
            if e.dtype == TensorDtype::Mxfp4 {
                let mut v = vec![0f32; e.shape[0] * e.shape[1]];
                cortiq_core::mxfp4::dequant(src, e.shape[0], e.shape[1], &mut v);
                v.iter_mut().for_each(|x| *x *= k);
                TensorSpec {
                    name: e.name.clone(),
                    dtype: TensorDtype::F32,
                    shape: e.shape.clone(),
                    data: f32_bytes(&v),
                }
            } else {
                TensorSpec {
                    name: e.name.clone(),
                    dtype: e.dtype,
                    shape: e.shape.clone(),
                    data: src.to_vec(),
                }
            }
        })
        .collect();
    CmfModel::write(
        out,
        &model.header,
        &specs,
        Some(&model.masks),
        model.vocab.as_deref(),
    )
    .unwrap();
}

/// `--quant fp4` on a synthetic DeepSeek-V4.1 checkpoint, through the real
/// converter, in ONE pass beside a `q4tp` output:
/// - every FP4 source expert matrix lands as `mxfp4` byte-identical to its
///   source planes (no float round trip); the FP8 shared expert is
///   OCP-quantized to `mxfp4`;
/// - every other tensor is byte-identical to the `q4tp` conversion;
/// - the q4tp output still decodes the experts through f32 (mixed outputs);
/// - the CPU forward of the fp4 file equals, bit for bit, the forward of the
///   same file with its FP4 tensors dequantized to f32 (exact contract), and
///   stays finite on the default A8W8 kernels — on a GPU host too: the
///   forwards are pinned to the host (`gpu::pause_gpu`, checked).
#[test]
fn dsv41_fp4_profile_is_byte_identical_and_forward_matches_f32_experts() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_convert_env();
    bound_device_budget();
    let src = write_dsv41_source("dsv41");
    let fp4_out = src.dir.join("fp4.cmf");
    let q4_out = src.dir.join("q4tp.cmf");
    run_convert_multi(
        src.dir.to_str().unwrap(),
        &[
            ("fp4".into(), fp4_out.to_str().unwrap().into()),
            ("q4tp".into(), q4_out.to_str().unwrap().into()),
        ],
        None,
        None,
        None,
        false,
        |_| {},
    )
    .unwrap();
    let fp4 = Arc::new(CmfModel::open(&fp4_out).unwrap());
    let q4 = Arc::new(CmfModel::open(&q4_out).unwrap());
    assert!(fp4.verify().is_empty(), "{:?}", fp4.verify());
    assert!(q4.verify().is_empty(), "{:?}", q4.verify());

    // Byte identity of every FP4 source matrix.
    assert_eq!(src.fp4.len(), LAYERS * NEXP * 3 + 3);
    for (name, (codes, scales)) in &src.fp4 {
        let e = fp4.tensor(name).unwrap_or_else(|| panic!("{name} missing"));
        assert_eq!(e.dtype, TensorDtype::Mxfp4, "{name}");
        let mut want = codes.clone();
        want.extend_from_slice(scales);
        assert_eq!(fp4.entry_bytes(e), &want[..], "{name}: not byte-identical");
        assert_eq!(
            q4.tensor(name).unwrap().dtype,
            TensorDtype::Q4TiledP,
            "{name} in q4tp"
        );
    }
    // The FP8 shared expert: mxfp4 by the OCP rule, within its error bound
    // of the FP8 decode (one E2M1 step of the group scale at most).
    for canon in ["gate_proj", "up_proj", "down_proj"] {
        let name = format!("model.layers.1.mlp.shared_expert.{canon}.weight");
        let e = fp4.tensor(&name).unwrap();
        assert_eq!(e.dtype, TensorDtype::Mxfp4, "{name}");
        let (rows, cols) = (e.shape[0], e.shape[1]);
        let mut got = vec![0f32; rows * cols];
        cortiq_core::mxfp4::dequant(fp4.entry_bytes(e), rows, cols, &mut got);
        let source = open_safetensors(&src.dir.join("model.safetensors")).unwrap();
        let stem = format!(
            "layers.1.ffn.shared_experts.{}",
            match canon {
                "gate_proj" => "w1",
                "up_proj" => "w3",
                _ => "w2",
            }
        );
        let w = source
            .tensors
            .iter()
            .find(|t| t.name == format!("{stem}.weight"))
            .unwrap();
        let s = source
            .tensors
            .iter()
            .find(|t| t.name == format!("{stem}.scale"))
            .unwrap();
        let fp8 = unpack_fp8_blocks(source.bytes(w), source.bytes(s), rows, cols, 32).unwrap();
        for g in 0..rows * cols / 32 {
            let grp = &fp8[g * 32..(g + 1) * 32];
            let amax = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
            let step = 2f32.powi(amax.log2().floor() as i32 - 2);
            for j in 0..32 {
                assert!(
                    (got[g * 32 + j] - grp[j]).abs() <= step * 2.0 + 1e-12,
                    "{name} g{g}"
                );
            }
        }
    }
    // Everything that is not an expert matrix: the q4tp profile's bytes.
    let mut skeleton = 0;
    for e in &q4.tensors {
        if fp4_expert_matrix(&e.name) {
            continue;
        }
        let f = fp4
            .tensor(&e.name)
            .unwrap_or_else(|| panic!("{} missing in fp4", e.name));
        assert_eq!((f.dtype, &f.shape), (e.dtype, &e.shape), "{}", e.name);
        assert_eq!(fp4.entry_bytes(f), q4.entry_bytes(e), "{}", e.name);
        skeleton += 1;
    }
    assert!(skeleton > 20);
    assert_eq!(fp4.tensors.len(), q4.tensors.len());
    assert_eq!(
        fp4.header
            .provenance
            .as_ref()
            .and_then(|p| p.get("weight_quant"))
            .and_then(|v| v.as_str()),
        Some("fp4")
    );
    assert_eq!(
        q4.header
            .provenance
            .as_ref()
            .and_then(|p| p.get("weight_quant"))
            .and_then(|v| v.as_str()),
        Some("q4tp")
    );

    // Forward: fp4 file vs its dequantized twin, exact contract.
    let twin_path = src.dir.join("twin.cmf");
    dequantized_twin(&fp4, &twin_path);
    let twin = Arc::new(CmfModel::open(&twin_path).unwrap());
    let ids = [1u32, 7, 3, 60, 12];
    let logits = |m: &Arc<CmfModel>| {
        let mut p = cortiq_engine::Pipeline::from_model(
            m,
            cortiq_engine::sampler::SamplerConfig::default(),
        )
        .unwrap();
        let out = p.forward_ids(&ids, None).unwrap();
        assert_eq!(
            bank_calls(&p),
            0,
            "the CPU-contract forward ran MoE calls on the device bank"
        );
        out
    };
    // The CPU contract: both forwards on the host. Where the wgpu backend is
    // up — self-selected on a Linux/Windows GPU host, `CMF_GPU=wgpu` on
    // macOS — the fp4 file's experts would take the dynamic mxfp4 bank while
    // the twin's f32 experts cannot, and the bits would differ by the bank's
    // summation order (the pod run on an RTX PRO 6000 did exactly that).
    // `CMF_GPU` is read once per process and this binary is shared, so
    // `CMF_GPU=0` cannot be set from here; `gpu::pause_gpu` is the
    // process-wide pin instead (every device path — the V4.1 bank and
    // attention tail, the per-op kernels — asks `gpu::enabled` first, and the
    // pause is a count of guards, so no other holder can lift it). The pin
    // is checked, not assumed: `enabled()` is false and no forward below
    // served a MoE call from the bank, so the comparison stays bit-exact.
    let cpu = cortiq_engine::gpu::pause_gpu();
    assert!(!cortiq_engine::gpu::enabled(), "the GPU pause did not hold");
    let (a, b) = cortiq_engine::qtensor::with_float_activations(|| (logits(&fp4), logits(&twin)));
    assert_eq!(a.len(), VOCAB);
    assert!(a.iter().all(|v| v.is_finite()));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b), "fp4 forward != dequantized-f32 forward");
    // ...and the experts do carry the result (a model without them differs).
    let zero_path = src.dir.join("zero.cmf");
    scaled_twin(&fp4, &zero_path, 0.0);
    let zero = Arc::new(CmfModel::open(&zero_path).unwrap());
    let z = cortiq_engine::qtensor::with_float_activations(|| logits(&zero));
    assert_ne!(bits(&a), bits(&z), "expert contribution is invisible");
    drop(zero);
    // Default kernels (A8W8 where the ISA has it): finite and close.
    let c = logits(&fp4);
    let span = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    let dmax = a
        .iter()
        .zip(&c)
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        c.iter().all(|v| v.is_finite()) && dmax <= 0.25 * span,
        "A8W8 drift {dmax} of {span}"
    );
    drop(cpu);
    drop((fp4, q4, twin));
    let _ = fs::remove_dir_all(&src.dir);
}

/// `--tensor-quant` with the plain q4tp profile also takes the raw path for
/// the matched experts, and refuses an FP4 source whose NaN scale would be
/// carried into the model.
#[test]
fn tensor_quant_mxfp4_override_repacks_and_refuses_nan_scales() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_convert_env();
    let src = write_dsv41_source("dsv41-override");
    let out = src.dir.join("q4tp-mx.cmf");
    set_tensor_quant_overrides(&["model.layers.0.mlp.experts.*=mxfp4".into()]).unwrap();
    let r = run_convert(
        src.dir.to_str().unwrap(),
        "q4tp",
        out.to_str().unwrap(),
        None,
        None,
        None,
        false,
        |_| {},
    );
    set_tensor_quant_overrides(&[]).unwrap();
    r.unwrap();
    let m = CmfModel::open(&out).unwrap();
    for (name, (codes, scales)) in &src.fp4 {
        let e = m.tensor(name).unwrap();
        if name.starts_with("model.layers.0.mlp.experts.") {
            assert_eq!(e.dtype, TensorDtype::Mxfp4, "{name}");
            let mut want = codes.clone();
            want.extend_from_slice(scales);
            assert_eq!(m.entry_bytes(e), &want[..]);
        } else {
            assert_eq!(e.dtype, TensorDtype::Q4TiledP, "{name}");
        }
    }
    drop(m);

    // A NaN E8M0 scale in a source FP4 expert: the conversion fails loudly.
    let st = open_safetensors(&src.dir.join("model.safetensors")).unwrap();
    let mut raw: Vec<Raw> = st
        .tensors
        .iter()
        .map(|t| {
            let dt: &'static str = match t.dtype.as_str() {
                "BF16" => "BF16",
                "F32" => "F32",
                "I8" => "I8",
                "F8_E8M0" => "F8_E8M0",
                "F8_E4M3" => "F8_E4M3",
                other => panic!("{other}"),
            };
            (t.name.clone(), dt, t.shape.clone(), st.bytes(t).to_vec())
        })
        .collect();
    drop(st);
    let victim = raw
        .iter_mut()
        .find(|t| t.0 == "layers.0.ffn.experts.1.w2.scale")
        .unwrap();
    victim.3[3] = 255;
    fs::write(src.dir.join("model.safetensors"), raw_st(&raw)).unwrap();
    let e = run_convert(
        src.dir.to_str().unwrap(),
        "fp4",
        src.dir.join("nan.cmf").to_str().unwrap(),
        None,
        None,
        None,
        false,
        |_| {},
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("NaN scale"), "{e}");
    let _ = fs::remove_dir_all(&src.dir);
}

/// The synthetic V4.1 `fp4` file end to end on the wgpu device: the
/// `Pipeline` forward routes its MoE through the dynamic GPU expert bank
/// (mxfp4 codec, `dsv4_global_*_mxfp4` kernels, host-resolved V4.1 route)
/// and its logits match the CPU forward of the same file.
///
/// Runs only where the wgpu backend was asked for — on macOS
/// `CMF_GPU=wgpu` (wgpu over Metal), on Linux/Windows `CMF_GPU=1` or
/// `wgpu` — and skips otherwise (CI without a GPU, macOS CI without the
/// variable); asked for and no context, it FAILS (the repo's
/// `selected_and_up`/`skip_or_fail` convention), so a broken driver on the
/// pod cannot pass as a skip. It does not set `CMF_GPU` itself: in this
/// binary that would move every later test onto the device. The CPU
/// reference pauses the device (`gpu::pause_gpu`), so both forwards run in
/// one process.
#[cfg(feature = "gpu")]
#[test]
fn dsv41_fp4_dynamic_gpu_moe_matches_the_cpu_forward() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_convert_env();
    // Before `selected_and_up` brings the backend up.
    bound_device_budget();
    use cortiq_engine::gpu_wgpu as gw;
    match gw::selected_and_up() {
        None => {
            gw::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!(
            "CMF_GPU={} selects wgpu but no wgpu context came up",
            std::env::var("CMF_GPU").unwrap_or_default()
        ),
        Some(true) => {}
    }
    if !gw::dsv4_global_moe_supported() {
        eprintln!("SKIPPED (dsv41 fp4 GPU e2e): no descriptor-indexed global expert bank here");
        return;
    }
    let src = write_dsv41_source("dsv41-gpu");
    let out = src.dir.join("fp4.cmf");
    run_convert(
        src.dir.to_str().unwrap(),
        "fp4",
        out.to_str().unwrap(),
        None,
        None,
        None,
        false,
        |_| {},
    )
    .unwrap();
    let fp4 = Arc::new(CmfModel::open(&out).unwrap());
    // A bank just above the allocator's 2-4 GiB workspace carve-out: 64
    // slots requested, which the pool caps at the model's experts (2 x (4 +
    // shared), whole S16 segments: 16 slots), instead of the default 75 % of
    // the card.
    let (gu, dn) = gw::MoeBankCodec::Mxfp4.lens(MI, DIM).unwrap();
    let gib = 1u64 << 30;
    let workspace = (gw::dsv4_vram_budget().unwrap() / 10).clamp(2 * gib, 4 * gib);
    let slots = workspace / (2 * gu + dn) as u64 + 64;
    unsafe {
        std::env::set_var("CMF_DSV41_EXPERT_SLOTS", slots.to_string());
        // The pool's creation line ("dynamic-pool capacity=… gu=mxfp4").
        std::env::set_var("CMF_DSV41_PROF", "1");
    }
    let ids = [1u32, 7, 3, 60, 12, 33, 5];
    let run = |m: &Arc<CmfModel>| {
        let mut p = cortiq_engine::Pipeline::from_model(
            m,
            cortiq_engine::sampler::SamplerConfig::default(),
        )
        .unwrap();
        let logits = p.forward_ids(&ids, None).unwrap();
        let stats = p.dsv41.as_ref().expect("V4.1 executor").3.gpu_moe_stats();
        (logits, stats)
    };
    let (cpu, cpu_stats) = {
        let _host = cortiq_engine::gpu::pause_gpu();
        cortiq_engine::qtensor::with_float_activations(|| run(&fp4))
    };
    let fills = gw::DSV4_FILLS.load(std::sync::atomic::Ordering::Relaxed);
    let (gpu, (calls, codec)) = cortiq_engine::qtensor::with_float_activations(|| run(&fp4));
    let fills = gw::DSV4_FILLS.load(std::sync::atomic::Ordering::Relaxed) - fills;
    unsafe {
        std::env::remove_var("CMF_DSV41_EXPERT_SLOTS");
        std::env::remove_var("CMF_DSV41_PROF");
    }
    assert_eq!(
        cpu_stats,
        (0, None),
        "the paused reference touched the device"
    );
    // Every MoE call of the forward (7 tokens x 2 layers) on the device bank.
    assert_eq!(
        codec,
        Some("mxfp4"),
        "the dynamic pool is not the mxfp4 bank"
    );
    assert_eq!(calls, (ids.len() * LAYERS) as u64, "gpu_moe calls");
    assert!(fills > 0, "no expert reached the bank");
    // The 64 requested slots are capped at the model's experts.
    assert_eq!(
        gw::dsv4_global_moe_geometry(&fp4).map(|(_, capacity)| capacity),
        Some((LAYERS * (NEXP + 1)).div_ceil(16) * 16),
        "the V4.1 pool is not capped at the model's expert count"
    );
    let (mut num, mut den) = (0f64, 0f64);
    for (g, c) in gpu.iter().zip(&cpu) {
        num += ((g - c) as f64).powi(2);
        den += (*c as f64).powi(2);
    }
    let nrmse = (num / den).sqrt();
    eprintln!(
        "dsv41 fp4 dynamic GPU MoE: {calls} bank calls, {fills} slot fills, logits nrmse vs CPU {nrmse:.3e}"
    );
    assert!(gpu.iter().all(|v| v.is_finite()));
    // Measured 6.37e-3 on wgpu-over-Metal (Apple M4; the review measured
    // 6.2e-3): the bank and the host sum each expert in a different order
    // and both round to BF16 at three boundaries per expert, so a few
    // roundings land one BF16 step apart. A wrong group/scale/slot index
    // costs O(1) (the kernel tests' mutations: nrmse 0.85-1.4).
    assert!(nrmse < 0.02, "logits nrmse {nrmse}");
    drop(fp4);
    let _ = fs::remove_dir_all(&src.dir);
}

// ── --exclude-tensors ───────────────────────────────────────────────────

#[test]
fn exclude_tensors_parses_and_counts() {
    assert_eq!(
        parse_tensor_exclusions(&["model.mtp.*, lm_head.weight".into(), "model.mtp.*".into()])
            .unwrap(),
        vec!["model.mtp.*".to_string(), "lm_head.weight".to_string()]
    );
    assert!(parse_tensor_exclusions(&["model.mtp.*,".into()]).is_err());
    assert!(parse_tensor_exclusions(&[" ".into()]).is_err());
    assert!(parse_tensor_exclusions(&[]).unwrap().is_empty());
    let ex = TensorExclusion::new(&["model.mtp.*".into(), "*.markov_head.*".into()]);
    assert!(!ex.is_empty());
    assert!(ex.skip_source("model.mtp.0.ffn.experts.3.w1.weight", 1000));
    assert!(ex.skip_source("model.mtp.0.ffn.experts.3.w1.scale", 64));
    assert!(!ex.skip_source("model.layers.0.mlp.experts.3.gate_proj.weight", 1000));
    assert!(!ex.skip_source("model.mtpx.0.w", 1));
    assert_eq!(ex.source_tensors.get(), 2);
    assert_eq!(ex.source_bytes.get(), 1064);
    assert_eq!(
        ex.hits.iter().map(|h| h.get()).collect::<Vec<_>>(),
        vec![2, 0]
    );
    assert!(ex.skip_output("x.markov_head.w"));
    assert!(!ex.skip_output("model.layers.1.mlp.gate.weight"));
    assert!(TensorExclusion::new(&[]).is_empty());
    assert!(!TensorExclusion::new(&[]).skip_output("model.mtp.0.norm.weight"));
}

/// The up-front check against the source index and the end-of-run report
/// of a resumed run: a pattern that matches no index name is flagged before
/// any shard is read; after a resume, a pattern whose tensors sat in shards
/// an earlier attempt finished gets a note, not a "matched nothing" warning.
#[test]
fn exclude_tensors_preview_and_resumed_report() {
    let ex = TensorExclusion::new(&["model.mtp.*".into(), "mtp.*".into()]);
    let index: Vec<String> = [
        "model.mtp.0.ffn.experts.0.w1.weight",
        "model.mtp.0.norm.weight",
        "model.layers.0.attn.wq_a.weight",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let lines = ex.preview_lines(&index);
    assert!(
        lines[0].contains("lists 2 matching tensors ('model.mtp.*' ×2, 'mtp.*' ×0)"),
        "{lines:?}"
    );
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[1].contains("warning: --exclude-tensors 'mtp.*' matches no tensor"), "{lines:?}");
    assert!(ex.preview_lines(&[]).is_empty());

    let ex = TensorExclusion::new(&["model.mtp.*".into()]);
    let fresh = ex.report_lines(0);
    assert!(fresh[0].contains("0 source tensors excluded"), "{fresh:?}");
    assert!(fresh[1].contains("warning: --exclude-tensors 'model.mtp.*' matched no tensor in this run"));
    let resumed = ex.report_lines(3);
    assert!(
        resumed[0].contains("resumed: these count only the shards this run converted; 3 source shards"),
        "{resumed:?}"
    );
    assert!(resumed[1].starts_with("  note: "), "{resumed:?}");
    assert!(!resumed.iter().any(|l| l.contains("warning")), "{resumed:?}");
}

/// A pattern over a name only a split produces (`up_proj` of a fused
/// `gate_up_proj`) holds at the writer, beside a source-level pattern; the
/// sibling split part stays.
#[test]
fn exclude_tensors_holds_for_split_produced_names() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_convert_env();
    let dir = tmp("exclude-split");
    let (h, inter, heads, kv, hd, vocab) = (64usize, 128usize, 2usize, 1usize, 32usize, 64usize);
    fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 1,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "max_position_embeddings": 512, "tie_word_embeddings": false
        })
        .to_string(),
    )
    .unwrap();
    let p = "model.layers.0";
    let mut t: Vec<Raw> = vec![
        (
            "model.embed_tokens.weight".into(),
            "BF16",
            vec![vocab, h],
            bf16_bytes(vocab * h, 1, 0.5),
        ),
        (
            "lm_head.weight".into(),
            "BF16",
            vec![vocab, h],
            bf16_bytes(vocab * h, 2, 0.5),
        ),
        (
            "model.norm.weight".into(),
            "BF16",
            vec![h],
            bf16_bytes(h, 3, 0.1),
        ),
    ];
    for (n, shape) in [
        ("input_layernorm.weight", vec![h]),
        ("post_attention_layernorm.weight", vec![h]),
        ("self_attn.q_norm.weight", vec![hd]),
        ("self_attn.k_norm.weight", vec![hd]),
        ("self_attn.q_proj.weight", vec![heads * hd, h]),
        ("self_attn.k_proj.weight", vec![kv * hd, h]),
        ("self_attn.v_proj.weight", vec![kv * hd, h]),
        ("self_attn.o_proj.weight", vec![h, heads * hd]),
        ("mlp.gate_up_proj.weight", vec![2 * inter, h]),
        ("mlp.down_proj.weight", vec![h, inter]),
    ] {
        let n_el = shape.iter().product();
        t.push((
            format!("{p}.{n}"),
            "BF16",
            shape,
            bf16_bytes(n_el, n_el as u64, 0.3),
        ));
    }
    fs::write(dir.join("model.safetensors"), raw_st(&t)).unwrap();
    let out = dir.join("q8.cmf");
    run_convert_multi_excluding(
        dir.to_str().unwrap(),
        &[("q8".into(), out.to_str().unwrap().into())],
        None,
        None,
        None,
        false,
        MimoTowers::TextOnly,
        &parse_tensor_exclusions(&["model.layers.0.mlp.up_proj.weight,lm_head.weight".into()])
            .unwrap(),
        |_| {},
    )
    .unwrap();
    let m = CmfModel::open(&out).unwrap();
    assert!(m.verify().is_empty(), "{:?}", m.verify());
    assert!(m.tensor("model.layers.0.mlp.gate_proj.weight").is_some());
    assert!(m.tensor("model.layers.0.mlp.down_proj.weight").is_some());
    assert!(m.tensor("model.layers.0.mlp.up_proj.weight").is_none());
    assert!(m.tensor("lm_head.weight").is_none());
    assert!(m.tensor("model.embed_tokens.weight").is_some());
    assert_eq!(
        m.header
            .provenance
            .as_ref()
            .and_then(|p| p.get("excluded_tensors")),
        Some(&serde_json::json!([
            "model.layers.0.mlp.up_proj.weight",
            "lm_head.weight"
        ]))
    );
    drop(m);
    let _ = fs::remove_dir_all(&dir);
}

/// `--exclude-tensors 'model.mtp.*'` on a synthetic DeepSeek-V4.1 checkpoint
/// that carries the MTP/DSpark stage (1 layer declared), through the real
/// converter, in one pass to `fp4` and (`--also-output`) `q4tp`:
/// - neither output holds a `model.mtp.*` tensor, both verify, and both
///   record `excluded_tensors` in the provenance; every other tensor is
///   byte-identical to the conversion without the flag (which does hold the
///   stage);
/// - the V4.1 loader takes the file without the stage and the CPU forward
///   of a few tokens equals, bit for bit, the forward of the file with it
///   (the runtime never executes the stage);
/// - an excluded tensor is never decoded: with a NaN scale planted in an
///   MTP expert, the conversion without the flag fails on it and the one
///   with the flag succeeds.
#[test]
fn dsv41_exclude_mtp_drops_the_stack_and_keeps_the_forward() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_convert_env();
    bound_device_budget();
    let src = write_dsv41_source_with("dsv41-mtp", true);
    let source = open_safetensors(&src.dir.join("model.safetensors")).unwrap();
    let mtp_source: Vec<String> = source
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("mtp."))
        .map(|t| t.name.clone())
        .collect();
    // Weights only: an E8M0 `.scale` plane rides with its weight.
    let mtp_written = mtp_source.iter().filter(|n| !n.ends_with(".scale")).count();
    assert!(mtp_written > 20, "fixture stage too small: {mtp_written}");
    drop(source);
    let path = |n: &str| src.dir.join(n).to_str().unwrap().to_string();

    // Without the flag: the stage is in the file.
    run_convert_multi(
        src.dir.to_str().unwrap(),
        &[("fp4".into(), path("full.cmf"))],
        None,
        None,
        None,
        false,
        |_| {},
    )
    .unwrap();
    let full = Arc::new(CmfModel::open(path("full.cmf")).unwrap());
    let in_full: Vec<&str> = full
        .tensors
        .iter()
        .map(|e| e.name.as_str())
        .filter(|n| n.starts_with("model.mtp."))
        .collect();
    assert_eq!(in_full.len(), mtp_written, "{in_full:?}");
    assert!(full.tensor("model.mtp.0.ffn.experts.2.w1.weight").is_some());
    assert!(
        full.tensor("model.mtp.0.markov_head.markov_w1.weight")
            .is_some()
    );

    // With it: one pass, two outputs.
    let exclude = parse_tensor_exclusions(&["model.mtp.*".into()]).unwrap();
    run_convert_multi_excluding(
        src.dir.to_str().unwrap(),
        &[
            ("fp4".into(), path("nomtp.cmf")),
            ("q4tp".into(), path("nomtp-q4tp.cmf")),
        ],
        None,
        None,
        None,
        false,
        MimoTowers::TextOnly,
        &exclude,
        |_| {},
    )
    .unwrap();
    let cut = Arc::new(CmfModel::open(path("nomtp.cmf")).unwrap());
    let cut_q4 = CmfModel::open(path("nomtp-q4tp.cmf")).unwrap();
    for (label, m) in [("fp4", &*cut), ("q4tp", &cut_q4)] {
        assert!(m.verify().is_empty(), "{label}: {:?}", m.verify());
        let stray: Vec<&str> = m
            .tensors
            .iter()
            .map(|e| e.name.as_str())
            .filter(|n| n.starts_with("model.mtp."))
            .collect();
        assert!(stray.is_empty(), "{label}: {stray:?}");
        assert_eq!(
            m.header
                .provenance
                .as_ref()
                .and_then(|p| p.get("excluded_tensors")),
            Some(&serde_json::json!(["model.mtp.*"])),
            "{label}"
        );
        assert_eq!(m.tensors.len(), full.tensors.len() - mtp_written, "{label}");
        // `info` must not claim the MTP head the header still declares.
        assert!(m.arch().mtp.is_some(), "{label}");
        assert_eq!(
            crate::mtp_excluded_by(m).as_deref(),
            Some("'model.mtp.*'"),
            "{label}"
        );
    }
    assert!(
        full.header
            .provenance
            .as_ref()
            .is_some_and(|p| p.get("excluded_tensors").is_none())
    );
    assert_eq!(crate::mtp_excluded_by(&full), None);
    for e in &cut.tensors {
        let f = full.tensor(&e.name).unwrap();
        assert_eq!((e.dtype, &e.shape), (f.dtype, &f.shape), "{}", e.name);
        assert_eq!(cut.entry_bytes(e), full.entry_bytes(f), "{}", e.name);
    }
    drop(cut_q4);

    // The V4.1 runtime loads the file without the stage; the forward is the
    // same bits as with it.
    let ids = [3u32, 9, 27, 1, 44, 8];
    let logits = |m: &Arc<CmfModel>| {
        let mut p = cortiq_engine::Pipeline::from_model(
            m,
            cortiq_engine::sampler::SamplerConfig::default(),
        )
        .unwrap();
        p.forward_ids(&ids, None).unwrap()
    };
    let (a, b) = {
        let _cpu = cortiq_engine::gpu::pause_gpu();
        cortiq_engine::qtensor::with_float_activations(|| (logits(&full), logits(&cut)))
    };
    assert_eq!(a.len(), VOCAB);
    assert!(b.iter().all(|v| v.is_finite()));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b), "the MTP stage changed the forward");
    drop((full, cut));

    // Never decoded: a NaN scale in an MTP expert fails the plain conversion
    // and passes unread under the flag.
    let st = open_safetensors(&src.dir.join("model.safetensors")).unwrap();
    let mut raw: Vec<Raw> = st
        .tensors
        .iter()
        .map(|t| {
            let dt: &'static str = match t.dtype.as_str() {
                "BF16" => "BF16",
                "F32" => "F32",
                "I8" => "I8",
                "F8_E8M0" => "F8_E8M0",
                "F8_E4M3" => "F8_E4M3",
                other => panic!("{other}"),
            };
            (t.name.clone(), dt, t.shape.clone(), st.bytes(t).to_vec())
        })
        .collect();
    drop(st);
    raw.iter_mut()
        .find(|t| t.0 == "mtp.0.ffn.experts.1.w2.scale")
        .unwrap()
        .3[5] = 255;
    fs::write(src.dir.join("model.safetensors"), raw_st(&raw)).unwrap();
    let e = run_convert(
        src.dir.to_str().unwrap(),
        "fp4",
        &path("nan.cmf"),
        None,
        None,
        None,
        false,
        |_| {},
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("NaN scale"), "{e}");
    run_convert_multi_excluding(
        src.dir.to_str().unwrap(),
        &[("fp4".into(), path("nan-nomtp.cmf"))],
        None,
        None,
        None,
        false,
        MimoTowers::TextOnly,
        &exclude,
        |_| {},
    )
    .unwrap();
    assert!(
        CmfModel::open(path("nan-nomtp.cmf"))
            .unwrap()
            .verify()
            .is_empty()
    );
    let _ = fs::remove_dir_all(&src.dir);
}

// ── NVIDIA ModelOpt NVFP4 source ────────────────────────────────────────

/// A tiny ModelOpt-NVFP4 Qwen3 (every linear NVFP4, lm_head BF16 as ModelOpt
/// excludes it) converts in one pass to f16, q4tp, q2tp and mxfp4: the f16
/// output holds exactly f16(e2m1 · e4m3 · weight_scale_2) — with an AWQ
/// pre-quant scale folded into the columns — and no calibration scalar or
/// scale companion reaches any output.
#[test]
fn nvfp4_checkpoint_converts_to_every_profile_with_the_formula_values() {
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    clear_convert_env();
    let dir = tmp("nvfp4");
    let (h, inter, heads, kv, hd, vocab) = (64usize, 128usize, 2usize, 1usize, 32usize, 64usize);
    fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "hidden_size": h, "intermediate_size": inter, "num_hidden_layers": 1,
            "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
            "max_position_embeddings": 512, "tie_word_embeddings": false,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float",
                    "group_size": 16}, "input_activations": {"num_bits": 4, "type": "float",
                    "group_size": 16}, "targets": ["Linear"]}},
                "ignore": ["lm_head"]}
        })
        .to_string(),
    )
    .unwrap();
    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let mut t: Vec<Raw> = vec![
        (
            "model.embed_tokens.weight".into(),
            "BF16",
            vec![vocab, h],
            bf16_bytes(vocab * h, 1, 0.5),
        ),
        (
            "lm_head.weight".into(),
            "BF16",
            vec![vocab, h],
            bf16_bytes(vocab * h, 2, 0.5),
        ),
        (
            "model.norm.weight".into(),
            "BF16",
            vec![h],
            bf16_bytes(h, 3, 0.1),
        ),
    ];
    let p = "model.layers.0";
    for (n, len) in [
        ("input_layernorm.weight", h),
        ("post_attention_layernorm.weight", h),
        ("self_attn.q_norm.weight", hd),
        ("self_attn.k_norm.weight", hd),
    ] {
        t.push((
            format!("{p}.{n}"),
            "BF16",
            vec![len],
            bf16_bytes(len, len as u64, 0.1),
        ));
    }
    let mut expect: HashMap<String, Vec<f32>> = HashMap::new();
    let linears = [
        ("self_attn.q_proj", heads * hd, h),
        ("self_attn.k_proj", kv * hd, h),
        ("self_attn.v_proj", kv * hd, h),
        ("self_attn.o_proj", h, heads * hd),
        ("mlp.gate_proj", inter, h),
        ("mlp.up_proj", inter, h),
        ("mlp.down_proj", h, inter),
    ];
    for (li, (n, rows, cols)) in linears.iter().enumerate() {
        let (rows, cols) = (*rows, *cols);
        let stem = format!("{p}.{n}");
        let packed = bytes(rows * cols / 2, 50 + li as u64);
        // Finite E4M3 block scales in a modest range.
        let sf: Vec<u8> = bytes(rows * cols / 16, 80 + li as u64)
            .iter()
            .map(|b| 0x30 + (b % 24))
            .collect();
        let global = 0.004_f32 * (1.0 + li as f32 * 0.37);
        t.push((
            format!("{stem}.weight"),
            "U8",
            vec![rows, cols / 2],
            packed.clone(),
        ));
        t.push((
            format!("{stem}.weight_scale"),
            "F8_E4M3",
            vec![rows, cols / 16],
            sf.clone(),
        ));
        t.push((
            format!("{stem}.weight_scale_2"),
            "F32",
            vec![],
            f32_bytes(&[global]),
        ));
        t.push((
            format!("{stem}.input_scale"),
            "F32",
            vec![],
            f32_bytes(&[0.5]),
        ));
        // The down projection carries an AWQ pre-quant scale.
        let pqs: Option<Vec<f32>> = (*n == "mlp.down_proj")
            .then(|| (0..cols).map(|j| 0.75 + (j % 5) as f32 * 0.125).collect());
        if let Some(v) = &pqs {
            t.push((
                format!("{stem}.pre_quant_scale"),
                "F32",
                vec![cols],
                f32_bytes(v),
            ));
        }
        let mut w = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let b = packed[(r * cols + c) / 2];
                let code = if c % 2 == 0 { b & 15 } else { b >> 4 };
                let s = (e4m3_ref(sf[(r * cols + c) / 16]) as f32) * global;
                let mut v = (e2m1_ref(code) as f32) * s;
                if let Some(pq) = &pqs {
                    v *= pq[c];
                }
                w[r * cols + c] = v;
            }
        }
        expect.insert(format!("{stem}.weight"), w);
    }
    t.push((
        format!("{p}.self_attn.k_proj.k_scale"),
        "F32",
        vec![],
        f32_bytes(&[1.0]),
    ));
    t.push((
        format!("{p}.self_attn.v_proj.v_scale"),
        "F32",
        vec![],
        f32_bytes(&[1.0]),
    ));
    fs::write(dir.join("model.safetensors"), raw_st(&t)).unwrap();

    let outs: Vec<(String, PathBuf)> = ["f16", "q4tp", "q2tp", "mxfp4"]
        .iter()
        .map(|q| (q.to_string(), dir.join(format!("{q}.cmf"))))
        .collect();
    let spec: Vec<(String, String)> = outs
        .iter()
        .map(|(q, p)| (q.clone(), p.to_str().unwrap().to_string()))
        .collect();
    run_convert_multi(
        dir.to_str().unwrap(),
        &spec,
        None,
        None,
        None,
        false,
        |_| {},
    )
    .unwrap();
    for (q, path) in &outs {
        let m = CmfModel::open(path).unwrap();
        assert!(m.verify().is_empty(), "{q}: {:?}", m.verify());
        for e in &m.tensors {
            assert!(
                !(e.name.ends_with("_scale")
                    || e.name.ends_with("_scale_2")
                    || e.name.contains("pre_quant")),
                "{q}: stray {}",
                e.name
            );
        }
        for (name, want) in &expect {
            let e = m
                .tensor(name)
                .unwrap_or_else(|| panic!("{q}: {name} missing"));
            let mut got = vec![0f32; want.len()];
            cortiq_core::quant::dequant_tensor(e, m.entry_bytes(e), &mut got).unwrap();
            match q.as_str() {
                "f16" => {
                    assert_eq!(e.dtype, TensorDtype::F16, "{name}");
                    for (i, (g, w)) in got.iter().zip(want).enumerate() {
                        let w16 = f16_to_f32(f32_to_f16(*w));
                        assert_eq!(g.to_bits(), w16.to_bits(), "{name}[{i}]: {g} vs {w16}");
                    }
                }
                "mxfp4" => {
                    assert_eq!(e.dtype, TensorDtype::Mxfp4, "{name}");
                    // The OCP re-encode stays within one grid step of the
                    // NVFP4 values (different block size, power-of-two scale).
                    for g in 0..want.len() / 32 {
                        let grp = &want[g * 32..(g + 1) * 32];
                        let amax = grp.iter().fold(0f32, |m, v| m.max(v.abs()));
                        let step = 2f32.powi(amax.log2().floor() as i32 - 2);
                        for j in 0..32 {
                            assert!((got[g * 32 + j] - grp[j]).abs() <= 2.0 * step, "{name}");
                        }
                    }
                }
                _ => {
                    // q4tp / q2tp: the profile's codecs on the decoded values.
                    let num: f64 = got
                        .iter()
                        .zip(want)
                        .map(|(a, b)| ((a - b) as f64).powi(2))
                        .sum();
                    let den: f64 = want.iter().map(|b| (*b as f64).powi(2)).sum();
                    let tol = if q == "q4tp" { 0.05 } else { 0.5 };
                    assert!(num / den < tol, "{q} {name}: rel. error {}", num / den);
                }
            }
        }
    }
    let _ = fs::remove_dir_all(&dir);
}
