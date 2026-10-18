//! A tiny synthetic DeepSeek-V4.1 checkpoint and an independent f64
//! reference forward written from the official `inference/model.py`,
//! `kernel.py` and `engram.py` (DeepSeek-V4.1-Flash-1017, MIT).
//!
//! The fixture has every structural feature the runtime has to get right:
//!   L0  ratio 0, window only, q4tp gate/up experts
//!   L1  ratio 2, KV source + index source (owns the index keys), Engram,
//!       q2tp gate/up experts
//!   L2  ratio 1, second KV source + index source + candidate source
//!   L3  ratio 1, index consumer of L2's keys, uses L2's candidates,
//!       q2tp gate/up experts
//!   L4  ratio 0 after the sources (window only), F32 attention linears
//!
//! The reference follows the BF16 checkpoint mode of `model.py` (weights
//! are already dequantized, so `linear()` is a plain BF16 GEMM): every
//! tensor the reference materializes as BF16 is rounded here as well, and
//! the explicit activation quantizers (window-KV FP8, compressed-KV FP4 with
//! E4M3 scales, index q/k FP4 with E8M0 scales) are re-implemented from the
//! kernel definitions without calling any cortiq helper. Accumulation is
//! f64, so the comparison measures the runtime's arithmetic, not a second
//! copy of it.
//!
//! Deliberate, documented differences between cortiq and the official
//! kernels are switchable through [`RefOpts`]; see `docs/dsv41/WAVE1_A.md`.

#![allow(dead_code)]

use cortiq_core::CMF_VERSION;
use cortiq_core::format::{CmfHeader, TensorSpec};
use cortiq_core::quant::{
    dequant_q2tp, dequant_q4tp, f32_to_f16, q2tp_sections, q4tp_put_code, q4tp_sections,
};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use serde_json::{Value, json};
use std::collections::HashMap;

pub const VOCAB: usize = 64;
pub const DIM: usize = 64;
pub const N_LAYERS: usize = 5;
pub const N_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const ROPE_DIM: usize = 16;
pub const Q_LORA: usize = 32;
pub const O_LORA: usize = 32;
pub const O_GROUPS: usize = 2;
pub const HC: usize = 4;
pub const EXPERTS: usize = 4;
pub const TOP_K: usize = 2;
pub const INTER: usize = 64;
pub const WINDOW: usize = 4;
pub const IH: usize = 2;
pub const ID: usize = 32;
pub const INDEX_TOPK: usize = 3;
pub const RATIOS: [usize; N_LAYERS] = [0, 2, 1, 1, 0];
pub const KV_SOURCES: [usize; 2] = [1, 2];
pub const INDEX_SOURCES: [usize; 3] = [1, 2, 3];
pub const CANDIDATE_SOURCE: usize = 2;
pub const CAND_BLOCKS: usize = 2;
pub const CAND_BS: usize = 2;
pub const ENGRAM_LAYER: usize = 1;
pub const ENGRAM_MAX_NGRAM: usize = 4;
pub const ENGRAM_HEADS: usize = 2;
pub const ENGRAM_HEAD_DIM: usize = 32;
pub const ENGRAM_TABLE_VOCAB: usize = 5;
pub const ENGRAM_ROWS: usize = 72;
pub const PAD_TOKEN: usize = 2;
pub const SWIGLU: f64 = 1.0;
pub const ROUTE_SCALE: f64 = 1.5;
pub const EPS: f64 = 1e-6;
pub const HC_EPS: f64 = 1e-6;
pub const SINKHORN: usize = 20;
pub const ROPE_THETA: f64 = 10_000.0;
pub const COMPRESS_THETA: f64 = 160_000.0;
pub const ROPE_FACTOR: f64 = 16.0;
pub const ORIG_SEQ: f64 = 65_536.0;
pub const BETA_FAST: f64 = 32.0;
pub const BETA_SLOW: f64 = 1.0;
/// The release's compressed tokenizer size. Engram layer 1 with this size
/// uses the published hash multipliers, so the reference hash below needs
/// no NumPy PCG64 port.
pub const RELEASE_COMPRESSED_VOCAB: usize = 99_092;
/// `np.random.default_rng(10007 * 1).integers(0, bound, 4) * 2 + 1` with
/// `bound = (i64::MAX // 99092) // 2`, as `engram.compute_hash_multipliers`
/// computes it (re-derived with NumPy 1.26 for this fixture).
pub const LAYER1_MULTIPLIERS: [u64; 4] = [
    76632096046245,
    4839876093313,
    35959672319349,
    73987337458391,
];

/// Twelve tokens: repeated bigrams exercise the Engram hash, the count
/// passes the window (4) and the index top-k (3) several times over.
pub const TOKENS: [u32; 12] = [3, 17, 42, 5, 17, 42, 9, 60, 3, 17, 33, 63];

/// Gain of the residual-writing projections (`wo_b`, expert `w2`). At a
/// unit gain this random network is chaotic: one BF16 rounding flip early
/// in the stream (f32 versus f64 accumulation landing on either side of a
/// rounding boundary) moves later FP4 cache codes and index picks and the
/// logits by 10-70 %. A quarter gain keeps the blocks residual-dominated,
/// so an isolated flip costs ~3e-3 while a real regression still moves
/// every position.
const OUT_GAIN: f64 = 0.25;

fn out_gain() -> f64 {
    OUT_GAIN
}

/// Deterministic splitmix64 stream.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Round to BF16, nearest-even, from f64 (one rounding, as a BF16 tensor
/// written from an exact accumulator).
pub fn bf(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let shift = 52 - 7;
    let half = (1u64 << (shift - 1)) - 1 + ((bits >> shift) & 1);
    f64::from_bits(bits.wrapping_add(half) & !((1u64 << shift) - 1))
}

pub fn bf32(x: f32) -> f32 {
    bf(x as f64) as f32
}

fn bf_vec(v: &mut [f64]) {
    for x in v {
        *x = bf(*x);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// F32 storage of BF16-representable values (a BF16 checkpoint weight).
    Bf16,
    /// F32 storage of full-precision values (the reference's FP32 params).
    F32,
    Q4tp,
    Q2tp,
}

pub struct Weight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

pub struct Fixture {
    pub specs: Vec<TensorSpec>,
    pub w: HashMap<String, Weight>,
    pub engram_bytes: Vec<u8>,
    pub engram_scales: Vec<u8>,
    pub source: Value,
    pub compressed_vocab: usize,
    /// `Some(CANDIDATE_SOURCE)`, or `None` written as
    /// `candidate_source_layer_id: -1` (pre-filtering off).
    pub candidate_source: Option<usize>,
    rng: Rng,
}

fn q4tp_payload(rng: &mut Rng, rows: usize, cols: usize, lo: f32, step: f32) -> Vec<u8> {
    let gpr = cols / 32;
    let (params_off, codes_off, stride) = q4tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    for byte in out[..params_off].iter_mut() {
        *byte = rng.next_u64() as u8;
    }
    for r in 0..rows {
        let p = params_off + r * 4;
        out[p..p + 2].copy_from_slice(&f32_to_f16(lo).to_le_bytes());
        out[p + 2..p + 4].copy_from_slice(&f32_to_f16(step).to_le_bytes());
        for g in 0..gpr {
            q4tp_put_code(
                &mut out[codes_off + r * stride..codes_off + (r + 1) * stride],
                g,
                (rng.next_u64() % 32) as usize,
            );
        }
    }
    out
}

fn q2tp_payload(rng: &mut Rng, rows: usize, cols: usize, lo: f32, step: f32) -> Vec<u8> {
    let gpr = cols / 32;
    let (params_off, codes_off, stride) = q2tp_sections(rows, cols);
    let mut out = vec![0u8; codes_off + rows * stride];
    for byte in out[..params_off].iter_mut() {
        *byte = rng.next_u64() as u8;
    }
    for r in 0..rows {
        let p = params_off + r * 4;
        out[p..p + 2].copy_from_slice(&f32_to_f16(lo).to_le_bytes());
        out[p + 2..p + 4].copy_from_slice(&f32_to_f16(step).to_le_bytes());
        for g in 0..gpr {
            q4tp_put_code(
                &mut out[codes_off + r * stride..codes_off + (r + 1) * stride],
                g,
                (rng.next_u64() % 32) as usize,
            );
        }
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Fixture {
    /// Build the fixture. `compressed_vocab` is the Engram compressed
    /// tokenizer size written into the config: the release value for the
    /// reference test (with an explicit token map), or 1 for a container
    /// loaded through `Pipeline` without a tokenizer (every id then decodes
    /// to the same empty key).
    pub fn build(compressed_vocab: usize) -> Self {
        Self::build_with(compressed_vocab, Some(CANDIDATE_SOURCE))
    }

    /// Same weights (one seed), with candidate pre-filtering on or off.
    pub fn build_with(compressed_vocab: usize, candidate_source: Option<usize>) -> Self {
        let mut fx = Self {
            specs: Vec::new(),
            w: HashMap::new(),
            engram_bytes: Vec::new(),
            engram_scales: Vec::new(),
            source: Value::Null,
            compressed_vocab,
            candidate_source,
            rng: Rng::new(0xD541_F1C5),
        };
        fx.source = source_config(compressed_vocab, candidate_source);
        fx.vec_bf16("model.embed_tokens.weight", &[VOCAB, DIM], 1.0, 0.0);
        fx.vec_bf16("model.norm.weight", &[DIM], 0.1, 1.0);
        fx.linear("lm_head.weight", VOCAB, DIM, Kind::Bf16);
        for li in 0..N_LAYERS {
            let p = format!("model.layers.{li}");
            let a = format!("{p}.self_attn");
            fx.vec_bf16(&format!("{p}.input_layernorm.weight"), &[DIM], 0.1, 1.0);
            fx.vec_bf16(
                &format!("{p}.post_attention_layernorm.weight"),
                &[DIM],
                0.1,
                1.0,
            );
            let attn = if li == 4 { Kind::Bf16 } else { Kind::Q4tp };
            fx.linear(&format!("{a}.wq_a.weight"), Q_LORA, DIM, attn);
            fx.vec_bf16(&format!("{a}.q_norm.weight"), &[Q_LORA], 0.1, 1.0);
            fx.linear(&format!("{a}.wq_b.weight"), N_HEADS * HEAD_DIM, Q_LORA, attn);
            fx.linear(&format!("{a}.wkv.weight"), HEAD_DIM, DIM, attn);
            fx.vec_bf16(&format!("{a}.kv_norm.weight"), &[HEAD_DIM], 0.1, 1.0);
            fx.linear(
                &format!("{a}.wo_a.weight"),
                O_GROUPS * O_LORA,
                N_HEADS * HEAD_DIM / O_GROUPS,
                attn,
            );
            fx.linear_gain(&format!("{a}.wo_b.weight"), DIM, O_GROUPS * O_LORA, attn, out_gain());
            fx.vec_f32(&format!("{a}.attn_sink"), &[N_HEADS], 1.0, 0.0);
            if KV_SOURCES.contains(&li) {
                let ck = if RATIOS[li] > 1 { Kind::F32 } else { Kind::Bf16 };
                fx.linear(&format!("{a}.compressor.wkv.weight"), HEAD_DIM, DIM, ck);
                if RATIOS[li] > 1 {
                    fx.linear(&format!("{a}.compressor.wgate.weight"), HEAD_DIM, DIM, ck);
                }
                fx.vec_bf16(&format!("{a}.compressor.norm.weight"), &[HEAD_DIM], 0.1, 1.0);
            }
            if INDEX_SOURCES.contains(&li) {
                fx.linear(&format!("{a}.indexer.wq_b.weight"), IH * ID, Q_LORA, Kind::Q4tp);
                fx.linear(&format!("{a}.indexer.weights_proj.weight"), IH, DIM, Kind::Bf16);
                if KV_SOURCES.contains(&li) {
                    fx.linear(&format!("{a}.indexer.wk.weight"), ID, HEAD_DIM, Kind::Bf16);
                    fx.vec_bf16(&format!("{a}.indexer.k_norm.weight"), &[ID], 0.1, 1.0);
                }
            }
            let mix = (2 + HC) * HC;
            for part in ["attn", "ffn"] {
                fx.vec_f32(&format!("{p}.hc_{part}_fn"), &[mix, HC * DIM], 0.05, 0.0);
                fx.vec_f32(&format!("{p}.hc_{part}_base"), &[mix], 0.5, 0.0);
                fx.vec_f32(&format!("{p}.hc_{part}_scale"), &[3], 0.2, 1.0);
            }
            // The router is an FP32 linear in the reference.
            fx.vec_bf16(&format!("{p}.mlp.gate.weight"), &[EXPERTS, DIM], 0.4, 0.0);
            fx.vec_f32(&format!("{p}.mlp.expert_bias"), &[EXPERTS], 0.1, 0.0);
            let gu = if li % 2 == 1 { Kind::Q2tp } else { Kind::Q4tp };
            for e in 0..EXPERTS {
                let x = format!("{p}.mlp.experts.{e}");
                fx.linear(&format!("{x}.gate_proj.weight"), INTER, DIM, gu);
                fx.linear(&format!("{x}.up_proj.weight"), INTER, DIM, gu);
                fx.linear_gain(&format!("{x}.down_proj.weight"), DIM, INTER, Kind::Q4tp, out_gain());
            }
            let x = format!("{p}.mlp.shared_experts");
            fx.linear(&format!("{x}.gate_proj.weight"), INTER, DIM, gu);
            fx.linear(&format!("{x}.up_proj.weight"), INTER, DIM, gu);
            fx.linear_gain(&format!("{x}.down_proj.weight"), DIM, INTER, Kind::Q4tp, out_gain());
            if li == ENGRAM_LAYER {
                fx.engram(&p);
            }
        }
        // Keep the last mapped quantized tensor away from the end-of-file
        // page edge (the runtime's no-copy weight cache wants that slack).
        fx.specs.push(TensorSpec {
            name: "pad".into(),
            dtype: TensorDtype::F32,
            shape: vec![4096, 2],
            data: vec![0; 4096 * 8],
        });
        fx
    }

    fn vec_bf16(&mut self, name: &str, shape: &[usize], std: f64, mean: f64) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| bf32((mean + std * self.rng.normal()) as f32))
            .collect();
        self.push_f32(name, shape, data);
    }

    fn vec_f32(&mut self, name: &str, shape: &[usize], std: f64, mean: f64) {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| (mean + std * self.rng.normal()) as f32)
            .collect();
        self.push_f32(name, shape, data);
    }

    fn push_f32(&mut self, name: &str, shape: &[usize], data: Vec<f32>) {
        let (rows, cols) = match shape {
            [n] => (1, *n),
            [r, c] => (*r, *c),
            _ => unreachable!(),
        };
        self.specs.push(TensorSpec {
            name: name.into(),
            dtype: TensorDtype::F32,
            shape: shape.to_vec(),
            data: f32_bytes(&data),
        });
        self.w.insert(name.into(), Weight { data, rows, cols });
    }

    fn linear(&mut self, name: &str, rows: usize, cols: usize, kind: Kind) {
        self.linear_gain(name, rows, cols, kind, 1.0);
    }

    fn linear_gain(&mut self, name: &str, rows: usize, cols: usize, kind: Kind, gain: f64) {
        let std = gain / (cols as f64).sqrt();
        match kind {
            Kind::Bf16 | Kind::F32 => {
                let data: Vec<f32> = (0..rows * cols)
                    .map(|_| {
                        let v = (std * self.rng.normal()) as f32;
                        if kind == Kind::Bf16 { bf32(v) } else { v }
                    })
                    .collect();
                self.push_f32(name, &[rows, cols], data);
            }
            Kind::Q4tp | Kind::Q2tp => {
                // Uniform nibbles have std 4.61, q2tp levels 1.118; aim the
                // mean rung at a 1/sqrt(fan-in) weight std.
                let level_std = if kind == Kind::Q4tp { 4.61 } else { 1.118 };
                let lo = ((std / level_std).log2() - 0.45) as f32;
                let step = 0.03f32;
                let (payload, dtype) = if kind == Kind::Q4tp {
                    (
                        q4tp_payload(&mut self.rng, rows, cols, lo, step),
                        TensorDtype::Q4TiledP,
                    )
                } else {
                    (
                        q2tp_payload(&mut self.rng, rows, cols, lo, step),
                        TensorDtype::Q2TiledP,
                    )
                };
                let mut data = vec![0.0f32; rows * cols];
                if kind == Kind::Q4tp {
                    dequant_q4tp(&payload, rows, cols, &mut data);
                } else {
                    dequant_q2tp(&payload, rows, cols, &mut data);
                }
                self.specs.push(TensorSpec {
                    name: name.into(),
                    dtype,
                    shape: vec![rows, cols],
                    data: payload,
                });
                self.w.insert(name.into(), Weight { data, rows, cols });
            }
        }
    }

    fn engram(&mut self, p: &str) {
        let cols = ENGRAM_HEAD_DIM;
        let mut bytes = vec![0u8; ENGRAM_ROWS * cols];
        for b in bytes.iter_mut() {
            let mut v = self.rng.next_u64() as u8;
            // 0x7f / 0xff are E4M3FN NaN; a real table has none.
            if v & 0x7f == 0x7f {
                v ^= 1;
            }
            *b = v;
        }
        let scales: Vec<u8> = (0..ENGRAM_ROWS * cols / 32)
            .map(|_| 117 + (self.rng.next_u64() % 3) as u8)
            .collect();
        self.specs.push(TensorSpec {
            name: format!("{p}.engram.embed.weight"),
            dtype: TensorDtype::U8,
            shape: vec![ENGRAM_ROWS, cols],
            data: bytes.clone(),
        });
        self.specs.push(TensorSpec {
            name: format!("{p}.engram.embed.scale"),
            dtype: TensorDtype::U8,
            shape: vec![ENGRAM_ROWS, cols / 32],
            data: scales.clone(),
        });
        self.engram_bytes = bytes;
        self.engram_scales = scales;
        let hash_cols = (ENGRAM_MAX_NGRAM - 1) * ENGRAM_HEADS;
        self.linear(
            &format!("{p}.engram.wkv.weight"),
            DIM * (HC + 1),
            hash_cols * cols,
            Kind::Q4tp,
        );
        self.vec_bf16(&format!("{p}.engram.q_weight"), &[HC, DIM], 0.1, 1.0);
        self.vec_bf16(&format!("{p}.engram.k_weight"), &[HC, DIM], 0.1, 1.0);
    }

    pub fn arch(&self) -> ModelArch {
        serde_json::from_value(json!({
            "arch_name": "deepseek_v41",
            "hidden_size": DIM,
            "intermediate_size": INTER,
            "num_layers": N_LAYERS,
            "num_attention_heads": N_HEADS,
            "num_kv_heads": 1,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "layer_types": vec!["FullAttention"; N_LAYERS],
            "rms_norm_eps": EPS,
            "max_position_embeddings": 4096,
            "linear_conv_kernel_dim": 0,
            "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
            "moe": {
                "num_experts": EXPERTS,
                "top_k": TOP_K,
                "moe_intermediate_size": INTER,
                "routed_scaling_factor": ROUTE_SCALE,
            },
            "deepseek_v41": self.source,
        }))
        .expect("fixture arch")
    }

    pub fn header(&self) -> CmfHeader {
        CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch: self.arch(),
            quant_type: QuantType::Q4Block,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
            routing: None,
            genome: None,
            lineage: Vec::new(),
            router: None,
            segments: Vec::new(),
        }
    }

    /// Write the container, open it and remove its temporary directory.
    /// The mapping keeps the unlinked file alive for the model's lifetime
    /// (unix), so nothing is left in the temp dir after the test.
    pub fn open(&self, tag: &str) -> std::sync::Arc<cortiq_core::CmfModel> {
        let path = self.write(tag);
        let model = cortiq_core::CmfModel::open(&path).expect("open fixture");
        if let Some(dir) = path.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
        std::sync::Arc::new(model)
    }

    /// Write the container into a fresh directory and return its path.
    /// Prefer `open`, which also cleans the directory up.
    pub fn write(&self, tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cmf-dsv41-fixture-{tag}-{}-{}",
            std::process::id(),
            std::thread::current()
                .name()
                .unwrap_or("t")
                .replace("::", "_")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dsv41_fixture.cmf");
        cortiq_core::CmfModel::write(&path, &self.header(), &self.specs, None, None)
            .expect("write fixture CMF");
        path
    }

    /// The compressed Engram token map handed to `dsv41::load`. For the
    /// release vocabulary it spreads the 64 fixture ids over the 99092-entry
    /// compressed space (max id 99091, so the size check holds).
    pub fn token_map(&self) -> Vec<u32> {
        if self.compressed_vocab == RELEASE_COMPRESSED_VOCAB {
            (0..VOCAB as u32)
                .map(|i| {
                    if i as usize == VOCAB - 1 {
                        (RELEASE_COMPRESSED_VOCAB - 1) as u32
                    } else {
                        i * 1543
                    }
                })
                .collect()
        } else {
            vec![0; VOCAB]
        }
    }

    fn wt(&self, name: &str) -> &Weight {
        self.w
            .get(name)
            .unwrap_or_else(|| panic!("fixture weight {name}"))
    }
}

/// The `config.json` preserved in the container, shaped like the release
/// file (`text_config` nesting, HF key spellings).
pub fn source_config(compressed_vocab: usize, candidate_source: Option<usize>) -> Value {
    let candidate = candidate_source.map_or(-1, |layer| layer as i64);
    json!({
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4"
        },
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": VOCAB,
            "hidden_size": DIM,
            "moe_intermediate_size": INTER,
            "num_hidden_layers": N_LAYERS,
            "num_attention_heads": N_HEADS,
            "num_key_value_heads": 1,
            "head_dim": HEAD_DIM,
            "qk_rope_head_dim": ROPE_DIM,
            "q_lora_rank": Q_LORA,
            "o_lora_rank": O_LORA,
            "o_groups": O_GROUPS,
            "hidden_act": "silu",
            "swiglu_limit": SWIGLU,
            "rms_norm_eps": EPS,
            "rope_theta": ROPE_THETA,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": ROPE_FACTOR,
                "beta_fast": BETA_FAST,
                "beta_slow": BETA_SLOW,
                "original_max_position_embeddings": ORIG_SEQ as u64
            },
            "n_routed_experts": EXPERTS,
            "n_shared_experts": 1,
            "num_experts_per_tok": TOP_K,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": true,
            "routed_scaling_factor": ROUTE_SCALE,
            "sliding_window": WINDOW,
            "compress_ratios": RATIOS,
            "compress_rope_theta": COMPRESS_THETA,
            "kv_source_layer_ids": KV_SOURCES,
            "index_source_layer_ids": INDEX_SOURCES,
            "index_n_heads": IH,
            "index_head_dim": ID,
            "index_topk": INDEX_TOPK,
            "candidate_source_layer_id": candidate,
            "candidate_topk_blocks": CAND_BLOCKS,
            "candidate_block_size": CAND_BS,
            "hc_mult": HC,
            "hc_sinkhorn_iters": SINKHORN,
            "hc_eps": HC_EPS,
            "engram_layer_ids": [ENGRAM_LAYER],
            "engram_num_embeddings": [ENGRAM_ROWS],
            "engram_max_ngram_size": ENGRAM_MAX_NGRAM,
            "engram_vocab_size": ENGRAM_TABLE_VOCAB,
            "engram_n_heads": ENGRAM_HEADS,
            "engram_head_dim": ENGRAM_HEAD_DIM,
            "engram_pad_token_id": PAD_TOKEN,
            "engram_compressed_vocab_size": compressed_vocab
        }
    })
}

// ───────────────────────── reference quantizers ─────────────────────────

fn round_half_even(x: f64) -> f64 {
    let f = x.floor();
    let d = x - f;
    if d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0) {
        f + 1.0
    } else {
        f
    }
}

/// `fast_log2_ceil` of kernel.py: exponent field plus one when any mantissa
/// bit is set (normal f32 inputs).
pub fn log2_ceil_f32(x: f32) -> i32 {
    let b = x.to_bits();
    ((b >> 23) & 0xff) as i32 - 127 + i32::from(b & 0x7f_ffff != 0)
}

/// Saturating E4M3FN round-to-nearest-even.
pub fn e4m3_rne(v: f64) -> f64 {
    let a = v.abs().min(448.0);
    if a == 0.0 {
        return 0.0;
    }
    let q = if a < 2f64.powi(-6) {
        2f64.powi(-9)
    } else {
        let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        2f64.powi(e - 3)
    };
    let r = (round_half_even(a / q) * q).min(448.0);
    if v < 0.0 { -r } else { r }
}

/// Saturating E2M1 round-to-nearest-even.
pub fn e2m1_rne(v: f64) -> f64 {
    let a = v.abs().min(6.0);
    let q = if a < 2.0 {
        0.5
    } else if a < 4.0 {
        1.0
    } else {
        2.0
    };
    let r = (round_half_even(a / q) * q).min(6.0);
    if v < 0.0 { -r } else { r }
}

/// `act_quant(x, 32, "ue8m0", e8m0, inplace=True)`.
pub fn act_quant_fp8(v: &mut [f64]) {
    for chunk in v.chunks_mut(32) {
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs() as f32));
        let amax = amax.max(1e-4f32);
        let s = 2f64.powi(log2_ceil_f32(amax * (1.0f32 / 448.0f32)));
        for x in chunk {
            *x = bf(e4m3_rne((*x / s).clamp(-448.0, 448.0)) * s);
        }
    }
}

/// `fp4_act_quant(x, 32, True)` with E8M0 scales (indexer q and k).
fn fp4_e8m0(v: &mut [f64]) {
    for chunk in v.chunks_mut(32) {
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs() as f32));
        let amax = amax.max(6.0 * 2f32.powi(-126));
        let s = 2f64.powi(log2_ceil_f32(amax * (1.0f32 / 6.0f32)));
        for x in chunk {
            *x = bf(e2m1_rne((*x / s).clamp(-6.0, 6.0)) * s);
        }
    }
}

/// `fp4_act_quant(latent, 16, True, scale_dtype=float8_e4m3fn)`.
fn fp4_e4m3(v: &mut [f64]) {
    for chunk in v.chunks_mut(16) {
        let amax = chunk.iter().fold(0.0f32, |m, &x| m.max(x.abs() as f32));
        let amax = amax.max(6.0 * 2f32.powi(-9));
        let s = e4m3_rne((amax / 6.0f32) as f64);
        for x in chunk {
            *x = bf(e2m1_rne((*x / s).clamp(-6.0, 6.0)) * s);
        }
    }
}

fn e4m3_decode(b: u8) -> f64 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0f) as i32;
    let m = (b & 7) as f64;
    if e == 0 {
        sign * m * 2f64.powi(-9)
    } else {
        sign * (1.0 + m / 8.0) * 2f64.powi(e - 7)
    }
}

// ───────────────────────────── reference model ─────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct RefOpts {
    /// `sparse_attn_kernel` casts the unnormalized probabilities to BF16
    /// (`acc_s_cast`) before the PV GEMM; cortiq keeps them in f32.
    pub bf16_attn_probs: bool,
    /// `sparse_attn` writes a BF16 `o` that the inverse RoPE then reads;
    /// cortiq rotates the f32 accumulator and rounds once afterwards.
    pub bf16_attn_out: bool,
    /// `act_quant(o)` before the grouped `wo_a` (reference commit 4dd75a8b).
    pub o_fp8: bool,
}

impl RefOpts {
    /// The runtime's documented semantics (the two attention BF16
    /// boundaries above are known, journaled differences).
    pub fn cortiq() -> Self {
        Self {
            bf16_attn_probs: false,
            bf16_attn_out: false,
            o_fp8: false,
        }
    }
    /// The official kernels exactly.
    pub fn kernels() -> Self {
        Self {
            bf16_attn_probs: true,
            bf16_attn_out: true,
            o_fp8: true,
        }
    }
}

fn mv(w: &Weight, x: &[f64]) -> Vec<f64> {
    assert_eq!(w.cols, x.len());
    (0..w.rows)
        .map(|r| {
            w.data[r * w.cols..(r + 1) * w.cols]
                .iter()
                .zip(x)
                .map(|(&a, &b)| a as f64 * b)
                .sum()
        })
        .collect()
}

fn rmsnorm(x: &mut [f64], w: &Weight) {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + EPS).sqrt();
    for (v, &g) in x.iter_mut().zip(&w.data) {
        *v = bf(*v * r * g as f64);
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `DSV41_REF_TRACE=1` prints the same stage summaries as the runtime's
/// `CMF_DSV41_TRACE=1`, for bisecting a disagreement.
fn tr(stage: &str, pos: usize, layer: Option<usize>, v: &[f64]) {
    if std::env::var_os("DSV41_REF_TRACE").is_none() {
        return;
    }
    let l2 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let head: Vec<f32> = v.iter().take(8).map(|&x| x as f32).collect();
    match layer {
        Some(l) => eprintln!("ref_trace stage={stage} position={pos} layer={l} l2={l2:.9e} head={head:?}"),
        None => eprintln!("ref_trace stage={stage} position={pos} l2={l2:.9e} head={head:?}"),
    }
}

/// `precompute_freqs_cis` angles per pair (f64).
pub fn ref_freqs(dim: usize, base: f64, orig: f64) -> Vec<f64> {
    let d = dim as f64;
    let mut f: Vec<f64> = (0..dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f64 / d))
        .collect();
    if orig > 0.0 {
        let corrected =
            |rot: f64| d * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
        let low = corrected(BETA_FAST).floor().max(0.0);
        let high = corrected(BETA_SLOW).ceil().min(d - 1.0);
        for (i, v) in f.iter_mut().enumerate() {
            let ramp = ((i as f64 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *v = *v / ROPE_FACTOR * (1.0 - smooth) + *v * smooth;
        }
    }
    f
}

/// `apply_rotary_emb` on the last `ROPE_DIM` elements, adjacent pairs.
fn rope(x: &mut [f64], freqs: &[f64], pos: usize, inverse: bool) {
    let base = x.len() - ROPE_DIM;
    for i in 0..ROPE_DIM / 2 {
        let th = pos as f64 * freqs[i];
        let (s, c) = th.sin_cos();
        let s = if inverse { -s } else { s };
        let a = x[base + 2 * i];
        let b = x[base + 2 * i + 1];
        x[base + 2 * i] = a * c - b * s;
        x[base + 2 * i + 1] = a * s + b * c;
    }
}

struct RefState {
    window: Vec<Vec<Vec<f64>>>,
    ckv: Vec<Vec<Vec<f64>>>,
    cscore: Vec<Vec<Vec<f64>>>,
    compress_kv: Vec<Vec<Vec<f64>>>,
    index_k: Vec<Vec<Vec<f64>>>,
    topk: Vec<usize>,
    candidates: Vec<bool>,
    history: Vec<i64>,
}

fn source_slot(li: usize) -> usize {
    KV_SOURCES
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s <= li)
        .map(|(i, _)| i)
        .max()
        .expect("ratio>0 layer after a KV source")
}

fn is_prime(v: u64) -> bool {
    v >= 2 && (2..).take_while(|d| d * d <= v).all(|d| v % d != 0)
}

/// `EngramLayout.from_args` primes for the single fixture layer.
fn engram_primes() -> Vec<Vec<u64>> {
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for _ in 0..ENGRAM_MAX_NGRAM - 1 {
        let mut current = ENGRAM_TABLE_VOCAB as u64 - 1;
        let mut sizes = Vec::new();
        for _ in 0..ENGRAM_HEADS {
            let mut c = current + 1;
            while !is_prime(c) || seen.contains(&c) {
                c += 1;
            }
            current = c;
            seen.push(c);
            sizes.push(c);
        }
        out.push(sizes);
    }
    out
}

/// `NgramHashState.forward` for one decode position.
fn engram_hash(history: &mut Vec<i64>, compressed: i64, pad: i64) -> Vec<usize> {
    history.push(compressed);
    let p = history.len() - 1;
    let primes = engram_primes();
    let flat: Vec<u64> = primes.iter().flatten().copied().collect();
    let mut offsets = vec![0u64; flat.len()];
    for i in 1..flat.len() {
        offsets[i] = offsets[i - 1] + flat[i - 1];
    }
    assert_eq!(flat.iter().sum::<u64>() as usize, ENGRAM_ROWS);
    let mut blocked = false;
    let mut products = Vec::new();
    for shift in 0..ENGRAM_MAX_NGRAM {
        let source = if p >= shift { history[p - shift] } else { -1 };
        blocked |= p < shift || source == -1;
        let tok = if blocked { pad } else { source };
        products.push(tok as u64 * LAYER1_MULTIPLIERS[shift]);
    }
    let mut rolling = products[0];
    let mut out = Vec::new();
    for i in 1..ENGRAM_MAX_NGRAM {
        rolling ^= products[i];
        for h in 0..ENGRAM_HEADS {
            let col = (i - 1) * ENGRAM_HEADS + h;
            out.push((rolling % primes[i - 1][h] + offsets[col]) as usize);
        }
    }
    out
}

fn hc_mixes(fx: &Fixture, h: &[f64], part: &str, li: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let p = format!("model.layers.{li}");
    let fn_w = fx.wt(&format!("{p}.hc_{part}_fn"));
    let base = &fx.wt(&format!("{p}.hc_{part}_base")).data;
    let scale = &fx.wt(&format!("{p}.hc_{part}_scale")).data;
    let ms = h.iter().map(|v| v * v).sum::<f64>() / h.len() as f64;
    let rs = 1.0 / (ms + EPS).sqrt();
    let mixes: Vec<f64> = mv(fn_w, h).into_iter().map(|v| v * rs).collect();
    let b = |i: usize| base[i] as f64;
    let s = |i: usize| scale[i] as f64;
    let pre: Vec<f64> = (0..HC)
        .map(|j| sigmoid(mixes[j] * s(0) + b(j)) + HC_EPS)
        .collect();
    let post: Vec<f64> = (0..HC)
        .map(|j| 2.0 * sigmoid(mixes[j + HC] * s(1) + b(j + HC)))
        .collect();
    let mut comb = vec![0.0f64; HC * HC];
    for j in 0..HC {
        for k in 0..HC {
            let i = j * HC + k + 2 * HC;
            comb[j * HC + k] = mixes[i] * s(2) + b(i);
        }
    }
    for j in 0..HC {
        let row = &mut comb[j * HC..(j + 1) * HC];
        let m = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|v| (v - m).exp()).sum();
        for v in row.iter_mut() {
            *v = (*v - m).exp() / sum + HC_EPS;
        }
    }
    let cols = |c: &mut [f64]| {
        for k in 0..HC {
            let s: f64 = (0..HC).map(|j| c[j * HC + k]).sum::<f64>() + HC_EPS;
            for j in 0..HC {
                c[j * HC + k] /= s;
            }
        }
    };
    let rows = |c: &mut [f64]| {
        for j in 0..HC {
            let s: f64 = c[j * HC..(j + 1) * HC].iter().sum::<f64>() + HC_EPS;
            for v in &mut c[j * HC..(j + 1) * HC] {
                *v /= s;
            }
        }
    };
    cols(&mut comb);
    for _ in 0..SINKHORN - 1 {
        rows(&mut comb);
        cols(&mut comb);
    }
    (pre, post, comb)
}

fn hc_pre(h: &[f64], pre: &[f64]) -> Vec<f64> {
    (0..DIM)
        .map(|d| bf((0..HC).map(|j| pre[j] * h[j * DIM + d]).sum()))
        .collect()
}

fn hc_post(x: &[f64], residual: &[f64], post: &[f64], comb: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0f64; HC * DIM];
    for j in 0..HC {
        for d in 0..DIM {
            let mut v = post[j] * x[d];
            for k in 0..HC {
                v += comb[k * HC + j] * residual[k * DIM + d];
            }
            y[j * DIM + d] = bf(v);
        }
    }
    y
}

fn engram(fx: &Fixture, h: &mut [f64], hashes: &[usize]) {
    let p = format!("model.layers.{ENGRAM_LAYER}.engram");
    let cols = ENGRAM_HEAD_DIM;
    let mut emb = Vec::with_capacity(hashes.len() * cols);
    for &row in hashes {
        for c in 0..cols {
            let s = 2f64.powi(fx.engram_scales[row * cols / 32 + c / 32] as i32 - 127);
            emb.push(bf(e4m3_decode(fx.engram_bytes[row * cols + c]) * s));
        }
    }
    let mut kv = mv(fx.wt(&format!("{p}.wkv.weight")), &emb);
    bf_vec(&mut kv);
    let qw = &fx.wt(&format!("{p}.q_weight")).data;
    let kw = &fx.wt(&format!("{p}.k_weight")).data;
    let value = &kv[HC * DIM..];
    for j in 0..HC {
        let hs = &h[j * DIM..(j + 1) * DIM];
        let key = &kv[j * DIM..(j + 1) * DIM];
        let hm = hs.iter().map(|v| v * v).sum::<f64>() / DIM as f64;
        let km = key.iter().map(|v| v * v).sum::<f64>() / DIM as f64;
        let rstd = 1.0 / (hm + EPS).sqrt() / (km + EPS).sqrt();
        let dot: f64 = (0..DIM)
            .map(|d| hs[d] * (qw[j * DIM + d] as f64 * kw[j * DIM + d] as f64) * key[d])
            .sum::<f64>()
            * rstd
            / (DIM as f64).sqrt();
        let root = dot.abs().max(1e-6).sqrt();
        let gate = sigmoid(if dot < 0.0 { -root } else { root });
        for d in 0..DIM {
            h[j * DIM + d] = bf(h[j * DIM + d] + gate * value[d]);
        }
    }
}

fn top_by_score(scores: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let mut out: Vec<usize> = order.into_iter().take(k).collect();
    out.sort_unstable();
    out
}

fn select_candidate_blocks(scores: &[f64], compress_len: usize) -> Vec<bool> {
    let width = scores.len();
    let nb = width.div_ceil(CAND_BS);
    let mut bs: Vec<f64> = (0..nb)
        .map(|b| {
            scores[b * CAND_BS..((b + 1) * CAND_BS).min(width)]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    bs[(compress_len - 1) / CAND_BS] = f64::INFINITY;
    let picked = top_by_score(&bs, CAND_BLOCKS.min(nb));
    let mut keep = vec![false; width];
    for b in picked {
        if bs[b] > f64::NEG_INFINITY {
            for k in keep
                .iter_mut()
                .take(((b + 1) * CAND_BS).min(width))
                .skip(b * CAND_BS)
            {
                *k = true;
            }
        }
    }
    keep
}

fn compressor(fx: &Fixture, st: &mut RefState, li: usize, si: usize, x: &[f64], p: usize) -> Option<Vec<f64>> {
    let a = format!("model.layers.{li}.self_attn.compressor");
    let r = RATIOS[li];
    let norm = fx.wt(&format!("{a}.norm.weight"));
    if r == 1 {
        let mut kv = mv(fx.wt(&format!("{a}.wkv.weight")), x);
        bf_vec(&mut kv);
        rmsnorm(&mut kv, norm);
        return Some(kv);
    }
    let kv = mv(fx.wt(&format!("{a}.wkv.weight")), x);
    let score = mv(fx.wt(&format!("{a}.wgate.weight")), x);
    if st.ckv[si].len() < r {
        st.ckv[si] = vec![vec![0.0; HEAD_DIM]; r];
        st.cscore[si] = vec![vec![f64::NEG_INFINITY; HEAD_DIM]; r];
    }
    st.ckv[si][p % r] = kv;
    st.cscore[si][p % r] = score;
    if (p + 1) % r != 0 {
        return None;
    }
    let mut pooled = vec![0.0f64; HEAD_DIM];
    for (d, out) in pooled.iter_mut().enumerate() {
        let m = (0..r)
            .map(|i| st.cscore[si][i][d])
            .fold(f64::NEG_INFINITY, f64::max);
        let den: f64 = (0..r).map(|i| (st.cscore[si][i][d] - m).exp()).sum();
        *out = (0..r)
            .map(|i| (st.cscore[si][i][d] - m).exp() / den * st.ckv[si][i][d])
            .sum();
    }
    bf_vec(&mut pooled);
    rmsnorm(&mut pooled, norm);
    Some(pooled)
}

#[allow(clippy::too_many_arguments)]
fn indexer(
    fx: &Fixture,
    st: &mut RefState,
    li: usize,
    si: usize,
    x: &[f64],
    qr: &[f64],
    latent: Option<&[f64]>,
    p: usize,
    compress_len: usize,
    freqs: &[f64],
) -> Vec<usize> {
    let a = format!("model.layers.{li}.self_attn.indexer");
    let r = RATIOS[li];
    if let (true, Some(lat)) = (KV_SOURCES.contains(&li), latent) {
        let mut k = mv(fx.wt(&format!("{a}.wk.weight")), lat);
        bf_vec(&mut k);
        rmsnorm(&mut k, fx.wt(&format!("{a}.k_norm.weight")));
        rope(&mut k, freqs, p + 1 - r, false);
        bf_vec(&mut k);
        fp4_e8m0(&mut k);
        st.index_k[si].push(k);
    }
    let mut q = mv(fx.wt(&format!("{a}.wq_b.weight")), qr);
    bf_vec(&mut q);
    for h in 0..IH {
        rope(&mut q[h * ID..(h + 1) * ID], freqs, p, false);
    }
    bf_vec(&mut q);
    fp4_e8m0(&mut q);
    let mut w = mv(fx.wt(&format!("{a}.weights_proj.weight")), x);
    bf_vec(&mut w);
    let sc = (ID as f64).powf(-0.5) * (IH as f64).powf(-0.5);
    for v in w.iter_mut() {
        *v = bf(*v * sc);
    }
    let mut scores = vec![f64::NEG_INFINITY; compress_len];
    for (t, score) in scores.iter_mut().enumerate() {
        let k = &st.index_k[si][t];
        let mut s = 0.0f64;
        for h in 0..IH {
            let d: f64 = q[h * ID..(h + 1) * ID].iter().zip(k).map(|(a, b)| a * b).sum();
            s += bf(bf(d).max(0.0) * w[h]);
        }
        *score = bf(s);
    }
    if fx.candidate_source == Some(li) {
        st.candidates = select_candidate_blocks(&scores, compress_len);
    } else if fx.candidate_source.is_some_and(|c| c < li) {
        for (t, score) in scores.iter_mut().enumerate() {
            if !st.candidates.get(t).copied().unwrap_or(false) {
                *score = f64::NEG_INFINITY;
            }
        }
    }
    let k = INDEX_TOPK.min(compress_len);
    let picked = top_by_score(&scores, k);
    assert!(
        picked.iter().all(|&i| scores[i].is_finite()),
        "fixture geometry must keep top-k inside the candidate set"
    );
    picked
}

fn sparse_attn(q: &[f64], rows: &[Vec<f64>], sink: f64, opts: RefOpts) -> Vec<f64> {
    let scale = 1.0 / (HEAD_DIM as f64).sqrt();
    let scores: Vec<f64> = rows
        .iter()
        .map(|k| q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() * scale)
        .collect();
    let m = scores.iter().copied().fold(-1e30f64, f64::max);
    let mut den = (sink - m).exp();
    let mut o = vec![0.0f64; HEAD_DIM];
    for (row, &s) in rows.iter().zip(&scores) {
        let p = (s - m).exp();
        den += p;
        let pv = if opts.bf16_attn_probs { bf(p) } else { p };
        for (dst, v) in o.iter_mut().zip(row) {
            *dst += pv * v;
        }
    }
    for v in o.iter_mut() {
        *v /= den;
    }
    o
}

#[allow(clippy::too_many_arguments)]
fn attention(
    fx: &Fixture,
    st: &mut RefState,
    li: usize,
    x: &[f64],
    p: usize,
    freqs_c: &[f64],
    freqs_w: &[f64],
    opts: RefOpts,
) -> Vec<f64> {
    let a = format!("model.layers.{li}.self_attn");
    let r = RATIOS[li];
    let freqs = if r > 0 { freqs_c } else { freqs_w };
    tr("x", p, Some(li), x);
    let mut qr = mv(fx.wt(&format!("{a}.wq_a.weight")), x);
    bf_vec(&mut qr);
    tr("wq_a", p, Some(li), &qr);
    rmsnorm(&mut qr, fx.wt(&format!("{a}.q_norm.weight")));
    tr("q_norm", p, Some(li), &qr);
    let mut q = mv(fx.wt(&format!("{a}.wq_b.weight")), &qr);
    bf_vec(&mut q);
    for h in 0..N_HEADS {
        rope(&mut q[h * HEAD_DIM..(h + 1) * HEAD_DIM], freqs, p, false);
    }
    bf_vec(&mut q);
    let mut kv = mv(fx.wt(&format!("{a}.wkv.weight")), x);
    bf_vec(&mut kv);
    tr("wkv", p, Some(li), &kv);
    rmsnorm(&mut kv, fx.wt(&format!("{a}.kv_norm.weight")));
    tr("kv_norm", p, Some(li), &kv);
    rope(&mut kv, freqs, p, false);
    bf_vec(&mut kv);
    act_quant_fp8(&mut kv);
    tr("kv", p, Some(li), &kv);
    tr("q", p, Some(li), &q);
    st.window[li].push(kv);
    if st.window[li].len() > WINDOW {
        st.window[li].remove(0);
    }
    let mut rows = st.window[li].clone();
    if r > 0 {
        let si = source_slot(li);
        let compress_len = (p + 1) / r;
        let latent = if KV_SOURCES.contains(&li) {
            compressor(fx, st, li, si, x, p)
        } else {
            None
        };
        let idxs = if INDEX_SOURCES.contains(&li) {
            let v = if compress_len == 0 {
                Vec::new()
            } else {
                indexer(
                    fx,
                    st,
                    li,
                    si,
                    x,
                    &qr,
                    latent.as_deref(),
                    p,
                    compress_len,
                    freqs,
                )
            };
            st.topk = v.clone();
            v
        } else {
            st.topk.clone()
        };
        if std::env::var_os("DSV41_REF_TRACE").is_some() {
            eprintln!("ref_trace stage=idxs position={p} layer={li} values={idxs:?}");
        }
        if let Some(mut lat) = latent {
            tr("latent", p, Some(li), &lat);
            rope(&mut lat, freqs, p + 1 - r, false);
            bf_vec(&mut lat);
            fp4_e4m3(&mut lat);
            st.compress_kv[si].push(lat);
        }
        for &i in &idxs {
            if i < compress_len {
                rows.push(st.compress_kv[si][i].clone());
            }
        }
    }
    let sink = &fx.wt(&format!("{a}.attn_sink")).data;
    let mut o = vec![0.0f64; N_HEADS * HEAD_DIM];
    for h in 0..N_HEADS {
        let oh = sparse_attn(
            &q[h * HEAD_DIM..(h + 1) * HEAD_DIM],
            &rows,
            sink[h] as f64,
            opts,
        );
        o[h * HEAD_DIM..(h + 1) * HEAD_DIM].copy_from_slice(&oh);
    }
    if opts.bf16_attn_out {
        bf_vec(&mut o);
    }
    for h in 0..N_HEADS {
        rope(&mut o[h * HEAD_DIM..(h + 1) * HEAD_DIM], freqs, p, true);
    }
    bf_vec(&mut o);
    if opts.o_fp8 {
        act_quant_fp8(&mut o);
    }
    let wo_a = fx.wt(&format!("{a}.wo_a.weight"));
    let per_group = N_HEADS * HEAD_DIM / O_GROUPS;
    let mut mid = vec![0.0f64; O_GROUPS * O_LORA];
    for (i, m) in mid.iter_mut().enumerate() {
        let g = i / O_LORA;
        *m = bf(wo_a.data[i * per_group..(i + 1) * per_group]
            .iter()
            .zip(&o[g * per_group..(g + 1) * per_group])
            .map(|(&w, &v)| w as f64 * v)
            .sum());
    }
    tr("attended", p, Some(li), &o);
    let mut out = mv(fx.wt(&format!("{a}.wo_b.weight")), &mid);
    bf_vec(&mut out);
    tr("attn", p, Some(li), &out);
    out
}

fn expert(fx: &Fixture, prefix: &str, x: &[f64], weight: Option<f64>) -> Vec<f64> {
    let mut g = mv(fx.wt(&format!("{prefix}.gate_proj.weight")), x);
    let mut u = mv(fx.wt(&format!("{prefix}.up_proj.weight")), x);
    bf_vec(&mut g);
    bf_vec(&mut u);
    let mut act = vec![0.0f64; INTER];
    for i in 0..INTER {
        let up = u[i].clamp(-SWIGLU, SWIGLU);
        let gate = g[i].min(SWIGLU);
        let mut v = gate * sigmoid(gate) * up;
        if let Some(w) = weight {
            v *= w;
        }
        act[i] = bf(v);
    }
    let mut y = mv(fx.wt(&format!("{prefix}.down_proj.weight")), &act);
    bf_vec(&mut y);
    y
}

fn moe(fx: &Fixture, li: usize, x: &[f64]) -> Vec<f64> {
    let p = format!("model.layers.{li}.mlp");
    let logits = mv(fx.wt(&format!("{p}.gate.weight")), x);
    let bias = &fx.wt(&format!("{p}.expert_bias")).data;
    let scores: Vec<f64> = logits
        .iter()
        .map(|&v| {
            // F.softplus: log1p(exp(x)), linear past the threshold 20.
            let sp = if v > 20.0 { v } else { v.exp().ln_1p() };
            sp.sqrt()
        })
        .collect();
    let shifted: Vec<f64> = scores
        .iter()
        .zip(bias)
        .map(|(s, &b)| s + b as f64)
        .collect();
    let mut picks: Vec<usize> = (0..EXPERTS).collect();
    picks.sort_by(|&a, &b| shifted[b].total_cmp(&shifted[a]).then(a.cmp(&b)));
    picks.truncate(TOP_K);
    if std::env::var_os("DSV41_REF_TRACE").is_some() {
        eprintln!("ref_trace stage=moe layer={li} logits={logits:?} scores={scores:?} picks={picks:?}");
    }
    let mut w: Vec<f64> = picks.iter().map(|&e| scores[e]).collect();
    if TOP_K > 1 {
        let s: f64 = w.iter().sum::<f64>() + 1e-20;
        for v in w.iter_mut() {
            *v /= s;
        }
    }
    let mut y = vec![0.0f64; DIM];
    for (&e, &we) in picks.iter().zip(&w) {
        let out = expert(fx, &format!("{p}.experts.{e}"), x, Some(we * ROUTE_SCALE));
        for (dst, v) in y.iter_mut().zip(out) {
            *dst += v;
        }
    }
    let shared = expert(fx, &format!("{p}.shared_experts"), x, None);
    for (dst, v) in y.iter_mut().zip(shared) {
        *dst += v;
    }
    bf_vec(&mut y);
    y
}

/// Logits of the reference model for every position of `tokens`, decoded
/// one token at a time (the runtime's prefill is token serial as well).
pub fn reference_logits(fx: &Fixture, tokens: &[u32], opts: RefOpts) -> Vec<Vec<f64>> {
    assert_eq!(fx.compressed_vocab, RELEASE_COMPRESSED_VOCAB);
    let map = fx.token_map();
    let pad = map[PAD_TOKEN] as i64;
    let freqs_c = ref_freqs(ROPE_DIM, COMPRESS_THETA, ORIG_SEQ);
    let freqs_w = ref_freqs(ROPE_DIM, ROPE_THETA, 0.0);
    let mut st = RefState {
        window: vec![Vec::new(); N_LAYERS],
        ckv: vec![Vec::new(); KV_SOURCES.len()],
        cscore: vec![Vec::new(); KV_SOURCES.len()],
        compress_kv: vec![Vec::new(); KV_SOURCES.len()],
        index_k: vec![Vec::new(); KV_SOURCES.len()],
        topk: Vec::new(),
        candidates: Vec::new(),
        history: Vec::new(),
    };
    let embed = fx.wt("model.embed_tokens.weight");
    let mut all = Vec::with_capacity(tokens.len());
    for (p, &t) in tokens.iter().enumerate() {
        let hashes = engram_hash(&mut st.history, map[t as usize] as i64, pad);
        let row: Vec<f64> = embed.data[t as usize * DIM..(t as usize + 1) * DIM]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let mut h: Vec<f64> = (0..HC).flat_map(|_| row.iter().copied()).collect();
        let mut pre_mix = vec![0.0f64; HC];
        pre_mix[0] = 1.0;
        for li in 0..N_LAYERS {
            let lp = format!("model.layers.{li}");
            if li == ENGRAM_LAYER {
                engram(fx, &mut h, &hashes);
            }
            let residual = h.clone();
            let (apre, apost, acomb) = hc_mixes(fx, &h, "attn", li);
            let mut x = hc_pre(&h, &pre_mix);
            rmsnorm(&mut x, fx.wt(&format!("{lp}.input_layernorm.weight")));
            let a = attention(fx, &mut st, li, &x, p, &freqs_c, &freqs_w, opts);
            h = hc_post(&a, &residual, &apost, &acomb);
            let residual = h.clone();
            let (fpre, fpost, fcomb) = hc_mixes(fx, &h, "ffn", li);
            let mut x = hc_pre(&h, &apre);
            rmsnorm(
                &mut x,
                fx.wt(&format!("{lp}.post_attention_layernorm.weight")),
            );
            let m = moe(fx, li, &x);
            h = hc_post(&m, &residual, &fpost, &fcomb);
            pre_mix = fpre;
            tr("block", p, Some(li), &h);
        }
        let mut x = hc_pre(&h, &pre_mix);
        rmsnorm(&mut x, fx.wt("model.norm.weight"));
        all.push(mv(fx.wt("lm_head.weight"), &x));
    }
    all
}

/// Worst relative logit error over all positions: max |a-b| / max |b|.
pub fn worst_rel(got: &[Vec<f32>], want: &[Vec<f64>]) -> (f64, usize) {
    let mut worst = (0.0f64, 0usize);
    for (p, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len());
        let scale = w.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
        let err = g
            .iter()
            .zip(w)
            .fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()));
        if err / scale > worst.0 {
            worst = (err / scale, p);
        }
    }
    worst
}

pub fn argmax32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

pub fn argmax64(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}
