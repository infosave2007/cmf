//! Native BERT forward on `cortiq_engine::fcd_ops::gemm_nt_host` (spec §1.4).
//!
//! The encoder of a decision file is a post-LN BERT (bge-small-en-v1.5 with
//! its MLP pruned for the release file) whose weights are ordinary F32 CMF
//! tensors `decision.encoder.*` ([`crate::manifest::EncoderConfig::weight_shapes`]).
//! φ_P of a text, one text per call, no padding and no mask:
//!
//! 1. ids = [`WordPiece::encode`] (`[CLS] … [SEP]`, at most `max_length`);
//! 2. embeddings `(word[id] + type[token_type]) + pos[i]` (the ONNX graph's
//!    order), LayerNorm;
//! 3. per layer: Q/K/V = `x·Wᵀ + b`; per head `S = (Q_h·K_hᵀ) / √head_dim`,
//!    softmax in f32 (max subtracted, `exp`, times `1/Σ`), `ctx_h = P·V_h`;
//!    `LN((ctx·Woᵀ + bo) + x)`; `h = GELU(x·Wiᵀ + bi)`; `LN((h·Wfᵀ + bf) + x)`;
//! 4. mean over all tokens (f32 sum in token order, divided by n);
//! 5. `v / (‖v‖ + 1e-12)`, then `v · (1/‖v‖)` when ‖v‖ > 1e-12 (the router's
//!    second L2), ‖v‖ = sqrt of the sequential f32 sum of squares.
//!
//! In the default CPU backend, every weight/attention product is
//! [`gemm_nt_host`] with no pool: the host f32 GEMM (Accelerate on macOS,
//! the engine's dot kernels elsewhere). It never initialises, probes or
//! dispatches to a GPU backend — unlike `gemm_nt`, which sends
//! `n·k·m ≥ 2^22` to wgpu (tf32-class on NVIDIA), i.e. every text of 29 or
//! more tokens here. The separate opt-in Metal/Vulkan graphs uses resident FP32
//! buffers, not that per-operation dispatch. LayerNorm accumulates in f64 and writes f32. GELU is
//! `((x·(erf(x/√2) + 1))·0.5` in f32 with the Numerical Recipes `erf`
//! (in f64) copied from `cortiq-engine/src/qwen3vis.rs`.
//!
//! [`EncoderExport`] reads the directory `tools/decision_export_encoder.py`
//! writes (`encoder.json`, one `.npy` per tensor, `vocab.txt`,
//! `tokenizer.json`) and produces the tensors of `cortiq decision init`,
//! including the φ_P of the golden texts; [`Encoder::verify_golden`] is the
//! load-time check of a decision file against them.

use crate::container::DecisionModel;
use crate::manifest::{
    self, ENCODER_GOLDEN_COUNT, ENCODER_KIND, ENCODER_PREFIX, EncoderConfig, EncoderRecord,
    EncoderSource, GOLDEN_TENSOR, NORMALIZATION, NormalizerRecord, POOLING, PRE_TOKENIZER,
    SpecialIds, TEMPLATE, TRUNCATION_DIRECTION, TensorDigest, TokenizerRecord, VOCAB_TENSOR,
};
use crate::unicode_tables;
use crate::wordpiece::WordPiece;
use anyhow::{Context, Result, bail, ensure};
use cortiq_core::{CmfModel, TensorDtype, TensorSpec};
use cortiq_engine::fcd_ops::gemm_nt_host;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(target_os = "macos")]
#[path = "bert_metal.rs"]
mod metal_backend;
#[cfg(feature = "vulkan")]
#[path = "bert_vulkan.rs"]
mod vulkan_backend;

/// Explicit opt-in: model bytes, golden tolerance and CPU default are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderDevice {
    Cpu,
    Metal,
    Vulkan,
}
impl EncoderDevice {
    pub fn from_env() -> Result<Self> {
        match std::env::var("CORTIQ_DECISION_DEVICE") {
            Err(std::env::VarError::NotPresent) => Ok(Self::Cpu),
            Ok(v) if v == "cpu" => Ok(Self::Cpu),
            Ok(v) if v == "metal" => Ok(Self::Metal),
            Ok(v) if v == "vulkan" => Ok(Self::Vulkan),
            _ => bail!("CORTIQ_DECISION_DEVICE must be cpu, metal or vulkan"),
        }
    }
}

/// Largest |Δ| between the stored golden φ_P and this build's that a decision
/// file may show before it is refused. The spec gives no tolerance; the golden
/// is written on the builder's machine and a GEMM on another platform (AVX-512
/// or NEON dot kernels instead of Accelerate) differs in the last bits
/// (measured parity between two f32 implementations: ~1e-7). A wrong
/// tokenizer, vocab or weight moves φ_P by ≥ 1e-2.
pub const GOLDEN_MAX_ABS: f32 = 1e-5;

/// Epsilon of the first L2 (fastembed `normalize`).
pub const L2_EPS: f32 = 1e-12;

// ------------------------------------------------------------------ numerics

/// `erf` by the Numerical Recipes rational form, copied from
/// `cortiq-engine/src/qwen3vis.rs` (1.2e-7 relative everywhere).
fn erf(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let ans = t
        * (-z * z - 1.265_512_23
            + t * (1.000_023_68
                + t * (0.374_091_96
                    + t * (0.096_784_18
                        + t * (-0.186_288_06
                            + t * (0.278_868_07
                                + t * (-1.135_203_98
                                    + t * (1.488_515_87
                                        + t * (-0.822_152_23 + t * 0.170_872_77)))))))))
            .exp();
    // `ans` is erfc(|x|); erfc is 2 − that on the negative side.
    if x >= 0.0 { 1.0 - ans } else { ans - 1.0 }
}

/// Exact (erf) GELU in the graph's f32 order: `((x · (erf(x/√2) + 1)) · 0.5`.
#[inline]
pub fn gelu(x: f32) -> f32 {
    let d = x / std::f32::consts::SQRT_2;
    let e = erf(d as f64) as f32;
    (x * (e + 1.0)) * 0.5
}

/// LayerNorm of rows of `w.len()` values: mean and variance accumulated in
/// f64, `((x − μ)·(1/√(σ² + ε))·w + b)` in f64, written as f32.
pub fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], eps: f64, dst: &mut [f32]) {
    let d = w.len();
    debug_assert_eq!(x.len(), dst.len());
    for (xr, dr) in x.chunks_exact(d).zip(dst.chunks_exact_mut(d)) {
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = xr
            .iter()
            .map(|&v| {
                let c = v as f64 - mean;
                c * c
            })
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            dr[i] = ((xr[i] as f64 - mean) * inv * w[i] as f64 + b[i] as f64) as f32;
        }
    }
}

/// Softmax of one row in f32: max subtracted, `exp`, multiplied by `1/Σ`
/// (sums in index order).
pub fn softmax_row(s: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in s.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in s.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in s.iter_mut() {
        *v *= inv;
    }
}

/// ‖v‖ as sqrt of the sequential f32 sum of squares (no FMA).
#[inline]
pub fn norm_seq_f32(v: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// The two L2 steps of φ_P: `v / (‖v‖ + 1e-12)` (fastembed), then
/// `v · (1/‖v‖)` when ‖v‖ > 1e-12 (the router's `linalg::normalize`).
pub fn normalize_phi_p(v: &mut [f32]) {
    let den = norm_seq_f32(v) + L2_EPS;
    for x in v.iter_mut() {
        *x /= den;
    }
    let n = norm_seq_f32(v);
    if n > 1e-12 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Mean over the rows of `hidden` (`n × dim`): f32 sum in row order, divided by n.
pub fn mean_pool(hidden: &[f32], dim: usize) -> Vec<f32> {
    let n = hidden.len() / dim;
    let mut out = vec![0.0f32; dim];
    for row in hidden.chunks_exact(dim) {
        for (o, &v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
    let nf = n as f32;
    for o in out.iter_mut() {
        *o /= nf;
    }
    out
}

// ------------------------------------------------------------------ model

/// Dimensions of a BERT encoder (from its [`EncoderConfig`]).
#[derive(Clone, Debug, PartialEq)]
pub struct BertDims {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub max_position: usize,
    pub vocab: usize,
    pub type_vocab: usize,
    pub token_type: usize,
    pub ln_eps: f64,
}

impl BertDims {
    pub fn from_config(c: &EncoderConfig) -> Result<Self> {
        c.validate()?;
        Ok(Self {
            layers: c.layers as usize,
            hidden: c.hidden as usize,
            heads: c.heads as usize,
            head_dim: c.head_dim as usize,
            intermediate: c.intermediate as usize,
            max_position: c.max_position as usize,
            vocab: c.vocab as usize,
            type_vocab: c.type_vocab as usize,
            token_type: c.token_type as usize,
            ln_eps: c.ln_eps,
        })
    }
}

/// One encoder weight (spec §2.2: "weights are read without copying when they
/// are 4-byte aligned, otherwise copied"): a view of the decision file's
/// mapping, which it keeps alive, or an owned copy (an unaligned tensor, a
/// big-endian host, or an export directory).
#[derive(Clone)]
pub enum Weights {
    Owned(Vec<f32>),
    /// Tensor `tensor` of `file`, 4-byte aligned (checked when built).
    Mapped {
        file: Arc<CmfModel>,
        tensor: usize,
    },
}

impl Weights {
    /// A view of tensor `tensor` of `file` when its bytes are 4-byte aligned
    /// little-endian f32, else a copy.
    pub fn of_tensor(file: &Arc<CmfModel>, tensor: usize) -> Self {
        let bytes = file.entry_bytes(&file.tensors[tensor]);
        if cfg!(target_endian = "little") && bytemuck::try_cast_slice::<u8, f32>(bytes).is_ok() {
            Self::Mapped {
                file: Arc::clone(file),
                tensor,
            }
        } else {
            Self::Owned(f32_from_le(bytes))
        }
    }

    /// Whether this weight is a view of the file (no copy).
    pub fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped { .. })
    }
}

impl From<Vec<f32>> for Weights {
    fn from(v: Vec<f32>) -> Self {
        Self::Owned(v)
    }
}

impl std::ops::Deref for Weights {
    type Target = [f32];

    fn deref(&self) -> &[f32] {
        match self {
            Self::Owned(v) => v,
            // Aligned and a multiple of 4 bytes: checked in `of_tensor`, and
            // the mapping never moves while `file` lives.
            Self::Mapped { file, tensor } => {
                bytemuck::cast_slice(file.entry_bytes(&file.tensors[*tensor]))
            }
        }
    }
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(v) => write!(f, "Owned({} values)", v.len()),
            Self::Mapped { file, tensor } => write!(
                f,
                "Mapped({}, {} values)",
                file.tensors[*tensor].name,
                self.len()
            ),
        }
    }
}

/// `y = x·Wᵀ + b`, W stored `[out, in]`.
#[derive(Clone, Debug)]
struct Linear {
    w: Weights,
    b: Weights,
    inp: usize,
    out: usize,
}

impl Linear {
    /// `y[n, out] = x[n, in]·Wᵀ + b` on the host GEMM.
    fn apply(&self, x: &[f32], n: usize, y: &mut [f32]) {
        gemm_nt_host(x, &self.w, y, n, self.inp, self.out, None);
        for row in y[..n * self.out].chunks_exact_mut(self.out) {
            for (v, &b) in row.iter_mut().zip(self.b.iter()) {
                *v += b;
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Norm {
    w: Weights,
    b: Weights,
}

#[derive(Clone, Debug)]
struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ln_attn: Norm,
    inter: Linear,
    out: Linear,
    ln_out: Norm,
}

/// A BERT encoder over f32 weights ([`Weights`]: views of a decision file's
/// mapping, or owned). `Send + Sync`; one forward per text.
#[derive(Clone, Debug)]
pub struct BertModel {
    dims: BertDims,
    word: Weights,
    pos: Weights,
    /// The `token_type` row of the token-type table.
    type_row: Vec<f32>,
    ln_emb: Norm,
    layers: Vec<Layer>,
}

impl BertModel {
    /// Build from the config and a getter of each weight by its name without
    /// `decision.encoder.` (e.g. `layer.0.attention.self.query.weight`); the
    /// shapes are the config's ([`EncoderConfig::weight_shapes`]).
    pub fn from_weights(
        config: &EncoderConfig,
        mut get: impl FnMut(&str) -> Result<Weights>,
    ) -> Result<Self> {
        let dims = BertDims::from_config(config)?;
        let shapes: BTreeMap<String, Vec<usize>> = config.weight_shapes().into_iter().collect();
        let mut take = |name: &str| -> Result<Weights> {
            let shape = shapes
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("'{name}' is not a weight of this config"))?;
            let v = get(name).with_context(|| format!("encoder weight '{name}'"))?;
            let want: usize = shape.iter().product();
            ensure!(
                v.len() == want,
                "encoder weight '{name}' has {} values, the shape {shape:?} wants {want}",
                v.len()
            );
            ensure!(
                v.iter().all(|x| x.is_finite()),
                "encoder weight '{name}' is not finite"
            );
            Ok(v)
        };
        let (h, f) = (dims.hidden, dims.intermediate);
        let mut lin = |p: &str, inp: usize, out: usize| -> Result<Linear> {
            Ok(Linear {
                w: take(&format!("{p}.weight"))?,
                b: take(&format!("{p}.bias"))?,
                inp,
                out,
            })
        };
        let mut layers = Vec::with_capacity(dims.layers);
        for l in 0..dims.layers {
            let p = format!("layer.{l}.");
            let q = lin(&format!("{p}attention.self.query"), h, h)?;
            let k = lin(&format!("{p}attention.self.key"), h, h)?;
            let v = lin(&format!("{p}attention.self.value"), h, h)?;
            let o = lin(&format!("{p}attention.output.dense"), h, h)?;
            let inter = lin(&format!("{p}intermediate.dense"), h, f)?;
            let out = lin(&format!("{p}output.dense"), f, h)?;
            layers.push((p, q, k, v, o, inter, out));
        }
        let mut norm = |p: &str| -> Result<Norm> {
            Ok(Norm {
                w: take(&format!("{p}.weight"))?,
                b: take(&format!("{p}.bias"))?,
            })
        };
        let mut built = Vec::with_capacity(dims.layers);
        for (p, q, k, v, o, inter, out) in layers {
            built.push(Layer {
                q,
                k,
                v,
                o,
                ln_attn: norm(&format!("{p}attention.output.LayerNorm"))?,
                inter,
                out,
                ln_out: norm(&format!("{p}output.LayerNorm"))?,
            });
        }
        let ln_emb = norm("embeddings.LayerNorm")?;
        let word = take("embeddings.word_embeddings.weight")?;
        let pos = take("embeddings.position_embeddings.weight")?;
        let types = take("embeddings.token_type_embeddings.weight")?;
        let t = dims.token_type;
        let type_row = types[t * h..(t + 1) * h].to_vec();
        Ok(Self {
            dims,
            word,
            pos,
            type_row,
            ln_emb,
            layers: built,
        })
    }

    /// The encoder of a decision file: every aligned weight is a view of the
    /// file's mapping (spec §2.2), which the model keeps alive; an unaligned
    /// one is copied.
    pub fn from_model(model: &DecisionModel) -> Result<Self> {
        let config = &model.representation().encoder.config;
        let file = model.base_file();
        Self::from_weights(config, |name| {
            let full = format!("{ENCODER_PREFIX}{name}");
            let i = file
                .tensor_index(&full)
                .ok_or_else(|| anyhow::anyhow!("tensor '{full}' is missing"))?;
            ensure!(
                file.tensors[i].dtype == TensorDtype::F32,
                "tensor '{full}' is not F32"
            );
            Ok(Weights::of_tensor(&file, i))
        })
    }

    /// f32 values of the weights this model holds as copies (the token-type row
    /// always; any weight that could not be mapped).
    pub fn copied_weight_values(&self) -> usize {
        let owned = |w: &Weights| if w.is_mapped() { 0 } else { w.len() };
        let mut n = self.type_row.len() + owned(&self.word) + owned(&self.pos);
        n += owned(&self.ln_emb.w) + owned(&self.ln_emb.b);
        for l in &self.layers {
            for lin in [&l.q, &l.k, &l.v, &l.o, &l.inter, &l.out] {
                n += owned(&lin.w) + owned(&lin.b);
            }
            for nm in [&l.ln_attn, &l.ln_out] {
                n += owned(&nm.w) + owned(&nm.b);
            }
        }
        n
    }

    pub fn dims(&self) -> &BertDims {
        &self.dims
    }

    /// `last_hidden_state` `[n, hidden]` of one sequence of ids (no padding,
    /// no mask). Panics when `ids` is empty, longer than `max_position` or holds
    /// an id outside the vocab.
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let d = &self.dims;
        let (n, h, f, nh, dh) = (ids.len(), d.hidden, d.intermediate, d.heads, d.head_dim);
        assert!(n >= 1, "empty token sequence");
        assert!(
            n <= d.max_position,
            "{n} tokens exceed max_position {}",
            d.max_position
        );
        // Embeddings: (word + type) + pos, then LayerNorm.
        let mut emb = vec![0.0f32; n * h];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            assert!(
                id < d.vocab,
                "token id {id} outside the vocab of {}",
                d.vocab
            );
            let w = &self.word[id * h..(id + 1) * h];
            let p = &self.pos[i * h..(i + 1) * h];
            for (j, e) in emb[i * h..(i + 1) * h].iter_mut().enumerate() {
                *e = (w[j] + self.type_row[j]) + p[j];
            }
        }
        let mut x = vec![0.0f32; n * h];
        layer_norm_rows(&emb, &self.ln_emb.w, &self.ln_emb.b, d.ln_eps, &mut x);

        let scale = (dh as f32).sqrt();
        let mut q = vec![0.0f32; n * h];
        let mut k = vec![0.0f32; n * h];
        let mut v = vec![0.0f32; n * h];
        let mut ctx = vec![0.0f32; n * h];
        let mut tmp = vec![0.0f32; n * h];
        let mut inter = vec![0.0f32; n * f];
        let mut qh = vec![0.0f32; n * dh];
        let mut kh = vec![0.0f32; n * dh];
        let mut vt = vec![0.0f32; dh * n];
        let mut ch = vec![0.0f32; n * dh];
        let mut s = vec![0.0f32; n * n];
        for layer in &self.layers {
            layer.q.apply(&x, n, &mut q);
            layer.k.apply(&x, n, &mut k);
            layer.v.apply(&x, n, &mut v);
            for head in 0..nh {
                let c0 = head * dh;
                for i in 0..n {
                    qh[i * dh..(i + 1) * dh].copy_from_slice(&q[i * h + c0..i * h + c0 + dh]);
                    kh[i * dh..(i + 1) * dh].copy_from_slice(&k[i * h + c0..i * h + c0 + dh]);
                    for c in 0..dh {
                        vt[c * n + i] = v[i * h + c0 + c];
                    }
                }
                // S = Q_h·K_hᵀ, then / √head_dim and a row softmax.
                gemm_nt_host(&qh, &kh, &mut s, n, dh, n, None);
                for row in s.chunks_exact_mut(n) {
                    for val in row.iter_mut() {
                        *val /= scale;
                    }
                    softmax_row(row);
                }
                // ctx_h = P·V_h.
                gemm_nt_host(&s, &vt, &mut ch, n, n, dh, None);
                for i in 0..n {
                    ctx[i * h + c0..i * h + c0 + dh].copy_from_slice(&ch[i * dh..(i + 1) * dh]);
                }
            }
            // LN((ctx·Woᵀ + bo) + x).
            layer.o.apply(&ctx, n, &mut tmp);
            for (t, &r) in tmp.iter_mut().zip(&x) {
                *t += r;
            }
            layer_norm_rows(&tmp, &layer.ln_attn.w, &layer.ln_attn.b, d.ln_eps, &mut x);
            // LN((GELU(x·Wiᵀ + bi)·Wfᵀ + bf) + x).
            layer.inter.apply(&x, n, &mut inter);
            for val in inter.iter_mut() {
                *val = gelu(*val);
            }
            layer.out.apply(&inter, n, &mut tmp);
            for (t, &r) in tmp.iter_mut().zip(&x) {
                *t += r;
            }
            layer_norm_rows(&tmp, &layer.ln_out.w, &layer.ln_out.b, d.ln_eps, &mut x);
        }
        x
    }

    /// φ_P of token ids: forward, mean over all tokens, both L2 steps.
    pub fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        let hidden = self.forward(ids);
        let mut v = mean_pool(&hidden, self.dims.hidden);
        normalize_phi_p(&mut v);
        v
    }
}

// ------------------------------------------------------------------ encoder

/// What [`Encoder::verify_golden`] measured.
#[derive(Clone, Debug, PartialEq)]
pub struct GoldenReport {
    pub rows: usize,
    /// Rows equal bit for bit.
    pub bit_exact_rows: usize,
    pub max_abs: f32,
}

/// Embedding and ordered per-skill reconstruction errors.
pub(crate) type EmbeddingAndErrors = (Vec<f32>, Vec<Vec<f32>>);

/// Tokenizer + BERT: text → φ_P.
#[derive(Clone, Debug)]
pub struct Encoder {
    tokenizer: WordPiece,
    model: BertModel,
    #[cfg(target_os = "macos")]
    metal: Option<Arc<parking_lot::Mutex<metal_backend::MetalEncoder>>>,
    #[cfg(feature = "vulkan")]
    vulkan: Option<Arc<parking_lot::Mutex<vulkan_backend::VulkanEncoder>>>,
    /// The record's `tables` differs from this build's (a warning).
    warnings: Vec<String>,
}

impl Encoder {
    pub fn new(tokenizer: WordPiece, model: BertModel) -> Result<Self> {
        ensure!(
            tokenizer.vocab_size() == model.dims.vocab,
            "vocab has {} tokens, the encoder config {}",
            tokenizer.vocab_size(),
            model.dims.vocab
        );
        ensure!(
            tokenizer.max_length() <= model.dims.max_position,
            "truncation max_length {} exceeds max_position {}",
            tokenizer.max_length(),
            model.dims.max_position
        );
        Ok(Self {
            tokenizer,
            model,
            #[cfg(target_os = "macos")]
            metal: None,
            #[cfg(feature = "vulkan")]
            vulkan: None,
            warnings: Vec::new(),
        })
    }

    /// The encoder of a decision file (weights copied out of the mmap). The
    /// golden φ_P is not checked here: [`Encoder::verify_golden`].
    pub fn from_model(model: &DecisionModel) -> Result<Self> {
        let rec = &model.representation().encoder;
        ensure!(
            rec.kind == ENCODER_KIND,
            "unsupported encoder kind '{}'",
            rec.kind
        );
        let tokenizer = WordPiece::from_record(model.vocab_bytes()?, &rec.tokenizer)?;
        let mut enc = Self::new(tokenizer, BertModel::from_model(model)?)?;
        if rec.tokenizer.tables != unicode_tables::SOURCE {
            enc.warnings.push(format!(
                "the encoder's tokenizer was recorded with Unicode tables '{}'; this build uses '{}'",
                rec.tokenizer.tables,
                unicode_tables::SOURCE
            ));
        }
        enc.with_device(EncoderDevice::from_env()?)
    }

    /// Select an execution device; GPU initialization errors never fall back to CPU.
    #[allow(unused_mut)]
    pub fn with_device(mut self, device: EncoderDevice) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            self.metal = match device {
                EncoderDevice::Cpu | EncoderDevice::Vulkan => None,
                EncoderDevice::Metal => Some(Arc::new(parking_lot::Mutex::new(
                    metal_backend::MetalEncoder::new(&self.model)?,
                ))),
            };
        }
        #[cfg(not(target_os = "macos"))]
        ensure!(
            device != EncoderDevice::Metal,
            "decision Metal requires macOS on Apple Silicon"
        );
        #[cfg(feature = "vulkan")]
        {
            self.vulkan = match device {
                EncoderDevice::Vulkan => Some(Arc::new(parking_lot::Mutex::new(
                    vulkan_backend::VulkanEncoder::new(&self.model)?,
                ))),
                _ => None,
            };
        }
        #[cfg(not(feature = "vulkan"))]
        ensure!(
            device != EncoderDevice::Vulkan,
            "rebuild cortiq-decision with --features vulkan"
        );
        Ok(self)
    }

    pub fn device_name(&self) -> String {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return format!("metal: {}", metal.lock().device_name());
        }
        #[cfg(feature = "vulkan")]
        if let Some(v) = &self.vulkan {
            return format!("vulkan: {}", v.lock().device_name());
        }
        "cpu".into()
    }

    /// Completed GPU command buffers, not an inference from environment flags.
    pub fn metal_submissions(&self) -> u64 {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return metal.lock().submissions();
        }
        0
    }

    pub fn vulkan_submissions(&self) -> u64 {
        #[cfg(feature = "vulkan")]
        if let Some(v) = &self.vulkan {
            return v.lock().submissions();
        }
        0
    }
    pub fn gpu_submissions(&self) -> u64 {
        self.metal_submissions() + self.vulkan_submissions()
    }

    pub fn tokenizer(&self) -> &WordPiece {
        &self.tokenizer
    }

    pub fn model(&self) -> &BertModel {
        &self.model
    }

    /// φ_P dimension.
    pub fn dim(&self) -> usize {
        self.model.dims.hidden
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// `[CLS] … [SEP]` ids of a text.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// φ_P of token ids.
    pub fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        self.try_embed_ids(ids).expect("decision encoder failed")
    }

    /// Fallible device execution, used by the request and golden-check paths.
    pub fn try_embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>> {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal {
            return metal.lock().embed_ids(ids);
        }
        #[cfg(feature = "vulkan")]
        if let Some(v) = &self.vulkan {
            return v.lock().embed_ids(ids);
        }
        Ok(self.model.embed_ids(ids))
    }

    /// Joint device path. None means explicitly mixed CPU/GPU backends, not
    /// a swallowed GPU failure. Empty skills require no reconstruction dispatch.
    pub(crate) fn try_embed_and_score(
        &self,
        ids: &[u32],
        hash: &[f32],
        packed: &[&crate::packed::Packed],
    ) -> Result<Option<EmbeddingAndErrors>> {
        #[cfg(target_os = "macos")]
        if let Some(metal) = &self.metal
            && packed.iter().all(|p| p.tasks() == 0 || p.metal.is_some())
        {
            return metal.lock().embed_and_score(ids, hash, packed).map(Some);
        }
        #[cfg(feature = "vulkan")]
        if let Some(v) = &self.vulkan {
            if packed.iter().all(|p| p.tasks() == 0 || p.vulkan.is_some()) {
                return v.lock().embed_and_score(ids, hash, packed).map(Some);
            }
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "vulkan")))]
        let _ = (ids, hash, packed);
        Ok(None)
    }

    /// φ_P of a text.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        self.embed_ids(&self.tokenize(text))
    }

    /// φ_P of each text, row-major `[texts.len(), dim]`.
    pub fn golden<S: AsRef<str>>(&self, texts: &[S]) -> Vec<f32> {
        let mut out = Vec::with_capacity(texts.len() * self.dim());
        for t in texts {
            out.extend(self.encode(t.as_ref()));
        }
        out
    }

    /// Compare φ_P of `texts` with `stored` (`[texts.len(), dim]`): refused
    /// when a value differs by more than [`GOLDEN_MAX_ABS`].
    pub fn check_golden<S: AsRef<str>>(&self, texts: &[S], stored: &[f32]) -> Result<GoldenReport> {
        let dim = self.dim();
        ensure!(
            stored.len() == texts.len() * dim,
            "golden holds {} values, {} texts × {dim} expected",
            stored.len(),
            texts.len()
        );
        let mut rep = GoldenReport {
            rows: texts.len(),
            bit_exact_rows: 0,
            max_abs: 0.0,
        };
        for (i, t) in texts.iter().enumerate() {
            let v = self.try_embed_ids(&self.tokenize(t.as_ref()))?;
            let s = &stored[i * dim..(i + 1) * dim];
            if v.iter().zip(s).all(|(a, b)| a.to_bits() == b.to_bits()) {
                rep.bit_exact_rows += 1;
            }
            for (a, b) in v.iter().zip(s) {
                let d = (a - b).abs();
                if !d.is_finite() {
                    bail!("encoder golden mismatch: text {i} is not finite");
                }
                rep.max_abs = rep.max_abs.max(d);
            }
        }
        ensure!(
            rep.max_abs <= GOLDEN_MAX_ABS,
            "encoder golden mismatch: max |Δφ_P| {:e} > {GOLDEN_MAX_ABS:e} over {} texts ({} bit-exact)",
            rep.max_abs,
            rep.rows,
            rep.bit_exact_rows
        );
        Ok(rep)
    }

    /// The load-time check of a decision file (spec §2.8): φ_P of the recorded
    /// golden texts against `decision.encoder.golden`.
    pub fn verify_golden(&self, model: &DecisionModel) -> Result<GoldenReport> {
        let rec = &model.representation().encoder;
        let stored = model.encoder_golden()?;
        self.check_golden(&rec.golden_texts, &stored)
    }
}

// ------------------------------------------------------------------ export directory

/// `format` of `encoder.json`.
pub const EXPORT_FORMAT: &str = "cortiq-decision-encoder-export/1";
/// The export's manifest file.
pub const EXPORT_MANIFEST: &str = "encoder.json";

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportTokenizer {
    file: String,
    vocab_file: String,
    unk: String,
    prefix: String,
    max_input_chars_per_word: u64,
    normalizer: NormalizerRecord,
    pre_tokenizer: String,
    template: String,
    ids: SpecialIds,
    max_length: u64,
    truncation_direction: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportTensor {
    name: String,
    file: String,
    shape: Vec<usize>,
    dtype: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifest {
    format: String,
    kind: String,
    source: EncoderSource,
    config: EncoderConfig,
    tokenizer: ExportTokenizer,
    tensors: Vec<ExportTensor>,
    /// Checks the export tool ran (e.g. the ORT parity) — informational.
    #[serde(default)]
    checks: Option<serde_json::Value>,
    /// Tool versions — informational.
    #[serde(default)]
    tools: Option<serde_json::Value>,
}

/// Parse a `.npy` holding little-endian f32 in C order: (shape, data bytes).
pub fn parse_npy_f32(bytes: &[u8]) -> Result<(Vec<usize>, &[u8])> {
    ensure!(
        bytes.len() >= 10 && &bytes[..6] == b"\x93NUMPY",
        "not a .npy file"
    );
    let major = bytes[6];
    let (hlen, off) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => {
            ensure!(bytes.len() >= 12, "truncated .npy header");
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12,
            )
        }
        v => bail!(".npy format version {v} is not supported"),
    };
    ensure!(bytes.len() >= off + hlen, "truncated .npy header");
    let header = std::str::from_utf8(&bytes[off..off + hlen])
        .map_err(|_| anyhow::anyhow!(".npy header is not text"))?;
    let field = |key: &str| -> Result<&str> {
        let pat = format!("'{key}':");
        let at = header
            .find(&pat)
            .ok_or_else(|| anyhow::anyhow!(".npy header has no {key}"))?;
        Ok(header[at + pat.len()..].trim_start())
    };
    let descr = field("descr")?;
    ensure!(
        descr.starts_with("'<f4'"),
        ".npy dtype must be '<f4' (little-endian f32)"
    );
    ensure!(
        field("fortran_order")?.starts_with("False"),
        ".npy must be C order"
    );
    let shape_s = field("shape")?;
    ensure!(shape_s.starts_with('('), ".npy shape is not a tuple");
    let close = shape_s
        .find(')')
        .ok_or_else(|| anyhow::anyhow!(".npy shape is not closed"))?;
    let mut shape = Vec::new();
    for part in shape_s[1..close].split(',') {
        let p = part.trim();
        if !p.is_empty() {
            shape.push(
                p.parse::<usize>()
                    .map_err(|_| anyhow::anyhow!(".npy shape entry '{p}'"))?,
            );
        }
    }
    let data = &bytes[off + hlen..];
    let want = 4 * shape.iter().product::<usize>();
    ensure!(
        data.len() == want,
        ".npy holds {} data bytes, the shape {shape:?} wants {want}",
        data.len()
    );
    Ok((shape, data))
}

fn f32_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The tensors `cortiq decision init` writes, with their record.
#[derive(Clone, Debug)]
pub struct InitTensors {
    /// The encoder record (`tensors_sha256` over `tensors`, golden texts set).
    pub record: EncoderRecord,
    /// Every `decision.encoder.*` tensor: the F32 weights, the golden φ_P and
    /// the U8 vocab.
    pub tensors: Vec<TensorSpec>,
    /// φ_P of the golden texts (the values of `decision.encoder.golden`).
    pub golden: Vec<f32>,
}

/// An encoder export directory (`tools/decision_export_encoder.py`), checked.
#[derive(Clone, Debug)]
pub struct EncoderExport {
    pub dir: PathBuf,
    pub source: EncoderSource,
    pub config: EncoderConfig,
    pub tokenizer: TokenizerRecord,
    /// Bytes of vocab.txt.
    pub vocab: Vec<u8>,
    /// Weights by name without `decision.encoder.`, as little-endian f32 bytes.
    weights: BTreeMap<String, (Vec<usize>, Vec<u8>)>,
    /// The export's own checks (informational).
    pub checks: Option<serde_json::Value>,
    /// Versions of the tools that wrote the export (informational).
    pub tools: Option<serde_json::Value>,
}

impl EncoderExport {
    /// Read and check an export directory: `format`, every sha256 (tensor data,
    /// vocab.txt, tokenizer.json), every shape against the config, and the
    /// tokenizer settings the native tokenizer implements.
    pub fn read(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mpath = dir.join(EXPORT_MANIFEST);
        let m: ExportManifest = serde_json::from_slice(
            &std::fs::read(&mpath).with_context(|| format!("read {}", mpath.display()))?,
        )
        .with_context(|| format!("parse {}", mpath.display()))?;
        ensure!(
            m.format == EXPORT_FORMAT,
            "{}: format '{}' is not {EXPORT_FORMAT}",
            mpath.display(),
            m.format
        );
        ensure!(
            m.kind == ENCODER_KIND,
            "unsupported encoder kind '{}'",
            m.kind
        );
        m.config.validate()?;
        let t = &m.tokenizer;
        ensure!(
            t.normalizer == NormalizerRecord::bert_uncased()
                && t.pre_tokenizer == PRE_TOKENIZER
                && t.template == TEMPLATE
                && t.truncation_direction == TRUNCATION_DIRECTION,
            "the export's tokenizer is not the uncased BERT WordPiece set-up this build implements"
        );
        let file = |name: &str| -> Result<PathBuf> {
            ensure!(
                !name.is_empty()
                    && !name.contains('/')
                    && !name.contains('\\')
                    && name != "."
                    && name != "..",
                "export file name '{name}' must be a plain file name"
            );
            Ok(dir.join(name))
        };
        let vocab = std::fs::read(file(&t.vocab_file)?)
            .with_context(|| format!("read {}", t.vocab_file))?;
        ensure!(
            sha256_hex(&vocab) == m.source.vocab_sha256,
            "{} does not match source.vocab_sha256",
            t.vocab_file
        );
        let tok_json = std::fs::read(file(&t.file)?).with_context(|| format!("read {}", t.file))?;
        ensure!(
            sha256_hex(&tok_json) == m.source.tokenizer_json_sha256,
            "{} does not match source.tokenizer_json_sha256",
            t.file
        );
        let tokenizer =
            TokenizerRecord::bert_uncased(t.ids.clone(), t.max_length, unicode_tables::SOURCE);
        let tokenizer = TokenizerRecord {
            unk: t.unk.clone(),
            prefix: t.prefix.clone(),
            max_input_chars_per_word: t.max_input_chars_per_word,
            ..tokenizer
        };
        let shapes: BTreeMap<String, Vec<usize>> = m.config.weight_shapes().into_iter().collect();
        let mut weights = BTreeMap::new();
        for e in &m.tensors {
            let Some(shape) = shapes.get(&e.name) else {
                bail!("'{}' is not a weight of this encoder config", e.name);
            };
            ensure!(
                e.dtype == "float32",
                "'{}': dtype {} (float32 expected)",
                e.name,
                e.dtype
            );
            ensure!(
                &e.shape == shape,
                "'{}': shape {:?}, the config wants {shape:?}",
                e.name,
                e.shape
            );
            let path = file(&e.file)?;
            let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
            let (npy_shape, data) =
                parse_npy_f32(&bytes).with_context(|| format!("{}", path.display()))?;
            ensure!(
                &npy_shape == shape,
                "{}: .npy shape {npy_shape:?}, the config wants {shape:?}",
                path.display()
            );
            ensure!(
                sha256_hex(data) == e.sha256,
                "{}: data sha256 differs from encoder.json",
                path.display()
            );
            ensure!(
                weights
                    .insert(e.name.clone(), (shape.clone(), data.to_vec()))
                    .is_none(),
                "'{}' is listed twice",
                e.name
            );
        }
        let missing: Vec<&String> = shapes
            .keys()
            .filter(|k| !weights.contains_key(*k))
            .collect();
        ensure!(missing.is_empty(), "export misses weights {missing:?}");
        Ok(Self {
            dir,
            source: m.source,
            config: m.config,
            tokenizer,
            vocab,
            weights,
            checks: m.checks,
            tools: m.tools,
        })
    }

    /// The tokenizer of this export.
    pub fn wordpiece(&self) -> Result<WordPiece> {
        WordPiece::from_record(&self.vocab, &self.tokenizer)
    }

    /// The encoder of this export (weights copied).
    pub fn encoder(&self) -> Result<Encoder> {
        let model = BertModel::from_weights(&self.config, |name| {
            let (_, bytes) = self
                .weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing weight '{name}'"))?;
            Ok(Weights::Owned(f32_from_le(bytes)))
        })?;
        Encoder::new(self.wordpiece()?, model)
    }

    /// The tensors and record of `cortiq decision init`: every weight under
    /// `decision.encoder.`, `decision.encoder.golden` (φ_P of `golden_texts`,
    /// [`manifest::ENCODER_GOLDEN_COUNT`] of them) and `decision.encoder.vocab`.
    pub fn init_tensors<S: AsRef<str>>(&self, golden_texts: &[S]) -> Result<InitTensors> {
        ensure!(
            golden_texts.len() == ENCODER_GOLDEN_COUNT,
            "{} golden texts given, {ENCODER_GOLDEN_COUNT} expected",
            golden_texts.len()
        );
        let encoder = self.encoder()?;
        let golden = encoder.golden(golden_texts);
        ensure!(
            golden.iter().all(|v| v.is_finite()),
            "golden φ_P is not finite"
        );
        let dim = encoder.dim();
        let mut tensors: Vec<TensorSpec> = self
            .weights
            .iter()
            .map(|(name, (shape, bytes))| TensorSpec {
                name: format!("{ENCODER_PREFIX}{name}"),
                dtype: TensorDtype::F32,
                shape: shape.clone(),
                data: bytes.clone(),
            })
            .collect();
        tensors.push(TensorSpec {
            name: GOLDEN_TENSOR.to_string(),
            dtype: TensorDtype::F32,
            shape: vec![ENCODER_GOLDEN_COUNT, dim],
            data: golden.iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
        tensors.push(TensorSpec {
            name: VOCAB_TENSOR.to_string(),
            dtype: TensorDtype::U8,
            shape: vec![self.vocab.len()],
            data: self.vocab.clone(),
        });
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let names: BTreeSet<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        ensure!(
            names.len() == tensors.len(),
            "duplicate encoder tensor names"
        );
        let tensors_sha256 = manifest::tensors_sha256(tensors.iter().map(|t| TensorDigest {
            name: &t.name,
            dtype: t.dtype,
            shape: &t.shape,
            data: &t.data,
        }));
        let record = EncoderRecord {
            kind: ENCODER_KIND.into(),
            source: self.source.clone(),
            config: self.config.clone(),
            tokenizer: self.tokenizer.clone(),
            pooling: POOLING.into(),
            normalization: NORMALIZATION.iter().map(|s| s.to_string()).collect(),
            dim: self.config.hidden,
            tensors_sha256,
            golden_texts: golden_texts
                .iter()
                .map(|s| s.as_ref().to_string())
                .collect(),
        };
        record.validate()?;
        Ok(InitTensors {
            record,
            tensors,
            golden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (-1.0, -0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (-3.0, -0.999_977_909_503_001_4),
        ] {
            assert!(
                (erf(x) - want).abs() < 2e-7,
                "erf({x}) = {} vs {want}",
                erf(x)
            );
        }
    }

    #[test]
    fn gelu_and_softmax() {
        assert_eq!(gelu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.158_655_25).abs() < 1e-6);
        let mut s = [1.0f32, 2.0, 3.0];
        softmax_row(&mut s);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn normalization_is_unit() {
        let mut v = vec![3.0f32, 4.0];
        normalize_phi_p(&mut v);
        assert!((norm_seq_f32(&v) - 1.0).abs() <= f32::EPSILON);
        let mut z = vec![0.0f32; 4];
        normalize_phi_p(&mut z);
        assert!(z.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn npy_header_parse() {
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
        let mut h = header.to_string();
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(' ');
        }
        h.push('\n');
        b.extend((h.len() as u16).to_le_bytes());
        b.extend(h.as_bytes());
        b.extend(std::iter::repeat_n(0u8, 24));
        let (shape, data) = parse_npy_f32(&b).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(data.len(), 24);
        assert!(parse_npy_f32(&b[..b.len() - 4]).is_err());
    }
}
