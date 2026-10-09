//! EmbeddingGemma 2 (`google/embeddinggemma-2`, `model_type:
//! embedding_gemma2`) — the text encoder of a 768-d multimodal embedding
//! model, read from a `.cmf` packed by `cortiq convert`.
//!
//! The forward reproduces transformers' `EmbeddingGemma2Model` followed by
//! sentence-transformers' mean pooling and L2 normalization:
//!
//! 1. **Tokens.** `[BOS] + tokenize(prompt + text) + [EOS]` (the tokenizer's
//!    own post-processor), capped at 8192 tokens. The task prompts are the
//!    sentence-transformers table carried in the header; `Document` is
//!    `title: {title | none} | text: `.
//! 2. **Embedding.** `E[id] · sqrt(512)`, in f32.
//! 3. **Per-layer inputs (projection-only PLE).** `P = RMSNorm_w(reshape(
//!    x · W_pleᵀ · 512^-½, [T, 24, 512]))` from the merged input embeddings,
//!    one shared norm weight. There is no per-layer token table.
//! 4. **24 bidirectional layers**, five sliding to one full:
//!    * sliding: 4 query heads over 2 key heads of 256, RoPE θ = 10⁴, a
//!      token sees keys with `|i - j| ≤ 512`;
//!    * full (layers 5, 11, 17, 23): 4 query heads over ONE key head of 512,
//!      RoPE θ = 10⁶ over the whole head, every token sees every token.
//!
//!    Attention is unscaled (`scaling = 1`, the q/k RMS-norms carry the
//!    scale), values are RMS-normalized without a weight, and the block is
//!    the Gemma sandwich: `h += N(attn(N(h)))`, `h += N(FFN(N(h)))` with a
//!    gelu-tanh gated FFN, then the per-layer-input gate
//!    `h += N(W_proj(gelu(W_gate h) · P_i))`, then `h *= layer_scalar_i`.
//!    Every RMS-norm multiplies by its weight plainly (not `1 + w`).
//! 5. **Head.** Final RMS-norm → 512→768 projection per token → mean over
//!    every token (BOS, prompt and EOS included) → L2. The projection is
//!    linear and bias-free, so it is applied once to the pooled mean.
//! 6. **Matryoshka.** The first 512/256/128 coordinates, L2-normalized again.
//!
//! Every step is checked against the float32 reference in
//! `tests/egemma2_parity.rs` (`CORTIQ_EGEMMA2_MODEL` + `CORTIQ_EGEMMA2_REF`).
//! The arithmetic is f32 on the host (Accelerate on macOS); sequences of a
//! batch are packed back to back so every token-wise projection is one GEMM,
//! and attention runs per sequence, so a batch embeds exactly as its
//! members would alone.

use crate::ltxdit::{Shared, gelu_tanh, rows};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::tokenizer::Tokenizer;
use cortiq_core::CmfModel;
use cortiq_core::types::TensorDtype;
use std::sync::{Arc, Mutex};

/// The architecture name a packed file carries.
pub const ARCH_NAME: &str = "embedding_gemma2";
/// The model's context, shared by every modality (the card's 8192).
pub const MAX_TOKENS: usize = 8192;
/// The Matryoshka dimensions the model was trained to truncate to.
pub const MATRYOSHKA_DIMS: [usize; 4] = [768, 512, 256, 128];
/// Sequences are packed into one forward up to this many tokens.
const PACK_TOKENS: usize = 8192;
/// Query rows per attention block: sliding layers (a block's key span is
/// its rows plus the window, so smaller blocks waste less) and full ones.
const QBLOCK: usize = 128;
const QBLOCK_FULL: usize = 512;
/// Sequences up to this long attend on the pool, one (sequence, head) per
/// work item; longer ones go through blocked GEMMs.
const SHORT_SEQ: usize = 96;

const EPS: f64 = 1e-6;

/// `CMF_EGEMMA2_PROF=1`: per-forward time split (projections / attention
/// core / the rest) on stderr.
mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static LIN: AtomicU64 = AtomicU64::new(0);
    pub static ATTN: AtomicU64 = AtomicU64::new(0);
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_EGEMMA2_PROF").is_ok_and(|v| v == "1"))
    }
    pub fn add(c: &AtomicU64, t: std::time::Instant) {
        c.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
    pub fn take(c: &AtomicU64) -> f64 {
        c.swap(0, Ordering::Relaxed) as f64 / 1e3
    }
}

/// The sentence-transformers prompt table of the release
/// (`config_sentence_transformers.json`), used when a file carries none.
pub const DEFAULT_PROMPTS: &[(&str, &str)] = &[
    ("BitextMining", "task: search result | query: "),
    ("Classification", "task: classification | query: "),
    ("Clustering", "task: clustering | query: "),
    ("CodeRetrieval", "task: code retrieval | query: "),
    ("Document", "title: none | text: "),
    ("FactChecking", "task: fact checking | query: "),
    ("InstructionRetrieval", "task: code retrieval | query: "),
    ("MultilabelClassification", "task: classification | query: "),
    ("PairClassification", "task: sentence similarity | query: "),
    ("QuestionAnswering", "task: question answering | query: "),
    ("Reranking", "task: search result | query: "),
    ("Retrieval", "task: search result | query: "),
    ("Retrieval-document", "title: none | text: "),
    ("Retrieval-query", "task: search result | query: "),
    ("STS", "task: sentence similarity | query: "),
    ("SearchQuery", "task: search result | query: "),
    ("SentenceSimilarity", "task: sentence similarity | query: "),
    ("Summarization", "task: sentence similarity | query: "),
    ("document", "title: none | text: "),
    ("query", "task: search result | query: "),
];

/// The prefix every Document prompt starts with; a title replaces `none`.
const DOC_PREFIX: &str = "title: none | text: ";

/// Is this container an EmbeddingGemma 2 pack?
pub fn is_embedding_gemma2(model: &CmfModel) -> bool {
    model.header.arch.arch_name == ARCH_NAME
}

// ------------------------------------------------------------ pure helpers

/// L2-normalize in place (a zero vector stays zero).
pub fn l2_normalize(v: &mut [f32]) {
    let n = v
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x = (*x as f64 / n) as f32;
        }
    }
}

/// Matryoshka truncation: the first `dim` coordinates, L2-normalized again.
/// `dim` must be one of [`MATRYOSHKA_DIMS`] (the model was trained for
/// those) and at most the vector's length.
pub fn matryoshka(v: &[f32], dim: usize) -> Result<Vec<f32>, String> {
    if !MATRYOSHKA_DIMS.contains(&dim) {
        return Err(format!(
            "dimensions {dim}: EmbeddingGemma 2 supports 768, 512, 256 or 128"
        ));
    }
    if dim > v.len() {
        return Err(format!("dimensions {dim} > embedding size {}", v.len()));
    }
    let mut out = v[..dim].to_vec();
    l2_normalize(&mut out);
    Ok(out)
}

/// Mean over the rows of `x` (`[n, d]`), accumulated in f64.
pub fn mean_pool(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut acc = vec![0f64; d];
    for row in x.chunks_exact(d).take(n) {
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v as f64;
        }
    }
    let inv = 1.0 / n.max(1) as f64;
    acc.into_iter().map(|a| (a * inv) as f32).collect()
}

/// Cosine similarity of two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    if aa == 0.0 || bb == 0.0 {
        0.0
    } else {
        ab / (aa.sqrt() * bb.sqrt())
    }
}

/// The task-prompt table.
#[derive(Clone, Debug)]
pub struct Prompts {
    table: Vec<(String, String)>,
}

impl Default for Prompts {
    fn default() -> Self {
        Prompts {
            table: DEFAULT_PROMPTS
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        }
    }
}

impl Prompts {
    /// From a sentence-transformers config (`{"prompts": {name: prefix}}`).
    pub fn from_st_config(cfg: &serde_json::Value) -> Option<Prompts> {
        let obj = cfg.get("prompts")?.as_object()?;
        let mut table: Vec<(String, String)> = obj
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string())))
            .collect();
        table.sort();
        if table.is_empty() {
            None
        } else {
            Some(Prompts { table })
        }
    }

    /// The prefix for `name`: exact match first, then case-insensitive.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.table
            .iter()
            .find(|(k, _)| k == name)
            .or_else(|| {
                self.table
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(name))
            })
            .map(|(_, v)| v.as_str())
    }

    pub fn names(&self) -> Vec<&str> {
        self.table.iter().map(|(k, _)| k.as_str()).collect()
    }

    /// The text the model sees for one input.
    ///
    /// * `prompt` (a raw prefix) wins over everything;
    /// * a named prompt is prepended; a Document prompt takes `title`
    ///   (`title: {title} | text: `, `title: none` without one);
    /// * a `title` with no prompt name means the Document prompt;
    /// * neither: the text as given (the release sets no default prompt).
    pub fn format(
        &self,
        text: &str,
        prompt_name: Option<&str>,
        title: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<String, String> {
        if let Some(p) = prompt {
            if title.is_some() {
                return Err("title applies to the Document prompt, not to a raw prompt".into());
            }
            return Ok(format!("{p}{text}"));
        }
        let prefix = match prompt_name {
            Some(n) => self.get(n).ok_or_else(|| {
                format!(
                    "unknown prompt name '{n}' (known: {})",
                    self.names().join(", ")
                )
            })?,
            None if title.is_some() => DOC_PREFIX,
            None => return Ok(text.to_string()),
        };
        match title {
            Some(t) => {
                let rest = prefix.strip_prefix("title: none").ok_or_else(|| {
                    format!(
                        "title applies to the Document prompt, not to '{}'",
                        prompt_name.unwrap_or("")
                    )
                })?;
                let t = t.trim();
                let t = if t.is_empty() { "none" } else { t };
                Ok(format!("title: {t}{rest}{text}"))
            }
            None => Ok(format!("{prefix}{text}")),
        }
    }
}

/// `[BOS] + ids + [EOS]`, the body truncated so the whole stays within
/// `max_tokens` (the special tokens are kept, as a truncating HF tokenizer
/// keeps them).
pub fn wrap_ids(ids: &[u32], bos: u32, eos: u32, max_tokens: usize) -> Vec<u32> {
    let body = ids.len().min(max_tokens.saturating_sub(2));
    let mut out = Vec::with_capacity(body + 2);
    out.push(bos);
    out.extend_from_slice(&ids[..body]);
    out.push(eos);
    out
}

// ------------------------------------------------------------ weights

/// A projection `y = x·Wᵀ`: owned f32 (the exact profiles dequantize once)
/// or a mapped quantized tensor on the engine's kernels.
pub(crate) enum Mat {
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Q(QTensor),
}

impl Mat {
    /// `dequant`: widen a quantized matrix to f32 once (Accelerate GEMM,
    /// 4 bytes a weight) instead of running the quantized kernels on it.
    pub(crate) fn load(model: &Arc<CmfModel>, name: &str, dequant: bool) -> Result<Mat, String> {
        let e = model
            .tensor(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if dequant && e.shape.len() == 2 {
            let (rows, cols) = (e.shape[0], e.shape[1]);
            return Ok(Mat::F32 {
                w: vecf(model, name)?,
                rows,
                cols,
            });
        }
        Ok(match QTensor::from_model(model, name)? {
            QTensor::F32 { data, rows, cols } => Mat::F32 {
                w: data,
                rows,
                cols,
            },
            q => Mat::Q(q),
        })
    }

    pub(crate) fn rows(&self) -> usize {
        match self {
            Mat::F32 { rows, .. } => *rows,
            Mat::Q(q) => q.rows(),
        }
    }

    pub(crate) fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let t0 = std::time::Instant::now();
        let mut out = vec![0f32; n * self.rows()];
        match self {
            Mat::F32 { w, rows, cols } => {
                crate::fcd_ops::gemm_nt_host(x, w, &mut out, n, *cols, *rows, pool)
            }
            Mat::Q(q) => q.matmat(x, n, &mut out, pool),
        }
        prof::add(&prof::LIN, t0);
        out
    }
}

/// The token table, read a row at a time. A bf16 table stays in the mapping
/// (half a gigabyte of f32 otherwise).
enum Table {
    Bf16 {
        model: Arc<CmfModel>,
        idx: usize,
        rows: usize,
        cols: usize,
    },
    Q(QTensor),
}

impl Table {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Table, String> {
        let idx = model
            .tensor_index(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let e = &model.tensors[idx];
        if e.shape.len() != 2 {
            return Err(format!("{name}: expected a 2-D table"));
        }
        if e.dtype == TensorDtype::Bf16 {
            return Ok(Table::Bf16 {
                model: model.clone(),
                idx,
                rows: e.shape[0],
                cols: e.shape[1],
            });
        }
        Ok(Table::Q(QTensor::from_model(model, name)?))
    }

    fn rows(&self) -> usize {
        match self {
            Table::Bf16 { rows, .. } => *rows,
            Table::Q(q) => q.rows(),
        }
    }

    fn row(&self, r: usize, dst: &mut [f32]) {
        match self {
            Table::Bf16 {
                model, idx, cols, ..
            } => {
                let bytes = model.entry_bytes(&model.tensors[*idx]);
                let src = &bytes[r * cols * 2..(r + 1) * cols * 2];
                for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                    *d = cortiq_core::quant::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                }
            }
            Table::Q(q) => q.row_f32(r, dst),
        }
    }
}

pub(crate) fn vecf(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    crate::dit::cmf_f32(model, name)
}

struct Layer {
    q: Mat,
    k: Mat,
    v: Mat,
    o: Mat,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    in_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    pre_ff_norm: Vec<f32>,
    post_ff_norm: Vec<f32>,
    gate: Mat,
    up: Mat,
    down: Mat,
    ple_gate: Mat,
    ple_proj: Mat,
    ple_norm: Vec<f32>,
    /// this layer's `[512, 512]` slice of the per-layer-input projection
    ple_in: Mat,
    scalar: f32,
    head_dim: usize,
    q_heads: usize,
    kv_heads: usize,
    /// `Some(w)`: a token attends to keys with `|i - j| <= w`
    window: Option<usize>,
    theta: f32,
}

/// RoPE tables for one `(head_dim, theta)`: `[pos, head_dim/2]`, computed
/// the way the reference does (f32 inverse frequencies, f32 angles).
struct Rot {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Rot {
    fn build(len: usize, head_dim: usize, theta: f32) -> Rot {
        let half = head_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / head_dim as f32))
            .collect();
        let mut cos = vec![0f32; len * half];
        let mut sin = vec![0f32; len * half];
        for p in 0..len {
            for (j, &f) in inv.iter().enumerate() {
                let a = (p as f32 * f) as f64;
                cos[p * half + j] = a.cos() as f32;
                sin[p * half + j] = a.sin() as f32;
            }
        }
        Rot { cos, sin, half }
    }

    /// `x·cos + rotate_half(x)·sin` over the whole head.
    fn apply(&self, pos: usize, x: &mut [f32]) {
        let h = self.half;
        let (c, s) = (
            &self.cos[pos * h..(pos + 1) * h],
            &self.sin[pos * h..(pos + 1) * h],
        );
        for i in 0..h {
            let (a, b) = (x[i], x[i + h]);
            x[i] = a * c[i] - b * s[i];
            x[i + h] = b * c[i] + a * s[i];
        }
    }
}

/// RMS-normalize `x` (`[n, d]`) row by row, times `w` when given.
pub(crate) fn rms_rows(x: &[f32], w: Option<&[f32]>, d: usize, pool: Option<&Pool>) -> Vec<f32> {
    let n = x.len() / d;
    let mut out = vec![0f32; x.len()];
    let dst = Shared(out.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let o = unsafe { dst.at(s * d, (e - s) * d) };
        for (orow, xrow) in o.chunks_exact_mut(d).zip(x[s * d..e * d].chunks_exact(d)) {
            rms_into(xrow, w, orow);
        }
    });
    out
}

#[inline]
pub(crate) fn rms_into(x: &[f32], w: Option<&[f32]>, out: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + EPS).sqrt();
    match w {
        Some(w) => {
            for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
                *o = (v as f64 * inv) as f32 * g;
            }
        }
        None => {
            for (o, &v) in out.iter_mut().zip(x) {
                *o = (v as f64 * inv) as f32;
            }
        }
    }
}

/// `h += rms_w(y)` row by row.
fn add_normed(h: &mut [f32], y: &[f32], w: &[f32], d: usize, pool: Option<&Pool>) {
    let n = h.len() / d;
    let dst = Shared(h.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let hh = unsafe { dst.at(s * d, (e - s) * d) };
        let mut tmp = vec![0f32; d];
        for (hrow, yrow) in hh.chunks_exact_mut(d).zip(y[s * d..e * d].chunks_exact(d)) {
            rms_into(yrow, Some(w), &mut tmp);
            for (a, &b) in hrow.iter_mut().zip(&tmp) {
                *a += b;
            }
        }
    });
}

/// `x[i] = gelu_tanh(x[i]) * y[i]`.
///
/// `0.5·x·(1 + tanh z)` is `x·σ(2z)`, so one exponential (NEON, |rel err| <
/// 2e-7) and one division replace libm's `tanh` — which, at 2560 values a
/// token a layer, was most of the forward outside the GEMMs.
#[inline]
fn gelu_mul_slice(x: &mut [f32], y: &[f32]) {
    #[allow(unused_mut)] // only the NEON body advances it
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every lane read/written is below `x.len()`, `y` is as long.
    unsafe {
        use core::arch::aarch64::*;
        debug_assert!(y.len() >= x.len());
        let k = vdupq_n_f32(-2.0 * 0.797_884_56);
        let c = vdupq_n_f32(0.044715);
        let one = vdupq_n_f32(1.0);
        while i + 4 <= x.len() {
            let v = vld1q_f32(x.as_ptr().add(i));
            let v3 = vmulq_f32(vmulq_f32(v, v), v);
            let e = crate::attention::vexpq_f32(vmulq_f32(k, vfmaq_f32(v, c, v3)));
            let g = vdivq_f32(v, vaddq_f32(one, e));
            vst1q_f32(
                x.as_mut_ptr().add(i),
                vmulq_f32(g, vld1q_f32(y.as_ptr().add(i))),
            );
            i += 4;
        }
    }
    for (a, &b) in x[i..].iter_mut().zip(&y[i..]) {
        *a = gelu_tanh(*a) * b;
    }
}

/// `a = gelu_tanh(a) * b`, elementwise, across the pool.
fn gelu_mul(a: &mut [f32], b: &[f32], pool: Option<&Pool>) {
    let n = a.len();
    let grain = 16384usize;
    let dst = Shared(a.as_mut_ptr());
    rows(pool, n.div_ceil(grain), &|s, e| {
        let (lo, hi) = (s * grain, (e * grain).min(n));
        let r = unsafe { dst.at(lo, hi - lo) };
        gelu_mul_slice(r, &b[lo..hi]);
    });
}

/// `y[n,m] = x[n,k] · w[m,k]ᵀ` where row `r` of `w` starts at `w[r·ldw]`.
fn gemm_nt_strided(x: &[f32], w: &[f32], ldw: usize, y: &mut [f32], n: usize, k: usize, m: usize) {
    assert!(m == 0 || w.len() >= (m - 1) * ldw + k);
    if ldw == k {
        return crate::fcd_ops::gemm_nt_host(x, &w[..m * k], y, n, k, m, None);
    }
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn cblas_sgemm(
                order: i32,
                ta: i32,
                tb: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                b: *const f32,
                ldb: i32,
                beta: f32,
                c: *mut f32,
                ldc: i32,
            );
        }
        // SAFETY: the bounds were checked above; row-major (101), x·wᵀ.
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                n as i32,
                m as i32,
                k as i32,
                1.0,
                x.as_ptr(),
                k as i32,
                w.as_ptr(),
                ldw as i32,
                0.0,
                y.as_mut_ptr(),
                m as i32,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let mut wc = vec![0f32; m * k];
        for r in 0..m {
            wc[r * k..(r + 1) * k].copy_from_slice(&w[r * ldw..r * ldw + k]);
        }
        crate::fcd_ops::gemm_nt_host(x, &wc, y, n, k, m, None);
    }
}

/// Softmax of `row` in place (finite inputs) — NEON on aarch64.
fn softmax(row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::attention::softmax_row(row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    softmax_scalar(row);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn softmax_scalar(row: &mut [f32]) {
    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0f64;
    for r in row.iter_mut() {
        *r = (*r - mx).exp();
        den += *r as f64;
    }
    let inv = (1.0 / den) as f32;
    for r in row.iter_mut() {
        *r *= inv;
    }
}

// ------------------------------------------------------------ the model

/// The EmbeddingGemma 2 text encoder (and the shared pieces every modality
/// path goes through: the tokenizer, the prompt table, the head).
pub struct EmbeddingGemma2 {
    tok: Tokenizer,
    prompts: Prompts,
    embed: Table,
    layers: Vec<Layer>,
    ple_norm: Vec<f32>,
    norm: Vec<f32>,
    head: Mat,
    hidden: usize,
    embed_scale: f32,
    n_ple: usize,
    pub dim: usize,
    pub max_tokens: usize,
    pub bos: u32,
    pub eos: u32,
    pool: Option<Arc<Pool>>,
    /// one forward at a time: the pool is not re-entrant
    busy: Mutex<()>,
    /// the codec profile the file was packed with (provenance `quant`)
    pub quant: String,
}

/// One text to embed.
#[derive(Clone, Debug, Default)]
pub struct TextInput {
    pub text: String,
    pub prompt_name: Option<String>,
    pub title: Option<String>,
    /// a raw prefix, instead of a named prompt
    pub prompt: Option<String>,
}

impl TextInput {
    pub fn plain(text: impl Into<String>) -> Self {
        TextInput {
            text: text.into(),
            ..Default::default()
        }
    }
}

impl EmbeddingGemma2 {
    /// Load from an EmbeddingGemma 2 pack. `pool`: the worker pool for the
    /// row-parallel work (`Pool::from_env()` when `None` is not wanted).
    pub fn load(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        if !is_embedding_gemma2(model) {
            return Err(format!(
                "not an EmbeddingGemma 2 file (arch '{}')",
                model.header.arch.arch_name
            ));
        }
        let prov = model
            .header
            .provenance
            .clone()
            .unwrap_or(serde_json::Value::Null);
        let eg = &prov["embedding_gemma2"];
        let tc = &eg["config"]["text_config"];
        let g = |k: &str, d: u64| tc.get(k).and_then(|v| v.as_u64()).unwrap_or(d) as usize;
        let hidden = g("hidden_size", 512);
        let n_layers = g("num_hidden_layers", 24);
        let n_heads = g("num_attention_heads", 4);
        let head_dim = g("head_dim", 256);
        let kv_heads = g("num_key_value_heads", 2);
        let window = g("sliding_window", 512);
        let types: Vec<bool> = match tc.get("layer_types").and_then(|v| v.as_array()) {
            Some(a) => a
                .iter()
                .map(|s| s.as_str() == Some("full_attention"))
                .collect(),
            None => (0..n_layers).map(|i| i % 6 == 5).collect(),
        };
        let theta = |kind: &str, d: f64| {
            tc["rope_parameters"][kind]["rope_theta"]
                .as_f64()
                .unwrap_or(d) as f32
        };
        let (theta_slide, theta_full) = (
            theta("sliding_attention", 1e4),
            theta("full_attention", 1e6),
        );
        let per_layer = &tc["per_layer_config"];
        let ple_dim = g("hidden_size_per_layer_input", 512);
        // A quantized text stack is widened to f32 at load: 130 M weights are
        // ~0.5 GB of RAM, and the f32 GEMM runs ~2.5x faster than the q8
        // kernels on these shapes (the token table stays mapped either way).
        // `CMF_EGEMMA2_LOWMEM=1` keeps the quantized kernels.
        let dequant = !std::env::var("CMF_EGEMMA2_LOWMEM").is_ok_and(|v| v == "1");

        // The per-layer-input projection, split into one [ple_dim, hidden]
        // matrix per layer (each layer reads only its own slice).
        let ple_all = vecf(
            model,
            "language_model.ple.per_layer_model_projection.weight",
        )?;
        if ple_all.len() != n_layers * ple_dim * hidden {
            return Err(format!(
                "per_layer_model_projection: {} values, expected {}",
                ple_all.len(),
                n_layers * ple_dim * hidden
            ));
        }

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("language_model.layers.{i}");
            let full = types.get(i).copied().unwrap_or(i % 6 == 5);
            let key = format!("{i:02}");
            let lc = &per_layer[key.as_str()];
            let hd = lc["head_dim"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(if full { head_dim * 2 } else { head_dim });
            let nkv = lc["num_key_value_heads"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(if full { 1 } else { kv_heads });
            let ld = |s: &str| Mat::load(model, &format!("{p}.{s}"), dequant);
            let q = ld("self_attn.q_proj.weight")?;
            let k = ld("self_attn.k_proj.weight")?;
            if q.rows() != n_heads * hd || k.rows() != nkv * hd {
                return Err(format!(
                    "layer {i}: q/k rows {}/{} do not match {n_heads}x{hd} / {nkv}x{hd}",
                    q.rows(),
                    k.rows()
                ));
            }
            layers.push(Layer {
                q,
                k,
                v: ld("self_attn.v_proj.weight")?,
                o: ld("self_attn.o_proj.weight")?,
                q_norm: vecf(model, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: vecf(model, &format!("{p}.self_attn.k_norm.weight"))?,
                in_norm: vecf(model, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: vecf(model, &format!("{p}.post_attention_layernorm.weight"))?,
                pre_ff_norm: vecf(model, &format!("{p}.pre_feedforward_layernorm.weight"))?,
                post_ff_norm: vecf(model, &format!("{p}.post_feedforward_layernorm.weight"))?,
                gate: ld("mlp.gate_proj.weight")?,
                up: ld("mlp.up_proj.weight")?,
                down: ld("mlp.down_proj.weight")?,
                ple_gate: ld("ple_block.per_layer_input_gate.weight")?,
                ple_proj: ld("ple_block.per_layer_projection.weight")?,
                ple_norm: vecf(
                    model,
                    &format!("{p}.ple_block.post_per_layer_input_norm.weight"),
                )?,
                ple_in: Mat::F32 {
                    w: ple_all[i * ple_dim * hidden..(i + 1) * ple_dim * hidden].to_vec(),
                    rows: ple_dim,
                    cols: hidden,
                },
                scalar: vecf(model, &format!("{p}.layer_scalar"))?[0],
                head_dim: hd,
                q_heads: n_heads,
                kv_heads: nkv,
                window: if full { None } else { Some(window) },
                theta: if full { theta_full } else { theta_slide },
            });
        }
        drop(ple_all);

        let vocab = model
            .vocab
            .as_deref()
            .ok_or("EmbeddingGemma 2 file has no embedded tokenizer")?;
        let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
        let cfg = &eg["config"];
        let bos = tc["bos_token_id"]
            .as_u64()
            .or_else(|| cfg["bos_token_id"].as_u64())
            .unwrap_or(2) as u32;
        let eos = tc["eos_token_id"]
            .as_u64()
            .or_else(|| cfg["eos_token_id"].as_u64())
            .unwrap_or(1) as u32;
        let prompts = Prompts::from_st_config(&eg["sentence_transformers"]).unwrap_or_default();
        let head = Mat::load(model, "language_model.embedding_projection.weight", true)?;
        let embed = Table::load(model, "language_model.embed_tokens.weight")?;
        let dim = head.rows();
        Ok(EmbeddingGemma2 {
            tok,
            prompts,
            embed,
            layers,
            ple_norm: vecf(model, "language_model.ple.per_layer_projection_norm.weight")?,
            norm: vecf(model, "language_model.norm.weight")?,
            head,
            hidden,
            embed_scale: (hidden as f32).sqrt(),
            n_ple: ple_dim,
            dim,
            max_tokens: eg["max_tokens"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(MAX_TOKENS),
            bos,
            eos,
            pool,
            busy: Mutex::new(()),
            quant: prov["quant"].as_str().unwrap_or("").to_string(),
        })
    }

    pub fn prompts(&self) -> &Prompts {
        &self.prompts
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }

    /// The model-facing text of one input (prompt and title applied).
    pub fn format(&self, input: &TextInput) -> Result<String, String> {
        self.prompts.format(
            &input.text,
            input.prompt_name.as_deref(),
            input.title.as_deref(),
            input.prompt.as_deref(),
        )
    }

    /// `[BOS] + tokens(text) + [EOS]`, capped at `max_tokens`.
    pub fn tokenize(&self, full_text: &str) -> Vec<u32> {
        wrap_ids(
            &self.tok.encode(full_text),
            self.bos,
            self.eos,
            self.max_tokens,
        )
    }

    /// Token ids of one input, ready for [`embed_ids`](Self::embed_ids).
    pub fn input_ids(&self, input: &TextInput) -> Result<Vec<u32>, String> {
        Ok(self.tokenize(&self.format(input)?))
    }

    /// Embed texts: unit-length 768-d vectors, one per input, in order.
    pub fn embed_texts(&self, inputs: &[TextInput]) -> Result<Vec<Vec<f32>>, String> {
        let ids: Vec<Vec<u32>> = inputs
            .iter()
            .map(|i| self.input_ids(i))
            .collect::<Result<_, _>>()?;
        self.embed_ids(&ids)
    }

    /// Embed already-tokenized sequences (each including BOS/EOS). Returns
    /// unit-length vectors of `self.dim`, one per sequence, in order.
    pub fn embed_ids(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>, String> {
        let vocab = self.embed.rows();
        for (i, s) in seqs.iter().enumerate() {
            if s.is_empty() {
                return Err(format!("input {i}: no tokens"));
            }
            if s.len() > self.max_tokens {
                return Err(format!(
                    "input {i}: {} tokens > the model's {} token context",
                    s.len(),
                    self.max_tokens
                ));
            }
            if let Some(&bad) = s.iter().find(|&&t| t as usize >= vocab) {
                return Err(format!("input {i}: token id {bad} outside the vocabulary"));
            }
        }
        let _g = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let mut out: Vec<Option<Vec<f32>>> = vec![None; seqs.len()];
        // Pack in input order up to PACK_TOKENS per forward.
        let mut start = 0usize;
        while start < seqs.len() {
            let mut end = start;
            let mut toks = 0usize;
            while end < seqs.len() && (end == start || toks + seqs[end].len() <= PACK_TOKENS) {
                toks += seqs[end].len();
                end += 1;
            }
            let x = self.embed_rows(&seqs[start..end]);
            let lens: Vec<usize> = seqs[start..end].iter().map(|s| s.len()).collect();
            let pooled = self.forward_pooled_locked(x, &lens);
            for (k, v) in pooled.into_iter().enumerate() {
                out[start + k] = Some(v);
            }
            start = end;
        }
        Ok(out.into_iter().map(|v| v.unwrap()).collect())
    }

    /// The scaled token embeddings of packed sequences, `[sum T, hidden]`
    /// (`E[id]·sqrt(hidden)`): the text half of a merged multimodal input.
    pub fn embed_rows(&self, seqs: &[Vec<u32>]) -> Vec<f32> {
        let d = self.hidden;
        let n: usize = seqs.iter().map(|s| s.len()).sum();
        let mut x = vec![0f32; n * d];
        for (row, &id) in x.chunks_exact_mut(d).zip(seqs.iter().flatten()) {
            self.embed.row(id as usize, row);
            for v in row.iter_mut() {
                *v *= self.embed_scale;
            }
        }
        x
    }

    /// The model width (512).
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The forward over merged input embeddings `x` (`[sum lens, hidden]`,
    /// sequences back to back — token rows from [`embed_rows`](Self::embed_rows),
    /// media soft tokens unscaled in their placeholder rows): one
    /// unit-length vector per sequence.
    pub fn embed_merged(&self, x: Vec<f32>, lens: &[usize]) -> Vec<Vec<f32>> {
        let _g = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        self.forward_pooled_locked(x, lens)
    }

    fn forward_pooled_locked(&self, x: Vec<f32>, lens: &[usize]) -> Vec<Vec<f32>> {
        let t_fwd = std::time::Instant::now();
        let pool = self.pool.as_deref();
        let d = self.hidden;
        let n: usize = lens.iter().sum();
        debug_assert_eq!(x.len(), n * d);
        let maxlen = lens.iter().copied().max().unwrap_or(0);
        // segment starts and per-row positions
        let mut segs = Vec::with_capacity(lens.len());
        let mut pos = Vec::with_capacity(n);
        let mut off = 0usize;
        for &l in lens {
            segs.push((off, l));
            pos.extend(0..l);
            off += l;
        }
        let mut rots: Vec<((usize, u32), Rot)> = Vec::new();
        for l in &self.layers {
            let key = (l.head_dim, l.theta.to_bits());
            if !rots.iter().any(|(k, _)| *k == key) {
                rots.push((key, Rot::build(maxlen, l.head_dim, l.theta)));
            }
        }
        let ple_scale = (d as f32).powf(-0.5);

        let mut h = x.clone();
        for l in &self.layers {
            let rot = &rots
                .iter()
                .find(|(k, _)| *k == (l.head_dim, l.theta.to_bits()))
                .unwrap()
                .1;
            // ── attention
            let a = rms_rows(&h, Some(&l.in_norm), d, pool);
            let attn = self.attention(l, &a, n, &segs, &pos, rot, pool);
            add_normed(&mut h, &attn, &l.post_attn_norm, d, pool);
            // ── gated FFN
            let m = rms_rows(&h, Some(&l.pre_ff_norm), d, pool);
            let mut g = l.gate.apply(&m, n, pool);
            let u = l.up.apply(&m, n, pool);
            gelu_mul(&mut g, &u, pool);
            let f = l.down.apply(&g, n, pool);
            add_normed(&mut h, &f, &l.post_ff_norm, d, pool);
            // ── per-layer input: this layer's slice of the PLE projection
            let mut pin = l.ple_in.apply(&x, n, pool);
            for v in pin.iter_mut() {
                *v *= ple_scale;
            }
            let pin = rms_rows(&pin, Some(&self.ple_norm), self.n_ple, pool);
            let mut z = l.ple_gate.apply(&h, n, pool);
            gelu_mul(&mut z, &pin, pool);
            let z = l.ple_proj.apply(&z, n, pool);
            add_normed(&mut h, &z, &l.ple_norm, d, pool);
            for v in h.iter_mut() {
                *v *= l.scalar;
            }
        }
        let hn = rms_rows(&h, Some(&self.norm), d, pool);
        let out = segs
            .iter()
            .map(|&(s, l)| {
                let mean = mean_pool(&hn[s * d..(s + l) * d], l, d);
                let mut v = self.head.apply(&mean, 1, pool);
                l2_normalize(&mut v);
                v
            })
            .collect();
        let (lin, attn) = (prof::take(&prof::LIN), prof::take(&prof::ATTN));
        if prof::on() {
            let total = t_fwd.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "egemma2 forward: {n} tokens / {} seqs in {total:.1} ms — projections {lin:.1}, attention core {attn:.1}, rest {:.1}",
                lens.len(),
                total - lin - attn
            );
        }
        out
    }

    /// Self-attention of one layer over packed sequences; `a` is the
    /// normalized input `[n, hidden]`. Returns `o_proj(attn)`, `[n, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        l: &Layer,
        a: &[f32],
        n: usize,
        segs: &[(usize, usize)],
        pos: &[usize],
        rot: &Rot,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let hd = l.head_dim;
        let (nq, nkv) = (l.q_heads, l.kv_heads);
        let (qw, kw) = (nq * hd, nkv * hd);
        let mut q = l.q.apply(a, n, pool);
        let mut k = l.k.apply(a, n, pool);
        let mut v = l.v.apply(a, n, pool);
        // q/k: RMS-norm with weight per head, then RoPE; v: RMS-norm, no weight
        {
            let (qp, kp, vp) = (
                Shared(q.as_mut_ptr()),
                Shared(k.as_mut_ptr()),
                Shared(v.as_mut_ptr()),
            );
            rows(pool, n, &|s, e| {
                let mut tmp = vec![0f32; hd];
                for t in s..e {
                    let qr = unsafe { qp.at(t * qw, qw) };
                    for hh in qr.chunks_exact_mut(hd) {
                        rms_into(hh, Some(&l.q_norm), &mut tmp);
                        hh.copy_from_slice(&tmp);
                        rot.apply(pos[t], hh);
                    }
                    let kr = unsafe { kp.at(t * kw, kw) };
                    for hh in kr.chunks_exact_mut(hd) {
                        rms_into(hh, Some(&l.k_norm), &mut tmp);
                        hh.copy_from_slice(&tmp);
                        rot.apply(pos[t], hh);
                    }
                    let vr = unsafe { vp.at(t * kw, kw) };
                    for hh in vr.chunks_exact_mut(hd) {
                        rms_into(hh, None, &mut tmp);
                        hh.copy_from_slice(&tmp);
                    }
                }
            });
        }
        let t_core = std::time::Instant::now();
        let mut out = vec![0f32; n * qw];
        let group = nq / nkv;
        // Short sequences: one (sequence, head) pair per work item, all of
        // them across the pool, each computed serially — a batch of queries
        // is thousands of tiny attentions, and a GEMM call (plus a pool
        // dispatch) per pair cost more than the arithmetic.
        let items: Vec<(usize, usize)> = segs
            .iter()
            .filter(|&&(_, len)| len <= SHORT_SEQ)
            .flat_map(|&(s0, len)| (0..nq).map(move |h| (s0 * nq + h, len)))
            .collect();
        if !items.is_empty() {
            let op = Shared(out.as_mut_ptr());
            let window = l.window;
            let (q, k, v) = (&q, &k, &v);
            rows(pool, items.len(), &|s, e| {
                let mut p = vec![0f32; SHORT_SEQ];
                for &(code, len) in &items[s..e] {
                    let (s0, qh) = (code / nq, code % nq);
                    let kvh = qh / group;
                    for i in 0..len {
                        let (lo, hi) = match window {
                            Some(w) => (i.saturating_sub(w), (i + w + 1).min(len)),
                            None => (0, len),
                        };
                        let qi = &q[(s0 + i) * qw + qh * hd..][..hd];
                        let row = &mut p[..hi - lo];
                        for (j, r) in (lo..hi).zip(row.iter_mut()) {
                            *r =
                                crate::attention::dot_f32(qi, &k[(s0 + j) * kw + kvh * hd..][..hd]);
                        }
                        softmax(row);
                        // SAFETY: (row, head) slices are disjoint across items
                        let o = unsafe { op.at((s0 + i) * qw + qh * hd, hd) };
                        o.iter_mut().for_each(|x| *x = 0.0);
                        for (j, &w) in (lo..hi).zip(row.iter()) {
                            crate::attention::axpy_f32(o, &v[(s0 + j) * kw + kvh * hd..][..hd], w);
                        }
                    }
                }
            });
        }
        for &(s0, len) in segs.iter().filter(|&&(_, len)| len > SHORT_SEQ) {
            for kvh in 0..nkv {
                // this key head's keys [len, hd] and values transposed [hd, len]
                let mut kh = vec![0f32; len * hd];
                let mut vt = vec![0f32; hd * len];
                for t in 0..len {
                    kh[t * hd..(t + 1) * hd].copy_from_slice(&k[(s0 + t) * kw + kvh * hd..][..hd]);
                    let vr = &v[(s0 + t) * kw + kvh * hd..][..hd];
                    for (c, &val) in vr.iter().enumerate() {
                        vt[c * len + t] = val;
                    }
                }
                for qh in kvh * group..(kvh + 1) * group {
                    let qblock = if l.window.is_some() {
                        QBLOCK
                    } else {
                        QBLOCK_FULL
                    };
                    let mut qb = vec![0f32; qblock.min(len) * hd];
                    let mut i0 = 0usize;
                    while i0 < len {
                        let i1 = (i0 + qblock).min(len);
                        let nb = i1 - i0;
                        let (k0, k1) = match l.window {
                            Some(w) => (i0.saturating_sub(w), (i1 - 1 + w + 1).min(len)),
                            None => (0, len),
                        };
                        let kr = k1 - k0;
                        for i in 0..nb {
                            qb[i * hd..(i + 1) * hd]
                                .copy_from_slice(&q[(s0 + i0 + i) * qw + qh * hd..][..hd]);
                        }
                        let mut sc = vec![0f32; nb * kr];
                        crate::fcd_ops::gemm_nt_host(
                            &qb[..nb * hd],
                            &kh[k0 * hd..k1 * hd],
                            &mut sc,
                            nb,
                            hd,
                            kr,
                            None,
                        );
                        // softmax over each row's allowed span; zero outside
                        let sp = Shared(sc.as_mut_ptr());
                        let window = l.window;
                        rows(pool, nb, &|s, e| {
                            for i in s..e {
                                let row = unsafe { sp.at(i * kr, kr) };
                                let qi = i0 + i;
                                let (lo, hi) = match window {
                                    Some(w) => (qi.saturating_sub(w), (qi + w + 1).min(len)),
                                    None => (0, len),
                                };
                                let (lo, hi) = (lo - k0, hi - k0);
                                softmax(&mut row[lo..hi]);
                                row[..lo].iter_mut().for_each(|x| *x = 0.0);
                                row[hi..].iter_mut().for_each(|x| *x = 0.0);
                            }
                        });
                        // P · V over V's columns k0..k1 (rows of `vt`, stride len)
                        let mut ob = vec![0f32; nb * hd];
                        gemm_nt_strided(&sc, &vt[k0..], len, &mut ob, nb, kr, hd);
                        for i in 0..nb {
                            out[(s0 + i0 + i) * qw + qh * hd..][..hd]
                                .copy_from_slice(&ob[i * hd..(i + 1) * hd]);
                        }
                        i0 = i1;
                    }
                }
            }
        }
        prof::add(&prof::ATTN, t_core);
        l.o.apply(&out, n, pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matryoshka_truncates_and_renormalizes() {
        let mut v: Vec<f32> = (0..768)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect();
        l2_normalize(&mut v);
        for d in MATRYOSHKA_DIMS {
            let t = matryoshka(&v, d).unwrap();
            assert_eq!(t.len(), d);
            let n: f64 = t.iter().map(|&x| (x as f64) * (x as f64)).sum();
            assert!((n - 1.0).abs() < 1e-6, "dim {d}: |v|² = {n}");
            // same direction as the plain prefix
            assert!(cosine(&t, &v[..d]) > 0.999_999);
        }
        assert_eq!(matryoshka(&v, 768).unwrap(), v);
        assert!(matryoshka(&v, 300).is_err());
        assert!(matryoshka(&v[..256], 512).is_err());
    }

    #[test]
    fn mean_pool_averages_rows() {
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(mean_pool(&x, 3, 2), vec![3.0, 4.0]);
        assert_eq!(mean_pool(&x, 1, 2), vec![1.0, 2.0]);
    }

    #[test]
    fn l2_normalize_unit_and_zero() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.6, 0.8]);
        let mut z = vec![0.0f32; 4];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0; 4]);
    }

    #[test]
    fn prompts_follow_sentence_transformers() {
        let p = Prompts::default();
        assert_eq!(
            p.format(
                "What causes the northern lights?",
                Some("SearchQuery"),
                None,
                None
            )
            .unwrap(),
            "task: search result | query: What causes the northern lights?"
        );
        assert_eq!(
            p.format("body", Some("Document"), None, None).unwrap(),
            "title: none | text: body"
        );
        assert_eq!(
            p.format("body", Some("Document"), Some("Aurora borealis"), None)
                .unwrap(),
            "title: Aurora borealis | text: body"
        );
        // a title alone means the Document prompt
        assert_eq!(
            p.format("body", None, Some("T"), None).unwrap(),
            "title: T | text: body"
        );
        assert_eq!(
            p.format("body", Some("document"), Some(""), None).unwrap(),
            "title: none | text: body"
        );
        // case-insensitive fallback, aliases
        assert_eq!(
            p.format("x", Some("sts"), None, None).unwrap(),
            "task: sentence similarity | query: x"
        );
        assert_eq!(
            p.format("x", Some("CodeRetrieval"), None, None).unwrap(),
            "task: code retrieval | query: x"
        );
        // no prompt: the text as is
        assert_eq!(p.format("plain", None, None, None).unwrap(), "plain");
        // raw prefix
        assert_eq!(
            p.format("x", Some("SearchQuery"), None, Some("custom: "))
                .unwrap(),
            "custom: x"
        );
        assert!(p.format("x", Some("NoSuchTask"), None, None).is_err());
        assert!(p.format("x", Some("SearchQuery"), Some("T"), None).is_err());
    }

    #[test]
    fn prompts_read_the_st_config() {
        let cfg =
            serde_json::json!({"prompts": {"query": "q: ", "document": "title: none | text: "}});
        let p = Prompts::from_st_config(&cfg).unwrap();
        assert_eq!(p.get("query"), Some("q: "));
        assert_eq!(
            p.format("b", Some("document"), Some("t"), None).unwrap(),
            "title: t | text: b"
        );
        assert!(Prompts::from_st_config(&serde_json::json!({})).is_none());
    }

    #[test]
    fn wrap_ids_adds_specials_and_caps() {
        assert_eq!(wrap_ids(&[5, 6], 2, 1, 8192), vec![2, 5, 6, 1]);
        let long: Vec<u32> = (10..30).collect();
        let w = wrap_ids(&long, 2, 1, 8);
        assert_eq!(w.len(), 8);
        assert_eq!(w[0], 2);
        assert_eq!(*w.last().unwrap(), 1);
        assert_eq!(&w[1..7], &long[..6]);
    }

    #[test]
    fn rope_rotates_halves() {
        let r = Rot::build(4, 8, 10000.0);
        let mut x = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        r.apply(0, &mut x);
        assert_eq!(x[0], 1.0); // position 0 is the identity
        r.apply(1, &mut x);
        // first frequency is 1 rad per position: (cos 1, ..., sin 1 at +half)
        assert!((x[0] - 1f32.cos()).abs() < 1e-6);
        assert!((x[4] - 1f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn softmax_is_a_distribution() {
        let mut r = vec![1.0f32, 2.0, 3.0, -50.0];
        softmax(&mut r);
        let s: f32 = r.iter().sum();
        assert!((s - 1.0).abs() < 1e-6);
        assert!(r[2] > r[1] && r[1] > r[0]);
    }
}
