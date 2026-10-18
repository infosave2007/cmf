//! DeepSeek-V4.1 runtime.
//!
//! V4.1 is close enough to V4 to share the small numerical kernels in
//! [`crate::dsv4`], but its cache contract is different: compressors and
//! index keys are published by source layers and consumed by later layers,
//! Engram tables are raw E4M3/E8M0 rows, and the mHC fold for attention uses
//! the previous sub-block's `pre` coefficients.  Keeping the implementation
//! here avoids making the V4 executor guess those rules from tensor names.
//!
//! The hot weights remain `QTensor` handles.  Experts therefore stay in the
//! CMF mmap and are touched only for the routes selected for the current
//! token.  The Engram tables deliberately do not go through `QTensor`: U8 is
//! the on-disk byte representation, and a full dequantisation would exceed
//! the host memory budget by hundreds of gigabytes.

use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::{CmfModel, TensorDtype};
use std::sync::Arc;

const DEAD: i64 = -1;
const FP8_NAN: u8 = 0x7f;
const E8M0_NAN: u8 = 0xff;

/// `CMF_DSV41_PROF=1` enables one compact, cumulative timing report for the
/// V4.1 path.  The timers are deliberately optional and token scoped: there
/// is no per-layer logging and the disabled path only tests the cached flag.
pub(crate) mod prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::Instant;

    pub static TOKENS: AtomicU64 = AtomicU64::new(0);
    pub static LAYER_VISITS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_NS: AtomicU64 = AtomicU64::new(0);
    pub static SPARSE_NS: AtomicU64 = AtomicU64::new(0);
    pub static ATTN_OTHER_NS: AtomicU64 = AtomicU64::new(0);
    pub static MOE_NS: AtomicU64 = AtomicU64::new(0);
    pub static ENGRAM_NS: AtomicU64 = AtomicU64::new(0);
    pub static HEAD_NS: AtomicU64 = AtomicU64::new(0);
    pub static TOTAL_NS: AtomicU64 = AtomicU64::new(0);
    pub static MOE_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_MOE_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static CPU_MOE_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static COLD_EXPERTS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_ATTN_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_ATTN_FALLBACKS: AtomicU64 = AtomicU64::new(0);
    static REPORTED: AtomicBool = AtomicBool::new(false);

    #[inline]
    pub fn on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_DSV41_PROF").is_ok_and(|v| v != "0"))
    }

    #[inline]
    pub fn start() -> Option<Instant> {
        on().then(Instant::now)
    }

    #[inline]
    pub fn add(dst: &AtomicU64, started: Option<Instant>) {
        if let Some(started) = started {
            dst.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn note_token(layers: usize) {
        if on() {
            TOKENS.fetch_add(1, Ordering::Relaxed);
            LAYER_VISITS.fetch_add(layers as u64, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn note_moe() {
        if on() {
            MOE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn note_gpu_moe() {
        if on() {
            GPU_MOE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn note_cpu_moe() {
        if on() {
            CPU_MOE_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn note_cold(count: usize) {
        if on() {
            COLD_EXPERTS.fetch_add(count as u64, Ordering::Relaxed);
        }
    }

    pub fn note_gpu_attn() {
        if on() {
            GPU_ATTN_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn note_gpu_attn_fallback() {
        if on() {
            GPU_ATTN_FALLBACKS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Print once at the normal CLI report point.  The dense number is the
    /// residual of the whole token after the explicitly timed attention,
    /// MoE, Engram, and final head stages, so it includes embeddings,
    /// hyper-connections, norms, routing, and other host work without
    /// double-counting nested operations.
    pub fn report() {
        if !on() || REPORTED.swap(true, Ordering::Relaxed) {
            return;
        }
        let tokens = TOKENS.load(Ordering::Relaxed).max(1) as f64;
        let layers = LAYER_VISITS.load(Ordering::Relaxed);
        let ns = |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64;
        let attn = ns(&ATTN_NS);
        let sparse = ns(&SPARSE_NS);
        let attn_other = ns(&ATTN_OTHER_NS);
        let moe = ns(&MOE_NS);
        let engram = ns(&ENGRAM_NS);
        let head = ns(&HEAD_NS);
        let total = ns(&TOTAL_NS);
        let dense = (total - attn - moe - engram - head).max(0.0);
        eprintln!(
            "[dsv41-profile] tokens={} layer_visits={} per_token_ms: attention={:.2} sparse_attend={:.2} attention_other={:.2} moe={:.2} engram={:.2} head={:.2} dense_other={:.2} total={:.2}",
            tokens as u64,
            layers,
            attn / 1e6 / tokens,
            sparse / 1e6 / tokens,
            attn_other / 1e6 / tokens,
            moe / 1e6 / tokens,
            engram / 1e6 / tokens,
            head / 1e6 / tokens,
            dense / 1e6 / tokens,
            total / 1e6 / tokens,
        );
        eprintln!(
            "[dsv41-profile] moe_calls={} gpu_moe_calls={} cpu_moe_calls={} cold_experts={} gpu_attn_calls={} gpu_attn_fallbacks={}",
            MOE_CALLS.load(Ordering::Relaxed),
            GPU_MOE_CALLS.load(Ordering::Relaxed),
            CPU_MOE_CALLS.load(Ordering::Relaxed),
            COLD_EXPERTS.load(Ordering::Relaxed),
            GPU_ATTN_CALLS.load(Ordering::Relaxed),
            GPU_ATTN_FALLBACKS.load(Ordering::Relaxed),
        );
        #[cfg(feature = "gpu")]
        {
            let fills = crate::gpu_wgpu::DSV4_FILLS.load(Ordering::Relaxed);
            let fill_bytes = crate::gpu_wgpu::DSV4_FILL_BYTES.load(Ordering::Relaxed);
            let fill_ns = crate::gpu_wgpu::DSV4_FILL_NS.load(Ordering::Relaxed);
            let submits = crate::gpu_wgpu::SUBMITS.load(Ordering::Relaxed);
            let passes = crate::gpu_wgpu::PASSES.load(Ordering::Relaxed);
            let upload_bytes = crate::gpu_wgpu::UPLOAD_BYTES.load(Ordering::Relaxed);
            let upload_ns = crate::gpu_wgpu::UPLOAD_NS.load(Ordering::Relaxed);
            eprintln!(
                "[dsv41-profile] gpu_counters fills={} fill_bytes={} submits={} passes={} upload_bytes={} upload_ms={:.2} refill_ms={:.2} resident_bytes={} vram_budget={}",
                fills,
                fill_bytes,
                submits,
                passes,
                upload_bytes,
                upload_ns as f64 / 1e6,
                fill_ns as f64 / 1e6,
                crate::gpu_wgpu::resident_bytes(),
                crate::gpu_wgpu::device_vram_budget(),
            );
            eprintln!(
                "[dsv41-profile] gpu_moe_host_encode_ms={:.2} gpu_moe_wait_ms={:.2} chain_encode_ms={:.2} chain_wait_ms={:.2} chain_layers={} chain_runs={}",
                crate::gpu_wgpu::MOE_ENC_NS.load(Ordering::Relaxed) as f64 / 1e6,
                crate::gpu_wgpu::MOE_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6,
                crate::gpu_wgpu::CHAIN_ENC_NS.load(Ordering::Relaxed) as f64 / 1e6,
                crate::gpu_wgpu::CHAIN_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6,
                crate::gpu_wgpu::CHAIN_LAYERS.load(Ordering::Relaxed),
                crate::gpu_wgpu::CHAIN_RUNS.load(Ordering::Relaxed),
            );
            eprintln!(
                "[dsv41-profile] gpu_attn_frame_encode_ms={:.2} gpu_attn_frame_wait_ms={:.2}",
                crate::gpu_wgpu::ATT_ENC_NS.load(Ordering::Relaxed) as f64 / 1e6,
                crate::gpu_wgpu::ATT_WAIT_NS.load(Ordering::Relaxed) as f64 / 1e6,
            );
        }
    }
}

/// Round an f32 through the BF16 format used by the reference model's
/// activation tensors.  The engine keeps working buffers in f32, so every
/// boundary where the reference writes a BF16 tensor has to make this
/// rounding explicit.  The add-and-mask form is round-to-nearest-even and
/// also handles negative values without a float conversion in the hot loop.
#[inline]
fn bf16_roundtrip(value: f32) -> f32 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round) & 0xffff_0000)
}

#[inline]
fn bf16_inplace(values: &mut [f32]) {
    for value in values {
        *value = bf16_roundtrip(*value);
    }
}

#[inline]
fn trace_stats(stage: &str, position: usize, layer: Option<usize>, v: &[f32]) {
    if std::env::var_os("CMF_DSV41_TRACE").is_none() {
        return;
    }
    let mean = v.iter().copied().sum::<f32>() / v.len().max(1) as f32;
    let sum = v.iter().copied().sum::<f32>();
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let l2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let full = std::env::var_os("CMF_DSV41_TRACE_FULL").is_some()
        && std::env::var("CMF_DSV41_TRACE_POS")
            .ok()
            .and_then(|p| p.parse::<usize>().ok())
            .is_some_and(|p| p == position);
    let head: Vec<f32> = if full {
        v.to_vec()
    } else {
        v.iter().take(8).copied().collect()
    };
    match layer {
        Some(layer) => eprintln!(
            "dsv41_trace stage={stage} position={position} layer={layer} mean={mean:.9e} sum={sum:.9e} min={min:.9e} max={max:.9e} l2={l2:.9e} head={head:?}"
        ),
        None => eprintln!(
            "dsv41_trace stage={stage} position={position} mean={mean:.9e} sum={sum:.9e} min={min:.9e} max={max:.9e} l2={l2:.9e} head={head:?}"
        ),
    }
}

#[inline]
fn trace_indices(position: usize, layer: usize, values: &[usize]) {
    if std::env::var_os("CMF_DSV41_TRACE").is_none() {
        return;
    }
    eprintln!("dsv41_trace stage=idxs position={position} layer={layer} values={values:?}");
}

#[inline]
fn trace_index_scores(
    position: usize,
    layer: usize,
    scores: &[f32],
    picked: &[usize],
    candidate: Option<&[bool]>,
    window_len: usize,
    compressed_len: usize,
    pending_len: usize,
) {
    if std::env::var_os("CMF_DSV41_TRACE_INDEX").is_none() {
        return;
    }
    if let Some(filter) = std::env::var("CMF_DSV41_TRACE_POS")
        .ok()
        .and_then(|p| p.parse::<usize>().ok())
    {
        if filter != position {
            return;
        }
    }
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then_with(|| a.cmp(&b)));
    let finite_ties: Vec<(usize, usize)> = (0..scores.len())
        .flat_map(|a| ((a + 1)..scores.len()).map(move |b| (a, b)))
        .filter(|&(a, b)| scores[a].is_finite() && scores[a].to_bits() == scores[b].to_bits())
        .collect();
    let top: Vec<(usize, f32)> = order.iter().take(12).map(|&i| (i, scores[i])).collect();
    let candidate_ids: Vec<usize> = candidate
        .map(|mask| {
            mask.iter()
                .enumerate()
                .filter_map(|(i, &keep)| keep.then_some(i))
                .collect()
        })
        .unwrap_or_default();
    eprintln!(
        "dsv41_trace stage=index_scores position={position} layer={layer} window_len={window_len} compressed_len={compressed_len} pending_len={pending_len} picked={picked:?} top={top:?} ties={finite_ties:?} candidate_ids={candidate_ids:?}"
    );
}

/// Emit one compact routing snapshot when diagnosing a discrete top-k
/// divergence against the reference.  This is deliberately opt-in and
/// position-filtered so it cannot add work to normal generation.  The full
/// shifted scores and the unbiased scores are included: ties in the biased
/// choice can therefore be distinguished from a genuinely different score
/// ordering.
#[inline]
fn trace_moe(
    position: usize,
    layer: usize,
    image: bool,
    logits: &[f32],
    scores: &[f32],
    bias: &[f32],
    picks: &[usize],
) {
    if std::env::var_os("CMF_DSV41_TRACE_MOE").is_none() {
        return;
    }
    if let Some(filter) = std::env::var("CMF_DSV41_TRACE_POS")
        .ok()
        .and_then(|p| p.parse::<usize>().ok())
    {
        if filter != position {
            return;
        }
    }
    let shifted: Vec<f32> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| s + bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let mut order: Vec<usize> = (0..shifted.len()).collect();
    order.sort_by(|&a, &b| {
        shifted[b]
            .partial_cmp(&shifted[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let ties: Vec<(usize, usize)> = (0..shifted.len())
        .flat_map(|a| ((a + 1)..shifted.len()).map(move |b| (a, b)))
        .filter(|&(a, b)| shifted[a].to_bits() == shifted[b].to_bits())
        .collect();
    eprintln!(
        "dsv41_trace stage=moe position={position} layer={layer} image={image} logits={logits:?} scores={scores:?} bias={bias:?} shifted={shifted:?} order={order:?} picks={picks:?} ties={ties:?}"
    );
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decode an OCP E4M3FN value.  The Engram payload stores these bytes as
/// U8, rather than as a generic CMF quantisation codec.
#[inline]
pub fn fp8_e4m3(byte: u8) -> f32 {
    if (byte & 0x7f) == FP8_NAN {
        return f32::NAN;
    }
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = (byte >> 3) & 0x0f;
    let mant = byte & 0x07;
    if exp == 0 {
        sign * (mant as f32) * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2.0f32.powi(exp as i32 - 7)
    }
}

/// Decode an E8M0 scale.  E8M0 has no sign or mantissa: the byte is a biased
/// exponent, with 127 representing one.  Zero is a finite subnormal scale
/// in the upstream implementation and is retained as 2^-127.
#[inline]
pub fn e8m0_scale(byte: u8) -> f32 {
    if byte == E8M0_NAN {
        f32::NAN
    } else {
        2.0f32.powi(byte as i32 - 127)
    }
}

/// Round a finite value to the representable E4M3FN value and decode it
/// again.  Activation quantisation in the reference is fused quantise plus
/// dequantise, so keeping this small scalar path here gives the CPU fallback
/// the same values that the GPU kernel writes back in-place.
#[inline]
fn round_ties_even(x: f32) -> i32 {
    let lo = x.floor() as i32;
    let frac = x - lo as f32;
    if frac > 0.5 || (frac == 0.5 && (lo & 1) != 0) {
        lo + 1
    } else {
        lo
    }
}

fn e4m3_round(x: f32) -> f32 {
    if !x.is_finite() {
        return if x.is_sign_negative() { -448.0 } else { 448.0 };
    }
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let ax = x.abs().min(448.0);
    if ax == 0.0 {
        return 0.0;
    }
    let byte = if ax < 2.0f32.powi(-6) {
        // Subnormals use a 2^-9 quantum.  A rounded mantissa of eight
        // carries into the smallest normal (0x08); clamping it to seven
        // loses that carry and creates a discontinuity at 2^-6.
        let mant = round_ties_even(ax * 2.0f32.powi(9));
        if mant >= 8 {
            sign | 0x08
        } else {
            sign | mant.clamp(0, 7) as u8
        }
    } else {
        // Read the unbiased binary exponent instead of relying on log2 at
        // exact powers of two.  The source cast is round-to-nearest-even;
        // f32::round is ties-away and changes half-way E4M3 codes.
        let exp = ((ax.to_bits() >> 23) & 0xff) as i32 - 127;
        let mut mant = round_ties_even((ax / 2.0f32.powi(exp) - 1.0) * 8.0);
        let mut exp = exp;
        if mant >= 8 {
            exp += 1;
            mant = 0;
        }
        let exp_field = exp + 7;
        if exp_field > 15 {
            // 0x7f/0xff is NaN in E4M3FN; 0x7e/0xfe is the largest finite
            // code and decodes to 448.  Values entering this helper are
            // clamped above, but retain saturation here for callers that
            // use the scalar rounder directly.
            sign | (15 << 3) | 6
        } else {
            // E4M3FN reserves mantissa seven at exponent 15 for NaN.  The
            // input clamp means a normal value reaches at most mantissa six;
            // keep the clamp as a defensive guard for direct callers.
            let mant = if exp_field == 15 { mant.min(6) } else { mant };
            sign | ((exp_field.clamp(1, 15) as u8) << 3) | (mant.clamp(0, 7) as u8)
        }
    };
    fp8_e4m3(byte)
}

fn round_e8m0_scale(raw: f32) -> f32 {
    if !raw.is_finite() || raw <= 0.0 {
        return 2.0f32.powi(-127);
    }
    // The upstream kernel uses fast_log2_ceil(amax / max), not nearest
    // rounding.  Ceil keeps every value representable after the scale is
    // quantised and is part of the MXFP contract.
    let exponent = raw.log2().ceil().clamp(-127.0, 127.0);
    2.0f32.powf(exponent)
}

/// A row-addressable E4M3/E8M0 tensor pair.  Only the rows selected by the
/// hash state are decoded, so the resident memory is O(hash columns), not
/// O(table rows).
#[derive(Clone)]
pub struct RawFp8Rows {
    model: Arc<CmfModel>,
    weight_idx: usize,
    scale_idx: usize,
    pub rows: usize,
    pub cols: usize,
}

impl RawFp8Rows {
    pub fn from_model(model: &Arc<CmfModel>, weight: &str, scale: &str) -> Result<Self, String> {
        let wi = model
            .tensor_index(weight)
            .ok_or_else(|| format!("Engram tensor '{weight}' not found"))?;
        let si = model
            .tensor_index(scale)
            .ok_or_else(|| format!("Engram tensor '{scale}' not found"))?;
        let we = &model.tensors[wi];
        let se = &model.tensors[si];
        if we.dtype != TensorDtype::U8 || se.dtype != TensorDtype::U8 {
            return Err(format!(
                "Engram rows must be U8, got {} and {}",
                we.dtype.name(),
                se.dtype.name()
            ));
        }
        if we.shape.len() != 2 || se.shape.len() != 2 {
            return Err("Engram rows must be two-dimensional".into());
        }
        let (rows, cols) = (we.shape[0], we.shape[1]);
        if cols == 0 || cols % 32 != 0 {
            return Err(format!(
                "Engram row width {cols} is not a positive multiple of 32"
            ));
        }
        if se.shape != vec![rows, cols / 32] {
            return Err(format!(
                "Engram scale shape {:?}, expected [{rows}, {}]",
                se.shape,
                cols / 32
            ));
        }
        if we.nbytes as usize != rows * cols || se.nbytes as usize != rows * cols / 32 {
            return Err(format!(
                "Engram U8 payload size mismatch for {weight}/{scale}"
            ));
        }
        Ok(Self {
            model: model.clone(),
            weight_idx: wi,
            scale_idx: si,
            rows,
            cols,
        })
    }

    /// Decode one row into `dst`.  Invalid NaN payloads are treated as zero
    /// at the edge of the runtime: an accidental poisoned row cannot turn a
    /// token's whole hidden state into NaNs, while the converter still keeps
    /// the raw bytes and the loader reports the shape/type contract.
    pub fn row_into(&self, row: usize, dst: &mut [f32]) {
        assert!(row < self.rows);
        assert_eq!(dst.len(), self.cols);
        let wb = self.model.entry_bytes(&self.model.tensors[self.weight_idx]);
        let sb = self.model.entry_bytes(&self.model.tensors[self.scale_idx]);
        for g in 0..self.cols / 32 {
            let scale = e8m0_scale(sb[row * (self.cols / 32) + g]);
            let scale = if scale.is_finite() { scale } else { 0.0 };
            let src = &wb[row * self.cols + g * 32..row * self.cols + (g + 1) * 32];
            for (out, &byte) in dst[g * 32..(g + 1) * 32].iter_mut().zip(src) {
                let v = fp8_e4m3(byte);
                *out = if v.is_finite() { v * scale } else { 0.0 };
            }
        }
    }
}

/// The exact source layout of the V4.1 n-gram tables.  The two configured
/// Engram layers have disjoint prime ranges, even though both tables use the
/// same nominal 16M vocabulary.
#[derive(Clone)]
pub struct EngramHash {
    pub layer_ids: Vec<usize>,
    pub max_ngram: usize,
    pub n_heads: usize,
    pub compressed_vocab: usize,
    pub pad_id: i64,
    pub primes: Vec<Vec<Vec<u64>>>,
    pub offsets: Vec<Vec<u64>>,
    pub token_map: Vec<u32>,
    pub multipliers: Vec<[u64; 4]>,
    history: Vec<i64>,
}

impl EngramHash {
    pub fn new(
        layer_ids: Vec<usize>,
        max_ngram: usize,
        n_heads: usize,
        table_vocab: usize,
        compressed_vocab: usize,
        pad_token: usize,
        token_map: Vec<u32>,
    ) -> Result<Self, String> {
        if max_ngram < 2 || n_heads == 0 || layer_ids.is_empty() {
            return Err("invalid Engram hash geometry".into());
        }
        let mapped_vocab = token_map
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        if mapped_vocab != compressed_vocab {
            return Err(format!(
                "Engram compressed vocabulary mismatch: tokenizer={mapped_vocab}, config={compressed_vocab}"
            ));
        }
        let mut seen = Vec::<u64>::new();
        let mut primes = Vec::with_capacity(layer_ids.len());
        let mut offsets = Vec::with_capacity(layer_ids.len());
        for _ in &layer_ids {
            let mut per_layer = Vec::with_capacity(max_ngram - 1);
            let mut per_offsets = Vec::with_capacity(max_ngram - 1);
            let mut offset = 0u64;
            for _ in 0..max_ngram - 1 {
                let mut per_n = Vec::with_capacity(n_heads);
                per_offsets.push(offset);
                let mut current = table_vocab.saturating_sub(1) as u64;
                for _ in 0..n_heads {
                    loop {
                        current = current.saturating_add(1);
                        if is_prime(current) && !seen.contains(&current) {
                            break;
                        }
                    }
                    seen.push(current);
                    per_n.push(current);
                    offset = offset.saturating_add(current);
                }
                per_layer.push(per_n);
            }
            primes.push(per_layer);
            offsets.push(per_offsets);
        }
        let pad_id = token_map.get(pad_token).copied().unwrap_or(2) as i64;
        let multipliers = layer_ids
            .iter()
            .map(|&id| hash_multipliers(id, max_ngram, compressed_vocab))
            .collect();
        Ok(Self {
            layer_ids,
            max_ngram,
            n_heads,
            compressed_vocab,
            pad_id,
            primes,
            offsets,
            token_map,
            multipliers,
            history: Vec::new(),
        })
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    /// Append one token and return one flattened hash column vector per
    /// configured layer. `participates=false` is used for image span tokens;
    /// dead tokens break every n-gram lookback just like the reference.
    pub fn push(&mut self, token: u32, participates: bool) -> Vec<Vec<usize>> {
        let mapped = self.token_map.get(token as usize).copied().unwrap_or(0) as i64;
        self.history.push(if participates { mapped } else { DEAD });
        let mut all = Vec::with_capacity(self.layer_ids.len());
        for (li, _) in self.layer_ids.iter().enumerate() {
            let mut cols = Vec::with_capacity((self.max_ngram - 1) * self.n_heads);
            for n in 1..self.max_ngram {
                let mut rolling = 0u64;
                let mut blocked = false;
                // `products[..., 0]` is the current token; each additional
                // lookback is XORed once.  The same rolling hash is then
                // placed in one disjoint prime range per head.
                for shift in 0..=n {
                    let (source, dead) = self.history_value(shift);
                    // The reference carries a blocked bit across the
                    // lookback loop.  Once a dead/image token or sequence
                    // boundary is encountered, every older term is the pad
                    // id too; replacing only the dead term would create a
                    // cross-image n-gram.
                    blocked |= dead;
                    let source = if blocked { self.pad_id } else { source };
                    rolling ^= (source as u64).wrapping_mul(self.multipliers[li][shift.min(3)]);
                }
                for h in 0..self.n_heads {
                    let p = self.primes[li][n - 1][h];
                    let off =
                        self.offsets[li][n - 1] + self.primes[li][n - 1][..h].iter().sum::<u64>();
                    cols.push((rolling % p + off) as usize);
                }
            }
            all.push(cols);
        }
        all
    }

    fn history_value(&self, shift: usize) -> (i64, bool) {
        if shift >= self.history.len() {
            (self.pad_id, true)
        } else {
            let v = self.history[self.history.len() - 1 - shift];
            if v == DEAD {
                (self.pad_id, true)
            } else {
                (v, false)
            }
        }
    }
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v % 2 == 0 {
        return v == 2;
    }
    let mut d = 3u64;
    while d <= v / d {
        if v % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Multipliers emitted by NumPy's PCG64 for the release's two Engram
/// layers.  These are constants in the upstream model contract.  For a
/// custom layer id, a SplitMix fallback keeps the runtime deterministic;
/// converted V4.1 files always use ids 1 and 14.
fn hash_multipliers(layer: usize, n: usize, vocab: usize) -> [u64; 4] {
    let known = match layer {
        1 => [
            76632096046245,
            4839876093313,
            35959672319349,
            73987337458391,
        ],
        14 => [
            67716810739261,
            51510806800915,
            30921347202721,
            82619226485591,
        ],
        _ => [0; 4],
    };
    if n <= 4 && known[0] != 0 && vocab == 99092 {
        return known;
    }
    let bound = ((i64::MAX as u128 / vocab.max(1) as u128) / 2) as u64;
    let mut x = 10007u64
        .wrapping_mul(layer as u64)
        .wrapping_add(0x9e3779b97f4a7c15);
    let mut out = [0u64; 4];
    for i in 0..n.min(4) {
        x = splitmix64(x);
        out[i] = (x % bound.max(1)) * 2 + 1;
    }
    out
}

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// V4.1 architecture geometry. Arrays are deliberately kept in this cfg,
/// because the source config is part of the converted model metadata and
/// there is no safe way to infer sharing from a missing tensor alone.
#[derive(Debug, Clone, PartialEq)]
pub struct Dsv41Cfg {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub n_routed_experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    /// Router temperature and whether selected scores are renormalized.
    /// V4.1 ships sqrt-softplus, temperature 1 and normalized top-k, but
    /// keeping these source fields here makes small fixtures and future
    /// checkpoints follow their preserved config instead of silently using
    /// release defaults.
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub window: usize,
    pub rope_theta: f32,
    pub compress_rope_theta: f32,
    pub rope_factor: f32,
    pub original_seq_len: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    /// Candidate pre-filtering source layer; `None` disables the two-level
    /// top-k (`candidate_source_layer_id < 0` in the source config).
    pub candidate_source: Option<usize>,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub compress_ratios: Vec<usize>,
    pub engram_layers: Vec<usize>,
    pub engram_vocab: usize,
    pub engram_embeddings: Vec<usize>,
    pub engram_max_ngram: usize,
    pub engram_heads: usize,
    pub engram_head_dim: usize,
    pub engram_compressed_vocab: usize,
    pub engram_pad_id: usize,
    pub vocab: usize,
}

impl Dsv41Cfg {
    pub fn ratio(&self, li: usize) -> usize {
        self.compress_ratios.get(li).copied().unwrap_or(0)
    }

    pub fn kv_source(&self, li: usize) -> Option<usize> {
        self.kv_sources.iter().copied().filter(|&s| s <= li).max()
    }

    pub fn index_source(&self, li: usize) -> Option<usize> {
        self.index_sources
            .iter()
            .copied()
            .filter(|&s| s <= li)
            .max()
    }

    /// Parse the preserved source `config.json` (the whole multimodal
    /// object, or its `text_config`) into the runtime geometry. `arch` only
    /// supplies fallbacks for keys a compact config omits. Returns the
    /// configuration and the backbone layer count.
    pub fn from_source(
        source: &serde_json::Value,
        arch: &cortiq_core::ModelArch,
    ) -> Result<(Self, usize), String> {
        let tc = source.get("text_config").unwrap_or(source);
        let usize_of = |key: &str, fallback: usize| {
            tc.get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(fallback)
        };
        let f32_of = |key: &str, fallback: f32| {
            tc.get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(fallback)
        };
        let usize_any = |keys: &[&str], fallback: usize| {
            keys.iter()
                .find_map(|key| tc.get(key).and_then(|v| v.as_u64()))
                .map(|v| v as usize)
                .unwrap_or(fallback)
        };
        let f32_any = |keys: &[&str], fallback: f32| {
            keys.iter()
                .find_map(|key| tc.get(key).and_then(|v| v.as_f64()))
                .map(|v| v as f32)
                .unwrap_or(fallback)
        };
        let bool_of = |key: &str, fallback: bool| {
            tc.get(key).and_then(|v| v.as_bool()).unwrap_or(fallback)
        };
        let array_of = |key: &str| -> Vec<usize> {
            tc.get(key)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect()
                })
                .unwrap_or_default()
        };
        let array_alias = |keys: &[&str]| -> Vec<usize> {
            keys.iter()
                .find_map(|key| {
                    let values = array_of(key);
                    (!values.is_empty()).then_some(values)
                })
                .unwrap_or_default()
        };
        let dim = usize_any(&["hidden_size", "dim"], arch.hidden_size);
        let n_layers = usize_any(&["num_hidden_layers", "n_layers"], arch.num_layers);
        let n_heads = usize_any(
            &["num_attention_heads", "n_heads"],
            arch.num_attention_heads,
        );
        let head_dim = usize_any(&["head_dim"], arch.head_dim);
        let rope_head_dim = usize_any(&["rope_head_dim", "qk_rope_head_dim"], 64.min(head_dim));
        let moe_inter = usize_any(
            &[
                "moe_intermediate_size",
                "moe_inter_dim",
                "intermediate_size",
            ],
            arch.intermediate_size,
        );
        let n_experts = usize_any(
            &["n_routed_experts"],
            arch.moe.as_ref().map(|m| m.num_experts).unwrap_or(384),
        );
        let top_k = usize_any(
            &["num_experts_per_tok", "n_activated_experts"],
            arch.moe.as_ref().map(|m| m.top_k).unwrap_or(6),
        );
        let mut ratios = array_of("compress_ratios");
        if ratios.len() >= n_layers {
            ratios.truncate(n_layers);
        } else {
            ratios = (0..n_layers)
                .map(|li| {
                    if (2..20).contains(&li) {
                        2
                    } else if (20..40).contains(&li) {
                        1
                    } else {
                        0
                    }
                })
                .collect();
        }
        let kv_sources = {
            let a = array_alias(&["kv_source_layers", "kv_source_layer_ids"]);
            if a.is_empty() { vec![2, 8, 14, 20] } else { a }
        };
        let index_sources = {
            let a = array_alias(&["index_source_layers", "index_source_layer_ids"]);
            if a.is_empty() {
                vec![2, 8, 14, 20, 24, 28, 32, 36]
            } else {
                a
            }
        };
        // The router: V4.1 scores experts with sqrt(softplus(logits)). The
        // reference also implements softmax and sigmoid scoring; this runtime
        // (CPU and the preweighted GPU bank) implements only sqrtsoftplus, so
        // any other value is refused rather than routed with the wrong rule.
        // An absent key is the release (and the reference default).
        if let Some(func) = tc.get("scoring_func").or_else(|| tc.get("score_func")) {
            if func.as_str() != Some("sqrtsoftplus") {
                return Err(format!(
                    "text_config.scoring_func = {func} is not supported; the DeepSeek-V4.1 \
                     runtime implements only \"sqrtsoftplus\" routing (sqrt(softplus(logits)))"
                ));
            }
        }
        // `candidate_source_layer < 0 turns it off` (reference ModelArgs). An
        // absent key keeps the release layer, like every other default here.
        let candidate_source = match ["candidate_source_layer", "candidate_source_layer_id"]
            .iter()
            .find_map(|key| tc.get(key))
        {
            None => Some(20),
            Some(v) => match v.as_i64() {
                Some(layer) if layer < 0 => None,
                Some(layer) => Some(layer as usize),
                None => {
                    return Err(format!(
                        "text_config.candidate_source_layer_id = {v} is not an integer"
                    ));
                }
            },
        };
        let engram_layers = array_of("engram_layer_ids");
        let engram_embeddings = array_of("engram_num_embeddings");
        let cfg = Dsv41Cfg {
            dim,
            n_heads,
            head_dim,
            rope_head_dim: rope_head_dim.min(head_dim) & !1,
            q_lora_rank: usize_of("q_lora_rank", 1280),
            o_lora_rank: usize_of("o_lora_rank", 1024),
            o_groups: usize_of("o_groups", 8),
            hc_mult: usize_of("hc_mult", 4),
            hc_sinkhorn_iters: usize_of("hc_sinkhorn_iters", 20),
            hc_eps: f32_of("hc_eps", 1e-6),
            norm_eps: f32_any(&["norm_eps", "rms_norm_eps"], arch.rms_norm_eps as f32),
            n_routed_experts: n_experts,
            top_k,
            moe_inter,
            gate_temp: f32_of("gate_temp", 1.0),
            norm_topk_prob: bool_of("norm_topk_prob", true),
            route_scale: f32_any(
                &["routed_scaling_factor", "route_scale"],
                arch.moe
                    .as_ref()
                    .and_then(|m| m.routed_scaling_factor)
                    .unwrap_or(1.5),
            ),
            swiglu_limit: f32_of("swiglu_limit", 10.0),
            window: usize_any(
                &["window_size", "sliding_window"],
                arch.sliding_window.unwrap_or(128),
            ),
            rope_theta: f32_any(&["rope_theta"], arch.rope_theta as f32),
            compress_rope_theta: f32_any(&["compress_rope_theta"], 160_000.0),
            rope_factor: f32_any(
                &["rope_factor"],
                tc.get("rope_scaling")
                    .and_then(|v| v.get("factor"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(16.0),
            ),
            original_seq_len: usize_any(
                &["original_seq_len", "original_max_position_embeddings"],
                tc.get("rope_scaling")
                    .and_then(|v| v.get("original_max_position_embeddings"))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(65_536),
            ),
            beta_fast: f32_any(
                &["beta_fast"],
                tc.get("rope_scaling")
                    .and_then(|v| v.get("beta_fast"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(32.0),
            ),
            beta_slow: f32_any(
                &["beta_slow"],
                tc.get("rope_scaling")
                    .and_then(|v| v.get("beta_slow"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(1.0),
            ),
            index_heads: usize_any(&["index_n_heads", "indexer_n_heads"], 32),
            index_head_dim: usize_any(&["index_head_dim", "indexer_head_dim"], 128),
            index_topk: usize_any(&["index_topk", "indexer_topk"], 512),
            candidate_source,
            candidate_topk_blocks: usize_any(&["candidate_topk_blocks"], 2048),
            candidate_block_size: usize_any(&["candidate_block_size"], 8),
            kv_sources,
            index_sources,
            compress_ratios: ratios,
            engram_layers,
            engram_vocab: usize_any(&["engram_vocab_size"], 16_000_000),
            engram_embeddings,
            engram_max_ngram: usize_any(&["engram_max_ngram_size"], 4),
            engram_heads: usize_any(&["engram_n_heads"], 8),
            engram_head_dim: usize_any(&["engram_head_dim"], 256),
            engram_compressed_vocab: usize_any(&["engram_compressed_vocab_size"], 99_092),
            engram_pad_id: usize_any(&["engram_pad_id", "engram_pad_token_id"], 2),
            vocab: usize_any(&["vocab_size"], arch.vocab_size),
        };
        Ok((cfg, n_layers))
    }
}

pub struct Dsv41Compressor {
    pub wkv: QTensor,
    pub wgate: Option<QTensor>,
    pub norm: Vec<f32>,
    pub ratio: usize,
}

pub struct Dsv41Indexer {
    pub wq_b: QTensor,
    pub weights_proj: QTensor,
    pub wk: Option<QTensor>,
    pub k_norm: Option<Vec<f32>>,
}

pub struct Dsv41Expert {
    pub w1: QTensor,
    pub w2: QTensor,
    pub w3: QTensor,
}

pub struct Dsv41Engram {
    pub embed: RawFp8Rows,
    pub wkv: QTensor,
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
}

pub struct Dsv41Layer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq_a: QTensor,
    pub q_norm: Vec<f32>,
    pub wq_b: QTensor,
    pub wkv: QTensor,
    pub kv_norm: Vec<f32>,
    pub wo_a: QTensor,
    pub wo_b: QTensor,
    pub attn_sink: Vec<f32>,
    pub compressor: Option<Dsv41Compressor>,
    pub indexer: Option<Dsv41Indexer>,
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: [f32; 3],
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: [f32; 3],
    pub gate: QTensor,
    pub gate_bias: Vec<f32>,
    pub gate_bias_vl: Option<Vec<f32>>,
    pub experts: Vec<Dsv41Expert>,
    pub shared: Dsv41Expert,
    pub engram: Option<Dsv41Engram>,
    /// Directory triples used by the shared DSV4 dynamic MoE bank.  An
    /// empty table means a synthetic/F32 fixture and deliberately falls
    /// back to the exact host expert loop.
    gpu_expert_ids: Vec<(usize, usize, usize)>,
    gpu_shared_ids: Option<(usize, usize, usize)>,
}

pub struct Dsv41Globals {
    pub embed: QTensor,
    pub norm: Vec<f32>,
    pub head: QTensor,
    pub inv_freq_compress: Vec<f32>,
    pub inv_freq_window: Vec<f32>,
}

/// Sequence state. `compressed` and `index_k` are keyed by source index,
/// rather than by every consumer layer; this is the CSA2 memory bound.
pub struct Dsv41State {
    pub pos: usize,
    pub window: Vec<Vec<f32>>,
    /// Compact global main KV and indexer-K streams keyed by CSA2 source.
    /// Rows are decoded only into bounded caller-owned scratch.
    pub packed: packed_kv::Dsv41PackedKvCache,
    pub pending_kv: Vec<Vec<f32>>,
    pub pending_score: Vec<Vec<f32>>,
    pub topk: Vec<usize>,
    /// Whether an index-source layer has published the shared sparse
    /// positions for the current sequence. An empty published list means
    /// "attend no compressed rows"; it must not be confused with the
    /// pre-indexer state, where the reference has no list yet and consumers
    /// use every visible row.
    pub topk_ready: bool,
    pub candidates: Vec<bool>,
    pub hash: Option<EngramHash>,
    /// Distinct key for the persistent device attention cache. DSV4 and
    /// DSV4.1 share the wgpu cache implementation but use different logical
    /// row contracts, so their sequence ids must never collide.
    #[cfg(feature = "gpu")]
    gpu_kv_id: u64,
    /// Model-wide segmented expert residency survives sequence resets, as
    /// in the established Qwen/Dsv4 dynamic MoE path.  The cache remains
    /// bounded by the GPU policy and cold experts complete on the host.
    #[cfg(feature = "gpu")]
    gpu_pool: Option<crate::qwen4_exp::QwenGpuPool>,
    /// GPTQ calibration capture (`cortiq quantize-moe`); `None` outside a
    /// capture. Survives [`Dsv41State::clear`]: [`capture_window`] owns it.
    pub moe_capture: Option<Dsv41MoeCapture>,
}

/// What a GPTQ calibration window records at layers `lo..hi`: every
/// token's MoE input (the RMS-normed mHC fold that `moe` receives, the input
/// of the routed experts and of the shared expert) and the experts its route
/// selected. Routing is resolved on the host in both the CPU and the dynamic
/// GPU MoE paths, so the record is taken before either runs.
#[derive(Debug, Clone, Default)]
pub struct Dsv41MoeCapture {
    pub lo: usize,
    pub hi: usize,
    /// End each token after layer `hi - 1`. No layer reads state written by
    /// a later layer (KV, index and candidate sources sit at or below their
    /// consumers), so the captured rows are the same and the rest of the
    /// stack and the head are skipped.
    pub stop_after_hi: bool,
    /// Per captured layer: `dim` f16 values per token.
    pub x: Vec<Vec<u16>>,
    /// Per captured layer: `top_k` routed expert ids per token, in route
    /// order (`u16::MAX` pads a route cut short by non-finite scores).
    pub picks: Vec<Vec<u16>>,
}

impl Dsv41State {
    pub fn new(cfg: &Dsv41Cfg, hash: Option<EngramHash>) -> Self {
        #[cfg(feature = "gpu")]
        let gpu_kv_id = {
            use std::sync::atomic::{AtomicU64, Ordering};
            static NEXT: AtomicU64 = AtomicU64::new(1);
            (0xD541u64 << 48) | NEXT.fetch_add(1, Ordering::Relaxed)
        };
        Self {
            pos: 0,
            window: vec![Vec::new(); cfg.compress_ratios.len()],
            packed: packed_kv::Dsv41PackedKvCache::new(
                cfg.kv_sources.len(),
                cfg.head_dim,
                cfg.index_head_dim,
            )
            .expect("valid V4.1 packed KV geometry"),
            pending_kv: vec![Vec::new(); cfg.kv_sources.len()],
            pending_score: vec![Vec::new(); cfg.kv_sources.len()],
            topk: Vec::new(),
            topk_ready: false,
            candidates: Vec::new(),
            hash,
            #[cfg(feature = "gpu")]
            gpu_kv_id,
            #[cfg(feature = "gpu")]
            gpu_pool: None,
            moe_capture: None,
        }
    }

    pub fn clear(&mut self) {
        #[cfg(feature = "gpu")]
        crate::gpu_wgpu::dsv4_cache_clear(self.gpu_kv_id);
        self.pos = 0;
        for v in &mut self.window {
            v.clear();
        }
        self.packed.clear();
        for v in &mut self.pending_kv {
            v.clear();
        }
        for v in &mut self.pending_score {
            v.clear();
        }
        self.topk.clear();
        self.topk_ready = false;
        self.candidates.clear();
        if let Some(h) = &mut self.hash {
            h.reset();
        }
    }
}

fn source_slot(sources: &[usize], li: usize) -> Option<usize> {
    sources
        .iter()
        .enumerate()
        .filter(|&(_, &s)| s <= li)
        .map(|(i, _)| i)
        .max()
}

fn rms(v: &mut [f32], w: &[f32], eps: f32) {
    // RMSNorm receives a BF16 tensor in the reference and returns another
    // BF16 tensor.  Rounding the input matters for folded residuals, while
    // rounding the result matters for every subsequent projection.
    bf16_inplace(v);
    crate::dsv4::rms_weighted(v, w, eps);
    bf16_inplace(v);
}

fn matvec(t: &QTensor, x: &[f32], out: &mut [f32], pool: Option<&Pool>) {
    t.matvec(x, out, pool);
}

#[inline]
fn matvec_bf16(t: &QTensor, x: &[f32], out: &mut [f32], pool: Option<&Pool>) {
    t.matvec(x, out, pool);
    bf16_inplace(out);
}

/// Opt-in V4.1 query fusion. The default remains the host-materialized query
/// path until the parent promotes the matched GPU/CPU profile. With this flag
/// the DSV4 frame consumes `qr` directly and performs wq_b, BF16 materializing,
/// and forward RoPE in its existing single encoder/readback.
#[inline]
fn v41_fused_q_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DSV41_FUSED_Q").is_ok_and(|value| value == "1"))
}

/// The V4.1 tail uses the proven DSV4 frame only when the selected backend is
/// live. The default follows the ordinary GPU selection; an explicit zero is
/// useful for the CPU reference and for the paired numerical proof.
#[cfg(feature = "gpu")]
fn gpu_attention_tail_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let requested = std::env::var("CMF_DSV41_GPU_ATTN")
            .map(|v| v != "0")
            .unwrap_or(true);
        let live = requested && crate::gpu::enabled_here() && crate::gpu_wgpu::adapter_up();
        if requested && !live && std::env::var("CMF_DSV41_GPU_ATTN").is_ok() {
            tracing::debug!("V4.1 GPU attention tail unavailable; retaining the CPU path");
        }
        live
    })
}

/// One-shot stage capture for the V4.1 adapter repair. The frame reads the
/// same tap names as the generic DSV4 proof, but the adapter falls through to
/// the CPU path after logging the GPU value so one call carries both sides.
fn gpu_tail_tap() -> Option<&'static str> {
    static TAP: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    TAP.get_or_init(|| {
        let tap = std::env::var("CMF_DSV41_TAIL_TAP").ok()?;
        matches!(tap.as_str(), "q" | "attn" | "mid" | "final").then_some(tap)
    })
    .as_deref()
}

fn f_any(model: &CmfModel, names: &[String]) -> Result<Vec<f32>, String> {
    for n in names {
        if model.tensor(n).is_some() {
            return crate::loader::load_f32(model, n, &crate::loader::Overlay::None);
        }
    }
    Err(format!("missing tensor (tried {})", names.join(", ")))
}

fn q_any(model: &Arc<CmfModel>, names: &[String]) -> Result<QTensor, String> {
    for n in names {
        if model.tensor(n).is_some() {
            return QTensor::from_model(model, n);
        }
    }
    Err(format!("missing tensor (tried {})", names.join(", ")))
}

fn scale3(model: &CmfModel, names: &[String]) -> Result<[f32; 3], String> {
    let v = f_any(model, names)?;
    if v.len() < 3 {
        return Err(format!(
            "scale tensor (tried {}) expected three values",
            names.join(", ")
        ));
    }
    Ok([v[0], v[1], v[2]])
}

fn load_expert(model: &Arc<CmfModel>, p: &str, e: usize) -> Result<Dsv41Expert, String> {
    let a = format!("{p}.experts.{e}");
    let w1 = q_any(
        model,
        &[format!("{a}.gate_proj.weight"), format!("{a}.w1.weight")],
    )?;
    let w2 = q_any(
        model,
        &[format!("{a}.down_proj.weight"), format!("{a}.w2.weight")],
    )?;
    let w3 = q_any(
        model,
        &[format!("{a}.up_proj.weight"), format!("{a}.w3.weight")],
    )?;
    Ok(Dsv41Expert { w1, w2, w3 })
}

fn load_shared(model: &Arc<CmfModel>, p: &str) -> Result<Dsv41Expert, String> {
    let mut names = Vec::new();
    for sh in ["shared_expert", "shared_experts"] {
        names.push(format!("{p}.{sh}.gate_proj.weight"));
        names.push(format!("{p}.{sh}.w1.weight"));
    }
    let w1 = q_any(model, &names)?;
    let mut names2 = Vec::new();
    for sh in ["shared_expert", "shared_experts"] {
        names2.push(format!("{p}.{sh}.down_proj.weight"));
        names2.push(format!("{p}.{sh}.w2.weight"));
    }
    let w2 = q_any(model, &names2)?;
    let mut names3 = Vec::new();
    for sh in ["shared_expert", "shared_experts"] {
        names3.push(format!("{p}.{sh}.up_proj.weight"));
        names3.push(format!("{p}.{sh}.w3.weight"));
    }
    let w3 = q_any(model, &names3)?;
    Ok(Dsv41Expert { w1, w2, w3 })
}

/// Load one V4.1 layer.  The loader accepts both the canonical names emitted
/// by the V4.1 converter and the upstream `ffn.*` spelling, which keeps toy
/// fixtures useful while the large source is converted in a separate job.
pub fn load_layer(model: &Arc<CmfModel>, cfg: &Dsv41Cfg, li: usize) -> Result<Dsv41Layer, String> {
    let p = format!("model.layers.{li}");
    let a = format!("{p}.self_attn");
    let q = |tail: &str| q_any(model, &[format!("{a}.{tail}"), format!("{p}.attn.{tail}")]);
    let attn_norm = f_any(
        model,
        &[
            format!("{p}.input_layernorm.weight"),
            format!("{p}.attn_norm.weight"),
        ],
    )?;
    let ffn_norm = f_any(
        model,
        &[
            format!("{p}.post_attention_layernorm.weight"),
            format!("{p}.ffn_norm.weight"),
        ],
    )?;
    let wq_a = q("wq_a.weight")?;
    let q_norm = f_any(
        model,
        &[
            format!("{a}.q_norm.weight"),
            format!("{p}.attn.q_norm.weight"),
        ],
    )?;
    let wq_b = q("wq_b.weight")?;
    let wkv = q("wkv.weight")?;
    let kv_norm = f_any(model, &[format!("{a}.kv_norm.weight")])?;
    let wo_a = q("wo_a.weight")?;
    let wo_b = q("wo_b.weight")?;
    let attn_sink = f_any(
        model,
        &[format!("{a}.attn_sink"), format!("{a}.attn_sink.weight")],
    )?;

    let ratio = cfg.ratio(li);
    let compressor = if cfg.kv_sources.contains(&li) {
        let cwkv = q_any(
            model,
            &[
                format!("{a}.compressor.wkv.weight"),
                format!("{p}.attn.compressor.wkv.weight"),
            ],
        )?;
        let wgate = if ratio > 1 {
            Some(q_any(
                model,
                &[
                    format!("{a}.compressor.wgate.weight"),
                    format!("{p}.attn.compressor.wgate.weight"),
                ],
            )?)
        } else {
            None
        };
        let norm = f_any(model, &[format!("{a}.compressor.norm.weight")])?;
        Some(Dsv41Compressor {
            wkv: cwkv,
            wgate,
            norm,
            ratio,
        })
    } else {
        None
    };

    let indexer = if cfg.index_sources.contains(&li) {
        let wq_bi = q_any(
            model,
            &[
                format!("{a}.indexer.wq_b.weight"),
                format!("{p}.attn.indexer.wq_b.weight"),
            ],
        )?;
        let weights_proj = q_any(
            model,
            &[
                format!("{a}.indexer.weights_proj.weight"),
                format!("{p}.attn.indexer.weights_proj.weight"),
            ],
        )?;
        let owns_k = cfg.kv_sources.contains(&li);
        let wk = if owns_k {
            Some(q_any(
                model,
                &[
                    format!("{a}.indexer.wk.weight"),
                    format!("{p}.attn.indexer.wk.weight"),
                ],
            )?)
        } else {
            None
        };
        let k_norm = if owns_k {
            Some(f_any(
                model,
                &[
                    format!("{a}.indexer.k_norm.weight"),
                    format!("{p}.attn.indexer.k_norm.weight"),
                ],
            )?)
        } else {
            None
        };
        Some(Dsv41Indexer {
            wq_b: wq_bi,
            weights_proj,
            wk,
            k_norm,
        })
    } else {
        None
    };

    let hc_attn_fn = f_any(model, &[format!("{p}.hc_attn_fn")])?;
    let hc_attn_base = f_any(model, &[format!("{p}.hc_attn_base")])?;
    let hc_attn_scale = scale3(
        model,
        &[format!("{p}.hc_attn_scale"), format!("{p}.hc_attn.scale")],
    )?;
    let hc_ffn_fn = f_any(model, &[format!("{p}.hc_ffn_fn")])?;
    let hc_ffn_base = f_any(model, &[format!("{p}.hc_ffn_base")])?;
    let hc_ffn_scale = scale3(
        model,
        &[format!("{p}.hc_ffn_scale"), format!("{p}.hc_ffn.scale")],
    )?;

    let gate = q_any(
        model,
        &[
            format!("{p}.mlp.gate.weight"),
            format!("{p}.ffn.gate.weight"),
        ],
    )?;
    let gate_bias = f_any(
        model,
        &[
            format!("{p}.mlp.expert_bias"),
            format!("{p}.ffn.gate.bias"),
            format!("{p}.mlp.gate.bias"),
        ],
    )?;
    let gate_bias_vl = f_any(
        model,
        &[
            format!("{p}.mlp.expert_bias_vl"),
            format!("{p}.ffn.gate.bias_vl"),
        ],
    )
    .ok();
    let mut experts = Vec::with_capacity(cfg.n_routed_experts);
    for e in 0..cfg.n_routed_experts {
        experts.push(
            load_expert(model, &format!("{p}.mlp"), e)
                .or_else(|_| load_expert(model, &format!("{p}.ffn"), e))?,
        );
    }
    let shared = load_shared(model, &format!("{p}.mlp"))
        .or_else(|_| load_shared(model, &format!("{p}.ffn")))?;
    let gpu_expert_ids = experts
        .iter()
        .map(|e| Some((e.w1.model_idx()?, e.w3.model_idx()?, e.w2.model_idx()?)))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    let gpu_shared_ids = match (
        shared.w1.model_idx(),
        shared.w3.model_idx(),
        shared.w2.model_idx(),
    ) {
        (Some(w1), Some(w3), Some(w2)) => Some((w1, w3, w2)),
        _ => None,
    };

    let engram = if cfg.engram_layers.contains(&li) {
        let ew = format!("{p}.engram.embed.weight");
        let es = format!("{p}.engram.embed.scale");
        let embed = RawFp8Rows::from_model(model, &ew, &es)?;
        if let Some(index) = cfg.engram_layers.iter().position(|&id| id == li) {
            if let Some(&expected_rows) = cfg.engram_embeddings.get(index) {
                if embed.rows != expected_rows {
                    return Err(format!(
                        "Engram layer {li} has {} rows, expected {expected_rows}",
                        embed.rows
                    ));
                }
            }
        }
        let wkv_e = q_any(model, &[format!("{p}.engram.wkv.weight")])?;
        let q_weight = f_any(model, &[format!("{p}.engram.q_weight")])?;
        let k_weight = f_any(model, &[format!("{p}.engram.k_weight")])?;
        Some(Dsv41Engram {
            embed,
            wkv: wkv_e,
            q_weight,
            k_weight,
        })
    } else {
        None
    };

    Ok(Dsv41Layer {
        attn_norm,
        ffn_norm,
        wq_a,
        q_norm,
        wq_b,
        wkv,
        kv_norm,
        wo_a,
        wo_b,
        attn_sink,
        compressor,
        indexer,
        hc_attn_fn,
        hc_attn_base,
        hc_attn_scale,
        hc_ffn_fn,
        hc_ffn_base,
        hc_ffn_scale,
        gate,
        gate_bias,
        gate_bias_vl,
        experts,
        shared,
        engram,
        gpu_expert_ids,
        gpu_shared_ids,
    })
}

/// Build the stack from canonical V4.1 tensors.  Frequency tables are kept
/// separate because compressed latents use the 160k base plus YaRN while the
/// raw sliding window uses the 10k base without YaRN.
pub fn load(
    model: &Arc<CmfModel>,
    cfg: &Dsv41Cfg,
    n_layers: usize,
    token_map: Vec<u32>,
) -> Result<(Dsv41Globals, Vec<Dsv41Layer>, Option<EngramHash>), String> {
    let q = |name: &str| QTensor::from_model(model, name);
    let globals = Dsv41Globals {
        embed: q("model.embed_tokens.weight")?,
        norm: crate::loader::load_f32(model, "model.norm.weight", &crate::loader::Overlay::None)?,
        head: q("lm_head.weight")?,
        inv_freq_compress: crate::attention::yarn_inv_freq(
            cfg.rope_head_dim,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.original_seq_len,
            cfg.beta_fast,
            cfg.beta_slow,
        ),
        inv_freq_window: crate::attention::rope_inv_freq(cfg.rope_head_dim, cfg.rope_theta),
    };
    let mut layers = Vec::with_capacity(n_layers);
    for li in 0..n_layers {
        layers.push(load_layer(model, cfg, li)?);
    }
    let hash = if !cfg.engram_layers.is_empty() {
        Some(EngramHash::new(
            cfg.engram_layers.clone(),
            cfg.engram_max_ngram,
            cfg.engram_heads,
            cfg.engram_vocab,
            cfg.engram_compressed_vocab,
            cfg.engram_pad_id,
            token_map,
        )?)
    } else {
        None
    };
    Ok((globals, layers, hash))
}

/// Build the compressed token lookup used by Engram.  The loader calls this
/// once; the returned vector is tiny compared with the mmap-backed tables.
pub fn token_map_from_model(model: &CmfModel, vocab: usize) -> Vec<u32> {
    let tokenizer = model
        .vocab
        .as_ref()
        .and_then(|bytes| crate::tokenizer::Tokenizer::from_bytes(bytes).ok());
    let mut out = Vec::with_capacity(vocab);
    let mut keys = std::collections::HashMap::<String, u32>::new();
    for id in 0..vocab {
        let text = tokenizer
            .as_ref()
            .map(|t| t.decode_token_for_hash(id as u32))
            .unwrap_or_default();
        // A partial UTF-8 byte token decodes to U+FFFD in the Rust
        // tokenizer. The reference keys those entries by their raw backend
        // token, otherwise unrelated byte fragments collapse together.
        let key = if text.contains('\u{fffd}') {
            tokenizer
                .as_ref()
                .map(|t| t.raw_token_for_hash(id as u32))
                .unwrap_or(text)
        } else {
            normalize_hash_token(&text)
        };
        let next = keys.len() as u32;
        out.push(*keys.entry(key).or_insert(next));
    }
    out
}

fn normalize_hash_token(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    // Match tokenizers' NFKC → NFD → StripAccents sequence. NFKD would
    // additionally decompose compatibility characters before composition,
    // which changes the compressed-vocabulary cardinality.
    let nfd: String = text.nfkc().nfd().collect();
    let mut out = String::with_capacity(nfd.len());
    let mut last_space = false;
    for c in nfd.chars() {
        if unicode_normalization::char::is_combining_mark(c) {
            continue;
        }
        if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
            if !last_space {
                out.push(' ');
            }
            last_space = true;
        } else {
            for lower in c.to_lowercase() {
                out.push(lower);
            }
            last_space = false;
        }
    }
    let trimmed = out.trim_matches(' ');
    if trimmed.is_empty() && out == " " {
        " ".to_string()
    } else if trimmed.is_empty() {
        text.to_string()
    } else {
        trimmed.to_string()
    }
}

fn apply_engram(
    e: &Dsv41Engram,
    h: &mut [f32],
    hashes: &[usize],
    cfg: &Dsv41Cfg,
    token_mask: bool,
    pool: Option<&Pool>,
) {
    if !token_mask {
        return;
    }
    let cols = hashes.len() * e.embed.cols;
    let mut emb = vec![0.0f32; cols];
    for (i, &row) in hashes.iter().enumerate() {
        e.embed.row_into(
            row.min(e.embed.rows.saturating_sub(1)),
            &mut emb[i * e.embed.cols..(i + 1) * e.embed.cols],
        );
    }
    // ParallelEngramEmbedding returns the dequantized lookup as BF16 before
    // the projection.  Keep the raw FP8 row representation, but materialise
    // this activation boundary explicitly in the f32 work buffer.
    bf16_inplace(&mut emb);
    let mut kv = vec![0.0f32; cfg.dim * (cfg.hc_mult + 1)];
    // Engram's projection uses the model BF16 dtype in the reference.
    matvec_bf16(&e.wkv, &emb, &mut kv, pool);
    let key = &kv[..cfg.dim * cfg.hc_mult];
    let value = &kv[cfg.dim * cfg.hc_mult..];
    for copy in 0..cfg.hc_mult {
        let hs = &h[copy * cfg.dim..(copy + 1) * cfg.dim];
        let ks = &key[copy * cfg.dim..(copy + 1) * cfg.dim];
        let qw = e
            .q_weight
            .get(copy * cfg.dim..(copy + 1) * cfg.dim)
            .unwrap_or(&[]);
        let kw = e
            .k_weight
            .get(copy * cfg.dim..(copy + 1) * cfg.dim)
            .unwrap_or(&[]);
        let hm = hs.iter().map(|v| v * v).sum::<f32>() / cfg.dim as f32;
        let km = ks.iter().map(|v| v * v).sum::<f32>() / cfg.dim as f32;
        let rstd = 1.0 / (hm + cfg.norm_eps).sqrt() / (km + cfg.norm_eps).sqrt();
        let mut dot = 0.0;
        for i in 0..cfg.dim {
            dot += hs[i]
                * ks[i]
                * qw.get(i).copied().unwrap_or(1.0)
                * kw.get(i).copied().unwrap_or(1.0);
        }
        dot *= rstd * (cfg.dim as f32).powf(-0.5);
        // torch.copysign(+sqrt(abs(dot)), dot) treats an exact zero as
        // positive. `signum()` would turn that case into zero and changes
        // the gate from sigmoid(sqrt(1e-6)) to sigmoid(0).
        let root = dot.abs().max(1e-6).sqrt();
        let gate = sigmoid(if dot.is_sign_negative() { -root } else { root });
        let dst = &mut h[copy * cfg.dim..(copy + 1) * cfg.dim];
        for i in 0..cfg.dim {
            dst[i] += gate * value[i];
        }
    }
    // The source returns `h + gate * value` cast back to the residual dtype;
    // leaving this f32 would make the next HC projection consume extra bits.
    bf16_inplace(h);
}

/// Run the isolated Engram projection for the release component oracle.
/// Kept hidden from generated documentation; this narrow entry point lets a
/// fixture exercise the mmap-backed raw E4M3/E8M0 lookup without constructing
/// the rest of a full V4.1 checkpoint.
#[doc(hidden)]
pub fn dsv41_apply_engram_for_test(
    e: &Dsv41Engram,
    h: &mut [f32],
    hashes: &[usize],
    cfg: &Dsv41Cfg,
    token_mask: bool,
) {
    apply_engram(e, h, hashes, cfg, token_mask, None);
}

fn compressor_step(
    cp: &Dsv41Compressor,
    x: &[f32],
    cfg: &Dsv41Cfg,
    pending_kv: &mut Vec<f32>,
    pending_score: &mut Vec<f32>,
    pool: Option<&Pool>,
) -> Option<Vec<f32>> {
    let mut kv = vec![0.0f32; cfg.head_dim];
    // Ratio-one compressors use the model BF16 linear.  The pooled
    // ratio>1 path intentionally keeps both projections in f32.
    if cp.ratio == 1 {
        matvec_bf16(&cp.wkv, x, &mut kv, pool);
    } else {
        matvec(&cp.wkv, x, &mut kv, pool);
    }
    if cp.ratio == 1 {
        rms(&mut kv, &cp.norm, cfg.norm_eps);
        return Some(kv);
    }
    let mut score = vec![0.0f32; cfg.head_dim];
    if let Some(wg) = &cp.wgate {
        matvec(wg, x, &mut score, pool);
    }
    pending_kv.extend_from_slice(&kv);
    pending_score.extend_from_slice(&score);
    if pending_kv.len() < cp.ratio * cfg.head_dim {
        return None;
    }
    let mut out = vec![0.0f32; cfg.head_dim];
    for d in 0..cfg.head_dim {
        let m = (0..cp.ratio)
            .map(|i| pending_score[i * cfg.head_dim + d])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut den = 0.0;
        for i in 0..cp.ratio {
            den += (pending_score[i * cfg.head_dim + d] - m).exp();
        }
        if den > 0.0 {
            for i in 0..cp.ratio {
                out[d] += (pending_score[i * cfg.head_dim + d] - m).exp() / den
                    * pending_kv[i * cfg.head_dim + d];
            }
        }
    }
    pending_kv.clear();
    pending_score.clear();
    rms(&mut out, &cp.norm, cfg.norm_eps);
    Some(out)
}

fn fp4_round(x: f32) -> f32 {
    let sign = if x.is_sign_negative() { -1.0 } else { 1.0 };
    let ax = x.abs();
    if !ax.is_finite() {
        return sign * 6.0;
    }
    const LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    // E2M1 has an alternating even/odd significand at these levels.  The
    // hardware/Torch cast is round-to-nearest-even, so a half-way value must
    // choose the even level (e.g. 0.75 -> 1, 1.75 -> 2, 3.5 -> 4), rather
    // than the lower level selected by a stable min-by tie.
    const EVEN: [bool; 8] = [true, false, true, false, true, false, true, false];
    let mut best = 0usize;
    let mut best_dist = ax;
    for i in 1..LEVELS.len() {
        let dist = (ax - LEVELS[i]).abs();
        if dist < best_dist || (dist == best_dist && EVEN[i] && !EVEN[best]) {
            best = i;
            best_dist = dist;
        }
    }
    sign * LEVELS[best]
}

fn fp4_inplace(v: &mut [f32], block: usize, e8m0_scale_fmt: bool) {
    for chunk in v.chunks_mut(block) {
        // Keep the same nonzero scale floors as the reference kernel.  The
        // E8M0 index path permits subnormal 2^-126, while compressed KV's
        // E4M3 scale path floors the amax at 6*2^-9 before the FP8 cast.
        // A single generic 1e-4 floor silently turns zero index blocks into
        // 2^-15 and changes every subsequent dot product.
        let min_amax = if e8m0_scale_fmt {
            6.0 * 2.0f32.powi(-126)
        } else {
            6.0 * 2.0f32.powi(-9)
        };
        let max = chunk.iter().map(|x| x.abs()).fold(min_amax, f32::max);
        let raw_scale = max / 6.0;
        let scale = if e8m0_scale_fmt {
            round_e8m0_scale(raw_scale)
        } else {
            // E4M3 scales use the same minimum as fp4_quant_kernel: an
            // all-zero block still carries 6*2^-9 before the cast.
            e4m3_round(raw_scale.max(6.0 * 2.0f32.powi(-9)))
        };
        if scale == 0.0 || !scale.is_finite() {
            chunk.fill(0.0);
            continue;
        }
        for x in chunk {
            *x = bf16_roundtrip(fp4_round(*x / scale) * scale);
        }
    }
}

/// `fast_log2_ceil` of the reference kernels: the biased exponent plus one
/// when any mantissa bit is set. Exact for normal inputs, unlike a rounded
/// `log2().ceil()` just above a power of two.
#[inline]
fn fast_log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0)
}

/// `act_quant(x, 32, "ue8m0", float8_e8m0fnu, inplace=True)`: per 32-value
/// block, `s = 2^ceil(log2(max(amax, 1e-4) / 448))` (the kernel multiplies by
/// the f32 constant 1/448), values quantized to E4M3FN (nearest-even,
/// saturating) and dequantized, stored as BF16.
fn fp8_activation_inplace(v: &mut [f32]) {
    const FP8_MAX_INV: f32 = 1.0 / 448.0;
    for chunk in v.chunks_mut(32) {
        let amax = chunk.iter().map(|x| x.abs()).fold(1e-4f32, f32::max);
        let scale = f32::from_bits(((fast_log2_ceil(amax * FP8_MAX_INV) + 127) as u32) << 23);
        for x in chunk {
            *x = bf16_roundtrip(e4m3_round((*x / scale).clamp(-448.0, 448.0)) * scale);
        }
    }
}

/// `CMF_DSV41_O_FP8=1`: quantize the attention output to FP8 (E4M3, 32-value
/// blocks, power-of-two scales) before the grouped `wo_a`, as the reference
/// does since commit 4dd75a8b (`act_quant(o, ...)` before the einsum). Off
/// by default until a run on the released weights shows its effect. The
/// device attention tail has no such quantizer yet, so the flag keeps V4.1
/// attention on the CPU tail.
fn v41_o_fp8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_DSV41_O_FP8").is_ok_and(|value| value == "1"))
}

/// The attention output boundary before `wo_a`: the reference views `o` as
/// `[groups, n_heads*head_dim/groups]` and quantizes the last axis in blocks
/// of 32, so with a 32-multiple group width the blocks are contiguous runs
/// of the flat output.
fn quantize_attention_output(attended: &mut [f32], o_groups: usize) {
    debug_assert!(
        attended.len() % o_groups.max(1) == 0 && (attended.len() / o_groups.max(1)) % 32 == 0,
        "act_quant(o) needs a 32-multiple group width"
    );
    fp8_activation_inplace(attended);
}

struct IndexResult {
    picked: Vec<usize>,
    scores: Vec<f32>,
}

fn update_index(
    ix: &Dsv41Indexer,
    x: &[f32],
    qr: &[f32],
    source_k: &packed_kv::PackedRows,
    cfg: &Dsv41Cfg,
    pos: usize,
    ratio: usize,
    inv_freq: &[f32],
    candidate_mask: Option<&[bool]>,
    pool: Option<&Pool>,
) -> IndexResult {
    let ih = cfg.index_heads;
    let id = cfg.index_head_dim;
    let mut q = vec![0.0f32; ih * id];
    matvec_bf16(&ix.wq_b, qr, &mut q, pool);
    // `q` is head-major. RoPE applies to the tail of each index head, not to
    // the tail of the flattened concatenation.
    for head in 0..ih {
        crate::dsv4::rope_tail(
            &mut q[head * id..(head + 1) * id],
            inv_freq,
            pos,
            cfg.rope_head_dim.min(id) & !1,
            false,
        );
    }
    // RoPE writes back into a BF16 tensor in the reference kernel.
    bf16_inplace(&mut q);
    fp4_inplace(&mut q, 32, true);
    let mut weights = vec![0.0f32; ih];
    matvec_bf16(&ix.weights_proj, x, &mut weights, pool);
    let sc = (id as f32).powf(-0.5) * (ih as f32).powf(-0.5);
    for w in &mut weights {
        // Torch keeps the scalar multiply in the BF16 tensor dtype.
        *w = bf16_roundtrip(*w * sc);
    }
    let n = source_k.rows();
    let mut scores = vec![f32::NEG_INFINITY; n];
    let mut k = vec![0.0f32; id];
    for t in 0..n {
        if candidate_mask.is_some_and(|m| !m.get(t).copied().unwrap_or(false)) {
            continue;
        }
        assert!(
            source_k.row_into(t, &mut k),
            "packed index-K row {t} is unavailable"
        );
        let k = &k[..];
        let mut s = 0.0;
        for head in 0..ih {
            // einsum emits a BF16 score; its dot-product accumulator is
            // wider, but the output is materialised before ReLU.
            let d = bf16_roundtrip(
                q[head * id..(head + 1) * id]
                    .iter()
                    .zip(k)
                    .map(|(a, b)| a * b)
                    .sum::<f32>(),
            )
            .max(0.0);
            // BF16 multiplication happens before the final head reduction.
            s += bf16_roundtrip(d * weights[head]);
        }
        if t >= (pos + 1) / ratio.max(1) {
            scores[t] = f32::NEG_INFINITY;
        } else {
            // torch.sum over the BF16 products returns a BF16 tensor.
            scores[t] = bf16_roundtrip(s);
        }
    }
    let mut picked = Vec::new();
    crate::dsv4::top_k_positions(&scores, cfg.index_topk, &mut picked);
    IndexResult { picked, scores }
}

fn attention(
    l: &Dsv41Layer,
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    li: usize,
    x: &[f32],
    inv_freq: &[f32],
    pool: Option<&Pool>,
    out: &mut [f32],
) -> Vec<f32> {
    let attn_other_t0 = prof::start();
    let mut qr = vec![0.0f32; cfg.q_lora_rank];
    let mut kv = vec![0.0f32; cfg.head_dim];
    // Both projections read the same hidden state.  Keep the two output
    // boundaries explicit, but hand the Q4TP pair to the existing
    // matvec_many dispatcher so one pool job owns the shared input walk.
    QTensor::matvec_many([&l.wq_a, &l.wkv], x, [&mut qr, &mut kv], pool);
    bf16_inplace(&mut qr);
    trace_stats("wq_a", st.pos, Some(li), &qr);
    rms(&mut qr, &l.q_norm, cfg.norm_eps);
    trace_stats("q_norm", st.pos, Some(li), &qr);
    bf16_inplace(&mut kv);
    trace_stats("wkv", st.pos, Some(li), &kv);
    rms(&mut kv, &l.kv_norm, cfg.norm_eps);
    trace_stats("kv_norm", st.pos, Some(li), &kv);
    crate::dsv4::rope_tail(&mut kv, inv_freq, st.pos, cfg.rope_head_dim, false);
    bf16_inplace(&mut kv);

    // A layer's `compress_ratio` controls whether it reads the shared
    // compressed stream at all. Layers after the last source can still have
    // a source slot in scope, but ratio-zero layers are window-only in the
    // reference and must not concatenate the old compressed cache.
    let source = if cfg.ratio(li) > 0 {
        source_slot(&cfg.kv_sources, li)
    } else {
        None
    };
    let mut latent = None;
    if let (Some(cp), Some(si)) = (&l.compressor, source) {
        latent = compressor_step(
            cp,
            x,
            cfg,
            &mut st.pending_kv[si],
            &mut st.pending_score[si],
            pool,
        );
        if let Some(v) = latent.as_ref() {
            trace_stats("latent", st.pos, Some(li), v);
            // The compressor publishes a pre-RoPE latent.  Keep that form
            // for the index-key owner, and write a separately rotated copy
            // to the attention cache.
            let latent_pos = st.pos + 1 - cfg.ratio(li).max(1);
            let mut stored = v.clone();
            crate::dsv4::rope_tail(&mut stored, inv_freq, latent_pos, cfg.rope_head_dim, false);
            bf16_inplace(&mut stored);
            let row_id = st
                .packed
                .push_main(si, &stored)
                .expect("valid main KV source slot");
            if std::env::var_os("CMF_DSV41_TRACE").is_some() {
                let mut quantized = vec![0.0f32; cfg.head_dim];
                assert!(st.packed.main_row_into(si, row_id, &mut quantized));
                trace_stats("compressed", st.pos, Some(li), &quantized);
            }
        }
    }
    // Index keys are derived from the unrotated latent, before attention's
    // RoPE write.  A source publishes one key; consumer indexers reuse it.
    if let (Some(ix), Some(si), Some(lat)) = (&l.indexer, source, latent.as_ref()) {
        if let (Some(wk), Some(kn)) = (&ix.wk, &ix.k_norm) {
            let latent_pos = st.pos + 1 - cfg.ratio(li).max(1);
            let mut k = vec![0.0f32; cfg.index_head_dim];
            matvec_bf16(wk, lat, &mut k, pool);
            rms(&mut k, kn, cfg.norm_eps);
            crate::dsv4::rope_tail(
                &mut k,
                inv_freq,
                latent_pos,
                cfg.rope_head_dim.min(cfg.index_head_dim) & !1,
                false,
            );
            bf16_inplace(&mut k);
            let row_id = st
                .packed
                .push_index(si, &k)
                .expect("valid index-K source slot");
            if std::env::var_os("CMF_DSV41_TRACE").is_some() {
                let mut quantized = vec![0.0f32; cfg.index_head_dim];
                assert!(st.packed.index_row_into(si, row_id, &mut quantized));
                trace_stats("index_k", st.pos, Some(li), &quantized);
            }
        }
        let _ = ix;
    }
    // Publish the sliding window before building the sparse position list.
    // Keep this mutable borrow scoped: attention below only needs immutable
    // slices, and can therefore read the shared compressed cache without
    // cloning it on every decode token.
    let win_len = {
        let w = &mut st.window[li];
        fp8_activation_inplace(&mut kv);
        // `act_quant(..., inplace=True)` stores the dequantized result as
        // BF16 even though its internal scale/value arithmetic is f32.
        bf16_inplace(&mut kv);
        trace_stats("kv", st.pos, Some(li), &kv);
        w.extend_from_slice(&kv);
        let cap = cfg.window * cfg.head_dim;
        if w.len() > cap {
            let drop = w.len() - cap;
            w.drain(..drop);
        }
        w.len() / cfg.head_dim
    };
    let mut idxs: Vec<usize> = (0..win_len).collect();
    if let Some(si) = source {
        let comp_len = st.packed.main_rows(si);
        if comp_len > 0 {
            if let Some(ix) = &l.indexer {
                let candidates = if cfg.candidate_source.is_some_and(|c| li > c)
                    && !st.candidates.is_empty()
                {
                    Some(st.candidates.as_slice())
                } else {
                    None
                };
                let result = update_index(
                    ix,
                    x,
                    &qr,
                    st.packed
                        .index_store(si)
                        .expect("index-K store for shared source"),
                    cfg,
                    st.pos,
                    cfg.ratio(li),
                    inv_freq,
                    candidates,
                    pool,
                );
                trace_indices(st.pos, li, &result.picked);
                if cfg.candidate_source == Some(li) {
                    st.candidates = candidate_blocks(
                        &result.scores,
                        cfg.candidate_topk_blocks,
                        cfg.candidate_block_size,
                    );
                    trace_index_scores(
                        st.pos,
                        li,
                        &result.scores,
                        &result.picked,
                        Some(st.candidates.as_slice()),
                        win_len,
                        comp_len,
                        st.pending_kv[si].len() / cfg.head_dim.max(1),
                    );
                } else {
                    trace_index_scores(
                        st.pos,
                        li,
                        &result.scores,
                        &result.picked,
                        candidates,
                        win_len,
                        comp_len,
                        st.pending_kv[si].len() / cfg.head_dim.max(1),
                    );
                }
                st.topk = result.picked;
                st.topk_ready = true;
            }
            let all_picks: Vec<usize> = (0..comp_len).collect();
            let picks: &[usize] = if cfg.index_source(li).is_some() && st.topk_ready {
                &st.topk
            } else {
                &all_picks
            };
            idxs.extend(
                picks
                    .iter()
                    .copied()
                    .filter(|&p| p < comp_len)
                    .map(|p| win_len + p),
            );
        }
    }

    // V4.1 has no second per-head query RMSNorm after wq_b. The default
    // path materialises the final query on the host so the GPU adapter and
    // CPU fallback share one exact BF16/RoPE value. The opt-in fused path
    // delays this allocation: its frame consumes `qr` and performs wq_b,
    // BF16 materialisation, and forward RoPE inside the existing encoder.
    let fused_q = v41_fused_q_enabled();
    let materialize_q = || {
        let mut q = vec![0.0f32; cfg.n_heads * cfg.head_dim];
        matvec_bf16(&l.wq_b, &qr, &mut q, pool);
        trace_stats("wq_b", st.pos, Some(li), &q);
        for h in 0..cfg.n_heads {
            let qh = &mut q[h * cfg.head_dim..(h + 1) * cfg.head_dim];
            crate::dsv4::rope_tail(qh, inv_freq, st.pos, cfg.rope_head_dim, false);
        }
        bf16_inplace(&mut q);
        trace_stats("q", st.pos, Some(li), &q);
        q
    };
    let mut q = if fused_q { None } else { Some(materialize_q()) };

    // Decode the selected global rows once. The resulting compact stream is
    // shared by every head and by both the device frame and CPU fallback;
    // no history-sized f32 materialization is needed for sparse attention.
    let selected_comp_positions: Vec<usize> = idxs
        .iter()
        .copied()
        .filter(|&p| p >= win_len)
        .map(|p| p - win_len)
        .collect();
    let mut selected_compressed = vec![0.0f32; selected_comp_positions.len() * cfg.head_dim];
    if let Some(si) = source {
        assert!(
            st.packed
                .gather_main_rows(si, &selected_comp_positions, &mut selected_compressed,),
            "packed main-K gather failed for source {si}"
        );
    }
    let mut packed_idxs = Vec::with_capacity(idxs.len());
    for &logical in &idxs {
        if logical < win_len {
            packed_idxs.push(logical);
        } else {
            let comp = logical - win_len;
            let selected = selected_comp_positions
                .iter()
                .position(|&row| row == comp)
                .expect("selected compressed row mapping");
            packed_idxs.push(win_len + selected);
        }
    }

    // ── proven DSV4 attention tail on the device ──
    // V4.1 keeps the compressor/index publication above on the host. Once
    // the bounded selected rows exist, the established DSV4 frame can perform
    // q projection (when fused), sparse attention with the per-head sink,
    // inverse RoPE, grouped wo_a and wo_b in one encoder/readback.
    // The cache is keyed by this state's private id and receives only the
    // current bounded window plus the selected compressed rows; no
    // whole-history upload or clone is created here.
    #[cfg(feature = "gpu")]
    if gpu_attention_tail_enabled() && !v41_o_fp8_enabled() {
        let n_comp = selected_comp_positions.len();
        let cache_cap = (cfg.window + n_comp.next_power_of_two().max(64)) * cfg.head_dim;
        let cache_ok =
            crate::gpu_wgpu::dsv4_cache_write(st.gpu_kv_id, li, 0, &st.window[li], cache_cap)
                && (selected_compressed.is_empty()
                    || crate::gpu_wgpu::dsv4_cache_write(
                        st.gpu_kv_id,
                        li,
                        cfg.window * cfg.head_dim,
                        &selected_compressed,
                        cache_cap,
                    ));
        let model = l.wq_b.model_arc();
        // The gathered cache is already in attention-list order, so logical
        // positions become compact row numbers. This keeps sink/index/window
        // semantics while bounding each upload to the rows actually used.
        let idx32: Vec<u32> = packed_idxs
            .iter()
            .map(|&p| {
                if p < win_len {
                    p as u32
                } else {
                    (cfg.window + p - win_len) as u32
                }
            })
            .collect();
        let diag_tap = gpu_tail_tap();
        let diag_len = diag_tap.map(|tap| match tap {
            "q" | "attn" => cfg.n_heads * cfg.head_dim,
            "mid" => cfg.o_groups * cfg.o_lora_rank,
            _ => cfg.dim,
        });
        let mut gpu_attended = vec![0.0f32; cfg.dim.max(diag_len.unwrap_or(0))];
        // `qn_in` is the already normalized LoRA-rank query. In fused mode
        // the frame owns wq_b/BF16/RoPE; otherwise retain the legacy final-q
        // upload. Both modes keep the same selected packed rows and output
        // buffer contract.
        let qn_in = fused_q.then_some(qr.as_slice());
        let q_in = q.as_deref();
        let gpu_ok = cache_ok
            && model.is_some_and(|model| {
                let w = crate::gpu_wgpu::Dsv4AttnW {
                    wq_a: l.wq_a.model_idx().unwrap_or(usize::MAX),
                    wq_b: l.wq_b.model_idx().unwrap_or(usize::MAX),
                    wo_a: l.wo_a.model_idx().unwrap_or(usize::MAX),
                    wo_b: l.wo_b.model_idx().unwrap_or(usize::MAX),
                    q_norm: &l.q_norm,
                    sink: &l.attn_sink,
                };
                let g = crate::gpu_wgpu::Dsv4AttnGeom {
                    dim: cfg.dim,
                    nh: cfg.n_heads,
                    hd: cfg.head_dim,
                    rd: cfg.rope_head_dim,
                    q_lora: cfg.q_lora_rank,
                    o_lora: cfg.o_lora_rank,
                    o_groups: cfg.o_groups,
                    eps: cfg.norm_eps,
                    scale: (cfg.head_dim as f32).powf(-0.5),
                    bf16: true,
                    q_rms: false,
                };
                crate::gpu_wgpu::dsv4_attn_frame(
                    &model,
                    &w,
                    g,
                    &[],
                    qn_in,
                    q_in,
                    st.gpu_kv_id,
                    li,
                    &idx32,
                    inv_freq,
                    st.pos,
                    None,
                    &mut gpu_attended,
                )
            });
        if gpu_ok {
            prof::note_gpu_attn();
            if let Some(tap) = diag_tap {
                let name = match tap {
                    "q" => "gpu_q",
                    "attn" => "gpu_attn",
                    "mid" => "gpu_mid",
                    _ => "gpu_tail",
                };
                trace_stats(name, st.pos, Some(li), &gpu_attended[..diag_len.unwrap()]);
            } else {
                // The GPU frame owns the entire attention body on this arm. Keep
                // the cumulative stage report additive with the CPU path: its
                // host time is the residual of the outer attention timer.
                prof::add(&prof::ATTN_OTHER_NS, attn_other_t0);
                bf16_inplace(&mut gpu_attended);
                out.copy_from_slice(&gpu_attended[..cfg.dim]);
                return qr;
            }
        }
        prof::note_gpu_attn_fallback();
    }

    // A disabled/unavailable GPU, a cache miss, or a diagnostic tap needs the
    // unchanged host query for the ordinary sparse CPU fallback. Keep this
    // allocation after the GPU attempt in fused mode so the successful path
    // has no host q materialization or q upload.
    if q.is_none() {
        q = Some(materialize_q());
    }
    let q = q
        .as_deref()
        .expect("V4.1 query must be materialized before CPU attention");

    let window = &st.window[li];
    let compressed = if selected_compressed.is_empty() {
        None
    } else {
        Some(selected_compressed.as_slice())
    };
    let mut attended = vec![0.0f32; cfg.n_heads * cfg.head_dim];
    prof::add(&prof::ATTN_OTHER_NS, attn_other_t0);
    let sparse_t0 = prof::start();
    for h in 0..cfg.n_heads {
        sparse_attend_split(
            &q[h * cfg.head_dim..(h + 1) * cfg.head_dim],
            window,
            compressed,
            &packed_idxs,
            l.attn_sink.get(h).copied().unwrap_or(0.0),
            (cfg.head_dim as f32).powf(-0.5),
            win_len,
            cfg.head_dim,
            &mut attended[h * cfg.head_dim..(h + 1) * cfg.head_dim],
        );
    }
    prof::add(&prof::SPARSE_NS, sparse_t0);
    let attn_other_t0 = prof::start();
    for h in 0..cfg.n_heads {
        crate::dsv4::rope_tail(
            &mut attended[h * cfg.head_dim..(h + 1) * cfg.head_dim],
            inv_freq,
            st.pos,
            cfg.rope_head_dim,
            true,
        );
        bf16_inplace(&mut attended[h * cfg.head_dim..(h + 1) * cfg.head_dim]);
    }
    if v41_o_fp8_enabled() {
        quantize_attention_output(&mut attended, cfg.o_groups);
    }
    trace_stats("attended", st.pos, Some(li), &attended);
    if gpu_tail_tap().is_some_and(|tap| tap == "mid") {
        let mut cpu_mid = vec![0.0f32; cfg.o_groups * cfg.o_lora_rank];
        let mut scratch = vec![0.0f32; l.wo_a.cols()];
        for (i, mid) in cpu_mid.iter_mut().enumerate() {
            let group = i / cfg.o_lora_rank;
            *mid = bf16_roundtrip(l.wo_a.row_dot(
                i,
                &attended[group * l.wo_a.cols()..(group + 1) * l.wo_a.cols()],
                &mut scratch,
            ));
        }
        trace_stats("cpu_mid", st.pos, Some(li), &cpu_mid);
    }
    crate::dsv4::o_project(
        &attended,
        // `wo_a` is the grouped BF16 einsum in the reference.  Its result
        // is materialised as BF16 before the second projection; rounding
        // only the final `wo_b` output leaves a wider f32 intermediate and
        // accumulates a measurable drift into the next block.
        &|r, v, scratch| bf16_roundtrip(l.wo_a.row_dot(r, v, scratch)),
        l.wo_a.cols(),
        &|m, o| l.wo_b.matvec(m, o, pool),
        cfg.o_groups,
        cfg.o_lora_rank,
        pool,
        out,
    );
    // The grouped wo_a/wo_b projection returns the model activation dtype.
    bf16_inplace(out);
    if gpu_tail_tap().is_some() {
        trace_stats("cpu_tail", st.pos, Some(li), out);
    }
    prof::add(&prof::ATTN_OTHER_NS, attn_other_t0);
    qr
}

/// Attention over the concatenated logical stream `window || compressed`
/// without materializing that concatenation.  A decode step can have a very
/// large compressed history, so cloning it once per layer would turn a
/// bounded sparse read into an O(context) copy before the actual O(top-k)
/// attention.  The position list still uses the same logical indices as the
/// reference helper; this function only changes how the selected value rows
/// are addressed.
fn sparse_attend_split(
    q: &[f32],
    window: &[f32],
    compressed: Option<&[f32]>,
    idxs: &[usize],
    sink: f32,
    scale: f32,
    window_len: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let row = |logical: usize| -> Option<&[f32]> {
        if logical < window_len {
            window.get(logical * head_dim..(logical + 1) * head_dim)
        } else {
            compressed.and_then(|c| {
                let p = logical - window_len;
                c.get(p * head_dim..(p + 1) * head_dim)
            })
        }
    };
    let mut m = sink;
    let mut scores = Vec::with_capacity(idxs.len());
    for &p in idxs {
        let Some(k) = row(p) else {
            scores.push(f32::NEG_INFINITY);
            continue;
        };
        let dot: f32 = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
        m = m.max(dot);
        scores.push(dot);
    }
    let mut denom = (sink - m).exp();
    out.fill(0.0);
    for (&p, &score) in idxs.iter().zip(&scores) {
        let Some(v) = row(p) else {
            continue;
        };
        let weight = (score - m).exp();
        denom += weight;
        for (dst, &value) in out.iter_mut().zip(v) {
            *dst += weight * value;
        }
    }
    if denom > 0.0 && denom.is_finite() {
        let inv = 1.0 / denom;
        for dst in out.iter_mut() {
            *dst *= inv;
        }
    }
}

fn candidate_blocks(scores: &[f32], top_blocks: usize, block: usize) -> Vec<bool> {
    let n = scores.len();
    let blocks = n.div_ceil(block.max(1));
    if blocks == 0 {
        return Vec::new();
    }
    let block = block.max(1);
    let mut order: Vec<usize> = (0..blocks).collect();
    // The source selects by the best reachable position in each block. A
    // stable index tie break keeps CPU and GPU fixtures deterministic.
    order.sort_by(|&a, &b| {
        let sa = scores[a * block..((a + 1) * block).min(n)]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sb = scores[b * block..((b + 1) * block).min(n)]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    let reachable = scores.iter().filter(|x| x.is_finite()).count();
    if reachable == 0 {
        return vec![false; n];
    }
    let last = scores
        .iter()
        .rposition(|x| x.is_finite())
        .map(|p| p / block)
        .unwrap_or(0);
    let mut keep = vec![false; blocks];
    let budget = top_blocks.min(blocks);
    let mut selected = 0usize;
    if budget > 0 {
        keep[last] = true;
        selected = 1;
    }
    for &b in &order {
        if selected >= budget {
            break;
        }
        if b == last || keep[b] {
            continue;
        }
        let block_score = scores[b * block..((b + 1) * block).min(n)]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        if block_score.is_finite() {
            keep[b] = true;
            selected += 1;
        }
    }
    let mut out = vec![false; n];
    for b in 0..blocks {
        if keep[b] {
            for i in b * block..((b + 1) * block).min(n) {
                out[i] = true;
            }
        }
    }
    out
}

#[cfg(feature = "gpu")]
fn dsv41_expert_cpu(
    expert: &Dsv41Expert,
    cfg: &Dsv41Cfg,
    x: &[f32],
    weight: f32,
    pool: Option<&Pool>,
    out: &mut [f32],
) {
    let mut g = vec![0.0f32; cfg.moe_inter];
    let mut u = vec![0.0f32; cfg.moe_inter];
    matvec_bf16(&expert.w1, x, &mut g, pool);
    matvec_bf16(&expert.w3, x, &mut u, pool);
    for i in 0..cfg.moe_inter {
        if cfg.swiglu_limit > 0.0 {
            g[i] = g[i].min(cfg.swiglu_limit);
            u[i] = u[i].clamp(-cfg.swiglu_limit, cfg.swiglu_limit);
        }
        g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i] * weight;
    }
    // The official Expert converts the weighted SwiGLU activation back to
    // the input dtype (BF16 in V4.1) before w2. `matvec_bf16` rounds its
    // output, but that is a separate boundary and cannot replace this cast.
    bf16_inplace(&mut g);
    let mut tmp = vec![0.0f32; cfg.dim];
    matvec_bf16(&expert.w2, &g, &mut tmp, pool);
    for (dst, value) in out.iter_mut().zip(tmp) {
        *dst += value;
    }
}

#[cfg(feature = "gpu")]
fn dsv41_cold_experts_cpu(
    layer: &Dsv41Layer,
    cfg: &Dsv41Cfg,
    x: &[f32],
    cold: &[(usize, f32)],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; cfg.dim];
    for &(expert, weight) in cold {
        if let Some(expert) = layer.experts.get(expert) {
            dsv41_expert_cpu(expert, cfg, x, weight, pool, &mut out);
        }
    }
    out
}

/// Execute V4.1's host-resolved route through the shared DSV4 segmented
/// global expert bank.  V4.1 routing stays on the host because its VL bias
/// and sqrt-softplus rule differ from the generic shader route.  Forced ids
/// plus preweighted values preserve that route exactly while the card fuses
/// gate/up/SwiGLU/down for resident experts and the host completes cold ones.
#[cfg(feature = "gpu")]
fn dynamic_moe_gpu(
    layer: &Dsv41Layer,
    cfg: &Dsv41Cfg,
    x: &[f32],
    pool: Option<&Pool>,
    logits: &[f32],
    picks: &[usize],
    weights: &[f32],
    layer_index: usize,
    n_layers: usize,
    state: &mut Dsv41State,
    out: &mut [f32],
) -> bool {
    if !crate::gpu::enabled_here()
        || !crate::gpu_wgpu::dsv4_global_moe_supported()
        || std::env::var("CMF_DSV41_DYNAMIC_MOE").as_deref() == Ok("0")
        || picks.len() != cfg.top_k
        || weights.len() != cfg.n_routed_experts
        || logits.len() != cfg.n_routed_experts
        || layer.gpu_expert_ids.len() != cfg.n_routed_experts
        || layer.gpu_shared_ids.is_none()
    {
        return false;
    }
    let first = match layer.experts.first() {
        Some(expert) => expert,
        None => return false,
    };
    let model = match first.w1.model_arc() {
        Some(model) => model,
        None => return false,
    };
    let gu_q2 = first.w1.model_dtype() == Some(TensorDtype::Q2TiledP)
        && first.w3.model_dtype() == Some(TensorDtype::Q2TiledP);
    if first.w2.model_dtype() != Some(TensorDtype::Q4TiledP) {
        return false;
    }
    // The global bank has one gate/up layout and one down layout.  Refuse a
    // mixed or synthetic layer rather than reinterpreting a tensor's bytes.
    let gu_dtype = if gu_q2 {
        TensorDtype::Q2TiledP
    } else {
        TensorDtype::Q4TiledP
    };
    let same_layout = layer.experts.iter().all(|expert| {
        expert
            .w1
            .model_arc()
            .is_some_and(|m| m.uid() == model.uid())
            && expert
                .w3
                .model_arc()
                .is_some_and(|m| m.uid() == model.uid())
            && expert
                .w2
                .model_arc()
                .is_some_and(|m| m.uid() == model.uid())
            && expert.w1.model_dtype() == Some(gu_dtype)
            && expert.w3.model_dtype() == Some(gu_dtype)
            && expert.w2.model_dtype() == Some(TensorDtype::Q4TiledP)
    });
    let shared_layout = layer
        .shared
        .w1
        .model_arc()
        .is_some_and(|m| m.uid() == model.uid())
        && layer
            .shared
            .w3
            .model_arc()
            .is_some_and(|m| m.uid() == model.uid())
        && layer
            .shared
            .w2
            .model_arc()
            .is_some_and(|m| m.uid() == model.uid())
        && layer.shared.w1.model_dtype() == Some(gu_dtype)
        && layer.shared.w3.model_dtype() == Some(gu_dtype)
        && layer.shared.w2.model_dtype() == Some(TensorDtype::Q4TiledP);
    if !same_layout || !shared_layout || layer_index >= n_layers {
        return false;
    }
    if state.gpu_pool.is_none() {
        state.gpu_pool = crate::qwen4_exp::QwenGpuPool::create_for_dsv41(
            &model,
            cfg.moe_inter,
            cfg.dim,
            n_layers,
            cfg.n_routed_experts,
            gu_q2,
        );
    }
    let (remap, shared_slot, segment_slots) = {
        let Some(gpu_pool) = state.gpu_pool.as_mut() else {
            return false;
        };
        let Some((remap, shared_slot)) = gpu_pool.ensure(
            &model,
            layer_index,
            picks,
            &layer.gpu_expert_ids,
            layer.gpu_shared_ids,
        ) else {
            return false;
        };
        (remap, shared_slot, gpu_pool.segment_slots)
    };
    let cold_ids: Vec<usize> = picks
        .iter()
        .copied()
        .filter(|&expert| remap.get(expert).copied() == Some(u32::MAX))
        .collect();
    let cold_jobs: Vec<(usize, f32)> = cold_ids
        .iter()
        .copied()
        .map(|expert| (expert, weights[expert]))
        .collect();
    let gpu_weights = crate::gpu_wgpu::Dsv4MoeW {
        router: &[],
        experts: &layer.gpu_expert_ids,
        logits,
        // In forced + preweighted mode this is V4.1's final route table;
        // the shader does not redo softplus, bias, or normalization.
        bias: Some(weights),
        mask: None,
        forced: Some(picks),
        remap: Some(&remap),
        global: Some(crate::gpu_wgpu::Dsv4GlobalMoe {
            pool_uid: model.uid(),
            shared_slot,
            segment_slots: segment_slots as u32,
        }),
        has_shared: true,
        shared_weight: 1.0,
        preweighted: true,
        qwen_softmax: false,
    };
    let geom = crate::gpu_wgpu::Dsv4MoeGeom {
        hidden: cfg.dim,
        inter: cfg.moe_inter,
        top_k: picks.len(),
        // We have already applied V4.1's route scale and denominator.
        route_scale: 1.0,
        swiglu_limit: cfg.swiglu_limit,
        gu_q2,
        bf16: true,
    };
    let mut gpu_out = vec![0.0f32; cfg.dim];
    let mut cold_from_gpu = Vec::new();
    let mut cold_x = Vec::new();
    let (frame_ok, cold_cpu) = std::thread::scope(|scope| {
        let cpu = (!cold_jobs.is_empty()).then(|| {
            // Cold experts deliberately stay on the host while resident
            // routes run on the device. Without the scope guard, QTensor's
            // small matvecs can re-enter the generic GPU backend and defeat
            // the intended overlap.
            scope.spawn(|| {
                crate::gpu::cpu_scope(|| dsv41_cold_experts_cpu(layer, cfg, x, &cold_jobs, pool))
            })
        });
        let ok = crate::gpu_wgpu::dsv4_moe_frame(
            &model,
            &gpu_weights,
            geom,
            x,
            &mut cold_from_gpu,
            &mut cold_x,
            None,
            None,
            &mut gpu_out,
        );
        let cpu_out = cpu
            .and_then(|job| job.join().ok())
            .unwrap_or_else(|| vec![0.0; cfg.dim]);
        (ok, cpu_out)
    });
    if !frame_ok
        || cold_from_gpu.len() != cold_jobs.len()
        || cold_from_gpu
            .iter()
            .map(|&(expert, _)| expert)
            .ne(cold_ids.iter().copied())
    {
        return false;
    }
    prof::note_cold(cold_jobs.len());
    for (dst, src) in gpu_out.iter_mut().zip(cold_cpu) {
        *dst += src;
    }
    // Match the source's BF16 boundary after routed and shared accumulation.
    bf16_inplace(&mut gpu_out);
    out[..cfg.dim].copy_from_slice(&gpu_out);
    true
}

fn moe(
    l: &Dsv41Layer,
    cfg: &Dsv41Cfg,
    x: &[f32],
    pool: Option<&Pool>,
    image: bool,
    position: usize,
    layer: usize,
    n_layers: usize,
    state: &mut Dsv41State,
    out: &mut [f32],
) {
    prof::note_moe();
    let mut logits = vec![0.0f32; cfg.n_routed_experts];
    matvec(&l.gate, x, &mut logits, pool);
    let bias = if image {
        l.gate_bias_vl.as_deref().unwrap_or(&l.gate_bias)
    } else {
        &l.gate_bias
    };
    let temperature = cfg.gate_temp.max(f32::MIN_POSITIVE);
    let scores: Vec<f32> = logits
        .iter()
        .map(|&v| sqrt_softplus(v / temperature))
        .collect();
    let mut shifted: Vec<f32> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| s + bias.get(i).copied().unwrap_or(0.0))
        .collect();
    let mut picks = Vec::with_capacity(cfg.top_k);
    for _ in 0..cfg.top_k.min(cfg.n_routed_experts) {
        let mut i = 0usize;
        let mut v = f32::NEG_INFINITY;
        for (j, &candidate) in shifted.iter().enumerate() {
            if candidate > v {
                i = j;
                v = candidate;
            }
        }
        if !v.is_finite() {
            break;
        }
        picks.push(i);
        shifted[i] = f32::NEG_INFINITY;
    }
    trace_moe(position, layer, image, &logits, &scores, bias, &picks);
    if let Some(cap) = state.moe_capture.as_mut() {
        if (cap.lo..cap.hi).contains(&layer) {
            let k = layer - cap.lo;
            cap.x[k].extend(x[..cfg.dim].iter().map(|&v| cortiq_core::quant::f32_to_f16(v)));
            cap.picks[k].extend(
                (0..cfg.top_k).map(|j| picks.get(j).map_or(u16::MAX, |&e| e as u16)),
            );
        }
    }
    // `weights /= weights.sum() + 1e-20` only when more than one expert is
    // chosen (Gate.forward); the epsilon is not norm_eps.
    let sum = picks.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20;
    let mut mix_weights = vec![0.0f32; cfg.n_routed_experts];
    if cfg.norm_topk_prob && cfg.top_k > 1 {
        for &e in &picks {
            mix_weights[e] = scores[e] / sum * cfg.route_scale;
        }
    } else {
        for &e in &picks {
            mix_weights[e] = scores[e] * cfg.route_scale;
        }
    }
    #[cfg(feature = "gpu")]
    if dynamic_moe_gpu(
        l,
        cfg,
        x,
        pool,
        &logits,
        &picks,
        &mix_weights,
        layer,
        n_layers,
        state,
        out,
    ) {
        prof::note_gpu_moe();
        return;
    }
    prof::note_cpu_moe();
    out.fill(0.0);
    let mut tmp = vec![0.0f32; cfg.dim];
    for &e in &picks {
        let ew = mix_weights[e];
        let ex = &l.experts[e];
        let mut g = vec![0.0f32; cfg.moe_inter];
        let mut u = vec![0.0f32; cfg.moe_inter];
        matvec_bf16(&ex.w1, x, &mut g, pool);
        matvec_bf16(&ex.w3, x, &mut u, pool);
        for i in 0..cfg.moe_inter {
            if cfg.swiglu_limit > 0.0 {
                g[i] = g[i].min(cfg.swiglu_limit);
                u[i] = u[i].clamp(-cfg.swiglu_limit, cfg.swiglu_limit);
            }
            g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i] * ew;
        }
        bf16_inplace(&mut g);
        matvec_bf16(&ex.w2, &g, &mut tmp, pool);
        for i in 0..cfg.dim {
            out[i] += tmp[i];
        }
    }
    let shared = &l.shared;
    let mut g = vec![0.0f32; cfg.moe_inter];
    let mut u = vec![0.0f32; cfg.moe_inter];
    matvec_bf16(&shared.w1, x, &mut g, pool);
    matvec_bf16(&shared.w3, x, &mut u, pool);
    for i in 0..cfg.moe_inter {
        if cfg.swiglu_limit > 0.0 {
            g[i] = g[i].min(cfg.swiglu_limit);
            u[i] = u[i].clamp(-cfg.swiglu_limit, cfg.swiglu_limit);
        }
        g[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
    }
    bf16_inplace(&mut g);
    matvec_bf16(&shared.w2, &g, &mut tmp, pool);
    for i in 0..cfg.dim {
        out[i] += tmp[i];
    }
    // MoE returns `y.type_as(x)` after accumulating routed and shared
    // branches in f32.
    bf16_inplace(out);
}

/// `F.softplus(x).sqrt()`: `log1p(exp(x))`, linear past torch's threshold
/// of 20. `ln_1p` keeps strongly negative logits positive; `(1 + e^x).ln()`
/// rounds them to an exact zero score below x ~ -17.
#[inline]
fn sqrt_softplus(v: f32) -> f32 {
    if v > 20.0 { v } else { v.exp().ln_1p() }.sqrt()
}

fn hc_mixes(
    x: &[f32],
    fn_w: &[f32],
    base: &[f32],
    scale: &[f32; 3],
    cfg: &Dsv41Cfg,
    pool: Option<&Pool>,
    pre: &mut [f32],
    post: &mut [f32],
    comb: &mut [f32],
) {
    let mix_n = (2 + cfg.hc_mult) * cfg.hc_mult;
    let mut mixes = vec![0.0f32; mix_n];
    crate::dsv4::hc_mixes(x, fn_w, mix_n, cfg.norm_eps, pool, &mut mixes);
    crate::dsv4::hc_split_sinkhorn(
        &mixes,
        scale,
        base,
        cfg.hc_mult,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        pre,
        post,
        comb,
    );
}

/// Run one token through the complete V4.1 stack.  `token_id` is used for
/// embedding and for Engram hashes; the returned vector is the final logits.
pub fn forward_token(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    token_id: u32,
    position: usize,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
) {
    forward_token_masked(
        globals, layers, cfg, st, token_id, position, true, pool, logits,
    )
}

pub fn forward_token_masked(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    token_id: u32,
    position: usize,
    participates: bool,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
) {
    forward_token_masked_with_embedding(
        globals,
        layers,
        cfg,
        st,
        token_id,
        position,
        participates,
        None,
        pool,
        logits,
        true,
    );
}

/// V4.1 image ingress uses the same text stack but replaces the embedding
/// row for image markers/projector outputs. `embedding` is `None` for normal
/// text tokens; `want_logits=false` is used for all non-final prefill rows.
pub fn forward_token_masked_with_embedding(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    token_id: u32,
    position: usize,
    participates: bool,
    embedding: Option<&[f32]>,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    forward_token_impl(
        globals,
        layers,
        cfg,
        st,
        token_id,
        position,
        participates,
        embedding,
        pool,
        logits,
        want_logits,
    );
}

fn forward_token_impl(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    token_id: u32,
    position: usize,
    participates: bool,
    embedding: Option<&[f32]>,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
    want_logits: bool,
) {
    let total_t0 = prof::start();
    prof::note_token(layers.len());
    st.pos = position;
    let hashes = st.hash.as_mut().map(|h| h.push(token_id, participates));
    let mut embed = vec![0.0f32; cfg.dim];
    if let Some(embedding) = embedding {
        assert_eq!(embedding.len(), cfg.dim, "V4.1 override embedding width");
        embed.copy_from_slice(embedding);
    } else {
        globals.embed.row_f32(
            (token_id as usize).min(globals.embed.rows().saturating_sub(1)),
            &mut embed,
        );
    }
    trace_stats("embed", position, None, &embed);
    let mut h = vec![0.0f32; cfg.hc_mult * cfg.dim];
    for copy in 0..cfg.hc_mult {
        h[copy * cfg.dim..(copy + 1) * cfg.dim].copy_from_slice(&embed);
    }
    let mut pre_mix = vec![0.0f32; cfg.hc_mult];
    pre_mix[0] = 1.0;
    for (li, l) in layers.iter().enumerate() {
        if let (Some(e), Some(all)) = (&l.engram, hashes.as_ref()) {
            if let Some(ix) = cfg.engram_layers.iter().position(|&id| id == li) {
                let engram_t0 = prof::start();
                apply_engram(e, &mut h, &all[ix], cfg, participates, pool);
                prof::add(&prof::ENGRAM_NS, engram_t0);
            }
        }
        let residual = h.clone();
        let mut ap = vec![0.0; cfg.hc_mult];
        let mut apo = vec![0.0; cfg.hc_mult];
        let mut ac = vec![0.0; cfg.hc_mult * cfg.hc_mult];
        hc_mixes(
            &h,
            &l.hc_attn_fn,
            &l.hc_attn_base,
            &l.hc_attn_scale,
            cfg,
            pool,
            &mut ap,
            &mut apo,
            &mut ac,
        );
        let mut folded = vec![0.0f32; cfg.dim];
        crate::dsv4::hc_fold(&h, &pre_mix, cfg.hc_mult, cfg.dim, &mut folded);
        rms(&mut folded, &l.attn_norm, cfg.norm_eps);
        let mut attn_out = vec![0.0f32; cfg.dim];
        let attn_t0 = prof::start();
        let _qr = attention(
            l,
            cfg,
            st,
            li,
            &folded,
            if cfg.ratio(li) > 0 {
                &globals.inv_freq_compress
            } else {
                &globals.inv_freq_window
            },
            pool,
            &mut attn_out,
        );
        prof::add(&prof::ATTN_NS, attn_t0);
        trace_stats("attn", position, Some(li), &attn_out);
        crate::dsv4::hc_expand(
            &attn_out,
            &residual,
            &apo,
            &ac,
            cfg.hc_mult,
            cfg.dim,
            &mut h,
        );
        bf16_inplace(&mut h);
        let residual2 = h.clone();
        let mut fp = vec![0.0; cfg.hc_mult];
        let mut fpo = vec![0.0; cfg.hc_mult];
        let mut fc = vec![0.0; cfg.hc_mult * cfg.hc_mult];
        hc_mixes(
            &h,
            &l.hc_ffn_fn,
            &l.hc_ffn_base,
            &l.hc_ffn_scale,
            cfg,
            pool,
            &mut fp,
            &mut fpo,
            &mut fc,
        );
        let mut ff = vec![0.0f32; cfg.dim];
        crate::dsv4::hc_fold(&h, &ap, cfg.hc_mult, cfg.dim, &mut folded);
        rms(&mut folded, &l.ffn_norm, cfg.norm_eps);
        // `participates=false` marks image span tokens.  Those tokens use
        // the VL routing correction bias while still bypassing Engram.
        let moe_t0 = prof::start();
        moe(
            l,
            cfg,
            &folded,
            pool,
            !participates,
            position,
            li,
            layers.len(),
            st,
            &mut ff,
        );
        prof::add(&prof::MOE_NS, moe_t0);
        crate::dsv4::hc_expand(&ff, &residual2, &fpo, &fc, cfg.hc_mult, cfg.dim, &mut h);
        bf16_inplace(&mut h);
        pre_mix.copy_from_slice(&fp);
        trace_stats("block", position, Some(li), &h);
        trace_stats("pre", position, Some(li), &pre_mix);
        if st
            .moe_capture
            .as_ref()
            .is_some_and(|c| c.stop_after_hi && li + 1 >= c.hi)
        {
            // Calibration capture: nothing at or above `hi` is recorded.
            logits.clear();
            prof::add(&prof::TOTAL_NS, total_t0);
            return;
        }
    }
    let head_t0 = prof::start();
    let mut final_h = vec![0.0f32; cfg.dim];
    crate::dsv4::hc_fold(&h, &pre_mix, cfg.hc_mult, cfg.dim, &mut final_h);
    rms(&mut final_h, &globals.norm, cfg.norm_eps);
    if want_logits {
        logits.resize(globals.head.rows(), 0.0);
        globals.head.matvec(&final_h, logits, pool);
        trace_stats("final", position, None, &final_h);
        trace_stats("logits", position, None, logits);
    } else {
        logits.clear();
    }
    prof::add(&prof::HEAD_NS, head_t0);
    prof::add(&prof::TOTAL_NS, total_t0);
}

/// Print the opt-in V4.1 stage profile from the CLI's normal completion path.
pub fn profile_report() {
    prof::report();
}

pub fn forward_chunk(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    ids: &[u32],
    start: usize,
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
) {
    for (i, &id) in ids.iter().enumerate() {
        forward_token_masked_with_embedding(
            globals,
            layers,
            cfg,
            st,
            id,
            start + i,
            true,
            None,
            pool,
            logits,
            i + 1 == ids.len(),
        );
    }
}

/// Prefill variant for multimodal ingress. Each optional row corresponds to
/// one token in `ids`; image rows are already produced by `VisionModel` and
/// are marked non-participating for Engram/vision routing.
pub fn forward_chunk_masked_with_embeddings(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    ids: &[u32],
    start: usize,
    embeddings: &[Option<Vec<f32>>],
    participates: &[bool],
    pool: Option<&Pool>,
    logits: &mut Vec<f32>,
) {
    assert_eq!(ids.len(), embeddings.len());
    assert_eq!(ids.len(), participates.len());
    for (i, &id) in ids.iter().enumerate() {
        forward_token_masked_with_embedding(
            globals,
            layers,
            cfg,
            st,
            id,
            start + i,
            participates[i],
            embeddings[i].as_deref(),
            pool,
            logits,
            i + 1 == ids.len(),
        );
    }
}

/// GPTQ calibration capture (`cortiq quantize-moe`): `ids` as a fresh
/// sequence from position 0, recording the MoE input rows and routed expert
/// ids of layers `lo..hi` ([`Dsv41MoeCapture`]); each token stops after
/// layer `hi - 1`. The sequence state is cleared before and after.
#[allow(clippy::too_many_arguments)]
pub fn capture_window(
    globals: &Dsv41Globals,
    layers: &[Dsv41Layer],
    cfg: &Dsv41Cfg,
    st: &mut Dsv41State,
    ids: &[u32],
    lo: usize,
    hi: usize,
    pool: Option<&Pool>,
) -> Result<Dsv41MoeCapture, String> {
    let hi = hi.min(layers.len());
    if lo >= hi {
        return Err(format!("capture range {lo}..{hi} is empty"));
    }
    if cfg.n_routed_experts > u16::MAX as usize {
        return Err(format!("{} experts do not fit a u16 id", cfg.n_routed_experts));
    }
    st.clear();
    st.moe_capture = Some(Dsv41MoeCapture {
        lo,
        hi,
        stop_after_hi: true,
        x: (lo..hi).map(|_| Vec::with_capacity(ids.len() * cfg.dim)).collect(),
        picks: (lo..hi).map(|_| Vec::with_capacity(ids.len() * cfg.top_k)).collect(),
    });
    let mut logits = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        forward_token_masked_with_embedding(
            globals,
            layers,
            cfg,
            st,
            id,
            pos,
            true,
            None,
            pool,
            &mut logits,
            false,
        );
    }
    let cap = st.moe_capture.take().unwrap_or_default();
    st.clear();
    for (k, (x, p)) in cap.x.iter().zip(&cap.picks).enumerate() {
        if x.len() != ids.len() * cfg.dim || p.len() != ids.len() * cfg.top_k {
            return Err(format!(
                "layer {}: captured {} rows / {} routes for {} tokens",
                lo + k,
                x.len() / cfg.dim.max(1),
                p.len() / cfg.top_k.max(1),
                ids.len()
            ));
        }
    }
    Ok(cap)
}

/// Compact, row-addressable CSA2 global KV storage.
///
/// V4.1 stores the post-RoPE main KV in MXFP4/E2M1 with one E4M3 scale
/// byte per 16 values.  Indexer K uses the same E2M1 nibbles with one E8M0
/// scale byte per 32 values.  Rows stay keyed by the shared CSA2 source;
/// callers can gather a selected set into one bounded scratch buffer and
/// reuse it for every attention head.
pub mod packed_kv {
    const FP4_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    const MAIN_BLOCK: usize = 16;
    const INDEX_BLOCK: usize = 32;

    /// A dense row store with two E2M1 values per byte and one scale byte per
    /// block.  `values` and `scales` are row-major, so an individual row can
    /// be decoded without touching any other history entry.
    #[derive(Clone, Debug)]
    pub struct PackedRows {
        cols: usize,
        block: usize,
        e8m0_scale: bool,
        value_bytes: usize,
        scale_bytes: usize,
        values: Vec<u8>,
        scales: Vec<u8>,
    }

    impl PackedRows {
        /// Construct a store. `cols` must be a multiple of `block` because
        /// the source MXFP formats carry exactly one scale for each block.
        pub fn new(cols: usize, block: usize, e8m0_scale: bool) -> Result<Self, String> {
            if cols == 0 || block == 0 || cols % block != 0 {
                return Err(format!(
                    "packed KV width {cols} must be a positive multiple of block {block}"
                ));
            }
            Ok(Self {
                cols,
                block,
                e8m0_scale,
                value_bytes: cols.div_ceil(2),
                scale_bytes: cols / block,
                values: Vec::new(),
                scales: Vec::new(),
            })
        }

        #[inline]
        pub fn cols(&self) -> usize {
            self.cols
        }

        #[inline]
        pub fn block(&self) -> usize {
            self.block
        }

        #[inline]
        pub fn rows(&self) -> usize {
            self.scales.len() / self.scale_bytes
        }

        #[inline]
        pub fn row_bytes(&self) -> usize {
            self.value_bytes + self.scale_bytes
        }

        #[inline]
        pub fn bytes(&self) -> usize {
            self.values.len() + self.scales.len()
        }

        #[inline]
        pub fn clear(&mut self) {
            self.values.clear();
            self.scales.clear();
        }

        /// Quantise and append one source row. The returned index is the
        /// stable logical row id used by `row_into` and `gather_rows`.
        /// Quantisation is deliberately performed here rather than by a
        /// caller-provided f32 mirror, so the packed representation itself is
        /// the canonical long-context storage.
        pub fn push(&mut self, row: &[f32]) -> usize {
            assert_eq!(row.len(), self.cols);
            let row_id = self.rows();
            let value_base = row_id * self.value_bytes;
            let scale_base = row_id * self.scale_bytes;
            self.values.resize(value_base + self.value_bytes, 0);
            self.scales.resize(scale_base + self.scale_bytes, 0);
            for block_id in 0..self.scale_bytes {
                let begin = block_id * self.block;
                let end = begin + self.block;
                let min_amax = if self.e8m0_scale {
                    6.0 * 2.0f32.powi(-126)
                } else {
                    6.0 * 2.0f32.powi(-9)
                };
                let max = row[begin..end]
                    .iter()
                    .map(|x| x.abs())
                    .fold(min_amax, f32::max);
                let raw_scale = max / 6.0;
                let scale = if self.e8m0_scale {
                    super::round_e8m0_scale(raw_scale)
                } else {
                    super::e4m3_round(raw_scale.max(6.0 * 2.0f32.powi(-9)))
                };
                self.scales[scale_base + block_id] = if self.e8m0_scale {
                    encode_e8m0_scale(scale)
                } else {
                    encode_e4m3_scale(scale)
                };
                for i in begin..end {
                    let code = fp4_code(row[i] / scale);
                    let dst = value_base + i / 2;
                    if i & 1 == 0 {
                        self.values[dst] = (self.values[dst] & 0xf0) | code;
                    } else {
                        self.values[dst] = (self.values[dst] & 0x0f) | (code << 4);
                    }
                }
            }
            row_id
        }

        /// Decode one row to the same BF16-materialised f32 values emitted by
        /// the source in-place quantiser. A signed E2M1 zero is retained.
        pub fn row_into(&self, row: usize, dst: &mut [f32]) -> bool {
            if row >= self.rows() || dst.len() != self.cols {
                return false;
            }
            let value_base = row * self.value_bytes;
            let scale_base = row * self.scale_bytes;
            for block_id in 0..self.scale_bytes {
                let scale_byte = self.scales[scale_base + block_id];
                let scale = if self.e8m0_scale {
                    super::e8m0_scale(scale_byte)
                } else {
                    super::fp8_e4m3(scale_byte)
                };
                let begin = block_id * self.block;
                let end = begin + self.block;
                for i in begin..end {
                    let code = if i & 1 == 0 {
                        self.values[value_base + i / 2] & 0x0f
                    } else {
                        self.values[value_base + i / 2] >> 4
                    };
                    let mag = FP4_LEVELS[(code & 0x07) as usize];
                    let signed = if code & 0x08 != 0 { -mag } else { mag };
                    // Multiplication by a positive scale does not reliably
                    // preserve a negative zero on every compiler path.
                    let mut value = signed * scale;
                    if mag == 0.0 && code & 0x08 != 0 {
                        value = -0.0;
                    }
                    dst[i] = super::bf16_roundtrip(value);
                }
            }
            true
        }

        /// Decode the requested rows once into contiguous row-major scratch.
        /// The caller can pass that same scratch to all attention heads.
        pub fn gather_rows(&self, rows: &[usize], dst: &mut [f32]) -> bool {
            if dst.len() != rows.len() * self.cols {
                return false;
            }
            for (i, &row) in rows.iter().enumerate() {
                if !self.row_into(row, &mut dst[i * self.cols..(i + 1) * self.cols]) {
                    return false;
                }
            }
            true
        }

        /// Decode all rows into caller-owned scratch. This is intended for
        /// bounded diagnostic/export paths; attention should use `gather_rows`.
        pub fn all_into(&self, dst: &mut [f32]) -> bool {
            if dst.len() != self.rows() * self.cols {
                return false;
            }
            self.gather_rows(&(0..self.rows()).collect::<Vec<_>>(), dst)
        }
    }

    /// The two CSA2 global streams. Main and index rows are each keyed by
    /// the source slot, so cross-layer reuse never duplicates history.
    #[derive(Clone, Debug)]
    pub struct Dsv41PackedKvCache {
        main: Vec<PackedRows>,
        index: Vec<PackedRows>,
    }

    impl Dsv41PackedKvCache {
        pub fn new(
            source_count: usize,
            main_cols: usize,
            index_cols: usize,
        ) -> Result<Self, String> {
            let mut main = Vec::with_capacity(source_count);
            let mut index = Vec::with_capacity(source_count);
            for _ in 0..source_count {
                main.push(PackedRows::new(main_cols, MAIN_BLOCK, false)?);
                index.push(PackedRows::new(index_cols, INDEX_BLOCK, true)?);
            }
            Ok(Self { main, index })
        }

        #[inline]
        pub fn main_store(&self, source: usize) -> Option<&PackedRows> {
            self.main.get(source)
        }

        #[inline]
        pub fn index_store(&self, source: usize) -> Option<&PackedRows> {
            self.index.get(source)
        }

        #[inline]
        pub fn main_rows(&self, source: usize) -> usize {
            self.main_store(source).map_or(0, PackedRows::rows)
        }

        #[inline]
        pub fn index_rows(&self, source: usize) -> usize {
            self.index_store(source).map_or(0, PackedRows::rows)
        }

        #[inline]
        pub fn main_row_bytes(&self, source: usize) -> usize {
            self.main_store(source).map_or(0, PackedRows::row_bytes)
        }

        #[inline]
        pub fn index_row_bytes(&self, source: usize) -> usize {
            self.index_store(source).map_or(0, PackedRows::row_bytes)
        }

        #[inline]
        pub fn main_bytes(&self) -> usize {
            self.main.iter().map(PackedRows::bytes).sum()
        }

        #[inline]
        pub fn index_bytes(&self) -> usize {
            self.index.iter().map(PackedRows::bytes).sum()
        }

        #[inline]
        pub fn bytes(&self) -> usize {
            self.main_bytes() + self.index_bytes()
        }

        pub fn clear(&mut self) {
            for store in &mut self.main {
                store.clear();
            }
            for store in &mut self.index {
                store.clear();
            }
        }

        pub fn push_main(&mut self, source: usize, row: &[f32]) -> Option<usize> {
            self.main.get_mut(source).map(|store| store.push(row))
        }

        pub fn push_index(&mut self, source: usize, row: &[f32]) -> Option<usize> {
            self.index.get_mut(source).map(|store| store.push(row))
        }

        pub fn main_row_into(&self, source: usize, row: usize, dst: &mut [f32]) -> bool {
            self.main_store(source)
                .is_some_and(|store| store.row_into(row, dst))
        }

        pub fn index_row_into(&self, source: usize, row: usize, dst: &mut [f32]) -> bool {
            self.index_store(source)
                .is_some_and(|store| store.row_into(row, dst))
        }

        /// Narrow GPU/CPU adapter: decode selected main rows exactly once.
        pub fn gather_main_rows(&self, source: usize, rows: &[usize], dst: &mut [f32]) -> bool {
            self.main_store(source)
                .is_some_and(|store| store.gather_rows(rows, dst))
        }

        /// Narrow indexer adapter; index scoring can stream one row at a time
        /// through `index_store` to avoid a history-sized f32 temporary.
        pub fn gather_index_rows(&self, source: usize, rows: &[usize], dst: &mut [f32]) -> bool {
            self.index_store(source)
                .is_some_and(|store| store.gather_rows(rows, dst))
        }
    }

    fn fp4_code(value: f32) -> u8 {
        let q = super::fp4_round(value);
        let sign = if q.is_sign_negative() { 0x08 } else { 0 };
        let aq = q.abs();
        let mag = FP4_LEVELS
            .iter()
            .position(|&level| level.to_bits() == aq.to_bits())
            .unwrap_or(7);
        sign | mag as u8
    }

    fn encode_e4m3_scale(scale: f32) -> u8 {
        // The source scale is already the result of e4m3_round. Exact lookup
        // avoids introducing a second tie rule in the storage adapter.
        (0u8..=0x7e)
            .find(|&byte| super::fp8_e4m3(byte).to_bits() == scale.to_bits())
            .unwrap_or_else(|| panic!("non-E4M3 scale {scale:?}"))
    }

    fn encode_e8m0_scale(scale: f32) -> u8 {
        if !scale.is_finite() || scale <= 0.0 {
            return 0;
        }
        let exponent = scale.log2().round() as i32;
        (exponent + 127).clamp(0, 254) as u8
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A fallback-only arch: every value differs from the release, so a key
    /// the parser fails to read from the source config shows up as a
    /// mismatch instead of silently agreeing.
    fn decoy_arch() -> cortiq_core::ModelArch {
        serde_json::from_value(serde_json::json!({
            "arch_name": "deepseek_v41",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_layers": 3,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "layer_types": ["FullAttention", "FullAttention", "FullAttention"],
            "rms_norm_eps": 0.5,
            "max_position_embeddings": 64,
            "linear_conv_kernel_dim": 0,
            "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
            "rope_theta": 7.0,
            "sliding_window": 3,
        }))
        .unwrap()
    }

    fn release_1017_cfg() -> Dsv41Cfg {
        let mut compress_ratios = vec![0, 0];
        compress_ratios.extend(std::iter::repeat_n(2, 18));
        compress_ratios.extend(std::iter::repeat_n(1, 20));
        Dsv41Cfg {
            dim: 5120,
            n_heads: 64,
            head_dim: 512,
            rope_head_dim: 64,
            q_lora_rank: 1280,
            o_lora_rank: 1024,
            o_groups: 8,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            norm_eps: 1e-20,
            n_routed_experts: 384,
            top_k: 6,
            moe_inter: 2304,
            gate_temp: 1.0,
            norm_topk_prob: true,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            window: 128,
            rope_theta: 10_000.0,
            compress_rope_theta: 160_000.0,
            rope_factor: 16.0,
            original_seq_len: 65_536,
            beta_fast: 32.0,
            beta_slow: 1.0,
            index_heads: 32,
            index_head_dim: 128,
            index_topk: 512,
            candidate_source: Some(20),
            candidate_topk_blocks: 2048,
            candidate_block_size: 8,
            kv_sources: vec![2, 8, 14, 20],
            index_sources: vec![2, 8, 14, 20, 24, 28, 32, 36],
            compress_ratios,
            engram_layers: vec![1, 14],
            engram_vocab: 16_000_000,
            engram_embeddings: vec![384_006_168, 384_016_682],
            engram_max_ngram: 4,
            engram_heads: 8,
            engram_head_dim: 256,
            engram_compressed_vocab: 99_092,
            engram_pad_id: 2,
            vocab: 129_280,
        }
    }

    /// The published DeepSeek-V4.1-Flash-1017 `config.json` (HF layout,
    /// `text_config`) and its `inference/config.json` (flat ModelArgs keys)
    /// both parse to the release geometry, independent of the arch
    /// fallbacks. The expected values were checked against the parser as it
    /// stood in cortiq 0.8.23.
    #[test]
    fn release_1017_configs_parse_to_release_geometry() {
        let hf: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/dsv41/config_1017.json")).unwrap();
        let flat: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/dsv41/inference_config_1017.json"
        ))
        .unwrap();
        for (name, source) in [("config.json", &hf), ("inference/config.json", &flat)] {
            let (cfg, n_layers) = Dsv41Cfg::from_source(source, &decoy_arch()).unwrap();
            assert_eq!(n_layers, 40, "{name}");
            assert_eq!(cfg, release_1017_cfg(), "{name}");
        }
    }

    fn release_text_config() -> serde_json::Value {
        let hf: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/dsv41/config_1017.json")).unwrap();
        hf["text_config"].clone()
    }

    #[test]
    fn scoring_func_other_than_sqrtsoftplus_is_refused() {
        for key in ["scoring_func", "score_func"] {
            for bad in [
                serde_json::json!("softmax"),
                serde_json::json!("sigmoid"),
                serde_json::json!(1),
            ] {
                let mut tc = release_text_config();
                tc.as_object_mut().unwrap().remove("scoring_func");
                tc[key] = bad.clone();
                let err = Dsv41Cfg::from_source(&tc, &decoy_arch()).unwrap_err();
                assert!(
                    err.contains("scoring_func") && err.contains("sqrtsoftplus"),
                    "{key}={bad}: {err}"
                );
            }
            let mut tc = release_text_config();
            tc.as_object_mut().unwrap().remove("scoring_func");
            tc[key] = serde_json::json!("sqrtsoftplus");
            assert!(Dsv41Cfg::from_source(&tc, &decoy_arch()).is_ok(), "{key}");
        }
        let mut tc = release_text_config();
        tc.as_object_mut().unwrap().remove("scoring_func");
        assert!(Dsv41Cfg::from_source(&tc, &decoy_arch()).is_ok(), "absent");
    }

    #[test]
    fn negative_candidate_source_disables_candidates() {
        for key in ["candidate_source_layer_id", "candidate_source_layer"] {
            let mut tc = release_text_config();
            tc.as_object_mut()
                .unwrap()
                .remove("candidate_source_layer_id");
            tc[key] = serde_json::json!(-1);
            let (cfg, _) = Dsv41Cfg::from_source(&tc, &decoy_arch()).unwrap();
            assert_eq!(cfg.candidate_source, None, "{key}");
            tc[key] = serde_json::json!(7);
            let (cfg, _) = Dsv41Cfg::from_source(&tc, &decoy_arch()).unwrap();
            assert_eq!(cfg.candidate_source, Some(7), "{key}");
            tc[key] = serde_json::json!("20");
            assert!(Dsv41Cfg::from_source(&tc, &decoy_arch()).is_err(), "{key}");
        }
        let mut tc = release_text_config();
        tc.as_object_mut()
            .unwrap()
            .remove("candidate_source_layer_id");
        let (cfg, _) = Dsv41Cfg::from_source(&tc, &decoy_arch()).unwrap();
        assert_eq!(cfg.candidate_source, Some(20), "absent keeps the release layer");
    }

    #[test]
    fn sqrt_softplus_keeps_negative_logits_positive() {
        for x in [-40.0f32, -30.0, -17.5, -3.0, 0.0, 1.5, 19.9, 20.0, 25.0] {
            let xd = x as f64;
            let want = if xd > 20.0 { xd } else { xd.exp().ln_1p() }.sqrt();
            let got = sqrt_softplus(x) as f64;
            assert!(got > 0.0, "x={x}");
            assert!((got - want).abs() <= want * 1e-6, "x={x}: {got} vs {want}");
        }
        // The old (1 + e^x).ln() form returned exactly zero here.
        assert_eq!((1.0f32 + (-30.0f32).exp()).ln(), 0.0);
    }

    /// kernel.py `act_quant(..., inplace=True)` written out independently:
    /// f64 arithmetic, an explicit E4M3FN grid with nearest-even ties, and
    /// the bit-level `fast_log2_ceil`/`fast_pow2` scale.
    fn act_quant_reference(block: &[f32]) -> Vec<f32> {
        let amax = block.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-4);
        let t = amax * (1.0f32 / 448.0);
        let bits = t.to_bits();
        let e = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
        let s = 2f64.powi(e);
        block
            .iter()
            .map(|&x| {
                let v = (x as f64 / s).clamp(-448.0, 448.0);
                let a = v.abs();
                let q = if a == 0.0 {
                    0.0
                } else {
                    let quantum = if a < 2f64.powi(-6) {
                        2f64.powi(-9)
                    } else {
                        2f64.powi(((a.to_bits() >> 52) & 0x7ff) as i32 - 1023 - 3)
                    };
                    let n = a / quantum;
                    let f = n.floor();
                    let r = if n - f > 0.5 || (n - f == 0.5 && f % 2.0 != 0.0) {
                        f + 1.0
                    } else {
                        f
                    };
                    (r * quantum).min(448.0)
                };
                let y = if v < 0.0 { -q } else { q } * s;
                // `T.Cast(bf16, ...)`: the product is BF16-exact already.
                assert_eq!(y as f32 as f64, y);
                y as f32
            })
            .collect()
    }

    #[test]
    fn fp8_activation_matches_kernel_act_quant() {
        let mut state = 0x5EED_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        // Random BF16 blocks over a wide dynamic range.
        for _ in 0..2000 {
            let exp = (next() % 40) as i32 - 30;
            blocks.push(
                (0..32)
                    .map(|_| {
                        let m = (next() % 65536) as f32 / 32768.0 - 1.0;
                        bf16_roundtrip(m * 2f32.powi(exp))
                    })
                    .collect(),
            );
        }
        // Edge blocks: all zeros (the 1e-4 amax floor), a signed zero, an
        // exact 448*2^k maximum (scale boundary), a maximum one f32 ulp above
        // it (where a rounded log2 picks the smaller scale), tiny values on
        // the E4M3 subnormal grid, and exact nearest-even ties.
        blocks.push(vec![0.0; 32]);
        let mut b = vec![-0.0f32; 32];
        b[3] = 1e-6;
        blocks.push(b);
        let mut b = vec![1.0f32; 32];
        b[0] = 448.0 * 2f32.powi(-3);
        blocks.push(b);
        let mut b = vec![0.3f32; 32];
        b[5] = f32::from_bits((448.0f32 * 2f32.powi(-10)).to_bits() + 1);
        // 1.5 subnormal quanta at the correct scale 2^-9 (tie -> 2), but an
        // exact 3 quanta at the 2^-10 a rounded log2 would choose.
        b[7] = 3.0 * 2f32.powi(-19);
        blocks.push(b);
        let mut b: Vec<f32> = (0..32).map(|i| i as f32 * 2f32.powi(-17)).collect();
        b[31] = 448.0 * 2f32.powi(-8);
        blocks.push(b);
        // With s = 1 (amax 448 / 2^0): 17 lies between 16 and 18 (tie ->
        // 16), 19 between 18 and 20 (tie -> 20), 0.0068359375 = 3.5*2^-9.
        let mut b = vec![17.0f32, 19.0, -17.0, 0.0068359375, 300.0, 448.0];
        b.resize(32, 1.0);
        blocks.push(b);
        for (i, block) in blocks.iter().enumerate() {
            let mut got = block.clone();
            fp8_activation_inplace(&mut got);
            let want = act_quant_reference(block);
            for (j, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), w.to_bits(), "block {i} value {j}: {g} vs {w}");
            }
        }
        let tie = &blocks[blocks.len() - 1];
        let mut q = tie.clone();
        fp8_activation_inplace(&mut q);
        assert_eq!(&q[..4], &[16.0, 20.0, -16.0, 4.0 * 2f32.powi(-9)]);
    }

    #[test]
    fn attention_output_quantizes_per_32_block() {
        // Two 64-wide groups: four independent blocks with their own scales.
        let mut o: Vec<f32> = (0..128)
            .map(|i| bf16_roundtrip(((i * 37 % 101) as f32 - 50.0) * 2f32.powi(-(i as i32 / 32) * 3)))
            .collect();
        let want: Vec<f32> = o.chunks(32).flat_map(act_quant_reference).collect();
        quantize_attention_output(&mut o, 2);
        assert_eq!(o, want);
    }

    #[test]
    fn fp8_reference_points() {
        assert_eq!(fp8_e4m3(0), 0.0);
        assert_eq!(fp8_e4m3(0x38), 1.0);
        assert_eq!(fp8_e4m3(0xb8), -1.0);
        assert_eq!(fp8_e4m3(0x3c), 1.5);
        assert_eq!(fp8_e4m3(0x7e), 448.0);
        assert!(fp8_e4m3(0x7f).is_nan());
        assert_eq!(e8m0_scale(127), 1.0);
        assert_eq!(e8m0_scale(128), 2.0);
    }

    #[test]
    fn fp8_rounds_finite_top_band_and_ties_even() {
        // Exponent field 15 has six finite mantissas.  The old scalar path
        // returned 448 for every value in this band, which corrupted 256..416.
        for (input, expected) in [
            (256.0, 256.0),
            (288.0, 288.0),
            (320.0, 320.0),
            (352.0, 352.0),
            (384.0, 384.0),
            (416.0, 416.0),
            (448.0, 448.0),
            (480.0, 448.0),
            (-320.0, -320.0),
        ] {
            assert_eq!(e4m3_round(input), expected, "input={input}");
        }
        // Half-way values choose the even mantissa, including the
        // subnormal-to-normal carry at 8 * 2^-9.
        let q = 2.0f32.powi(-9);
        assert_eq!(e4m3_round(0.5 * q), 0.0);
        assert_eq!(e4m3_round(1.5 * q), 2.0 * q);
        assert_eq!(e4m3_round(7.5 * q), 8.0 * q);
        assert_eq!(e4m3_round(272.0), 256.0);
        assert_eq!(e4m3_round(304.0), 320.0);
        assert_eq!(e4m3_round(432.0), 448.0);
    }

    #[test]
    fn fp4_rounds_ties_to_even_level() {
        assert_eq!(fp4_round(0.25), 0.0);
        assert_eq!(fp4_round(0.75), 1.0);
        assert_eq!(fp4_round(1.25), 1.0);
        assert_eq!(fp4_round(1.75), 2.0);
        assert_eq!(fp4_round(2.5), 2.0);
        assert_eq!(fp4_round(3.5), 4.0);
        assert_eq!(fp4_round(5.0), 4.0);
        assert_eq!(fp4_round(-0.75), -1.0);
    }

    #[test]
    fn release_hash_multipliers_are_stable() {
        assert_eq!(
            hash_multipliers(1, 4, 99092),
            [
                76632096046245,
                4839876093313,
                35959672319349,
                73987337458391
            ]
        );
        assert_eq!(
            hash_multipliers(14, 4, 99092),
            [
                67716810739261,
                51510806800915,
                30921347202721,
                82619226485591
            ]
        );
    }

    #[test]
    fn release_hash_layout_is_disjoint_and_stable() {
        let h = EngramHash::new(
            vec![1, 14],
            4,
            8,
            16_000_000,
            99_092,
            2,
            (0..99_092).map(|v| v as u32).collect(),
        )
        .unwrap();
        assert_eq!(
            h.primes[0][0],
            [
                16_000_057, 16_000_079, 16_000_081, 16_000_097, 16_000_121, 16_000_129, 16_000_133,
                16_000_183,
            ]
        );
        // `offsets[n]` is the flattened start of n-gram order n.  Each
        // order owns eight disjoint prime ranges, so the starts are the
        // cumulative sums before orders 1, 2, and 3 in the oracle layout.
        assert_eq!(h.offsets[0], [0, 128_000_880, 256_002_934]);
        assert_eq!(h.primes[1][2][7], 16_000_889);
        assert_eq!(h.offsets[1], [0, 128_004_290, 256_009_984]);
    }

    #[test]
    fn dead_tokens_break_ngrams() {
        let map = (0..16).collect::<Vec<u32>>();
        let mut h = EngramHash::new(vec![1], 4, 2, 16, 16, 2, map).unwrap();
        let first = h.push(3, true);
        let second = h.push(4, false);
        let third = h.push(5, true);
        assert_ne!(first, second);
        assert_ne!(second, third);
        assert_eq!(h.history.last().copied(), Some(5));
    }

    #[test]
    fn candidate_mask_is_bounded() {
        let mut scores = vec![f32::NEG_INFINITY; 32];
        scores[1] = 1.0;
        scores[2] = 2.0;
        scores[9] = 3.0;
        scores[31] = 0.0;
        let m = candidate_blocks(&scores, 2, 8);
        assert_eq!(m.len(), 32);
        assert!(m.iter().filter(|&&x| x).count() <= 24);
    }

    #[test]
    fn split_attention_matches_materialized_stream() {
        let window = vec![
            0.10, 0.20, 0.30, 0.40, // window row 0
            -0.20, 0.30, 0.50, -0.10, // window row 1
        ];
        let compressed = vec![
            0.70, -0.40, 0.10, 0.20, // compressed row 0
            -0.30, 0.60, 0.20, 0.90, // compressed row 1
        ];
        let idxs = [0, 2, 3];
        let mut stream = window.clone();
        stream.extend_from_slice(&compressed);
        let q = [0.25, -0.10, 0.40, 0.05];
        let mut expected = [0.0; 4];
        crate::dsv4::sparse_attend(&q, &stream, &idxs, -0.2, 0.5, 4, &mut expected);
        let mut actual = [0.0; 4];
        sparse_attend_split(
            &q,
            &window,
            Some(&compressed),
            &idxs,
            -0.2,
            0.5,
            2,
            4,
            &mut actual,
        );
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-7, "split={a} materialized={b}");
        }
    }

    #[test]
    fn packed_rows_match_source_fp4_and_preserve_signed_zero() {
        let mut main_input = (0..32)
            .map(|i| (i as f32 - 13.0) * 0.137)
            .collect::<Vec<_>>();
        main_input[0] = -0.0;
        main_input[1] = 0.25;
        main_input[16] = 2688.0;
        let mut main_ref = main_input.clone();
        fp4_inplace(&mut main_ref, 16, false);
        let mut main = packed_kv::PackedRows::new(32, 16, false).unwrap();
        assert_eq!(main.push(&main_input), 0);
        let mut main_decoded = vec![0.0f32; 32];
        assert!(main.row_into(0, &mut main_decoded));
        for (i, (&expected, &actual)) in main_ref.iter().zip(&main_decoded).enumerate() {
            assert_eq!(expected.to_bits(), actual.to_bits(), "main row value {i}");
        }
        assert!(main_decoded[0].is_sign_negative());

        let mut index_input = (0..64)
            .map(|i| ((i as f32 * 0.03125).sin()) * 3.0)
            .collect::<Vec<_>>();
        index_input[32] = -0.0;
        let mut index_ref = index_input.clone();
        fp4_inplace(&mut index_ref, 32, true);
        let mut index = packed_kv::PackedRows::new(64, 32, true).unwrap();
        index.push(&index_input);
        let mut index_decoded = vec![0.0f32; 64];
        assert!(index.row_into(0, &mut index_decoded));
        for (i, (&expected, &actual)) in index_ref.iter().zip(&index_decoded).enumerate() {
            assert_eq!(expected.to_bits(), actual.to_bits(), "index row value {i}");
        }
        assert!(index_decoded[32].is_sign_negative());
        assert_eq!(main.row_bytes(), 18);
        assert_eq!(index.row_bytes(), 34);
    }

    #[test]
    fn packed_cache_gathers_once_and_tracks_source_rows() {
        let mut cache = packed_kv::Dsv41PackedKvCache::new(2, 32, 64).unwrap();
        let a = (0..32).map(|i| i as f32 * 0.01).collect::<Vec<_>>();
        let b = (0..32).map(|i| -(i as f32) * 0.02).collect::<Vec<_>>();
        let k = (0..64)
            .map(|i| (i as f32 - 20.0) * 0.03)
            .collect::<Vec<_>>();
        assert_eq!(cache.push_main(1, &a), Some(0));
        assert_eq!(cache.push_main(1, &b), Some(1));
        assert_eq!(cache.push_index(1, &k), Some(0));
        assert_eq!(cache.main_rows(0), 0);
        assert_eq!(cache.main_rows(1), 2);
        assert_eq!(cache.index_rows(1), 1);
        let mut gathered = vec![0.0f32; 64];
        assert!(cache.gather_main_rows(1, &[1, 0], &mut gathered));
        let mut row = vec![0.0f32; 32];
        assert!(cache.main_row_into(1, 1, &mut row));
        assert_eq!(&gathered[..32], &row[..]);
        assert_eq!(cache.main_row_bytes(1), 18);
        assert_eq!(cache.index_row_bytes(1), 34);
        assert_eq!(cache.bytes(), 2 * 18 + 34);
    }

    #[test]
    fn v41_global_row_bytes_match_890_byte_slope() {
        let cache = packed_kv::Dsv41PackedKvCache::new(4, 512, 128).unwrap();
        assert_eq!(cache.main_row_bytes(0), 512 / 2 + 512 / 16);
        assert_eq!(cache.index_row_bytes(0), 128 / 2 + 128 / 32);
        // The release source has 2.5 global source rows per token across its
        // four CSA2 source layers: 2.5 * (288 + 68) = 890 bytes/token.
        assert_eq!(
            2.5 * (cache.main_row_bytes(0) + cache.index_row_bytes(0)) as f32,
            890.0
        );
    }

    #[test]
    fn fp4_scales_keep_reference_zero_floors() {
        assert_eq!(round_e8m0_scale(2.0f32.powi(-126)), 2.0f32.powi(-126));
        assert_eq!(e4m3_round(2.0f32.powi(-9)), 2.0f32.powi(-9));
        let mut index = vec![0.0; 32];
        fp4_inplace(&mut index, 32, true);
        // E8M0's all-zero block floor is 2^-126, rather than the generic
        // activation floor used by the FP8 path.
        assert_eq!(index, vec![0.0; 32]);
        let mut compressed = vec![0.0; 16];
        fp4_inplace(&mut compressed, 16, false);
        assert_eq!(compressed, vec![0.0; 16]);
    }

    /// A tiny window-only V4.1 stack of f32 tensors (no compressor, indexer
    /// or Engram): enough for the full per-token forward on the host.
    fn tiny_stack(n_layers: usize) -> (Dsv41Globals, Vec<Dsv41Layer>, Dsv41Cfg) {
        let cfg = Dsv41Cfg {
            dim: 32,
            n_heads: 2,
            head_dim: 16,
            rope_head_dim: 4,
            q_lora_rank: 8,
            o_lora_rank: 4,
            o_groups: 2,
            hc_mult: 2,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-6,
            norm_eps: 1e-6,
            n_routed_experts: 8,
            top_k: 2,
            moe_inter: 16,
            gate_temp: 1.0,
            norm_topk_prob: true,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            window: 8,
            rope_theta: 10_000.0,
            compress_rope_theta: 160_000.0,
            rope_factor: 1.0,
            original_seq_len: 64,
            beta_fast: 32.0,
            beta_slow: 1.0,
            index_heads: 1,
            index_head_dim: 32,
            index_topk: 4,
            candidate_source: None,
            candidate_topk_blocks: 2,
            candidate_block_size: 8,
            kv_sources: Vec::new(),
            index_sources: Vec::new(),
            compress_ratios: vec![0; n_layers],
            engram_layers: Vec::new(),
            engram_vocab: 0,
            engram_embeddings: Vec::new(),
            engram_max_ngram: 4,
            engram_heads: 1,
            engram_head_dim: 16,
            engram_compressed_vocab: 0,
            engram_pad_id: 2,
            vocab: 16,
        };
        let mut seed = 0x5eed_u64;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = splitmix64(seed);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
                })
                .collect()
        };
        let q = |v: Vec<f32>, r: usize, c: usize| QTensor::from_f32(v, r, c);
        let (d, hc) = (cfg.dim, cfg.hc_mult);
        let mix_n = (2 + hc) * hc;
        let globals = Dsv41Globals {
            embed: q(rnd(cfg.vocab * d, 1.0), cfg.vocab, d),
            norm: vec![1.0; d],
            head: q(rnd(cfg.vocab * d, 0.3), cfg.vocab, d),
            inv_freq_compress: vec![1.0, 0.01],
            inv_freq_window: vec![1.0, 0.01],
        };
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let mut expert = |rnd: &mut dyn FnMut(usize, f32) -> Vec<f32>| Dsv41Expert {
                w1: q(rnd(cfg.moe_inter * d, 0.4), cfg.moe_inter, d),
                w2: q(rnd(d * cfg.moe_inter, 0.4), d, cfg.moe_inter),
                w3: q(rnd(cfg.moe_inter * d, 0.4), cfg.moe_inter, d),
            };
            let experts = (0..cfg.n_routed_experts).map(|_| expert(&mut rnd)).collect();
            let shared = expert(&mut rnd);
            let per_group = cfg.n_heads * cfg.head_dim / cfg.o_groups;
            layers.push(Dsv41Layer {
                attn_norm: vec![1.0; d],
                ffn_norm: vec![1.0; d],
                wq_a: q(rnd(cfg.q_lora_rank * d, 0.4), cfg.q_lora_rank, d),
                q_norm: vec![1.0; cfg.q_lora_rank],
                wq_b: q(rnd(cfg.n_heads * cfg.head_dim * cfg.q_lora_rank, 0.4), cfg.n_heads * cfg.head_dim, cfg.q_lora_rank),
                wkv: q(rnd(cfg.head_dim * d, 0.4), cfg.head_dim, d),
                kv_norm: vec![1.0; cfg.head_dim],
                wo_a: q(rnd(cfg.o_groups * cfg.o_lora_rank * per_group, 0.4), cfg.o_groups * cfg.o_lora_rank, per_group),
                wo_b: q(rnd(d * cfg.o_groups * cfg.o_lora_rank, 0.4), d, cfg.o_groups * cfg.o_lora_rank),
                attn_sink: vec![0.0; cfg.n_heads],
                compressor: None,
                indexer: None,
                hc_attn_fn: rnd(mix_n * hc * d, 0.1),
                hc_attn_base: rnd(mix_n, 0.5),
                hc_attn_scale: [1.0, 1.0, 1.0],
                hc_ffn_fn: rnd(mix_n * hc * d, 0.1),
                hc_ffn_base: rnd(mix_n, 0.5),
                hc_ffn_scale: [1.0, 1.0, 1.0],
                gate: q(rnd(cfg.n_routed_experts * d, 1.0), cfg.n_routed_experts, d),
                gate_bias: rnd(cfg.n_routed_experts, 0.2),
                gate_bias_vl: None,
                experts,
                shared,
                engram: None,
                gpu_expert_ids: Vec::new(),
                gpu_shared_ids: None,
            });
        }
        (globals, layers, cfg)
    }

    /// The quantize-moe capture records the MoE input and the route of
    /// every token at the captured layers through the real forward; ending
    /// each token after the last captured layer leaves the record unchanged.
    #[test]
    fn moe_capture_records_inputs_and_routes() {
        let (g, layers, cfg) = tiny_stack(3);
        let ids: Vec<u32> = (0..13).map(|i| (i * 5 % 16) as u32).collect();
        let mut st = Dsv41State::new(&cfg, None);
        let cap = capture_window(&g, &layers, &cfg, &mut st, &ids, 1, 2, None).unwrap();
        assert_eq!((cap.lo, cap.hi), (1, 2));
        assert_eq!(cap.x.len(), 1);
        assert_eq!(cap.x[0].len(), ids.len() * cfg.dim);
        assert_eq!(cap.picks[0].len(), ids.len() * cfg.top_k);
        assert!(st.moe_capture.is_none(), "the capture is handed back, not left on the state");

        // the same record through the whole stack (no early stop)
        st.clear();
        st.moe_capture = Some(Dsv41MoeCapture {
            lo: 0,
            hi: 2,
            stop_after_hi: false,
            x: vec![Vec::new(); 2],
            picks: vec![Vec::new(); 2],
        });
        let mut logits = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            forward_token_masked_with_embedding(&g, &layers, &cfg, &mut st, id, pos, true, None, None, &mut logits, true);
        }
        assert_eq!(logits.len(), cfg.vocab, "a full forward still produces logits");
        let full = st.moe_capture.take().unwrap();
        assert_eq!(full.x[1], cap.x[0], "early stop changes no captured row");
        assert_eq!(full.picks[1], cap.picks[0]);
        assert_ne!(full.x[0], full.x[1], "layers capture their own inputs");

        // the route is V4.1's: sqrt-softplus scores + bias, top-k, from the captured row
        let l = &layers[1];
        for t in 0..ids.len() {
            let x: Vec<f32> = cap.x[0][t * cfg.dim..(t + 1) * cfg.dim]
                .iter()
                .map(|&h| cortiq_core::quant::f16_to_f32(h))
                .collect();
            let mut lg = vec![0f32; cfg.n_routed_experts];
            l.gate.matvec(&x, &mut lg, None);
            let mut sh: Vec<(f32, usize)> = lg
                .iter()
                .enumerate()
                .map(|(e, &v)| ((1.0 + v.exp()).ln().sqrt() + l.gate_bias[e], e))
                .collect();
            sh.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let want: Vec<u16> = sh[..cfg.top_k].iter().map(|&(_, e)| e as u16).collect();
            assert_eq!(&cap.picks[0][t * cfg.top_k..(t + 1) * cfg.top_k], &want[..], "token {t}");
        }
    }
}
