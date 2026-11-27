//! GigaAM-v4 speech recognition (ai-sage/GigaAM-v4): a Zapformer encoder
//! over 64 log-mel bins with two recognition heads on top — `e2e` (case,
//! punctuation, digits) and `raw` (lower-case words) — as RNN-T or CTC.
//!
//! The encoder is three stacks (768 → 1024 at half the frame rate → 768)
//! behind a causal convolutional front end that takes 100 mel frames a
//! second down to 25. A layer is: feed-forward, gated rotary self-attention
//! whose queries and keys come from the layer's input, a gated depthwise
//! convolution with a global mean, a second feed-forward, and a blend with
//! the layer input under an RMS norm.
//!
//! A clip of up to 30 s is one segment. A longer recording is cut by the
//! file's voice-activity model ([`vad`]) into speech segments of at most
//! 30 s, and the segments go through the encoder as one ragged batch: every
//! matrix product sees all their frames at once, attention and the
//! convolutions stay inside a segment, and nothing is padded.
//!
//! Products run on the selected device when one is up (`CMF_GPU`), and on
//! the host through `ops::matmul` otherwise: weight rows are widened a
//! cache-sized tile at a time and multiplied by the frame batch with a
//! float GEMM.

mod decode;
mod ops;
pub mod vad;

use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::CmfModel;
use cortiq_core::TensorDtype;
use cortiq_core::quant::{bf16_to_f32, f16_to_f32};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

pub use decode::Word;

/// The architecture name in the file header.
pub const ARCH_NAME: &str = "gigaam";
/// Sample rate of the model's input.
pub const SR: usize = 16_000;
/// Samples in one segment at most: a longer clip is cut by the VAD.
pub const MAX_SEGMENT: usize = 30 * SR;

const N_FFT: usize = 320;
const HOP: usize = 160;
const QDIM: usize = 32;
const ENC: &str = "model.encoder.encoder";
const EMBED: &str = "model.encoder.encoder_embed";

/// Is this file a GigaAM checkpoint?
pub fn is_gigaam(model: &CmfModel) -> bool {
    model.header.arch.arch_name == ARCH_NAME
}

// ---------------------------------------------------------------------
// tensors
// ---------------------------------------------------------------------

pub(crate) fn tensor_f32(m: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let e = m.tensor(name).ok_or_else(|| format!("missing GigaAM tensor {name}"))?;
    let b = m.entry_bytes(e);
    let n = e.n_elems();
    match e.dtype {
        TensorDtype::F32 if b.len() == n * 4 => Ok(b.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect()),
        TensorDtype::F16 if b.len() == n * 2 => Ok(b.chunks_exact(2).map(|x| f16_to_f32(u16::from_le_bytes([x[0], x[1]]))).collect()),
        TensorDtype::Bf16 if b.len() == n * 2 => Ok(b.chunks_exact(2).map(|x| bf16_to_f32(u16::from_le_bytes([x[0], x[1]]))).collect()),
        _ => Err(format!("{name}: expected a float tensor, got {:?}", e.dtype)),
    }
}

fn scalar(m: &CmfModel, name: &str) -> Result<f32, String> {
    tensor_f32(m, name)?.first().copied().ok_or_else(|| format!("{name}: empty"))
}

/// How the host multiplies by a weight.
enum HostW {
    /// floats in memory (an exact file, or a small matrix)
    Dense(Vec<f32>),
    /// the file's int8 codes, with the row scales and the column field
    Q8 { row_scale: Vec<f32>, col_field: Vec<f32> },
    /// any other codec, through the tensor's own row decoder
    Rows,
}

/// One product `y = x·Wᵀ + b` over a batch of rows.
pub(crate) struct Lin {
    model: Arc<CmfModel>,
    idx: usize,
    pub rows: usize,
    pub cols: usize,
    host: HostW,
    bias: Vec<f32>,
    /// the tensor as the device paths take it (built on first use)
    dev: OnceLock<Option<QTensor>>,
}

impl Lin {
    fn load(m: &Arc<CmfModel>, prefix: &str) -> Result<Self, String> {
        Self::load_named(m, &format!("{prefix}.weight"), Some(&format!("{prefix}.bias")))
    }

    fn load_named(m: &Arc<CmfModel>, weight: &str, bias: Option<&str>) -> Result<Self, String> {
        let idx = m.tensor_index(weight).ok_or_else(|| format!("missing GigaAM tensor {weight}"))?;
        let e = &m.tensors[idx];
        // a 1×1 convolution is a matrix with a trailing unit axis
        let shape: Vec<usize> = e.shape.iter().copied().filter(|d| *d != 1).collect();
        let (rows, cols) = match (e.shape.len(), shape.as_slice()) {
            (2, _) => (e.shape[0], e.shape[1]),
            (_, [r, c]) => (*r, *c),
            (_, [c]) => (1, *c),
            _ => return Err(format!("{weight}: expected a matrix, got {:?}", e.shape)),
        };
        let host = match e.dtype {
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::Bf16 => HostW::Dense(tensor_f32(m, weight)?),
            TensorDtype::Q8Row | TensorDtype::Q8_2f if rows * cols < 1 << 16 => {
                // a small matrix: floats cost nothing and skip the tile
                let q = QTensor::from_model(m, weight)?;
                let mut d = vec![0f32; rows * cols];
                for (r, row) in d.chunks_exact_mut(cols).enumerate() {
                    q.row_f32(r, row);
                }
                HostW::Dense(d)
            }
            TensorDtype::Q8Row | TensorDtype::Q8_2f => {
                let b = m.entry_bytes(e);
                let f16s = |off: usize, n: usize| -> Vec<f32> {
                    (0..n).map(|i| f16_to_f32(u16::from_le_bytes([b[off + 2 * i], b[off + 2 * i + 1]]))).collect()
                };
                let row_scale = f16s(rows * cols, rows);
                let col_field = if e.dtype == TensorDtype::Q8_2f { f16s(rows * cols + rows * 2, cols) } else { Vec::new() };
                HostW::Q8 { row_scale, col_field }
            }
            _ => HostW::Rows,
        };
        let bias = match bias {
            Some(b) if m.tensor(b).is_some() => tensor_f32(m, b)?,
            _ => Vec::new(),
        };
        if !bias.is_empty() && bias.len() != rows {
            return Err(format!("{weight}: bias of {} for {rows} rows", bias.len()));
        }
        let mut lin = Self { model: m.clone(), idx, rows, cols, host, bias, dev: OnceLock::new() };
        if matches!(lin.host, HostW::Rows) && lin.qtensor().is_none() {
            return Err(format!("{weight}: codec {:?} is not supported here", e.dtype));
        }
        // `CMF_GIGAAM_DENSE=1`: keep every weight as floats in memory (four
        // bytes a weight, about 0.95 GB for the model) — a short clip then
        // pays no widening at all; a four-bit file is always expanded on the
        // host, its row decoder being slower than the product it feeds.
        let dense_all = std::env::var("CMF_GIGAAM_DENSE").is_ok_and(|v| v == "1");
        let host_only = !crate::gpu::enabled_here();
        if (dense_all && matches!(lin.host, HostW::Q8 { .. })) || (matches!(lin.host, HostW::Rows) && (dense_all || host_only)) {
            let mut d = vec![0f32; rows * cols];
            crate::gpu::GaWeight::fill(&lin, &mut d);
            lin.host = HostW::Dense(d);
        }
        Ok(lin)
    }

    /// The weight as a `QTensor` over the file (quantized codecs only).
    fn qtensor(&self) -> Option<&QTensor> {
        self.dev
            .get_or_init(|| {
                let e = &self.model.tensors[self.idx];
                if e.shape.len() != 2 || matches!(self.host, HostW::Dense(_)) {
                    return None;
                }
                QTensor::from_model(&self.model, &e.name).ok()
            })
            .as_ref()
    }

    /// This product for a device that keeps the weight itself.
    fn ga(&self) -> crate::gpu::GaLin<'_> {
        crate::gpu::GaLin { rows: self.rows, cols: self.cols, weight: self, bias: &self.bias }
    }

    /// `y[n, rows] = x[n, cols]·Wᵀ + b`.
    fn apply(&self, x: &[f32], n: usize, ex: &Exec) -> Vec<f32> {
        debug_assert_eq!(x.len(), n * self.cols);
        let mut y = vec![0f32; n * self.rows];
        let t0 = ex.prof.then(Instant::now);
        let on_device = ex.gpu
            && n >= 8
            && n * self.rows * self.cols >= ex.gpu_min_macs
            && self.qtensor().is_some_and(|q| q.device_matmat(x, n, &mut y));
        if !on_device {
            let w = match &self.host {
                HostW::Dense(d) => ops::Weight::Dense(d),
                HostW::Q8 { row_scale, col_field } => {
                    let e = &self.model.tensors[self.idx];
                    ops::Weight::Q8 { codes: &self.model.entry_bytes(e)[..self.rows * self.cols], row_scale, col_field }
                }
                HostW::Rows => ops::Weight::Rows(self.qtensor().expect("checked at load")),
            };
            ops::matmul(&w, self.rows, self.cols, x, n, &mut y, ex.pool);
        }
        if let Some(t0) = t0 {
            ex.add(if on_device { Slot::GemmDev } else { Slot::GemmHost }, t0);
        }
        if !self.bias.is_empty() {
            let (rows, yp) = (self.rows, ops::Shared(y.as_mut_ptr()));
            ops::par(ex.pool, n, 256, &|a, b| {
                // SAFETY: disjoint row ranges of `y`.
                let part = unsafe { yp.slice(a * rows, (b - a) * rows) };
                for row in part.chunks_exact_mut(rows) {
                    ops::axpy(row, 1.0, &self.bias);
                }
            });
        }
        y
    }
}

impl crate::gpu::GaWeight for Lin {
    fn fill(&self, dst: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        debug_assert_eq!(dst.len(), rows * cols);
        match &self.host {
            HostW::Dense(d) => dst.copy_from_slice(d),
            HostW::Q8 { row_scale, col_field } => {
                let codes = &self.model.entry_bytes(&self.model.tensors[self.idx])[..rows * cols];
                for (r, row) in dst.chunks_exact_mut(cols).enumerate() {
                    ops::widen_i8(row, &codes[r * cols..(r + 1) * cols]);
                    let rs = row_scale[r];
                    if col_field.is_empty() {
                        row.iter_mut().for_each(|v| *v *= rs);
                    } else {
                        row.iter_mut().zip(col_field).for_each(|(v, &c)| *v *= rs * c);
                    }
                }
            }
            HostW::Rows => {
                let q = self.qtensor().expect("checked at load");
                for (r, row) in dst.chunks_exact_mut(cols).enumerate() {
                    q.row_f32(r, row);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// execution context and timers
// ---------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Slot {
    GemmHost,
    GemmDev,
    Attention,
    Conv,
    Glue,
}
const SLOTS: usize = 5;
const SLOT_NAMES: [&str; SLOTS] = ["products (host)", "products (device)", "attention", "convolution", "activations and norms"];

/// What a forward pass runs on.
pub(crate) struct Exec<'a> {
    pub pool: Option<&'a Pool>,
    /// send products and attention to the device
    pub gpu: bool,
    /// a product smaller than this many multiply-adds stays on the host
    pub gpu_min_macs: usize,
    /// try the whole encoder on the device before the per-product path
    pub resident: bool,
    prof: bool,
    times: [std::sync::atomic::AtomicU64; SLOTS],
}

impl<'a> Exec<'a> {
    pub(crate) fn new(pool: Option<&'a Pool>) -> Self {
        let gpu = crate::gpu::enabled_here() && std::env::var("CMF_GIGAAM_GPU").as_deref() != Ok("0");
        let gpu_min_macs = std::env::var("CMF_GIGAAM_GPU_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 23);
        Self {
            pool,
            gpu,
            gpu_min_macs,
            resident: gpu && std::env::var("CMF_GIGAAM_RESIDENT").as_deref() != Ok("0"),
            prof: std::env::var("CMF_GIGAAM_PROF").is_ok_and(|v| v == "1"),
            times: Default::default(),
        }
    }

    fn add(&self, slot: Slot, since: Instant) {
        self.times[slot as usize].fetch_add(since.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    fn timed<R>(&self, slot: Slot, f: impl FnOnce() -> R) -> R {
        if !self.prof {
            return f();
        }
        let t0 = Instant::now();
        let r = f();
        self.add(slot, t0);
        r
    }

    fn report(&self) {
        if !self.prof {
            return;
        }
        let parts: Vec<String> = (0..SLOTS)
            .map(|i| (SLOT_NAMES[i], self.times[i].swap(0, std::sync::atomic::Ordering::Relaxed)))
            .filter(|(_, ns)| *ns > 0)
            .map(|(name, ns)| format!("{name} {:.1} ms", ns as f64 / 1e6))
            .collect();
        eprintln!("gigaam encoder: {}", parts.join(" | "));
    }
}

// ---------------------------------------------------------------------
// front end: log-mel and the convolutional subsampling
// ---------------------------------------------------------------------

/// The mel front end: 20 ms Hann windows every 10 ms, no centering, the
/// power spectrum through the checkpoint's own filter bank, natural log.
struct Mel {
    window: Vec<f32>,
    /// per mel bin: `(first frequency bin, weights)`
    bands: Vec<(usize, Vec<f32>)>,
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl Mel {
    fn load(m: &CmfModel, prefix: &str) -> Result<Self, String> {
        let window = tensor_f32(m, &format!("{prefix}.featurizer.0.spectrogram.window"))?;
        if window.len() != N_FFT {
            return Err(format!("GigaAM front end: a window of {} samples (expected {N_FFT})", window.len()));
        }
        let fb_name = format!("{prefix}.featurizer.0.mel_scale.fb");
        let fb = tensor_f32(m, &fb_name)?;
        let shape = &m.tensor(&fb_name).unwrap().shape;
        let bins = N_FFT / 2 + 1;
        if shape.len() != 2 || shape[0] != bins {
            return Err(format!("{fb_name}: shape {shape:?}, expected [{bins}, mel]"));
        }
        let n_mel = shape[1];
        let bands = (0..n_mel)
            .map(|mb| {
                let col: Vec<f32> = (0..bins).map(|k| fb[k * n_mel + mb]).collect();
                let first = col.iter().position(|v| *v != 0.0).unwrap_or(0);
                let last = col.iter().rposition(|v| *v != 0.0).map_or(first, |l| l + 1);
                (first, col[first..last].to_vec())
            })
            .collect();
        Ok(Self { window, bands, fft: rustfft::FftPlanner::<f32>::new().plan_fft_forward(N_FFT) })
    }

    fn n_mel(&self) -> usize {
        self.bands.len()
    }

    /// Mel frames of `samples`.
    fn frames(samples: usize) -> usize {
        if samples < N_FFT { 0 } else { (samples - N_FFT) / HOP + 1 }
    }

    /// `[frames, mel]` log-mel features.
    fn features(&self, samples: &[f32], pool: Option<&Pool>) -> Vec<f32> {
        let (frames, n_mel) = (Self::frames(samples.len()), self.n_mel());
        let mut out = vec![0f32; frames * n_mel];
        let op = ops::Shared(out.as_mut_ptr());
        ops::par(pool, frames, 256, &|a, b| {
            use rustfft::num_complex::Complex;
            let mut buf = vec![Complex::new(0f32, 0f32); N_FFT];
            let mut scratch = vec![Complex::new(0f32, 0f32); self.fft.get_inplace_scratch_len()];
            let mut power = vec![0f32; N_FFT / 2 + 1];
            for t in a..b {
                let frame = &samples[t * HOP..t * HOP + N_FFT];
                for ((c, &s), &w) in buf.iter_mut().zip(frame).zip(&self.window) {
                    *c = Complex::new(s * w, 0.0);
                }
                self.fft.process_with_scratch(&mut buf, &mut scratch);
                for (p, c) in power.iter_mut().zip(&buf) {
                    *p = c.norm_sqr();
                }
                // SAFETY: frame `t` belongs to this range alone.
                let row = unsafe { op.slice(t * n_mel, n_mel) };
                for (v, (first, w)) in row.iter_mut().zip(&self.bands) {
                    let e: f32 = w.iter().zip(&power[*first..]).map(|(w, p)| w * p).sum();
                    *v = e.clamp(1e-9, 1e9).ln();
                }
            }
        });
        out
    }
}

/// The causal convolutional front end: 3×3, 3×3 stride 2, 3×3 stride 2, a
/// ConvNeXt block, and a projection of the 64 × 15 map to the model width.
struct Embed {
    conv: [(Vec<f32>, Vec<f32>, usize); 3],
    dw: (Vec<f32>, Vec<f32>),
    pw1: Lin,
    pw2: Lin,
    out: Lin,
}

impl Embed {
    fn load(m: &Arc<CmfModel>) -> Result<Self, String> {
        let conv = |i: usize| -> Result<(Vec<f32>, Vec<f32>, usize), String> {
            let name = format!("{EMBED}.conv{i}.weight");
            let co = m.tensor(&name).ok_or_else(|| format!("missing GigaAM tensor {name}"))?.shape[0];
            Ok((tensor_f32(m, &name)?, tensor_f32(m, &format!("{EMBED}.conv{i}.bias"))?, co))
        };
        Ok(Self {
            conv: [conv(1)?, conv(2)?, conv(3)?],
            dw: (
                tensor_f32(m, &format!("{EMBED}.convnext.depthwise_conv.weight"))?,
                tensor_f32(m, &format!("{EMBED}.convnext.depthwise_conv.bias"))?,
            ),
            pw1: Lin::load(m, &format!("{EMBED}.convnext.pointwise_conv1"))?,
            pw2: Lin::load(m, &format!("{EMBED}.convnext.pointwise_conv2"))?,
            out: Lin::load(m, &format!("{EMBED}.out"))?,
        })
    }

    /// The front end for a device that runs it itself.
    fn ga(&self, n_mel: usize) -> crate::gpu::GaFront<'_> {
        let c = |i: usize| (self.conv[i].0.as_slice(), self.conv[i].1.as_slice(), self.conv[i].2);
        crate::gpu::GaFront { n_mel, conv: [c(0), c(1), c(2)], dw: (&self.dw.0, &self.dw.1), pw1: self.pw1.ga(), pw2: self.pw2.ga(), out: self.out.ga() }
    }

    /// Encoder frames for `frames` mel frames.
    fn out_frames(frames: usize) -> usize {
        frames.div_ceil(2).div_ceil(2)
    }

    /// `[mel frames, mel]` features of one segment to `[T, width]` rows.
    fn forward(&self, feats: &[f32], frames: usize, n_mel: usize, ex: &Exec) -> Vec<f32> {
        // [1, t, f]: the features are already time-major
        let mut x = ops::Vol { c: 1, t: frames, f: n_mel, d: feats.to_vec() };
        for (i, (w, b, co)) in self.conv.iter().enumerate() {
            x = if i == 0 {
                ops::conv3x3(&x, w, b, *co, 1, (2, 0), 1, ex.pool)
            } else {
                ops::conv3x3(&x, w, b, *co, 2, (1, 1), 0, ex.pool)
            };
            silu_par(&mut x.d, ex.pool);
        }
        // ConvNeXt: depthwise 3×3, 1×1 up, SiLU, 1×1 down, residual
        let y = ops::depthwise3x3_causal(&x, &self.dw.0, &self.dw.1, ex.pool);
        let (c, t, f) = (x.c, x.t, x.f);
        let n = t * f;
        let mut rows = vec![0f32; n * c];
        for ch in 0..c {
            for p in 0..n {
                rows[p * c + ch] = y.d[ch * n + p];
            }
        }
        let mut h = self.pw1.apply(&rows, n, ex);
        silu_par(&mut h, ex.pool);
        let h = self.pw2.apply(&h, n, ex);
        // [t, c·f] rows for the projection: channel-major inside a frame
        let mut flat = vec![0f32; t * c * f];
        for tt in 0..t {
            for ch in 0..c {
                for ff in 0..f {
                    flat[(tt * c + ch) * f + ff] = x.d[(ch * t + tt) * f + ff] + h[(tt * f + ff) * c + ch];
                }
            }
        }
        self.out.apply(&flat, t, ex)
    }
}

fn silu_par(x: &mut [f32], pool: Option<&Pool>) {
    let xp = ops::Shared(x.as_mut_ptr());
    ops::par(pool, x.len().div_ceil(4096), 8, &|a, b| {
        let end = (b * 4096).min(x.len());
        // SAFETY: disjoint ranges of `x`.
        ops::silu(unsafe { xp.slice(a * 4096, end - a * 4096) });
    });
}

// ---------------------------------------------------------------------
// the encoder
// ---------------------------------------------------------------------

struct Ff {
    inp: Lin,
    out: Lin,
}

impl Ff {
    fn load(m: &Arc<CmfModel>, p: &str) -> Result<Self, String> {
        Ok(Self { inp: Lin::load(m, &format!("{p}.in_proj"))?, out: Lin::load(m, &format!("{p}.out_proj"))? })
    }

    fn forward(&self, x: &[f32], n: usize, ex: &Exec) -> Vec<f32> {
        let mut h = self.inp.apply(x, n, ex);
        ex.timed(Slot::Glue, || silu_par(&mut h, ex.pool));
        self.out.apply(&h, n, ex)
    }
}

struct Attn {
    in_norm: Vec<f32>,
    qk: Lin,
    vg: Lin,
    out: Lin,
    heads: usize,
}

struct ConvMod {
    inp: Lin,
    dw_w: Vec<f32>,
    /// the same taps as `[kernel][channel]`
    taps: Vec<f32>,
    dw_b: Vec<f32>,
    kernel: usize,
    mean_w: Vec<f32>,
    out: Lin,
}

struct Layer {
    ff1: Ff,
    attn: Attn,
    conv: Option<ConvMod>,
    ff2: Option<Ff>,
    norm: Vec<f32>,
}

struct Stack {
    /// frames folded into one row
    ds: usize,
    dim: usize,
    proj: Lin,
    proj_t: Lin,
    residual_scales: Vec<f32>,
    input_scale: f32,
    layers: Vec<Layer>,
}

/// `out[r] = x[r] · rsqrt(mean(x[r]²) + eps) ⊙ w` over rows.
fn rms_rows(x: &[f32], dim: usize, w: &[f32], eps: f32, pool: Option<&Pool>) -> Vec<f32> {
    let n = x.len() / dim;
    let mut out = vec![0f32; x.len()];
    let op = ops::Shared(out.as_mut_ptr());
    ops::par(pool, n, 128, &|a, b| {
        for r in a..b {
            let src = &x[r * dim..(r + 1) * dim];
            let inv = 1.0 / (ops::sumsq(src) / dim as f32 + eps).sqrt();
            // SAFETY: row `r` belongs to this range alone.
            let dst = unsafe { op.slice(r * dim, dim) };
            for ((d, &v), &w) in dst.iter_mut().zip(src).zip(w) {
                *d = v * inv * w;
            }
        }
    });
    out
}

fn add_rows(x: &mut [f32], y: &[f32], pool: Option<&Pool>) {
    let xp = ops::Shared(x.as_mut_ptr());
    ops::par(pool, x.len().div_ceil(8192), 8, &|a, b| {
        let (s, e) = (a * 8192, (b * 8192).min(x.len()));
        // SAFETY: disjoint ranges of `x`.
        ops::axpy(unsafe { xp.slice(s, e - s) }, 1.0, &y[s..e]);
    });
}

impl Attn {
    fn load(m: &Arc<CmfModel>, p: &str, heads: usize) -> Result<Self, String> {
        let a = Self {
            in_norm: tensor_f32(m, &format!("{p}.in_norm.weight"))?,
            qk: Lin::load(m, &format!("{p}.qk_in_proj"))?,
            vg: Lin::load(m, &format!("{p}.vg_in_proj"))?,
            out: Lin::load(m, &format!("{p}.out_proj"))?,
            heads,
        };
        let hw = heads * QDIM;
        if a.qk.rows != 2 * hw || a.vg.rows != 3 * hw || a.out.cols != hw {
            return Err(format!("{p}: attention shapes do not match {heads} heads of {QDIM}"));
        }
        Ok(a)
    }

    /// `x_qk` is the layer input (queries and keys), `x_vg` the stream after
    /// the first feed-forward (values and both gates). Rows are the segments
    /// of `segs` back to back; attention stays inside a segment.
    fn forward(&self, x_qk: &[f32], x_vg: &[f32], segs: &[usize], rope: &Rope, ex: &Exec) -> Vec<f32> {
        let n: usize = segs.iter().sum();
        let dim = self.in_norm.len();
        let (heads, hw) = (self.heads, self.heads * QDIM);
        let normed = ex.timed(Slot::Glue, || rms_rows(x_qk, dim, &self.in_norm, 1e-6, ex.pool));
        let qk = self.qk.apply(&normed, n, ex);
        drop(normed);
        let vg = self.vg.apply(x_vg, n, ex);
        let mut ctx = vec![0f32; n * hw];
        let t0 = ex.prof.then(Instant::now);
        let scale = (QDIM as f32).powf(-0.5);
        let mut row0 = 0;
        for &t in segs {
            assert!(t <= rope.positions(), "a GigaAM segment of {t} frames is past the rotary table");
            // head-major planes [head][frame][32]: rotated queries and keys, gated values
            let mut qh = vec![0f32; heads * t * QDIM];
            let mut kh = vec![0f32; heads * t * QDIM];
            let mut vh = vec![0f32; heads * t * QDIM];
            {
                let (qp, kp, vp) = (ops::Shared(qh.as_mut_ptr()), ops::Shared(kh.as_mut_ptr()), ops::Shared(vh.as_mut_ptr()));
                ops::par(ex.pool, t, 64, &|a, b| {
                    let mut gated = vec![0f32; hw];
                    for i in a..b {
                        let r = row0 + i;
                        let (q, k) = qk[r * 2 * hw..(r + 1) * 2 * hw].split_at(hw);
                        let v = &vg[r * 3 * hw..r * 3 * hw + hw];
                        let gate_in = &vg[r * 3 * hw + hw..r * 3 * hw + 2 * hw];
                        gated.copy_from_slice(v);
                        ops::mul_sigmoid(&mut gated, gate_in);
                        let (cos, sin) = rope.at(i);
                        for h in 0..heads {
                            // SAFETY: frame `i` of every head belongs to this range alone.
                            let (qd, kd, vd) = unsafe {
                                let at = (h * t + i) * QDIM;
                                (qp.slice(at, QDIM), kp.slice(at, QDIM), vp.slice(at, QDIM))
                            };
                            rotate(&q[h * QDIM..(h + 1) * QDIM], cos, sin, qd);
                            rotate(&k[h * QDIM..(h + 1) * QDIM], cos, sin, kd);
                            vd.copy_from_slice(&gated[h * QDIM..(h + 1) * QDIM]);
                        }
                    }
                });
            }
            let out = &mut ctx[row0 * hw..(row0 + t) * hw];
            let on_device = ex.gpu && t >= ex.attn_min() && crate::gpu::dit_attention(&qh, &kh, &vh, heads, heads, t, QDIM, scale, out);
            if !on_device {
                attention_host(&qh, &kh, &vh, heads, t, scale, out, ex.pool);
            }
            row0 += t;
        }
        // gate the context, project
        {
            let cp = ops::Shared(ctx.as_mut_ptr());
            ops::par(ex.pool, n, 128, &|a, b| {
                for r in a..b {
                    // SAFETY: row `r` belongs to this range alone.
                    let row = unsafe { cp.slice(r * hw, hw) };
                    ops::mul_sigmoid(row, &vg[r * 3 * hw + 2 * hw..(r + 1) * 3 * hw]);
                }
            });
        }
        if let Some(t0) = t0 {
            ex.add(Slot::Attention, t0);
        }
        self.out.apply(&ctx, n, ex)
    }
}

impl Exec<'_> {
    /// Segments shorter than this attend on the host.
    fn attn_min(&self) -> usize {
        64
    }
}

/// `dst = rope(x)`: pairs `(2i, 2i+1)` rotated by the position's angles.
#[inline]
fn rotate(x: &[f32], cos: &[f32], sin: &[f32], dst: &mut [f32]) {
    for i in 0..QDIM / 2 {
        let (x0, x1) = (x[2 * i], x[2 * i + 1]);
        dst[2 * i] = x0 * cos[i] - x1 * sin[i];
        dst[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
    }
}

/// Full attention of one segment on the host: per head `softmax(scale·q·kᵀ)·v`.
/// `out` is `[frame][head·32]`.
#[allow(clippy::too_many_arguments)]
fn attention_host(qh: &[f32], kh: &[f32], vh: &[f32], heads: usize, t: usize, scale: f32, out: &mut [f32], pool: Option<&Pool>) {
    let hw = heads * QDIM;
    let op = ops::Shared(out.as_mut_ptr());
    let one = |h: usize, scores: &mut Vec<f32>, ctx: &mut Vec<f32>| {
        let plane = h * t * QDIM..(h + 1) * t * QDIM;
        scores.resize(t * t, 0.0);
        ctx.resize(t * QDIM, 0.0);
        ops::gemm_nt(&qh[plane.clone()], &kh[plane.clone()], scores, t, t, QDIM, None);
        for row in scores.chunks_exact_mut(t) {
            ops::softmax_scaled(row, scale);
        }
        ops::gemm_nn(scores, &vh[plane], ctx, t, QDIM, t, None);
        for i in 0..t {
            // SAFETY: the head's columns of every row belong to this head alone.
            unsafe { op.slice(i * hw + h * QDIM, QDIM) }.copy_from_slice(&ctx[i * QDIM..(i + 1) * QDIM]);
        }
    };
    ops::par(pool, heads, 2, &|a, b| {
        let (mut scores, mut ctx) = (Vec::new(), Vec::new());
        for h in a..b {
            one(h, &mut scores, &mut ctx);
        }
    });
}

/// The rotary table: `[position][16]` cosines and sines, as the checkpoint
/// stores them.
struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    fn load(m: &CmfModel) -> Result<Self, String> {
        let name = |part: &str| format!("{ENC}.rope_bank.tables.ed{QDIM}_S10000_b10000.{part}");
        let (cos, sin) = match (m.tensor(&name("cos")), m.tensor(&name("sin"))) {
            (Some(_), Some(_)) => (tensor_f32(m, &name("cos"))?, tensor_f32(m, &name("sin"))?),
            _ => {
                let inv: Vec<f32> = (0..QDIM / 2).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / QDIM as f32)).collect();
                let mut cos = Vec::with_capacity(10_000 * QDIM / 2);
                let mut sin = Vec::with_capacity(10_000 * QDIM / 2);
                for p in 0..10_000 {
                    for f in &inv {
                        let a = p as f32 * f;
                        cos.push(a.cos());
                        sin.push(a.sin());
                    }
                }
                (cos, sin)
            }
        };
        Ok(Self { cos, sin })
    }

    fn positions(&self) -> usize {
        self.cos.len() / (QDIM / 2)
    }

    fn at(&self, pos: usize) -> (&[f32], &[f32]) {
        let h = QDIM / 2;
        (&self.cos[pos * h..(pos + 1) * h], &self.sin[pos * h..(pos + 1) * h])
    }
}

impl ConvMod {
    fn load(m: &Arc<CmfModel>, p: &str, kernel: usize) -> Result<Self, String> {
        let dw_w = tensor_f32(m, &format!("{p}.depthwise_conv.weight"))?;
        let d = dw_w.len() / kernel.max(1);
        let mut taps = vec![0f32; dw_w.len()];
        for c in 0..d {
            for j in 0..kernel {
                taps[j * d + c] = dw_w[c * kernel + j];
            }
        }
        Ok(Self {
            inp: Lin::load(m, &format!("{p}.in_proj"))?,
            dw_w,
            taps,
            dw_b: tensor_f32(m, &format!("{p}.depthwise_conv.bias"))?,
            kernel,
            mean_w: tensor_f32(m, &format!("{p}.weighted_mean.weights"))?,
            out: Lin::load(m, &format!("{p}.out_proj"))?,
        })
    }

    fn forward(&self, x: &[f32], segs: &[usize], ex: &Exec) -> Vec<f32> {
        let n: usize = segs.iter().sum();
        let d = self.mean_w.len();
        let xg = self.inp.apply(x, n, ex);
        let t0 = ex.prof.then(Instant::now);
        // split the stream from its gate
        let mut xs = vec![0f32; n * d];
        for (dst, src) in xs.chunks_exact_mut(d).zip(xg.chunks_exact(2 * d)) {
            dst.copy_from_slice(&src[..d]);
        }
        let mut y = vec![0f32; n * d];
        let mut row0 = 0;
        for &t in segs {
            let seg = &xs[row0 * d..(row0 + t) * d];
            // the segment's mean, weighted per channel
            let mut mean = vec![0f32; d];
            for row in seg.chunks_exact(d) {
                ops::axpy(&mut mean, 1.0, row);
            }
            for (m, &w) in mean.iter_mut().zip(&self.mean_w) {
                *m = *m / t as f32 * w;
            }
            let out = &mut y[row0 * d..(row0 + t) * d];
            ops::depthwise1d(seg, t, d, &self.dw_w, &self.dw_b, self.kernel, out, ex.pool);
            let op = ops::Shared(out.as_mut_ptr());
            ops::par(ex.pool, t, 64, &|a, b| {
                for i in a..b {
                    // SAFETY: row `i` belongs to this range alone.
                    let row = unsafe { op.slice(i * d, d) };
                    ops::add_mul_sigmoid_silu(row, &mean, &xg[(row0 + i) * 2 * d + d..(row0 + i + 1) * 2 * d]);
                }
            });
            row0 += t;
        }
        if let Some(t0) = t0 {
            ex.add(Slot::Conv, t0);
        }
        self.out.apply(&y, n, ex)
    }
}

impl Layer {
    fn forward(&self, x: Vec<f32>, dim: usize, segs: &[usize], rope: &Rope, ex: &Exec) -> Vec<f32> {
        let n: usize = segs.iter().sum();
        let mut s = x.clone();
        add_rows(&mut s, &self.ff1.forward(&x, n, ex), ex.pool);
        let a = self.attn.forward(&x, &s, segs, rope, ex);
        add_rows(&mut s, &a, ex.pool);
        if let Some(conv) = &self.conv {
            let c = conv.forward(&s, segs, ex);
            add_rows(&mut s, &c, ex.pool);
        }
        if let Some(ff2) = &self.ff2 {
            let f = ff2.forward(&s, n, ex);
            add_rows(&mut s, &f, ex.pool);
        }
        ex.timed(Slot::Glue, || {
            // norm(residual + ¼·(stream − residual))
            let sp = ops::Shared(s.as_mut_ptr());
            ops::par(ex.pool, n, 128, &|a, b| {
                // SAFETY: disjoint row ranges of `s`.
                let part = unsafe { sp.slice(a * dim, (b - a) * dim) };
                for (v, &r) in part.iter_mut().zip(&x[a * dim..b * dim]) {
                    *v = r + 0.25 * (*v - r);
                }
            });
            rms_rows(&s, dim, &self.norm, 1e-6, ex.pool)
        })
    }
}

impl Stack {
    fn forward(&self, x_in: &[f32], segs: &[usize], rope: &Rope, ex: &Exec) -> Vec<f32> {
        let n: usize = segs.iter().sum();
        let mut x = self.proj.apply(x_in, n, ex);
        let mut bypass: Vec<f32> = x.iter().map(|v| v * self.residual_scales[0]).collect();
        for v in &mut x {
            *v *= self.input_scale;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, self.dim, segs, rope, ex);
            let (bp, w) = (ops::Shared(bypass.as_mut_ptr()), self.residual_scales[i + 1]);
            ops::par(ex.pool, bypass.len().div_ceil(8192), 8, &|a, b| {
                let (s, e) = (a * 8192, (b * 8192).min(x.len()));
                // SAFETY: disjoint ranges of `bypass`.
                ops::axpy(unsafe { bp.slice(s, e - s) }, w, &x[s..e]);
            });
        }
        let mut out = self.proj_t.apply(&bypass, n, ex);
        add_rows(&mut out, x_in, ex.pool);
        out
    }
}

struct Encoder {
    stacks: Vec<Stack>,
    rope: Rope,
    out_scale: f32,
    out_eps: f32,
    width: usize,
    /// the model's key in a device weight cache
    key: u64,
    /// the device declined the whole-encoder path: do not ask again
    resident_off: std::sync::atomic::AtomicBool,
}

impl Encoder {
    fn load(m: &Arc<CmfModel>, cfg: &serde_json::Value) -> Result<Self, String> {
        let tuple = |key: &str| -> Result<Vec<usize>, String> {
            let v = match cfg.get(key) {
                Some(serde_json::Value::String(s)) => s.split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<Vec<_>, _>>().ok(),
                Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| vec![v as usize]),
                _ => None,
            };
            v.filter(|v| !v.is_empty()).ok_or_else(|| format!("GigaAM config: encoder.{key}"))
        };
        let ds = tuple("downsampling_factor")?;
        let n = ds.len();
        let per = |key: &str| -> Result<Vec<usize>, String> {
            let v = tuple(key)?;
            match v.len() {
                1 => Ok(vec![v[0]; n]),
                l if l == n => Ok(v),
                _ => Err(format!("GigaAM config: encoder.{key} has {} entries for {n} stacks", v.len())),
            }
        };
        let (dims, layers, heads, kernels) = (per("encoder_dim")?, per("num_encoder_layers")?, per("num_heads")?, per("conv_params")?);
        if per("query_head_dim")?.iter().chain(&per("value_head_dim")?).any(|d| *d != QDIM) {
            return Err(format!("GigaAM: only {QDIM}-wide attention heads are supported"));
        }
        if cfg.get("activation").and_then(|v| v.as_str()).is_some_and(|a| !a.eq_ignore_ascii_case("silu")) {
            return Err("GigaAM: only the SiLU activation is supported".into());
        }
        if ds.iter().any(|d| *d != 1 && *d != 2) {
            return Err("GigaAM: stacks at 1× or 2× the frame step only".into());
        }
        let width = dims[0];
        let mut stacks = Vec::with_capacity(n);
        for s in 0..n {
            let p = format!("{ENC}.encoders.{s}");
            let mut ls = Vec::with_capacity(layers[s]);
            for l in 0..layers[s] {
                let lp = format!("{p}.layers.{l}");
                let (conv, ff2) = if kernels[s] > 0 {
                    (Some(ConvMod::load(m, &format!("{lp}.conv_module"), kernels[s])?), Some(Ff::load(m, &format!("{lp}.feed_forward2"))?))
                } else {
                    (None, None)
                };
                ls.push(Layer {
                    ff1: Ff::load(m, &format!("{lp}.feed_forward1"))?,
                    attn: Attn::load(m, &format!("{lp}.self_attn"), heads[s])?,
                    conv,
                    ff2,
                    norm: tensor_f32(m, &format!("{lp}.norm.weight"))?,
                });
            }
            let proj = Lin::load_named(m, &format!("{p}.proj.weight"), None)?;
            let proj_t = Lin::load_named(m, &format!("{p}.proj_t.weight"), None)?;
            if proj.cols != ds[s] * width || proj.rows != dims[s] || proj_t.rows != proj.cols {
                return Err(format!("{p}: projection {}×{} does not fit the stack", proj.rows, proj.cols));
            }
            let residual_scales = tensor_f32(m, &format!("{p}.residual_scales"))?;
            if residual_scales.len() != layers[s] + 1 {
                return Err(format!("{p}: {} residual scales for {} layers", residual_scales.len(), layers[s]));
            }
            stacks.push(Stack { ds: ds[s], dim: dims[s], proj, proj_t, residual_scales, input_scale: scalar(m, &format!("{p}.input_scale"))?, layers: ls });
        }
        Ok(Self {
            stacks,
            rope: Rope::load(m)?,
            out_scale: scalar(m, &format!("{ENC}.out_norm.scale"))?,
            out_eps: scalar(m, &format!("{ENC}.out_norm.eps"))?,
            width,
            key: m.uid(),
            resident_off: Default::default(),
        })
    }

    /// The stacks as the device's whole-encoder path takes them.
    fn ga_args(&self, prof: bool) -> crate::gpu::GigaamArgs<'_> {
        use crate::gpu::{GaConv, GaLayer, GaStack, GigaamArgs};
        let stacks = self
            .stacks
            .iter()
            .map(|st| GaStack {
                fold: st.ds,
                dim: st.dim,
                heads: st.layers.first().map_or(1, |l| l.attn.heads),
                proj: st.proj.ga(),
                proj_t: st.proj_t.ga(),
                residual_scales: &st.residual_scales,
                input_scale: st.input_scale,
                layers: st
                    .layers
                    .iter()
                    .map(|l| GaLayer {
                        ff1_in: l.ff1.inp.ga(),
                        ff1_out: l.ff1.out.ga(),
                        in_norm: &l.attn.in_norm,
                        qk: l.attn.qk.ga(),
                        vg: l.attn.vg.ga(),
                        attn_out: l.attn.out.ga(),
                        conv: l.conv.as_ref().map(|c| GaConv { inp: c.inp.ga(), taps: &c.taps, bias: &c.dw_b, kernel: c.kernel, mean_w: &c.mean_w, out: c.out.ga() }),
                        ff2: l.ff2.as_ref().map(|f| (f.inp.ga(), f.out.ga())),
                        norm: &l.norm,
                    })
                    .collect(),
            })
            .collect();
        GigaamArgs { key: self.key, width: self.width, head_dim: QDIM, stacks, cos: &self.rope.cos, sin: &self.rope.sin, out_scale: self.out_scale, out_eps: self.out_eps, prof, front: None }
    }

    /// The whole encoder on the device, a segment at a time; `None` when it declines.
    fn forward_resident(&self, x: &[f32], segs: &[usize], ex: &Exec) -> Option<Vec<f32>> {
        use std::sync::atomic::Ordering;
        if !ex.resident || self.resident_off.load(Ordering::Relaxed) {
            return None;
        }
        let args = self.ga_args(ex.prof);
        let w = self.width;
        let mut out = x.to_vec();
        let mut row0 = 0;
        for &t in segs {
            if !crate::gpu::gigaam_encode(&args, &mut out[row0 * w..(row0 + t) * w], t) {
                self.resident_off.store(true, Ordering::Relaxed);
                return None;
            }
            row0 += t;
        }
        Some(out)
    }

    /// `[Σ segs, width]` front-end rows to encoder output rows.
    fn forward(&self, mut x: Vec<f32>, segs: &[usize], ex: &Exec) -> Vec<f32> {
        let w = self.width;
        if let Some(out) = self.forward_resident(&x, segs, ex) {
            return out;
        }
        for stack in &self.stacks {
            if stack.ds == 1 {
                x = stack.forward(&x, segs, &self.rope, ex);
                continue;
            }
            // fold pairs of frames into rows (an odd last frame pairs with zeros)
            let half: Vec<usize> = segs.iter().map(|t| t.div_ceil(2)).collect();
            let nh: usize = half.iter().sum();
            let mut folded = vec![0f32; nh * 2 * w];
            let (mut src, mut dst) = (0, 0);
            for &t in segs {
                folded[dst * 2 * w..dst * 2 * w + t * w].copy_from_slice(&x[src * w..(src + t) * w]);
                src += t;
                dst += t.div_ceil(2);
            }
            let y = stack.forward(&folded, &half, &self.rope, ex);
            let (mut src, mut dst) = (0, 0);
            for &t in segs {
                x[dst * w..(dst + t) * w].copy_from_slice(&y[src * 2 * w..src * 2 * w + t * w]);
                dst += t;
                src += t.div_ceil(2);
            }
        }
        ex.timed(Slot::Glue, || {
            let (scale, eps2) = (self.out_scale, self.out_eps * self.out_eps);
            let xp = ops::Shared(x.as_mut_ptr());
            ops::par(ex.pool, x.len() / w, 128, &|a, b| {
                for r in a..b {
                    // SAFETY: row `r` belongs to this range alone.
                    let row = unsafe { xp.slice(r * w, w) };
                    let k = scale / (ops::sumsq(row) / w as f32 + eps2).sqrt();
                    for v in row.iter_mut() {
                        *v *= k;
                    }
                }
            });
        });
        x
    }
}

// ---------------------------------------------------------------------
// the transcriber
// ---------------------------------------------------------------------

/// Which transcripts to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heads {
    /// formatted text: case, punctuation, digits
    E2e,
    /// lower-case words
    Raw,
    Both,
}

/// Transcription settings.
#[derive(Clone, Debug)]
pub struct Options {
    pub heads: Heads,
    /// word start and end times
    pub words: bool,
    /// cut the recording by voice activity even when it fits one segment
    pub force_vad: bool,
    /// encoder frames in one batch of segments at most
    pub batch_frames: usize,
    pub vad: vad::Settings,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            heads: Heads::E2e,
            words: false,
            force_vad: false,
            batch_frames: std::env::var("CMF_GIGAAM_BATCH_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(4500),
            vad: vad::Settings::default(),
        }
    }
}

/// One stretch of speech and what was said in it.
#[derive(Clone, Debug)]
pub struct Segment {
    /// seconds from the start of the recording
    pub start: f64,
    pub end: f64,
    pub text: String,
    /// present when [`Options::words`] was set (times from the start of the recording)
    pub words: Vec<Word>,
}

/// The transcript of one head.
#[derive(Clone, Debug, Default)]
pub struct Transcript {
    pub text: String,
    pub segments: Vec<Segment>,
}

/// What [`Transcriber::transcribe`] returns.
#[derive(Clone, Debug, Default)]
pub struct Output {
    /// formatted text (when asked for)
    pub e2e: Option<Transcript>,
    /// lower-case words (when asked for)
    pub raw: Option<Transcript>,
    pub audio_s: f64,
    /// seconds of speech sent through the encoder
    pub speech_s: f64,
    pub vad_s: f64,
    pub features_s: f64,
    pub encoder_s: f64,
    pub decode_s: f64,
}

impl Output {
    /// The transcript a caller who asked for one head wants: formatted when present.
    pub fn text(&self) -> &str {
        self.e2e.as_ref().or(self.raw.as_ref()).map_or("", |t| t.text.as_str())
    }

    /// Seconds of work: everything after the audio was read.
    pub fn total_s(&self) -> f64 {
        self.vad_s + self.features_s + self.encoder_s + self.decode_s
    }
}

/// A loaded GigaAM model.
pub struct Transcriber {
    mel: Mel,
    embed: Embed,
    encoder: Encoder,
    e2e: decode::Head,
    raw: decode::Head,
    vad: Option<vad::Vad>,
    /// the device declined the front end: the host runs it
    front_dev_off: std::sync::atomic::AtomicBool,
    /// `rnnt` or `ctc`
    pub revision: String,
}

impl Transcriber {
    pub fn load(model: &Arc<CmfModel>) -> Result<Self, String> {
        if !is_gigaam(model) {
            return Err(format!("not a GigaAM file (arch '{}')", model.header.arch.arch_name));
        }
        let meta = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("gigaam"))
            .ok_or("GigaAM file without its config (header provenance.gigaam)")?;
        let cfg = meta.pointer("/config/cfg").ok_or("GigaAM file: no config.cfg")?;
        let enc_cfg = cfg.get("encoder").ok_or("GigaAM config: no encoder")?;
        let mel = Mel::load(model, "model.preprocessor")?;
        let head = |name: &str| decode::Head::load(model, cfg, meta.get("tokenizers"), name);
        let vad = match meta.get("vad_config").filter(|v| !v.is_null()) {
            Some(vc) if model.tensor("vad.head.decoder_layers.0.weight").is_some() => Some(vad::Vad::load(model, vc)?),
            _ => None,
        };
        Ok(Self {
            mel,
            embed: Embed::load(model)?,
            encoder: Encoder::load(model, enc_cfg)?,
            e2e: head("e2e")?,
            raw: head("raw")?,
            vad,
            front_dev_off: Default::default(),
            revision: meta.get("revision").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        })
    }

    /// Does the file carry the voice-activity model (recordings over 30 s)?
    pub fn has_vad(&self) -> bool {
        self.vad.is_some()
    }

    /// Encoder output of one clip of at most 30 s: `[T, width]` rows.
    pub fn encode(&self, samples: &[f32], pool: Option<&Pool>) -> Result<(Vec<f32>, usize), String> {
        if samples.len() > MAX_SEGMENT {
            return Err("encode takes one segment of at most 30 s".into());
        }
        let ex = Exec::new(pool);
        let (x, segs) = self.front(&[samples], &ex);
        let t = segs.first().copied().unwrap_or(0);
        if t == 0 {
            return Ok((Vec::new(), 0));
        }
        Ok((self.encoder.forward(x, &segs, &ex), t))
    }

    /// Features and the front end of each segment: `[Σ T, width]` rows and
    /// the frame counts (zero for a segment shorter than one window).
    fn front(&self, segments: &[&[f32]], ex: &Exec) -> (Vec<f32>, Vec<usize>) {
        let n_mel = self.mel.n_mel();
        let mut rows = Vec::new();
        let mut segs = Vec::with_capacity(segments.len());
        for s in segments {
            let frames = Mel::frames(s.len());
            if frames == 0 {
                segs.push(0);
                continue;
            }
            let feats = self.mel.features(s, ex.pool);
            let x = self.embed.forward(&feats, frames, n_mel, ex);
            debug_assert_eq!(x.len(), Embed::out_frames(frames) * self.encoder.width);
            segs.push(Embed::out_frames(frames));
            rows.extend_from_slice(&x);
        }
        (rows, segs)
    }

    /// Features on the host, the front end and the stacks on the device, a
    /// segment at a time: the encoder rows of the batch and its frame counts,
    /// with the seconds spent on features and on the device. `None` when the
    /// device has no such path (nothing was computed that the host path
    /// would not compute again).
    fn encode_on_device(&self, segments: &[&[f32]], ex: &Exec) -> Option<(Vec<f32>, Vec<usize>, f64, f64)> {
        use std::sync::atomic::Ordering;
        if !ex.resident || self.encoder.resident_off.load(Ordering::Relaxed) || self.front_dev_off.load(Ordering::Relaxed) {
            return None;
        }
        let mut args = self.encoder.ga_args(ex.prof);
        args.front = Some(self.embed.ga(self.mel.n_mel()));
        let w = self.encoder.width;
        let (mut rows, mut segs) = (Vec::new(), Vec::with_capacity(segments.len()));
        let (mut feat_s, mut enc_s) = (0.0, 0.0);
        for s in segments {
            let frames = Mel::frames(s.len());
            let t = Embed::out_frames(frames);
            segs.push(t);
            if t == 0 {
                continue;
            }
            let t0 = Instant::now();
            let feats = self.mel.features(s, ex.pool);
            feat_s += t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let at = rows.len();
            rows.resize(at + t * w, 0.0);
            if !crate::gpu::gigaam_encode_feats(&args, &feats, frames, &mut rows[at..]) {
                self.front_dev_off.store(true, Ordering::Relaxed);
                return None;
            }
            enc_s += t1.elapsed().as_secs_f64();
        }
        Some((rows, segs, feat_s, enc_s))
    }

    /// Transcribe mono 16 kHz samples.
    pub fn transcribe(&self, samples: &[f32], opts: &Options, pool: Option<&Pool>) -> Result<Output, String> {
        let mut out = Output { audio_s: samples.len() as f64 / SR as f64, ..Default::default() };
        // where the speech is: the whole clip, or the VAD's segments
        let t0 = Instant::now();
        let bounds: Vec<(f64, f64)> = if samples.len() <= MAX_SEGMENT && !opts.force_vad {
            vec![(0.0, out.audio_s)]
        } else {
            let vad = self.vad.as_ref().ok_or(
                "this recording is longer than 30 s and the file has no voice-activity model to cut it \
                 (convert from a checkpoint directory that holds vad.safetensors and vad_config.json)",
            )?;
            vad.segments(samples, &opts.vad, pool)
        };
        out.vad_s = t0.elapsed().as_secs_f64();
        let clips: Vec<&[f32]> = bounds
            .iter()
            .map(|&(s, e)| {
                let a = ((s * SR as f64).round_ties_even() as usize).min(samples.len());
                let b = ((e * SR as f64).round_ties_even() as usize).clamp(a, samples.len());
                &samples[a..b]
            })
            .collect();
        let want = |h: Heads| opts.heads == h || opts.heads == Heads::Both;
        let mut e2e = want(Heads::E2e).then(Transcript::default);
        let mut raw = want(Heads::Raw).then(Transcript::default);
        let ex = Exec::new(pool);
        let w = self.encoder.width;
        let mut i = 0;
        while i < clips.len() {
            // a batch: segments until the frame budget is spent
            let mut j = i;
            let mut frames = 0;
            while j < clips.len() && (j == i || frames + Embed::out_frames(Mel::frames(clips[j].len())) <= opts.batch_frames) {
                frames += Embed::out_frames(Mel::frames(clips[j].len()));
                j += 1;
            }
            let (enc, segs) = match self.encode_on_device(&clips[i..j], &ex) {
                Some((enc, segs, feat_s, enc_s)) => {
                    out.features_s += feat_s;
                    out.encoder_s += enc_s;
                    (enc, segs)
                }
                None => {
                    let t1 = Instant::now();
                    let (x, segs) = self.front(&clips[i..j], &ex);
                    out.features_s += t1.elapsed().as_secs_f64();
                    let t2 = Instant::now();
                    let live: Vec<usize> = segs.iter().copied().filter(|t| *t > 0).collect();
                    let enc = if live.is_empty() { Vec::new() } else { self.encoder.forward(x, &live, &ex) };
                    out.encoder_s += t2.elapsed().as_secs_f64();
                    (enc, segs)
                }
            };
            let t3 = Instant::now();
            // every (segment, head) decodes on its own: spread them over the pool
            let mut jobs: Vec<(usize, usize, &decode::Head)> = Vec::new();
            let mut row0 = 0;
            let mut starts = Vec::with_capacity(segs.len());
            for &t in &segs {
                starts.push(row0);
                row0 += t;
            }
            for (k, &t) in segs.iter().enumerate() {
                if t == 0 {
                    continue;
                }
                if e2e.is_some() {
                    jobs.push((k, 0, &self.e2e));
                }
                if raw.is_some() {
                    jobs.push((k, 1, &self.raw));
                }
            }
            let results: Vec<std::sync::Mutex<Option<decode::Decoded>>> = jobs.iter().map(|_| std::sync::Mutex::new(None)).collect();
            // one job a worker; inside a job the products are a thread's own (a segment's greedy search is a
            // few hundred small products: handing each to the pool costs more than it saves)
            ops::par(pool, jobs.len(), 2, &|a, b| {
                for q in a..b {
                    let (k, _, head) = jobs[q];
                    let rows = &enc[starts[k] * w..(starts[k] + segs[k]) * w];
                    *results[q].lock().unwrap() = Some(head.decode(rows, segs[k], w, None));
                }
            });
            for (q, (k, which, head)) in jobs.iter().enumerate() {
                let d = results[q].lock().unwrap().take().expect("decoded");
                let (start, end) = bounds[i + k];
                let words = if opts.words {
                    let shift = clips[i + k].len() as f64 / SR as f64 / segs[*k] as f64;
                    head.words(&d, shift, start)
                } else {
                    Vec::new()
                };
                let seg = Segment { start, end, text: head.text(&d.ids), words };
                match which {
                    0 => e2e.as_mut().unwrap().segments.push(seg),
                    _ => raw.as_mut().unwrap().segments.push(seg),
                }
            }
            out.decode_s += t3.elapsed().as_secs_f64();
            out.speech_s += clips[i..j].iter().map(|c| c.len() as f64 / SR as f64).sum::<f64>();
            i = j;
        }
        ex.report();
        for t in [&mut e2e, &mut raw].into_iter().flatten() {
            t.text = t.segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        }
        out.e2e = e2e;
        out.raw = raw;
        Ok(out)
    }

    /// Parity tooling: the front-end rows, every stack's output and the
    /// encoder output of one clip of at most 30 s, as `(name, rows, width)`.
    pub fn trace(&self, samples: &[f32], pool: Option<&Pool>) -> Result<Vec<(String, Vec<f32>, usize)>, String> {
        if samples.len() > MAX_SEGMENT {
            return Err("trace takes one segment of at most 30 s".into());
        }
        let ex = Exec::new(pool);
        let frames = Mel::frames(samples.len());
        let feats = self.mel.features(samples, pool);
        let mut out = vec![("features".to_string(), feats.clone(), self.mel.n_mel())];
        let x = self.embed.forward(&feats, frames, self.mel.n_mel(), &ex);
        let w = self.encoder.width;
        out.push(("frontend".into(), x.clone(), w));
        let t = x.len() / w;
        let mut cur = x;
        for (s, stack) in self.encoder.stacks.iter().enumerate() {
            if stack.ds == 1 {
                cur = stack.forward(&cur, &[t], &self.encoder.rope, &ex);
            } else {
                let mut folded = vec![0f32; t.div_ceil(2) * 2 * w];
                folded[..t * w].copy_from_slice(&cur);
                let y = stack.forward(&folded, &[t.div_ceil(2)], &self.encoder.rope, &ex);
                cur = y[..t * w].to_vec();
            }
            out.push((format!("stack{s}"), cur.clone(), w));
        }
        let (enc, _) = self.encode(samples, pool)?;
        out.push(("encoded".into(), enc, w));
        Ok(out)
    }
}

/// Mono 16 kHz samples of an audio file: WAV here, other containers through `ffmpeg`.
pub fn read_audio(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_audio(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Mono 16 kHz samples of an audio file's bytes.
pub fn decode_audio(bytes: &[u8]) -> Result<Vec<f32>, String> {
    match pcm16_mono_16k(bytes) {
        Some(samples) => Ok(samples),
        None => crate::egemma2_audio::decode_audio(bytes),
    }
}

/// The samples of a plain 16-bit PCM, mono, 16 kHz WAV — the format the
/// model takes as it is, read without the general decoder's passes.
fn pcm16_mono_16k(b: &[u8]) -> Option<Vec<f32>> {
    if b.len() < 44 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return None;
    }
    let mut plain = false;
    let mut i = 12;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().ok()?) as usize;
        let st = i + 8;
        match &b[i..i + 4] {
            b"fmt " if st + 16 <= b.len() => {
                let u16_at = |o: usize| u16::from_le_bytes([b[st + o], b[st + o + 1]]);
                let rate = u32::from_le_bytes(b[st + 4..st + 8].try_into().ok()?);
                plain = u16_at(0) == 1 && u16_at(2) == 1 && rate as usize == SR && u16_at(14) == 16;
            }
            b"data" => {
                if !plain {
                    return None;
                }
                let end = st.saturating_add(len).min(b.len());
                return Some(b[st..end].chunks_exact(2).map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0).collect());
            }
            _ => {}
        }
        i = st.saturating_add(len).saturating_add(len & 1);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_interleaved_pairs() {
        let x: Vec<f32> = (0..QDIM).map(|i| i as f32 + 1.0).collect();
        let cos: Vec<f32> = (0..QDIM / 2).map(|i| (0.1 * i as f32).cos()).collect();
        let sin: Vec<f32> = (0..QDIM / 2).map(|i| (0.1 * i as f32).sin()).collect();
        let mut y = vec![0f32; QDIM];
        rotate(&x, &cos, &sin, &mut y);
        assert_eq!((y[0], y[1]), (1.0, 2.0));
        assert!((y[2] - (3.0 * cos[1] - 4.0 * sin[1])).abs() < 1e-6);
        assert!((y[3] - (3.0 * sin[1] + 4.0 * cos[1])).abs() < 1e-6);
        // a rotation keeps the pair's length
        for i in 0..QDIM / 2 {
            let (a, b) = (x[2 * i].hypot(x[2 * i + 1]), y[2 * i].hypot(y[2 * i + 1]));
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn plain_wav_is_read_directly() {
        let mut wav = Vec::new();
        let data: Vec<u8> = [0i16, 16384, -32768, 32767].iter().flat_map(|v| v.to_le_bytes()).collect();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        for v in [1u16, 1] {
            wav.extend_from_slice(&v.to_le_bytes());
        }
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        for v in [2u16, 16] {
            wav.extend_from_slice(&v.to_le_bytes());
        }
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
        wav.extend_from_slice(&data);
        assert_eq!(pcm16_mono_16k(&wav).unwrap(), vec![0.0, 0.5, -1.0, 32767.0 / 32768.0]);
        // another rate goes to the general decoder
        wav[24..28].copy_from_slice(&8000u32.to_le_bytes());
        assert!(pcm16_mono_16k(&wav).is_none());
        assert!(pcm16_mono_16k(b"not a wav").is_none());
    }

    #[test]
    fn frame_counts_follow_the_front_end() {
        assert_eq!(Mel::frames(319), 0);
        assert_eq!(Mel::frames(320), 1);
        assert_eq!(Mel::frames(71_136), 443);
        assert_eq!(Embed::out_frames(443), 111);
        assert_eq!(Embed::out_frames(1), 1);
        assert_eq!(Embed::out_frames(MAX_SEGMENT / HOP - 1), 750);
    }

    #[test]
    fn host_attention_is_softmax_weighted_values() {
        let (heads, t) = (2usize, 5usize);
        let fill = |seed: usize| -> Vec<f32> { (0..heads * t * QDIM).map(|i| (((i * 7 + seed * 13) % 23) as f32 - 11.0) * 0.05).collect() };
        let (q, k, v) = (fill(1), fill(2), fill(3));
        let scale = (QDIM as f32).powf(-0.5);
        let mut out = vec![0f32; t * heads * QDIM];
        attention_host(&q, &k, &v, heads, t, scale, &mut out, None);
        for h in 0..heads {
            for i in 0..t {
                let qi = &q[(h * t + i) * QDIM..][..QDIM];
                let s: Vec<f32> = (0..t).map(|j| scale * qi.iter().zip(&k[(h * t + j) * QDIM..][..QDIM]).map(|(a, b)| a * b).sum::<f32>()).collect();
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = s.iter().map(|x| (x - m).exp()).sum();
                for d in 0..QDIM {
                    let want: f32 = (0..t).map(|j| (s[j] - m).exp() / z * v[(h * t + j) * QDIM + d]).sum();
                    assert!((out[i * heads * QDIM + h * QDIM + d] - want).abs() < 1e-5);
                }
            }
        }
    }
}
