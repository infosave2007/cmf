//! MiniMax-H3's audio VAE decoder: BigVGAN at 32 kHz, stereo.
//!
//! 32 latent channels at 40 frames a second become a waveform at 800
//! samples a frame, through seven transposed-convolution stages and
//! their AMP residual blocks. The two stereo channels are two
//! independent mono passes, which is how the reference batches them.
//!
//! The activations are the interesting part and the easy thing to get
//! subtly wrong: every nonlinearity is wrapped in a 2× kaiser-sinc
//! upsample, the pointwise SnakeBeta, and a 2× lowpass back down. The
//! filter is designed here rather than shipped, from the same
//! `kaiser_sinc_filter1d(cutoff, half_width, 12)` the reference calls,
//! so it cannot drift out of step with a checkpoint that does not
//! contain it.

use crate::pool::Pool;

/// Where an audio decode goes (`CMF_AVAE_TIME=1`). The video decoder
/// went a whole session untuned because nothing measured it; this one
/// already misled once, when the shape of the per-channel FIR promised
/// more than the 9.2 s → 8.1 s it delivered.
pub static AVAE_TIME: [std::sync::atomic::AtomicU64; 6] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn atime_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_AVAE_TIME").is_ok())
}

fn atime(slot: usize, t: std::time::Instant) {
    if atime_on() {
        AVAE_TIME[slot].fetch_add(
            t.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// One line per phase, sorted by cost.
pub fn avae_time_report() -> Option<String> {
    if !atime_on() {
        return None;
    }
    const NAMES: [&str; 6] = [
        "dec_in",
        "conv_pre",
        "upsamples",
        "resblocks",
        "act_post",
        "conv_post",
    ];
    let mut v: Vec<(u64, &str)> = AVAE_TIME
        .iter()
        .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        .zip(NAMES)
        .collect();
    let total: u64 = v.iter().map(|(u, _)| u).sum();
    if total == 0 {
        return None;
    }
    v.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = format!("audio vae phases (total {:.1} s):\n", total as f64 / 1e6);
    for (us, name) in v {
        out.push_str(&format!(
            "  {name:<12} {:>6.1} s  {:>5.1}%\n",
            us as f64 / 1e6,
            100.0 * us as f64 / total as f64
        ));
    }
    Some(out)
}
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Both resampling filters in the alias-free activation are designed at
/// this kernel length.
const FILTER_LEN: usize = 12;

struct Conv1d {
    w: Vec<f32>, // [out, in, k]
    b: Option<Vec<f32>>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
    pad: usize,
    dilation: usize,
}

impl Conv1d {
    fn load(
        model: &Arc<CmfModel>,
        name: &str,
        pad: usize,
        dilation: usize,
    ) -> Result<Self, String> {
        let e = model
            .tensor(&format!("{name}.weight"))
            .ok_or_else(|| format!("missing {name}.weight"))?;
        let w = crate::dit::cmf_f32(model, &format!("{name}.weight"))?;
        let b = crate::dit::cmf_f32(model, &format!("{name}.bias")).ok();
        Ok(Self {
            out_ch: e.shape[0],
            in_ch: e.shape[1],
            k: e.shape[2],
            w,
            b,
            pad,
            dilation,
        })
    }

    /// `x` is `[in_ch, n]`; the result is `[out_ch, n]` for the
    /// paddings used here (all `same`).
    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let out_n = (n + 2 * self.pad).saturating_sub(self.dilation * (self.k - 1));
        let mut out = vec![0f32; self.out_ch * out_n];
        let ptr = SendPtr(out.as_mut_ptr());
        // Each (channel, time slice) pair is independent, so time is
        // tiled until there are enough rows for any pool. Kept because
        // it is free and bit-identical — but MEASURED SMALL: 8.6 s →
        // 8.2 s on a stage that is 95.4% these convolutions. That is
        // the finding. The resblocks are not starved of parallelism,
        // they are arithmetic: ~8 s of dilated convolution on the CPU
        // while the card is idle. The next move is the device, not a
        // finer split — everything else in this render already went
        // that way, and this is the last host-bound stage left.
        //
        // The shape it wants, so the next pass is implementation and not
        // research: this convolution IS a GEMM under im2col. Build
        // `[in_ch·k × out_n]` where row `(i, j)` holds
        // `x[i, t + j·dilation - pad]` (zero outside), and the weights
        // are ALREADY `[out_ch × in_ch·k]` in exactly that order —
        // `w[(o·in_ch + i)·k + j]` — so nothing needs repacking. Then
        // `out = W · col` with a bias per row. im2col is a gather, which
        // the card does far better than it does the branch in this loop.
        //
        // The f32 device GEMM already exists and is public INSIDE the
        // backend — `gpu_wgpu::gemm_nt_f32(x, w, y, n, k, m)`, tensor
        // cores when the card has them. It is simply not re-exported
        // through `gpu.rs`, which is where every other caller goes. Two
        // conditions come with it and both suit this decoder: it refuses
        // under `CMF_BAKE_GPU=0` or strict-f32, and it refuses jobs
        // below n·k·m = 4M — these convolutions are far above that
        // (out_ch × in_ch·k × out_n runs to hundreds of millions).
        // So the work is: a facade re-export, an im2col buffer, and a
        // parity check against this loop.
        // im2col + the device GEMM: `out[out_ch × out_n] = W · col`,
        // with W already `[out_ch × in_ch·k]` in the order this loop
        // reads it. Opt-in (`CMF_AVAE_CONV_GPU=1`) until measured — the
        // column buffer is in_ch·k × out_n floats, which is the price.
        if std::env::var("CMF_AVAE_CONV_GPU").as_deref() != Ok("0") && crate::gpu::enabled_here() {
            let kk = self.in_ch * self.k;
            // First ask the device to expand the columns itself. That is
            // strictly less work than the host arm below: `x` is k times
            // smaller than the column buffer, the transposed copy is not
            // built at all, and the tiling the 2 GiB binding limit forces
            // happens without a host allocation per tile. The host arm
            // stays for the backends and shapes the kernel refuses.
            {
                let mut yt = vec![0f32; out_n * self.out_ch];
                if crate::gpu::conv1d_gemm(
                    x,
                    &self.w,
                    self.in_ch,
                    self.out_ch,
                    n,
                    self.k,
                    self.pad,
                    self.dilation,
                    out_n,
                    &mut yt,
                ) {
                    for o in 0..self.out_ch {
                        let bias = self.b.as_ref().map_or(0.0, |b| b[o]);
                        // SAFETY: single-threaded here.
                        let dst = unsafe { ptr.row(o * out_n, out_n) };
                        for (t, d) in dst.iter_mut().enumerate() {
                            *d = yt[t * self.out_ch + o] + bias;
                        }
                    }
                    return out;
                }
            }
            // The column buffer is kk x out_n floats and out_n is the
            // AUDIO length, so it grows with the song: 20 s through this
            // decoder's last stage wants 2.37 GB and Vulkan refuses a
            // binding over 2 GiB — `Buffer binding range 2147483648
            // exceeds limit 2147483644`, which is what a 20-second render
            // died on. Tile the time axis so one pass always fits, and
            // keep the device arm instead of falling back to the host for
            // exactly the lengths that need it most.
            const MAX_COL_FLOATS: usize = 96 << 20; // 384 MB a tile
            let span = (MAX_COL_FLOATS / kk.max(1)).max(1).min(out_n);
            if span < out_n {
                let mut done = 0usize;
                while done < out_n {
                    let w = span.min(out_n - done);
                    // Build the window with its zeros already in it, so
                    // the sub-convolution runs at pad = 0 and both edges
                    // are right by construction: a tile that clamped its
                    // input instead would lose the right-hand padding the
                    // untiled path applies, and the tail would be short.
                    let need = w + self.dilation * (self.k - 1);
                    let mut sub = vec![0f32; self.in_ch * need];
                    for i in 0..self.in_ch {
                        let src = &x[i * n..(i + 1) * n];
                        let dst = &mut sub[i * need..(i + 1) * need];
                        for (u, d) in dst.iter_mut().enumerate() {
                            let p = (done + u) as isize - self.pad as isize;
                            if p >= 0 && (p as usize) < n {
                                *d = src[p as usize];
                            }
                        }
                    }
                    let piece = Self {
                        w: self.w.clone(),
                        b: self.b.clone(),
                        out_ch: self.out_ch,
                        in_ch: self.in_ch,
                        k: self.k,
                        pad: 0,
                        dilation: self.dilation,
                    }
                    .apply(&sub, need, pool);
                    for o in 0..self.out_ch {
                        // SAFETY: single-threaded here.
                        let dst = unsafe { ptr.row(o * out_n + done, w) };
                        dst.copy_from_slice(&piece[o * w..(o + 1) * w]);
                    }
                    done += w;
                }
                return out;
            }
            if let Some(col_len) = kk.checked_mul(out_n) {
                let mut col = vec![0f32; col_len];
                let pc = SendPtr(col.as_mut_ptr());
                let fill = |lo: usize, hi: usize| {
                    for r in lo..hi {
                        let (i, j) = (r / self.k, r % self.k);
                        let src = &x[i * n..(i + 1) * n];
                        // SAFETY: workers own disjoint rows of `col`.
                        let dst = unsafe { pc.row(r * out_n, out_n) };
                        for (t, d) in dst.iter_mut().enumerate() {
                            let p = (t + j * self.dilation) as isize - self.pad as isize;
                            *d = if p >= 0 && (p as usize) < n {
                                src[p as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                };
                match pool {
                    Some(p) => p.run_rows(kk, &fill),
                    None => fill(0, kk),
                }
                // colᵀ is [out_n × kk]; the GEMM wants x[n×k]·wᵀ[m×k],
                // so n = out_n, k = kk, m = out_ch — and `col` is built
                // row-major over kk, so it is transposed on the way in.
                let mut colt = vec![0f32; col_len];
                for r in 0..kk {
                    for t in 0..out_n {
                        colt[t * kk + r] = col[r * out_n + t];
                    }
                }
                let mut yt = vec![0f32; out_n * self.out_ch];
                if crate::gpu::gemm_nt_f32(&colt, &self.w, &mut yt, out_n, kk, self.out_ch) {
                    for o in 0..self.out_ch {
                        let bias = self.b.as_ref().map_or(0.0, |b| b[o]);
                        // SAFETY: single-threaded here.
                        let dst = unsafe { ptr.row(o * out_n, out_n) };
                        for (t, d) in dst.iter_mut().enumerate() {
                            *d = yt[t * self.out_ch + o] + bias;
                        }
                    }
                    return out;
                }
            }
        }
        let tiles = (128 / self.out_ch.max(1)).max(1);
        let tile = out_n.div_ceil(tiles.max(1));
        let rows = self.out_ch * tiles;
        let work = |lo: usize, hi: usize| {
            for r in lo..hi {
                let o = r / tiles;
                let t0 = (r - o * tiles) * tile;
                if t0 >= out_n {
                    continue;
                }
                let len = tile.min(out_n - t0);
                // SAFETY: workers own disjoint (channel, time) slices.
                let dst = unsafe { ptr.row(o * out_n + t0, len) };
                let bias = self.b.as_ref().map_or(0.0, |b| b[o]);
                dst.fill(bias);
                for i in 0..self.in_ch {
                    let ker =
                        &self.w[(o * self.in_ch + i) * self.k..(o * self.in_ch + i + 1) * self.k];
                    let src = &x[i * n..(i + 1) * n];
                    for (tt, d) in dst.iter_mut().enumerate() {
                        let t = t0 + tt;
                        let mut acc = 0f32;
                        for (j, &kv) in ker.iter().enumerate() {
                            let p = (t + j * self.dilation) as isize - self.pad as isize;
                            if p >= 0 && (p as usize) < n {
                                acc += kv * src[p as usize];
                            }
                        }
                        *d += acc;
                    }
                }
            }
        };
        match pool {
            Some(p) => p.run_rows(rows, &work),
            None => work(0, rows),
        }
        out
    }
}

struct ConvT1d {
    w: Vec<f32>, // [in, out, k]
    b: Vec<f32>,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    stride: usize,
    pad: usize,
}

impl ConvT1d {
    fn load(model: &Arc<CmfModel>, name: &str, stride: usize) -> Result<Self, String> {
        let e = model
            .tensor(&format!("{name}.weight"))
            .ok_or_else(|| format!("missing {name}.weight"))?;
        let (in_ch, out_ch, k) = (e.shape[0], e.shape[1], e.shape[2]);
        Ok(Self {
            w: crate::dit::cmf_f32(model, &format!("{name}.weight"))?,
            b: crate::dit::cmf_f32(model, &format!("{name}.bias"))?,
            in_ch,
            out_ch,
            k,
            stride,
            pad: (k - stride) / 2,
        })
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let full = (n - 1) * self.stride + self.k;
        let out_n = full - 2 * self.pad;
        let mut out = vec![0f32; self.out_ch * out_n];
        let ptr = SendPtr(out.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            for o in lo..hi {
                // SAFETY: workers own disjoint output channels.
                let dst = unsafe { ptr.row(o * out_n, out_n) };
                dst.fill(self.b[o]);
                for i in 0..self.in_ch {
                    let ker =
                        &self.w[(i * self.out_ch + o) * self.k..(i * self.out_ch + o + 1) * self.k];
                    let src = &x[i * n..(i + 1) * n];
                    for (t, &sv) in src.iter().enumerate() {
                        if sv == 0.0 {
                            continue;
                        }
                        let base = t * self.stride;
                        for (j, &kv) in ker.iter().enumerate() {
                            let p = base + j;
                            if p >= self.pad && p - self.pad < out_n {
                                dst[p - self.pad] += sv * kv;
                            }
                        }
                    }
                }
            }
        };
        match pool {
            Some(p) => p.run_rows(self.out_ch, &work),
            None => work(0, self.out_ch),
        }
        out
    }
}

/// Plain Snake, `x + sin²(α·x)/α`, with α used EXACTLY as stored.
///
/// The distinction from `SnakeBeta` below is not cosmetic: H3's BigVGAN
/// keeps α and β in log scale and this file exponentiates them on load,
/// while MiniMax-Music-3's decoder reads its α straight
/// (`(alpha + 1e-9).reciprocal() * sin(alpha * x)**2`). Feeding one
/// model's parameters to the other's loader silently raises the
/// activation to an exponent, and the only symptom is worse audio.
struct Snake {
    alpha: Vec<f32>,
}

impl Snake {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        Ok(Self {
            alpha: crate::dit::cmf_f32(model, &format!("{name}.alpha"))?,
        })
    }

    fn apply(&self, x: &mut [f32], n: usize) {
        for (c, row) in x.chunks_exact_mut(n).enumerate() {
            let a = self.alpha[c];
            let inv = 1.0 / (a + 1e-9);
            for v in row.iter_mut() {
                let s = (a * *v).sin();
                *v += s * s * inv;
            }
        }
    }
}

/// One residual unit: Snake → dilated 7-tap → Snake → 1-tap, added back.
struct DavUnit {
    a1: Snake,
    c1: Conv1d,
    a2: Snake,
    c2: Conv1d,
}

impl DavUnit {
    fn load(model: &Arc<CmfModel>, p: &str, dilation: usize) -> Result<Self, String> {
        Ok(Self {
            a1: Snake::load(model, &format!("{p}.block.0"))?,
            c1: Conv1d::load(model, &format!("{p}.block.1"), 3 * dilation, dilation)?,
            a2: Snake::load(model, &format!("{p}.block.2"))?,
            c2: Conv1d::load(model, &format!("{p}.block.3"), 0, 1)?,
        })
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut h = x.to_vec();
        self.a1.apply(&mut h, n);
        let mut h = self.c1.apply(&h, n, pool);
        self.a2.apply(&mut h, n);
        let r = self.c2.apply(&h, n, pool);
        // Same padding throughout, so the residual lines up without the
        // reference's centre-crop; assert rather than trust that.
        debug_assert_eq!(r.len(), x.len());
        x.iter().zip(&r).map(|(a, b)| a + b).collect()
    }
}

/// Snake → transposed conv (×stride) → three residual units at
/// dilations 1, 3, 9.
struct DavStage {
    act: Snake,
    up: ConvT1d,
    units: Vec<DavUnit>,
}

impl DavStage {
    fn load(model: &Arc<CmfModel>, p: &str, stride: usize) -> Result<Self, String> {
        Ok(Self {
            act: Snake::load(model, &format!("{p}.block.0"))?,
            up: ConvT1d::load(model, &format!("{p}.block.1"), stride)?,
            units: [1usize, 3, 9]
                .iter()
                .enumerate()
                .map(|(i, &d)| DavUnit::load(model, &format!("{p}.block.{}", i + 2), d))
                .collect::<Result<_, _>>()?,
        })
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> (Vec<f32>, usize) {
        let mut h = x.to_vec();
        self.act.apply(&mut h, n);
        let mut h = self.up.apply(&h, n, pool);
        let n = n * self.up.stride;
        for u in &self.units {
            h = u.apply(&h, n, pool);
        }
        (h, n)
    }
}

/// MiniMax-Music-3's DAV decoder: latent → 44.1 kHz stereo.
///
/// The 128 latent channels are a STEREO PAIR of 64: the reference folds
/// `[b, 128, t]` to `[b·2, 64, t]`, decodes mono, and unfolds. Reading
/// 128 as one wide latent — which the vocoder config's `latent_channels`
/// invites — decodes noise at half the length.
pub struct Music3Dav {
    dec_in: Conv1d,
    conv_pre: Conv1d,
    stages: Vec<DavStage>,
    act_post: Snake,
    conv_post: Conv1d,
}

impl Music3Dav {
    pub const STRIDES: [usize; 4] = [8, 8, 4, 2];
    /// Audio samples per latent frame: 8·8·4·2.
    pub const HOP: usize = 512;
    pub const SAMPLE_RATE: usize = 44100;

    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        Ok(Self {
            dec_in: Conv1d::load(model, "mvae.dec_in_proj", 0, 1)?,
            conv_pre: Conv1d::load(model, "mvae.decoder.model.0", 3, 1)?,
            stages: Self::STRIDES
                .iter()
                .enumerate()
                .map(|(i, &s)| DavStage::load(model, &format!("mvae.decoder.model.{}", i + 1), s))
                .collect::<Result<_, _>>()?,
            act_post: Snake::load(model, "mvae.decoder.model.5")?,
            conv_post: Conv1d::load(model, "mvae.decoder.model.6", 3, 1)?,
        })
    }

    /// `latent` is `[128, frames]`; the result is interleaved stereo of
    /// `frames · 512` samples per channel.
    pub fn decode(&self, latent: &[f32], frames: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut chans: Vec<Vec<f32>> = Vec::with_capacity(2);
        for half in 0..2 {
            let src = &latent[half * 64 * frames..(half + 1) * 64 * frames];
            let mut h = self.dec_in.apply(src, frames, pool);
            h = self.conv_pre.apply(&h, frames, pool);
            let mut n = frames;
            for st in &self.stages {
                let (nh, nn) = st.apply(&h, n, pool);
                h = nh;
                n = nn;
            }
            self.act_post.apply(&mut h, n);
            let w = self.conv_post.apply(&h, n, pool);
            // The reference ends in tanh; without it a loud latent
            // clips as a wrap rather than a limit.
            chans.push(w.iter().map(|v| v.tanh()).collect());
        }
        let n = chans[0].len();
        let mut out = vec![0f32; n * 2];
        for (i, o) in out.chunks_exact_mut(2).enumerate() {
            o[0] = chans[0][i];
            o[1] = chans[1][i];
        }
        out
    }
}

/// `x + sin²(α·x)/β`, with α and β stored in log scale.
struct SnakeBeta {
    alpha: Vec<f32>,
    beta: Vec<f32>,
}

impl SnakeBeta {
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        Ok(Self {
            alpha: crate::dit::cmf_f32(model, &format!("{name}.alpha"))?
                .iter()
                .map(|v| v.exp())
                .collect(),
            beta: crate::dit::cmf_f32(model, &format!("{name}.beta"))?
                .iter()
                .map(|v| v.exp())
                .collect(),
        })
    }

    fn apply(&self, x: &mut [f32], n: usize) {
        for (c, row) in x.chunks_exact_mut(n).enumerate() {
            let (a, b) = (self.alpha[c], 1.0 / (self.beta[c] + 1e-9));
            for v in row.iter_mut() {
                let s = (a * *v).sin();
                *v += s * s * b;
            }
        }
    }
}

fn bessel_i0(x: f64) -> f64 {
    // Series; the argument here is ~4.7, where a dozen terms is exact
    // to double precision.
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..40 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

/// The reference's `kaiser_sinc_filter1d`, normalized to unit sum.
fn kaiser_sinc(cutoff: f64, half_width: f64, k: usize) -> Vec<f32> {
    let half = k / 2;
    let delta_f = 4.0 * half_width;
    let a = 2.285 * (half as f64 - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if a > 50.0 {
        0.1102 * (a - 8.7)
    } else if a >= 21.0 {
        0.5842 * (a - 21.0).powf(0.4) + 0.078_86 * (a - 21.0)
    } else {
        0.0
    };
    let denom = bessel_i0(beta);
    let n = k as f64 - 1.0;
    let mut f: Vec<f64> = (0..k)
        .map(|i| {
            let r = (2.0 * i as f64 / n) - 1.0;
            let win = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / denom;
            // even length: sample points sit on half-integers
            let t = -(half as f64) + i as f64 + 0.5;
            2.0 * cutoff * win * sinc(2.0 * cutoff * t)
        })
        .collect();
    let s: f64 = f.iter().sum();
    for v in f.iter_mut() {
        *v /= s;
    }
    f.into_iter().map(|v| v as f32).collect()
}

/// Replicate-pad, then a per-channel FIR.
///
/// MEASURED, and smaller than it looked: parallelizing this took the
/// audio stage 9.2 s → 8.1 s, output bit-identical. So the FIR is not
/// where that stage's time goes — the rest is in the convolutions,
/// which DO use the pool but split over OUTPUT CHANNELS, and this
/// decoder narrows to a handful of them near the output where the
/// samples are longest. Splitting those over TIME instead is the next
/// thing to try, and it wants a phase profiler first: this decoder
/// still has none.
///
/// Original note: the audio decoder is 9.2 s of a 95.6 s render — the same share the video decoder had
/// before its host loops met the thread pool (38.3 s → 16.6 s). This
/// function is the shape that fix wanted: every channel is independent
/// and the whole thing runs on one thread. The convolutions above DO
/// use the pool, but they split over OUTPUT CHANNELS, and this decoder
/// narrows to a handful of them near the output — where the samples
/// are longest and the parallelism collapses exactly when it is needed.
///
/// It needs a `pool` argument threaded from `Activation1d`/`AudioVae`
/// and a per-worker `buf` instead of the shared one. Measure first with
/// a phase profiler like `CMF_VAE3D_PROF`: this decoder has none, and
/// the video one went untuned for a whole session precisely because
/// nothing measured it.
#[allow(clippy::too_many_arguments)]
fn fir_pad(
    x: &[f32],
    ch: usize,
    n: usize,
    f: &[f32],
    pad_l: usize,
    pad_r: usize,
    stride: usize,
    pool: Option<&Pool>,
) -> (Vec<f32>, usize) {
    let padded = n + pad_l + pad_r;
    let out_n = (padded - f.len()) / stride + 1;
    let mut out = vec![0f32; ch * out_n];
    struct P(*mut f32);
    // SAFETY: each channel owns `out[c*out_n .. (c+1)*out_n]` and no
    // two workers take the same channel.
    unsafe impl Send for P {}
    unsafe impl Sync for P {}
    impl P {
        // Through a method, so the closure captures the WRAPPER and not
        // the bare pointer — 2021 captures disjoint fields, and a
        // captured `*mut f32` is neither Send nor Sync.
        #[allow(clippy::mut_from_ref)]
        pub(crate) unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
            unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
        }
    }
    let po = P(out.as_mut_ptr());
    let work = |lo: usize, hi: usize| {
        // Per worker, not shared: the old single `buf` is what kept this
        // on one thread.
        let mut buf = vec![0f32; padded];
        for c in lo..hi {
            let src = &x[c * n..(c + 1) * n];
            for (i, b) in buf.iter_mut().enumerate() {
                let p = i as isize - pad_l as isize;
                *b = src[p.clamp(0, n as isize - 1) as usize];
            }
            let dst = unsafe { po.row(c * out_n, out_n) };
            for (t, d) in dst.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (j, &kv) in f.iter().enumerate() {
                    acc += kv * buf[t * stride + j];
                }
                *d = acc;
            }
        }
    };
    match pool {
        Some(p) => p.run_rows(ch, &work),
        None => work(0, ch),
    }
    (out, out_n)
}

/// Upsample ×2, apply, downsample ×2 — the anti-aliased activation.
struct Activation1d {
    act: SnakeBeta,
    up: Vec<f32>,
    down: Vec<f32>,
}

impl Activation1d {
    /// `name` is the Activation1d module, not its `.act`. The release
    /// ships both resampling filters as buffers — 254 of them — so read
    /// them rather than re-designing them, and keep `kaiser_sinc` as
    /// the fallback for a checkpoint that drops them.
    fn load(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        let designed = || kaiser_sinc(0.25, 0.3, FILTER_LEN);
        Ok(Self {
            act: SnakeBeta::load(model, &format!("{name}.act"))?,
            up: crate::dit::cmf_f32(model, &format!("{name}.upsample.filter"))
                .unwrap_or_else(|_| designed()),
            down: crate::dit::cmf_f32(model, &format!("{name}.downsample.lowpass.filter"))
                .unwrap_or_else(|_| designed()),
        })
    }

    fn apply(&self, x: &[f32], ch: usize, n: usize, pool: Option<&Pool>) -> (Vec<f32>, usize) {
        // conv_transpose1d(pad(x, 5, 5), filter, stride 2) · 2, then the
        // 15-sample margins the reference trims off each end.
        let pad = FILTER_LEN / 2 - 1;
        let pad_l = pad * 2 + (FILTER_LEN - 2) / 2;
        let pad_r = pad * 2 + (FILTER_LEN - 2).div_ceil(2);
        let pn = n + 2 * pad;
        let full = (pn - 1) * 2 + FILTER_LEN;
        let mut up = vec![0f32; ch * full];
        for c in 0..ch {
            let src = &x[c * n..(c + 1) * n];
            let dst = &mut up[c * full..(c + 1) * full];
            for i in 0..pn {
                let p = i as isize - pad as isize;
                let v = src[p.clamp(0, n as isize - 1) as usize] * 2.0;
                if v == 0.0 {
                    continue;
                }
                for (j, &kv) in self.up.iter().enumerate() {
                    dst[i * 2 + j] += v * kv;
                }
            }
        }
        let keep = full - pad_l - pad_r;
        let mut mid = vec![0f32; ch * keep];
        for c in 0..ch {
            mid[c * keep..(c + 1) * keep]
                .copy_from_slice(&up[c * full + pad_l..c * full + pad_l + keep]);
        }
        self.act.apply(&mut mid, keep);
        // LowPassFilter1d at stride 2: even kernel pads 5 left, 6 right.
        fir_pad(
            &mid,
            ch,
            keep,
            &self.down,
            FILTER_LEN / 2 - 1,
            FILTER_LEN / 2,
            2,
            pool,
        )
    }
}

struct AmpBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    acts: Vec<Activation1d>,
}

pub struct AudioVae {
    dec_in: Conv1d,
    conv_pre: Conv1d,
    ups: Vec<ConvT1d>,
    resblocks: Vec<AmpBlock>,
    act_post: Activation1d,
    conv_post: Conv1d,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    pool: Option<Arc<Pool>>,
    n_kernels: usize,
    pub sample_rate: usize,
}

fn get_padding(k: usize, d: usize) -> usize {
    (k * d - d) / 2
}

impl AudioVae {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("avae.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("avae.config_json: {e}"))?;
        let rates: Vec<usize> = cfg["upsample_rates"]
            .as_array()
            .ok_or("upsample_rates")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(1) as usize)
            .collect();
        let rk: Vec<usize> = cfg["resblock_kernel_sizes"]
            .as_array()
            .ok_or("resblock_kernel_sizes")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(3) as usize)
            .collect();
        let rd: Vec<Vec<usize>> = cfg["resblock_dilation_sizes"]
            .as_array()
            .ok_or("resblock_dilation_sizes")?
            .iter()
            .map(|a| {
                a.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(1) as usize)
                    .collect()
            })
            .collect();

        let mut ups = Vec::new();
        for (i, &u) in rates.iter().enumerate() {
            ups.push(ConvT1d::load(model, &format!("avae.decoder.ups.{i}.0"), u)?);
        }
        let mut resblocks = Vec::new();
        for i in 0..rates.len() {
            for (j, (&k, d)) in rk.iter().zip(&rd).enumerate() {
                let p = format!("avae.decoder.resblocks.{}", i * rk.len() + j);
                let convs1 = (0..d.len())
                    .map(|q| {
                        Conv1d::load(
                            model,
                            &format!("{p}.convs1.{q}"),
                            get_padding(k, d[q]),
                            d[q],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let convs2 = (0..d.len())
                    .map(|q| Conv1d::load(model, &format!("{p}.convs2.{q}"), get_padding(k, 1), 1))
                    .collect::<Result<Vec<_>, _>>()?;
                let acts = (0..convs1.len() + convs2.len())
                    .map(|q| Activation1d::load(model, &format!("{p}.activations.{q}")))
                    .collect::<Result<Vec<_>, _>>()?;
                resblocks.push(AmpBlock {
                    convs1,
                    convs2,
                    acts,
                });
            }
        }
        Ok(Self {
            dec_in: Conv1d::load(model, "avae.dec_in_proj", 0, 1)?,
            conv_pre: Conv1d::load(model, "avae.decoder.conv_pre", 3, 1)?,
            ups,
            resblocks,
            act_post: Activation1d::load(model, "avae.decoder.activation_post")?,
            conv_post: Conv1d::load(model, "avae.decoder.conv_post", 3, 1)?,
            latents_mean: crate::dit::cmf_f32(model, "avae.latents_mean")?,
            latents_std: crate::dit::cmf_f32(model, "avae.latents_std")?,
            pool: Pool::from_env(),
            n_kernels: rk.len(),
            sample_rate: cfg["sample_rate"].as_u64().unwrap_or(32000) as usize,
        })
    }

    /// Normalized latents `[C, 2, T]` → stereo `[2, L]` in [-1, 1].
    ///
    /// THE REMAINING WIN IS THIS LOOP: the two stereo channels are two
    /// complete, independent decoder passes and they run one after the
    /// other. That is a factor of two sitting outside the exact place
    /// the inside cannot use — the convolutions parallelize over output
    /// channels, and this decoder narrows to a handful of those near the
    /// output where the samples are longest (measured: parallelizing the
    /// per-channel FIR bought only 9.2 s → 8.1 s, so the time is in the
    /// convolutions, not the filter).
    ///
    /// The nesting question is ANSWERED, and the answer forbids the
    /// obvious version: `Pool` keeps ONE job slot (`inner.slot`), and
    /// `run()` writes it on the stated assumption that no job is in
    /// flight. Two concurrent callers would overwrite each other's job.
    /// So wrapping this loop in a `thread::scope` while the passes still
    /// call into the pool is a data race, not an optimization.
    ///
    /// Two shapes remain. Drive the pool from the OUTSIDE — one row per
    /// stereo channel, `None` inside — which is correct but caps the
    /// whole decode at two threads and would lose the wide layers what
    /// it wins the narrow ones. Or split the narrow convolutions over
    /// TIME instead of output channels, which keeps every thread busy
    /// at both ends. Measure before choosing: this decoder still has no
    /// phase profiler, and the FIR already proved the shape of the code
    /// is a poor guide to where its seconds are.
    pub fn decode(&self, z: &[f32], c: usize, t: usize) -> (Vec<f32>, usize) {
        let pool = self.pool.as_deref();
        let mut chans: Vec<Vec<f32>> = Vec::with_capacity(2);
        for ch in 0..2 {
            let mut lat = vec![0f32; c * t];
            for ci in 0..c {
                let (m, s) = (self.latents_mean[ci], self.latents_std[ci]);
                for ti in 0..t {
                    lat[ci * t + ti] = z[(ci * 2 + ch) * t + ti] * s + m;
                }
            }
            // `CMF_AVAE_PROF=1`: per-stage rms, to diff against the
            // reference stage by stage rather than at the waveform.
            let prof = std::env::var_os("CMF_AVAE_PROF").is_some();
            let rms = |x: &[f32]| {
                (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
            };
            let tt = std::time::Instant::now();
            let mut x = self.dec_in.apply(&lat, t, pool);
            atime(0, tt);
            let mut n = t;
            if prof {
                eprintln!("ch{ch} dec_in rms {:.6e} n {n}", rms(&x));
            }
            let tt = std::time::Instant::now();
            x = self.conv_pre.apply(&x, n, pool);
            atime(1, tt);
            if prof {
                eprintln!("ch{ch} conv_pre rms {:.6e} n {n}", rms(&x));
            }
            for i in 0..self.ups.len() {
                let up = &self.ups[i];
                let tt = std::time::Instant::now();
                x = up.apply(&x, n, pool);
                atime(2, tt);
                n = (n - 1) * up.stride + up.k - 2 * up.pad;
                let ch_n = up.out_ch;
                let mut acc = vec![0f32; ch_n * n];
                for j in 0..self.n_kernels {
                    let tt = std::time::Instant::now();
                    let r = self.resblocks[i * self.n_kernels + j].apply(&x, ch_n, n, pool);
                    atime(3, tt);
                    for (a, b) in acc.iter_mut().zip(&r) {
                        *a += b;
                    }
                }
                let inv = 1.0 / self.n_kernels as f32;
                for v in acc.iter_mut() {
                    *v *= inv;
                }
                x = acc;
                if prof {
                    eprintln!("ch{ch} up{i} rms {:.6e} ch {ch_n} n {n}", rms(&x));
                }
            }
            let last_ch = self.ups[self.ups.len() - 1].out_ch;
            let tt = std::time::Instant::now();
            let (mut y, yn) = self.act_post.apply(&x, last_ch, n, pool);
            atime(4, tt);
            let tt = std::time::Instant::now();
            y = self.conv_post.apply(&y, yn, pool);
            atime(5, tt);
            for v in y.iter_mut() {
                *v = v.clamp(-1.0, 1.0);
            }
            chans.push(y);
            n = yn;
            let _ = n;
        }
        let len = chans[0].len().min(chans[1].len());
        let mut out = vec![0f32; 2 * len];
        for (ch, c) in chans.iter().enumerate() {
            out[ch * len..(ch + 1) * len].copy_from_slice(&c[..len]);
        }
        (out, len)
    }
}

impl AmpBlock {
    fn apply(&self, x: &[f32], ch: usize, n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut cur = x.to_vec();
        for i in 0..self.convs1.len() {
            let (a1, a2) = (&self.acts[i * 2], &self.acts[i * 2 + 1]);
            let (xt, tn) = a1.apply(&cur, ch, n, pool);
            let xt = self.convs1[i].apply(&xt, tn, pool);
            let (xt, tn2) = a2.apply(&xt, ch, tn, pool);
            let xt = self.convs2[i].apply(&xt, tn2, pool);
            for (a, b) in cur.iter_mut().zip(&xt) {
                *a += b;
            }
        }
        cur
    }
}

/// Test hook: the designed 12-tap resampling filter.
#[doc(hidden)]
pub fn kaiser_sinc_for_test() -> Vec<f32> {
    kaiser_sinc(0.25, 0.3, FILTER_LEN)
}

pub(crate) struct SendPtr(pub *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode through the packed MiniMax-Music-3 vocoder and check the
    /// two things the reference fixes exactly: 512 samples per latent
    /// frame per side, and a `tanh` range. `CMF_MUSIC3_VAE=<file.cmf>`
    /// points at a pack; without it there is nothing to test against.
    #[test]
    fn music3_dav_decodes_to_the_reference_geometry() {
        let Ok(p) = std::env::var("CMF_MUSIC3_VAE") else {
            eprintln!("CMF_MUSIC3_VAE unset — skipping Music-3 vocoder test");
            return;
        };
        let model = Arc::new(CmfModel::open(&p).expect("open packed vocoder"));
        let dav = Music3Dav::from_cmf(&model).expect("load DAV");
        let frames = 12usize;
        // A latent with structure rather than noise: a decoder that has
        // silently lost a stage still returns *something* for noise.
        let latent: Vec<f32> = (0..128 * frames)
            .map(|i| {
                let (c, t) = (i / frames, i % frames);
                0.4 * ((c as f32 * 0.13 + t as f32 * 0.7).sin())
            })
            .collect();
        let pcm = dav.decode(&latent, frames, None);
        assert_eq!(
            pcm.len(),
            frames * Music3Dav::HOP * 2,
            "512 samples a frame, two sides interleaved"
        );
        assert!(pcm.iter().all(|v| v.is_finite()), "non-finite sample");
        assert!(
            pcm.iter().all(|v| v.abs() <= 1.0),
            "tanh range violated: {}",
            pcm.iter().fold(0f32, |m, v| m.max(v.abs()))
        );
        // Silence would also satisfy the above; a working stack moves.
        let rms = (pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len() as f32).sqrt();
        assert!(rms > 1e-4, "decoded to near-silence, rms {rms}");
        let (l, r): (Vec<f32>, Vec<f32>) = pcm.chunks_exact(2).map(|c| (c[0], c[1])).unzip();
        let d = l
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            d > 0.0,
            "both sides identical — the 128 latent was not split"
        );
        eprintln!(
            "music3 dav: {} samples/side, rms {rms:.4}, L-R max {d:.4}",
            l.len()
        );
    }

    #[test]
    fn the_resampling_filter_is_the_references() {
        let f = kaiser_sinc(0.25, 0.3, FILTER_LEN);
        assert_eq!(f.len(), FILTER_LEN);
        // Unit sum: without it a constant input leaks amplitude, which
        // is the whole reason the reference normalizes.
        assert!((f.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Symmetric about the centre, and its peak is at the centre.
        for i in 0..FILTER_LEN / 2 {
            assert!(
                (f[i] - f[FILTER_LEN - 1 - i]).abs() < 1e-6,
                "asymmetric at {i}"
            );
        }
        let peak = f.iter().cloned().fold(f32::MIN, f32::max);
        assert!((f[5] - peak).abs() < 1e-6);
    }

    #[test]
    fn bessel_i0_matches_known_values() {
        // The third is the β the 12-tap filter's Kaiser window is
        // designed at, so it is the value that actually gets used.
        for (x, want) in [(0.0, 1.0), (1.0, 1.266_065_878), (4.664, 20.204_6)] {
            let got = bessel_i0(x);
            assert!((got - want).abs() < 1e-3 * want.max(1.0), "I0({x}) = {got}");
        }
    }
}
