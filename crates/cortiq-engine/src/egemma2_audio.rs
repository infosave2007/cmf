//! EmbeddingGemma 2 audio: the `gemma4_audio` tower and the `<|audio|>`
//! path into the shared 768-d space, read from the same `.cmf` as the text
//! encoder (`audio_tower.*`, `embed_audio.*`).
//!
//! The chain reproduces transformers' `Gemma4AudioFeatureExtractor`,
//! `Gemma4AudioModel` and `EmbeddingGemma2MultimodalEmbedder`:
//!
//! 1. **Waveform.** Mono f32 at 16 kHz ([`decode_audio`] mixes channels down
//!    and resamples other rates; WAV is decoded natively, anything else goes
//!    through `ffmpeg` when it is installed). Clips are cut at 30 s
//!    (480 000 samples), as the feature extractor's default truncation does.
//! 2. **Log-mel.** 160 zeros in front ("semicausal"), 320-sample frames every
//!    160 samples, periodic Hann window, |rfft| at 512 points, 128 HTK mel
//!    bands over 0–8 kHz (no filter norm), `ln(mel + 0.001)`. A frame is
//!    kept only while its window lies inside the clip, so a clip of `N`
//!    samples gives `⌈(N − 160)/160⌉` frames; the reference pads the batch
//!    to a multiple of 128 samples and masks those frames out, which is the
//!    same as never computing them.
//! 3. **Subsampling.** Two 3×3 stride-2 convolutions (1→128→32 channels,
//!    zero padding 1, no bias), each followed by a bias-free LayerNorm over
//!    the channels and ReLU; the `[T/4, 32 freq, 32 ch]` grid flattens
//!    (frequency-major) to 1024 and is projected 1024→1024. `⌈frames/4⌉`
//!    soft tokens: 25 per second.
//! 4. **12 Conformer layers** (width 1024, 8 heads of 128), each
//!    `x = FFN₁(x)`, chunked local attention, light conv, `FFN₂`, `RMSNorm`:
//!    * FFN: `x + ½·N(W₂ silu(W₁ N(x)))`;
//!    * attention: a token sees itself and the 11 tokens before it (none
//!      after); `q·(log2 e)/√128·softplus(per_dim_scale)`, `k·log2(1+e)`,
//!      logits `q·k + q·R[d]` with `R[d]` the relative key of distance `d`
//!      (a sinusoid through `relative_k_proj`), soft-capped at 50 by tanh;
//!      `x + N(W_post attn)`;
//!    * light conv: `x + W_end silu(N(dwconv₅(glu(W_start N(x)))))`, the
//!      depthwise convolution causal (4 taps back);
//!    * every projection but `relative_k_proj` clamps its input and output
//!      to the trained bounds the checkpoint stores (`input_min/max`,
//!      `output_min/max`).
//!
//!    Every RMS-norm multiplies by its weight plainly (not `1 + w`).
//! 5. **Into the text space.** `W_out x + b` (1024→1536), a weightless
//!    RMS-norm, then `embed_audio` 1536→512: one row per `<|audio|>`
//!    placeholder, written into the token sequence unscaled — the text
//!    encoder ([`EmbeddingGemma2::embed_merged`]) does the rest.
//!
//! The sequence of an audio-only input is `[BOS] <|audio> <|audio|>×n
//! <audio|> [EOS]` (no task prompt: media take none). An input with text
//! carries one `<|audio|>` per clip, each expanded the same way in place.
//!
//! Checked against the float32 reference in `tests/egemma2_audio_parity.rs`.

use crate::egemma2::{EmbeddingGemma2, Mat, TextInput, rms_into, rms_rows, vecf};
use crate::ltxdit::{Shared, rows};
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::{Arc, Mutex};

/// The rate every clip is brought to.
pub const SAMPLE_RATE: u32 = 16_000;
/// The feature extractor's default truncation: 30 s.
pub const MAX_SAMPLES: usize = 480_000;
/// Analysis window and hop, in samples (20 ms / 10 ms).
pub const FRAME: usize = 320;
pub const HOP: usize = 160;
/// FFT size (the frame is zero-padded to it).
pub const N_FFT: usize = 512;
/// Mel bands.
pub const N_MELS: usize = 128;
/// Added before the log.
pub const MEL_FLOOR: f64 = 1e-3;
/// The text placeholder of one clip.
pub const PLACEHOLDER: &str = "<|audio|>";
/// Token ids of the release (`config.json`): `audio_token_id`,
/// `boa_token_id`, `eoa_token_index`.
pub const AUDIO_TOKEN: u32 = 258_881;
pub const BOA_TOKEN: u32 = 256_000;
pub const EOA_TOKEN: u32 = 258_883;

const EPS: f64 = 1e-6;
/// Soft tokens per packed tower forward.
const PACK_SOFT: usize = 4096;

/// `CMF_EGEMMA2_PROF=1`: the time split of every audio embedding call on
/// stderr (front end, subsampling, projections, attention core, text).
mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static MEL: AtomicU64 = AtomicU64::new(0);
    pub static SUB: AtomicU64 = AtomicU64::new(0);
    pub static LIN: AtomicU64 = AtomicU64::new(0);
    pub static ATT: AtomicU64 = AtomicU64::new(0);
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

// ───────────────────────────── sizes ─────────────────────────────

/// Mel frames of a clip of `n` samples (after the 30 s cut).
pub fn num_frames(n: usize) -> usize {
    let n = n.min(MAX_SAMPLES);
    if n <= HOP { 0 } else { (n - HOP).div_ceil(HOP) }
}

/// Soft tokens of a clip of `n` samples: two stride-2 convolutions over the
/// frames. 25 per second, at most 750.
pub fn num_tokens(n: usize) -> usize {
    num_frames(n).div_ceil(2).div_ceil(2)
}

// ───────────────────────────── decoding ─────────────────────────────

/// Mono f32 at 16 kHz from an audio file's bytes. WAV (PCM 8/16/24/32,
/// float 32/64, any channel count and rate) is decoded here: channels are
/// averaged, then a rate other than 16 kHz goes through [`resample_hq`].
/// Other containers (mp3, flac, ogg, m4a, …) are decoded by `ffmpeg` to
/// float WAV at their own rate and come through the same path.
pub fn decode_audio(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let wav = if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        crate::mimo_audio::decode_wav(bytes)?
    } else {
        crate::mimo_audio::decode_wav(&ffmpeg_to_wav(bytes)?)?
    };
    Ok(to_mono_16k(&wav))
}

/// Read and decode an audio file (see [`decode_audio`]).
pub fn read_audio(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_audio(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Mono 16 kHz from a request's audio source: a string (a `data:` URL, an
/// `http(s)://` URL, or a local path, `file://` optional), or an object
/// `{"data": <base64>, "format"?}`, `{"url": …}`, `{"path": …}`, or the
/// OpenAI chat shape `{"input_audio": {"data", "format"}}`.
pub fn load_audio_source(v: &serde_json::Value) -> Result<Vec<f32>, String> {
    use serde_json::{Value, json};
    let record = match v {
        Value::String(s) => json!({ "url": s }),
        Value::Object(m) => {
            if let Some(inner) = m.get("input_audio") {
                return load_audio_source(inner);
            }
            if let Some(p) = m.get("path").and_then(Value::as_str) {
                json!({ "url": p })
            } else {
                v.clone()
            }
        }
        _ => return Err("an audio source must be a string or an object".into()),
    };
    let bytes = crate::media::load_image_bytes(&record).map_err(|e| e.replace("image", "audio"))?;
    decode_audio(&bytes)
}

/// Average the channels, then resample to 16 kHz.
pub fn to_mono_16k(wav: &crate::mimo_audio::Wav) -> Vec<f32> {
    let n = wav.frames();
    let mono: Vec<f32> = match wav.channels.len() {
        0 => Vec::new(),
        1 => wav.channels[0].clone(),
        c => (0..n)
            .map(|i| wav.channels.iter().map(|ch| ch[i]).sum::<f32>() / c as f32)
            .collect(),
    };
    resample_hq(&mono, wav.sample_rate, SAMPLE_RATE)
}

/// Zero crossings of the sinc on each side, cut-off as a fraction of the
/// lower Nyquist, and the Kaiser β of [`resample_hq`].
const RS_WIDTH: f64 = 64.0;
const RS_ROLLOFF: f64 = 0.958;
const RS_BETA: f64 = 12.8;
/// Above this many table entries the taps are computed on the fly.
const RS_TABLE_MAX: usize = 1 << 22;

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let (mut term, mut sum) = (1.0f64, 1.0f64);
    for k in 1..200 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Band-limited resampling: a polyphase Kaiser-windowed sinc (64 zero
/// crossings a side, β 12.8, cut-off at 0.958 of the lower Nyquist),
/// linear phase, `⌈to·N/from⌉` samples, f64 accumulation.
///
/// It is shaped after the resampler the reference ecosystem actually uses —
/// librosa's default `soxr_hq`, which `transformers.audio_utils.load_audio`
/// calls — and lands within 0.002 (8 kHz) / 0.0006 (44.1 and 48 kHz) mean
/// absolute log-mel of it, against 0.35 / 0.013 for torchaudio's default
/// Hann sinc (width 6), whose images leak into the empty band of an
/// upsampled clip where the log-mel floor makes every bit of energy count.
pub fn resample_hq(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let g = {
        let (mut a, mut b) = (from as u64, to as u64);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let (o, nw) = ((from as u64 / g) as usize, (to as u64 / g) as usize);
    let len = x.len();
    let target = ((nw as u128 * len as u128).div_ceil(o as u128)) as usize;
    let base = o.min(nw) as f64 * RS_ROLLOFF;
    let w = (RS_WIDTH * o as f64 / base).ceil() as usize;
    let taps = 2 * w + o;
    let i0b = bessel_i0(RS_BETA);
    let scale = base / o as f64;
    let coef = |j: usize, i: usize| -> f64 {
        let t = (-(j as f64) / nw as f64 + (i as f64 - w as f64) / o as f64) * base;
        if t.abs() >= RS_WIDTH {
            return 0.0;
        }
        let r = t / RS_WIDTH;
        let win = bessel_i0(RS_BETA * (1.0 - r * r).sqrt()) / i0b;
        let pt = t * std::f64::consts::PI;
        let sinc = if pt == 0.0 { 1.0 } else { pt.sin() / pt };
        sinc * win * scale
    };
    // per phase: the span of taps inside the window, and their values
    let table = nw * taps <= RS_TABLE_MAX;
    let span = |j: usize| -> (usize, usize) {
        let c = w as f64 + o as f64 * j as f64 / nw as f64;
        let h = RS_WIDTH * o as f64 / base;
        let lo = (c - h).floor().max(0.0) as usize;
        let hi = ((c + h).ceil() as usize + 1).min(taps);
        (lo, hi)
    };
    let rows: Vec<(usize, Vec<f64>)> = if table {
        (0..nw)
            .map(|j| {
                let (lo, hi) = span(j);
                (lo, (lo..hi).map(|i| coef(j, i)).collect())
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut out = vec![0f32; target];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let chunk = target.div_ceil(threads).max(4096);
    std::thread::scope(|sc| {
        for (ci, dst) in out.chunks_mut(chunk).enumerate() {
            let (rows, coef, span) = (&rows, &coef, &span);
            sc.spawn(move || {
                let mut tmp = Vec::new();
                for (k, y) in dst.iter_mut().enumerate() {
                    let n_out = ci * chunk + k;
                    let (f, j) = (n_out / nw, n_out % nw);
                    let (lo, ks): (usize, &[f64]) = if table {
                        (rows[j].0, &rows[j].1)
                    } else {
                        let (lo, hi) = span(j);
                        tmp.clear();
                        tmp.extend((lo..hi).map(|i| coef(j, i)));
                        (lo, &tmp)
                    };
                    // padded index p = f·o + i holds x[p − w]
                    let p0 = (f * o + lo) as isize - w as isize;
                    let mut acc = 0f64;
                    for (m, &kv) in ks.iter().enumerate() {
                        let p = p0 + m as isize;
                        if p >= 0 && (p as usize) < len {
                            acc += kv * x[p as usize] as f64;
                        }
                    }
                    *y = acc as f32;
                }
            });
        }
    });
    out
}

/// Any container `ffmpeg` reads → 32-bit float WAV at the source rate and
/// channel count (our own resampler and mixdown run after).
fn ffmpeg_to_wav(bytes: &[u8]) -> Result<Vec<u8>, String> {
    // A temp file, not a pipe: mp4/m4a keep their index at the end.
    let dir = std::env::temp_dir();
    let src = dir.join(format!(
        "cortiq-egemma2-audio-{}-{}.in",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    std::fs::write(&src, bytes).map_err(|e| format!("audio: temp file: {e}"))?;
    let out = std::process::Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(&src)
        .args(["-vn", "-f", "wav", "-acodec", "pcm_f32le", "pipe:1"])
        .output();
    let _ = std::fs::remove_file(&src);
    match out {
        Err(_) => Err(
            "audio: not a WAV file, and ffmpeg (needed for other formats) is not installed".into(),
        ),
        Ok(o) if !o.status.success() || o.stdout.len() < 44 => {
            // the last line says why; the earlier ones name the temp file
            let err = String::from_utf8_lossy(&o.stderr);
            let why = err
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("")
                .trim();
            Err(format!(
                "audio: not a WAV file, and ffmpeg could not decode it either ({why})"
            ))
        }
        Ok(o) => Ok(o.stdout),
    }
}

// ───────────────────────────── log-mel ─────────────────────────────

/// `np.linspace(a, b, n)`: `a + i·step`, the last point exactly `b`.
fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let step = (b - a) / (n - 1) as f64;
    let mut v: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
    v[n - 1] = b;
    v
}

/// transformers' `mel_filter_bank(257, 128, 0, 8000, 16000, norm=None,
/// mel_scale="htk")` in f64, `[257 freq][128 mel]`.
pub fn mel_filter_bank() -> Vec<f64> {
    let n_freq = N_FFT / 2 + 1;
    let hz2mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel2hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let mels = linspace(hz2mel(0.0), hz2mel(8000.0), N_MELS + 2);
    let ff: Vec<f64> = mels.iter().map(|&m| mel2hz(m)).collect();
    let fft = linspace(0.0, (SAMPLE_RATE / 2) as f64, n_freq);
    let mut out = vec![0f64; n_freq * N_MELS];
    for (k, &f) in fft.iter().enumerate() {
        for m in 0..N_MELS {
            let down = -(ff[m] - f) / (ff[m + 1] - ff[m]);
            let up = (ff[m + 2] - f) / (ff[m + 2] - ff[m + 1]);
            out[k * N_MELS + m] = 0f64.max(down.min(up));
        }
    }
    out
}

/// The feature extractor: Hann window, mel bank, FFT plan.
pub struct MelFrontend {
    window: Vec<f32>,
    /// `[257][128]`
    filters: Vec<f64>,
    fft: Arc<dyn rustfft::Fft<f64>>,
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontend {
    pub fn new() -> Self {
        // periodic Hann (`window_function(320)`), stored as float32
        let window = (0..FRAME)
            .map(|n| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos()) as f32
            })
            .collect();
        let fft = rustfft::FftPlanner::<f64>::new().plan_fft_forward(N_FFT);
        MelFrontend {
            window,
            filters: mel_filter_bank(),
            fft,
        }
    }

    /// Log-mel rows `[frames, 128]` of a mono 16 kHz clip, and the frame
    /// count. The window product is f32 (as the reference's float32
    /// frames), the FFT f64 with the magnitude rounded to f32 (numpy's
    /// float32 rfft), the mel sum and log f64.
    pub fn log_mel(&self, wave: &[f32], pool: Option<&Pool>) -> (Vec<f32>, usize) {
        let wave = &wave[..wave.len().min(MAX_SAMPLES)];
        let n = num_frames(wave.len());
        let mut out = vec![0f32; n * N_MELS];
        let dst = Shared(out.as_mut_ptr());
        let n_freq = N_FFT / 2 + 1;
        rows(pool, n, &|s, e| {
            let o = unsafe { dst.at(s * N_MELS, (e - s) * N_MELS) };
            let mut buf = vec![rustfft::num_complex::Complex::<f64>::default(); N_FFT];
            let mut scratch = vec![
                rustfft::num_complex::Complex::<f64>::default();
                self.fft.get_inplace_scratch_len()
            ];
            let mut mag = vec![0f64; n_freq];
            for (i, row) in (s..e).zip(o.chunks_exact_mut(N_MELS)) {
                for (j, b) in buf.iter_mut().enumerate() {
                    // padded sample p = i·hop + j is wave[p − 160]
                    let v = if j < FRAME {
                        let p = i * HOP + j;
                        let x = if p >= HOP { wave[p - HOP] } else { 0.0 };
                        (x * self.window[j]) as f64
                    } else {
                        0.0
                    };
                    *b = rustfft::num_complex::Complex::new(v, 0.0);
                }
                self.fft.process_with_scratch(&mut buf, &mut scratch);
                for (m, c) in mag.iter_mut().zip(&buf[..n_freq]) {
                    *m = (c.re.hypot(c.im) as f32) as f64;
                }
                for (b, r) in row.iter_mut().enumerate() {
                    let mut acc = 0f64;
                    for (k, &m) in mag.iter().enumerate() {
                        acc += m * self.filters[k * N_MELS + b];
                    }
                    *r = (acc + MEL_FLOOR).ln() as f32;
                }
            }
        });
        (out, n)
    }
}

// ───────────────────────────── the tower ─────────────────────────────

/// A projection that clamps its input and output to stored bounds
/// (`Gemma4ClippableLinear`).
struct ClipLin {
    w: Mat,
    lo_in: f32,
    hi_in: f32,
    lo_out: f32,
    hi_out: f32,
}

fn scalar(model: &CmfModel, name: &str, default: f32) -> Result<f32, String> {
    if model.tensor(name).is_none() {
        return Ok(default);
    }
    Ok(vecf(model, name)?.first().copied().unwrap_or(default))
}

impl ClipLin {
    fn load(model: &Arc<CmfModel>, p: &str, dequant: bool) -> Result<Self, String> {
        Ok(ClipLin {
            w: Mat::load(model, &format!("{p}.linear.weight"), dequant)?,
            lo_in: scalar(model, &format!("{p}.input_min"), f32::NEG_INFINITY)?,
            hi_in: scalar(model, &format!("{p}.input_max"), f32::INFINITY)?,
            lo_out: scalar(model, &format!("{p}.output_min"), f32::NEG_INFINITY)?,
            hi_out: scalar(model, &format!("{p}.output_max"), f32::INFINITY)?,
        })
    }

    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut xc: Vec<f32>;
        let x = if self.lo_in.is_finite() || self.hi_in.is_finite() {
            xc = x.to_vec();
            let (lo, hi) = (self.lo_in, self.hi_in);
            par_chunks(&mut xc, pool, &|_, c| {
                for v in c {
                    *v = v.clamp(lo, hi);
                }
            });
            &xc[..]
        } else {
            x
        };
        let t0 = std::time::Instant::now();
        let mut y = self.w.apply(x, n, pool);
        prof::add(&prof::LIN, t0);
        if self.lo_out.is_finite() || self.hi_out.is_finite() {
            let (lo, hi) = (self.lo_out, self.hi_out);
            par_chunks(&mut y, pool, &|_, c| {
                for v in c {
                    *v = v.clamp(lo, hi);
                }
            });
        }
        y
    }
}

struct Ffn {
    pre: Vec<f32>,
    l1: ClipLin,
    l2: ClipLin,
    post: Vec<f32>,
}

struct ALayer {
    ff1: Ffn,
    ff2: Ffn,
    norm_pre_attn: Vec<f32>,
    norm_post_attn: Vec<f32>,
    norm_out: Vec<f32>,
    q: ClipLin,
    k: ClipLin,
    v: ClipLin,
    post: ClipLin,
    /// `softplus(per_dim_scale)`, `[head_dim]`
    q_dim: Vec<f32>,
    /// relative keys by distance, `[distances, heads·head_dim]`
    rel: Vec<f32>,
    lc_pre: Vec<f32>,
    lc_start: ClipLin,
    /// depthwise taps `[channels][kernel]`
    lc_dw: Vec<f32>,
    lc_norm: Vec<f32>,
    lc_end: ClipLin,
}

/// The `gemma4_audio` encoder with its projection into the text width.
pub struct AudioTower {
    frontend: MelFrontend,
    /// `[128 out][1 in][3][3]`
    conv0: Vec<f32>,
    ln0: Vec<f32>,
    /// `[32 out][128 in][3][3]`
    conv1: Vec<f32>,
    ln1: Vec<f32>,
    c0: usize,
    c1: usize,
    in_proj: Mat,
    layers: Vec<ALayer>,
    out_proj: Mat,
    out_bias: Vec<f32>,
    embed: Mat,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    /// a token sees keys at distances `0..left`
    left: usize,
    kernel: usize,
    softcap: f32,
    residual_weight: f32,
    q_scale: f32,
    k_scale: f32,
    /// the text width the soft tokens land in (512)
    pub out_dim: usize,
    pub audio_token: u32,
    pub boa_token: u32,
    pub eoa_token: u32,
    pool: Option<Arc<Pool>>,
    busy: Mutex<()>,
}

/// Does this pack carry the audio tower?
pub fn has_audio(model: &CmfModel) -> bool {
    model
        .tensor("audio_tower.subsample_conv_projection.layer0.conv.weight")
        .is_some()
        && model
            .tensor("embed_audio.embedding_projection.weight")
            .is_some()
}

impl AudioTower {
    /// Load the tower from an EmbeddingGemma 2 pack. A quantized pack's
    /// matrices are widened to f32 at load (~1.2 GB, the f32 GEMM is the
    /// fast path); `CMF_EGEMMA2_LOWMEM=1` keeps the quantized kernels.
    pub fn load(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        if !crate::egemma2::is_embedding_gemma2(model) {
            return Err("not an EmbeddingGemma 2 file".into());
        }
        if !has_audio(model) {
            return Err(
                "this EmbeddingGemma 2 file has no audio tower (packed with CMF_EGEMMA2_TEXT_ONLY=1?)"
                    .into(),
            );
        }
        let prov = model
            .header
            .provenance
            .clone()
            .unwrap_or(serde_json::Value::Null);
        let eg = &prov["embedding_gemma2"];
        let cfg = &eg["config"];
        let ac = &cfg["audio_config"];
        check_frontend(&eg["preprocessor"])?;
        let gu = |k: &str, d: u64| ac.get(k).and_then(|v| v.as_u64()).unwrap_or(d) as usize;
        let gf = |k: &str, d: f64| ac.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let hidden = gu("hidden_size", 1024);
        let heads = gu("num_attention_heads", 8);
        let n_layers = gu("num_hidden_layers", 12);
        let chunk = gu("attention_chunk_size", 12);
        let ctx_left = gu("attention_context_left", 13);
        let ctx_right = gu("attention_context_right", 0);
        if ctx_right != 0 {
            return Err(format!(
                "audio: attention_context_right {ctx_right} is not supported (0 in the release)"
            ));
        }
        let head_dim = hidden / heads;
        let left = ctx_left.saturating_sub(1).max(1);
        let dequant = !std::env::var("CMF_EGEMMA2_LOWMEM").is_ok_and(|v| v == "1");

        // Sinusoids of the relative distances: row r is distance r, laid
        // out [sin × hidden/2, cos × hidden/2] (`Gemma4AudioRelPositionalEncoding`,
        // which lists them from the far end; the shift maps distance d to
        // its row). f32 throughout, as the reference.
        let n_ts = hidden / 2;
        let incr = (10000f64).ln() / (n_ts.max(2) - 1) as f64;
        let inv: Vec<f32> = (0..n_ts)
            .map(|i| ((i as f32) * (-incr as f32)).exp())
            .collect();
        let n_dist = (chunk + ctx_left - 1 + ctx_right) / 2 + 1;
        if n_dist < left {
            return Err("audio: relative position table shorter than the attention span".into());
        }
        let mut pe = vec![0f32; n_dist * hidden];
        for d in 0..n_dist {
            for (i, &f) in inv.iter().enumerate() {
                let t = d as f32 * f;
                pe[d * hidden + i] = t.sin();
                pe[d * hidden + n_ts + i] = t.cos();
            }
        }

        let p0 = "audio_tower.subsample_conv_projection";
        let conv0 = vecf(model, &format!("{p0}.layer0.conv.weight"))?;
        let conv1 = vecf(model, &format!("{p0}.layer1.conv.weight"))?;
        let ln0 = vecf(model, &format!("{p0}.layer0.norm.weight"))?;
        let ln1 = vecf(model, &format!("{p0}.layer1.norm.weight"))?;
        let (c0, c1) = (ln0.len(), ln1.len());
        if conv0.len() != c0 * 9 || conv1.len() != c1 * c0 * 9 {
            return Err("audio: unexpected subsampling convolution shapes".into());
        }
        let in_proj = Mat::load(model, &format!("{p0}.input_proj_linear.weight"), dequant)?;
        if in_proj.rows() != hidden {
            return Err("audio: input projection width mismatch".into());
        }

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("audio_tower.layers.{i}");
            let ffn = |f: &str| -> Result<Ffn, String> {
                Ok(Ffn {
                    pre: vecf(model, &format!("{p}.{f}.pre_layer_norm.weight"))?,
                    l1: ClipLin::load(model, &format!("{p}.{f}.ffw_layer_1"), dequant)?,
                    l2: ClipLin::load(model, &format!("{p}.{f}.ffw_layer_2"), dequant)?,
                    post: vecf(model, &format!("{p}.{f}.post_layer_norm.weight"))?,
                })
            };
            let pds = vecf(model, &format!("{p}.self_attn.per_dim_scale"))?;
            if pds.len() != head_dim {
                return Err(format!(
                    "audio layer {i}: per_dim_scale has {} values",
                    pds.len()
                ));
            }
            // softplus as torch: x > 20 passes through
            let q_dim = pds
                .iter()
                .map(|&x| {
                    if x > 20.0 {
                        x
                    } else {
                        ((x as f64).exp().ln_1p()) as f32
                    }
                })
                .collect();
            let relk = Mat::load(
                model,
                &format!("{p}.self_attn.relative_k_proj.weight"),
                true,
            )?;
            let rel = relk.apply(&pe, n_dist, None);
            let lc_dw = vecf(model, &format!("{p}.lconv1d.depthwise_conv1d.weight"))?;
            layers.push(ALayer {
                ff1: ffn("feed_forward1")?,
                ff2: ffn("feed_forward2")?,
                norm_pre_attn: vecf(model, &format!("{p}.norm_pre_attn.weight"))?,
                norm_post_attn: vecf(model, &format!("{p}.norm_post_attn.weight"))?,
                norm_out: vecf(model, &format!("{p}.norm_out.weight"))?,
                q: ClipLin::load(model, &format!("{p}.self_attn.q_proj"), dequant)?,
                k: ClipLin::load(model, &format!("{p}.self_attn.k_proj"), dequant)?,
                v: ClipLin::load(model, &format!("{p}.self_attn.v_proj"), dequant)?,
                post: ClipLin::load(model, &format!("{p}.self_attn.post"), dequant)?,
                q_dim,
                rel,
                lc_pre: vecf(model, &format!("{p}.lconv1d.pre_layer_norm.weight"))?,
                lc_start: ClipLin::load(model, &format!("{p}.lconv1d.linear_start"), dequant)?,
                lc_dw,
                lc_norm: vecf(model, &format!("{p}.lconv1d.conv_norm.weight"))?,
                lc_end: ClipLin::load(model, &format!("{p}.lconv1d.linear_end"), dequant)?,
            });
        }
        let kernel = layers
            .first()
            .map(|l| l.lc_dw.len() / hidden)
            .unwrap_or(gu("conv_kernel_size", 5));
        let out_proj = Mat::load(model, "audio_tower.output_proj.weight", dequant)?;
        let out_bias = vecf(model, "audio_tower.output_proj.bias")?;
        let embed = Mat::load(model, "embed_audio.embedding_projection.weight", true)?;
        let tok = |k: &str, d: u32| cfg.get(k).and_then(|v| v.as_u64()).map_or(d, |v| v as u32);
        Ok(AudioTower {
            frontend: MelFrontend::new(),
            conv0,
            ln0,
            conv1,
            ln1,
            c0,
            c1,
            in_proj,
            layers,
            out_bias,
            out_dim: embed.rows(),
            out_proj,
            embed,
            hidden,
            heads,
            head_dim,
            left,
            kernel,
            softcap: gf("attention_logit_cap", 50.0) as f32,
            residual_weight: gf("residual_weight", 0.5) as f32,
            q_scale: ((head_dim as f64).powf(-0.5) / std::f64::consts::LN_2) as f32,
            k_scale: ((1.0 + std::f64::consts::E).ln() / std::f64::consts::LN_2) as f32,
            audio_token: tok("audio_token_id", AUDIO_TOKEN),
            boa_token: tok("boa_token_id", BOA_TOKEN),
            eoa_token: tok("eoa_token_index", tok("eoa_token_id", EOA_TOKEN)),
            pool,
            busy: Mutex::new(()),
        })
    }

    /// Log-mel features of a mono 16 kHz clip (`[frames, 128]`, frames).
    pub fn log_mel(&self, wave: &[f32]) -> (Vec<f32>, usize) {
        self.frontend.log_mel(wave, self.pool.as_deref())
    }

    /// Soft tokens of several clips (mono 16 kHz), `[num_tokens(len), 512]`
    /// each, in order. Clips are packed through the tower together.
    pub fn soft_tokens(&self, clips: &[&[f32]]) -> Vec<Vec<f32>> {
        let _g = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        let t0 = std::time::Instant::now();
        let mels: Vec<(Vec<f32>, usize)> = clips.iter().map(|c| self.log_mel(c)).collect();
        prof::add(&prof::MEL, t0);
        let mut out = Vec::with_capacity(clips.len());
        let mut start = 0usize;
        while start < mels.len() {
            let mut end = start;
            let mut toks = 0usize;
            while end < mels.len() {
                let t = mels[end].1.div_ceil(4);
                if end > start && toks + t > PACK_SOFT {
                    break;
                }
                toks += t;
                end += 1;
            }
            let part: Vec<(&[f32], usize)> =
                mels[start..end].iter().map(|(m, n)| (&m[..], *n)).collect();
            out.extend(self.encode_locked(&part, None));
            start = end;
        }
        out
    }

    /// The tower over log-mel features (`[frames, 128]` each): the soft
    /// tokens of each clip. `trace` collects, per call, the subsampling
    /// output, every layer's output and the tower output (all packed).
    pub fn encode_mels(
        &self,
        mels: &[(&[f32], usize)],
        trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<Vec<f32>> {
        let _g = self.busy.lock().unwrap_or_else(|e| e.into_inner());
        self.encode_locked(mels, trace)
    }

    fn encode_locked(
        &self,
        mels: &[(&[f32], usize)],
        mut trace: Option<&mut Vec<Vec<f32>>>,
    ) -> Vec<Vec<f32>> {
        let pool = self.pool.as_deref();
        let d = self.hidden;
        // ── subsampling, clip by clip, rows packed back to back
        let mut x = Vec::new();
        let mut lens = Vec::with_capacity(mels.len());
        for &(m, frames) in mels {
            let t0 = std::time::Instant::now();
            let (sub, t) = self.subsample(m, frames, pool);
            prof::add(&prof::SUB, t0);
            x.extend_from_slice(&sub);
            lens.push(t);
        }
        let n: usize = lens.iter().sum();
        let out_dim = self.out_dim;
        if n == 0 {
            return lens.iter().map(|_| Vec::new()).collect();
        }
        // position of every row inside its clip
        let mut pos = Vec::with_capacity(n);
        for &l in &lens {
            pos.extend(0..l);
        }
        let t0 = std::time::Instant::now();
        let mut h = self.in_proj.apply(&x, n, pool);
        prof::add(&prof::LIN, t0);
        drop(x);
        if let Some(t) = trace.as_deref_mut() {
            t.push(h.clone());
        }
        for l in &self.layers {
            self.ffn(&l.ff1, &mut h, n, pool);
            // ── attention
            let a = rms_rows(&h, Some(&l.norm_pre_attn), d, pool);
            let o = self.attention(l, &a, n, &pos, pool);
            add_normed(&mut h, &o, &l.norm_post_attn, 1.0, d, pool);
            // ── light conv
            let a = rms_rows(&h, Some(&l.lc_pre), d, pool);
            let mut g = l.lc_start.apply(&a, n, pool);
            let gl = glu(&mut g, n, d, pool);
            drop(g);
            let mut c = self.dwconv(&gl, &l.lc_dw, n, &pos, pool);
            drop(gl);
            norm_silu(&mut c, &l.lc_norm, d, pool);
            let e = l.lc_end.apply(&c, n, pool);
            par_chunks(&mut h, pool, &|off, c| {
                for (hv, ev) in c.iter_mut().zip(&e[off..]) {
                    *hv = ev + *hv;
                }
            });
            self.ffn(&l.ff2, &mut h, n, pool);
            h = rms_rows(&h, Some(&l.norm_out), d, pool);
            if let Some(t) = trace.as_deref_mut() {
                t.push(h.clone());
            }
        }
        let t0 = std::time::Instant::now();
        let mut o = self.out_proj.apply(&h, n, pool);
        prof::add(&prof::LIN, t0);
        let od = self.out_bias.len();
        let bias = &self.out_bias;
        par_chunks(&mut o, pool, &|off, c| {
            for (i, v) in c.iter_mut().enumerate() {
                *v += bias[(off + i) % od];
            }
        });
        if let Some(t) = trace.as_deref_mut() {
            t.push(o.clone());
        }
        let on = rms_rows(&o, None, od, pool);
        let t0 = std::time::Instant::now();
        let e = self.embed.apply(&on, n, pool);
        prof::add(&prof::LIN, t0);
        let mut res = Vec::with_capacity(lens.len());
        let mut off = 0usize;
        for &l in &lens {
            res.push(e[off * out_dim..(off + l) * out_dim].to_vec());
            off += l;
        }
        res
    }

    /// The two stride-2 convolutions (+ LayerNorm, ReLU) over one clip's
    /// mel rows: `[⌈frames/4⌉, 32·32]` in (frequency, channel) order, before
    /// the input projection.
    fn subsample(&self, mel: &[f32], frames: usize, pool: Option<&Pool>) -> (Vec<f32>, usize) {
        let (c0, c1) = (self.c0, self.c1);
        let f0 = N_MELS;
        let t1 = frames.div_ceil(2);
        let f1 = f0.div_ceil(2);
        // layer 0: 1 → c0 channels, out[t][f][c] (channels last)
        let mut y0 = vec![0f32; t1 * f1 * c0];
        let dst = Shared(y0.as_mut_ptr());
        rows(pool, t1, &|s, e| {
            let o = unsafe { dst.at(s * f1 * c0, (e - s) * f1 * c0) };
            let mut patch = [0f32; 9];
            for t in s..e {
                for f in 0..f1 {
                    for dt in 0..3 {
                        for df in 0..3 {
                            let (ti, fi) = ((2 * t + dt) as isize - 1, (2 * f + df) as isize - 1);
                            patch[dt * 3 + df] = if ti >= 0
                                && (ti as usize) < frames
                                && fi >= 0
                                && (fi as usize) < f0
                            {
                                mel[ti as usize * f0 + fi as usize]
                            } else {
                                0.0
                            };
                        }
                    }
                    let row = &mut o[((t - s) * f1 + f) * c0..((t - s) * f1 + f + 1) * c0];
                    for (c, r) in row.iter_mut().enumerate() {
                        let w = &self.conv0[c * 9..(c + 1) * 9];
                        let mut acc = 0f32;
                        for k in 0..9 {
                            acc += w[k] * patch[k];
                        }
                        *r = acc;
                    }
                    layer_norm_relu(row, &self.ln0);
                }
            }
        });
        // layer 1: c0 → c1 channels, im2col by time blocks + GEMM
        let t2 = t1.div_ceil(2);
        let f2 = f1.div_ceil(2);
        let kk = c0 * 9;
        let mut y1 = vec![0f32; t2 * f2 * c1];
        const TB: usize = 32;
        let mut col = vec![0f32; TB * f2 * kk];
        for tb in (0..t2).step_by(TB) {
            let te = (tb + TB).min(t2);
            let nr = (te - tb) * f2;
            let cp = Shared(col.as_mut_ptr());
            rows(pool, te - tb, &|s, e| {
                let cc = unsafe { cp.at(s * f2 * kk, (e - s) * f2 * kk) };
                for tt in s..e {
                    let t = tb + tt;
                    for f in 0..f2 {
                        let r = &mut cc[((tt - s) * f2 + f) * kk..((tt - s) * f2 + f + 1) * kk];
                        for dt in 0..3 {
                            for df in 0..3 {
                                let (ti, fi) =
                                    ((2 * t + dt) as isize - 1, (2 * f + df) as isize - 1);
                                let inside =
                                    ti >= 0 && (ti as usize) < t1 && fi >= 0 && (fi as usize) < f1;
                                for ci in 0..c0 {
                                    r[ci * 9 + dt * 3 + df] = if inside {
                                        y0[(ti as usize * f1 + fi as usize) * c0 + ci]
                                    } else {
                                        0.0
                                    };
                                }
                            }
                        }
                    }
                }
            });
            crate::fcd_ops::gemm_nt_host(
                &col[..nr * kk],
                &self.conv1,
                &mut y1[tb * f2 * c1..te * f2 * c1],
                nr,
                kk,
                c1,
                pool,
            );
        }
        for row in y1.chunks_exact_mut(c1) {
            layer_norm_relu(row, &self.ln1);
        }
        (y1, t2)
    }

    /// `h += ½·N_post(W₂ silu(W₁ N_pre(h)))`.
    fn ffn(&self, f: &Ffn, h: &mut [f32], n: usize, pool: Option<&Pool>) {
        let d = self.hidden;
        let a = rms_rows(h, Some(&f.pre), d, pool);
        let mut u = f.l1.apply(&a, n, pool);
        silu_inplace(&mut u, pool);
        let y = f.l2.apply(&u, n, pool);
        add_normed(h, &y, &f.post, self.residual_weight, d, pool);
    }

    /// Chunked local attention over packed rows (`pos` = index in the clip).
    fn attention(
        &self,
        l: &ALayer,
        a: &[f32],
        n: usize,
        pos: &[usize],
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let (nh, hd, d) = (self.heads, self.head_dim, self.hidden);
        let mut q = l.q.apply(a, n, pool);
        let mut k = l.k.apply(a, n, pool);
        let v = l.v.apply(a, n, pool);
        let (qs, ks) = (self.q_scale, self.k_scale);
        let qd = &l.q_dim;
        par_chunks(&mut q, pool, &|off, c| {
            for (i, x) in c.iter_mut().enumerate() {
                *x = (*x * qs) * qd[(off + i) % hd];
            }
        });
        par_chunks(&mut k, pool, &|_, c| {
            for x in c {
                *x *= ks;
            }
        });
        let t_core = std::time::Instant::now();
        let mut out = vec![0f32; n * d];
        let dst = Shared(out.as_mut_ptr());
        let (left, cap) = (self.left, self.softcap);
        rows(pool, n, &|s, e| {
            let o = unsafe { dst.at(s * d, (e - s) * d) };
            let mut w = vec![0f32; left];
            for t in s..e {
                let span = (pos[t] + 1).min(left);
                for hh in 0..nh {
                    let qv = &q[t * d + hh * hd..t * d + (hh + 1) * hd];
                    let mut mx = f32::NEG_INFINITY;
                    for (dist, wd) in w.iter_mut().enumerate().take(span) {
                        let kv = &k[(t - dist) * d + hh * hd..(t - dist) * d + (hh + 1) * hd];
                        let rv = &l.rel[dist * d + hh * hd..dist * d + (hh + 1) * hd];
                        let (mut ac, mut bd) = (0f32, 0f32);
                        for j in 0..hd {
                            ac += qv[j] * kv[j];
                            bd += qv[j] * rv[j];
                        }
                        let z = ((ac + bd) / cap).tanh() * cap;
                        *wd = z;
                        mx = mx.max(z);
                    }
                    let mut den = 0f32;
                    for wd in w.iter_mut().take(span) {
                        *wd = (*wd - mx).exp();
                        den += *wd;
                    }
                    let ov = &mut o[(t - s) * d + hh * hd..(t - s) * d + (hh + 1) * hd];
                    ov.fill(0.0);
                    for (dist, &wd) in w.iter().enumerate().take(span) {
                        let p = wd / den;
                        let vv = &v[(t - dist) * d + hh * hd..(t - dist) * d + (hh + 1) * hd];
                        for j in 0..hd {
                            ov[j] += p * vv[j];
                        }
                    }
                }
            }
        });
        prof::add(&prof::ATT, t_core);
        l.post.apply(&out, n, pool)
    }

    /// Causal depthwise convolution (`kernel` taps ending at the row).
    fn dwconv(
        &self,
        x: &[f32],
        w: &[f32],
        n: usize,
        pos: &[usize],
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let (d, kn) = (self.hidden, self.kernel);
        let mut out = vec![0f32; n * d];
        let dst = Shared(out.as_mut_ptr());
        rows(pool, n, &|s, e| {
            let o = unsafe { dst.at(s * d, (e - s) * d) };
            for t in s..e {
                let row = &mut o[(t - s) * d..(t - s + 1) * d];
                for (k, back) in (0..kn).zip((0..kn).rev()) {
                    if back > pos[t] {
                        continue;
                    }
                    let src = &x[(t - back) * d..(t - back + 1) * d];
                    for c in 0..d {
                        row[c] += w[c * kn + k] * src[c];
                    }
                }
            }
        });
        out
    }
}

/// Is the preprocessor the one this front end implements?
fn check_frontend(pp: &serde_json::Value) -> Result<(), String> {
    if pp.is_null() {
        return Ok(());
    }
    let f = |k: &str| pp.get(k).and_then(|v| v.as_f64());
    let want: [(&str, f64); 9] = [
        ("feature_size", N_MELS as f64),
        ("sampling_rate", SAMPLE_RATE as f64),
        ("frame_length", FRAME as f64),
        ("hop_length", HOP as f64),
        ("fft_length", N_FFT as f64),
        ("mel_floor", MEL_FLOOR),
        ("preemphasis", 0.0),
        ("dither", 0.0),
        ("input_scale_factor", 1.0),
    ];
    for (k, v) in want {
        if let Some(got) = f(k) {
            if (got - v).abs() > 1e-9 {
                return Err(format!(
                    "audio: preprocessor {k} = {got}, this build implements {v}"
                ));
            }
        }
    }
    for k in ["per_bin_mean", "per_bin_stddev"] {
        if pp.get(k).is_some_and(|v| !v.is_null()) {
            return Err(format!("audio: preprocessor {k} is set; not implemented"));
        }
    }
    Ok(())
}

/// `f(offset, slice)` over `x` in parallel slices.
fn par_chunks(x: &mut [f32], pool: Option<&Pool>, f: &(dyn Fn(usize, &mut [f32]) + Sync)) {
    let n = x.len();
    let grain = 16384usize;
    let dst = Shared(x.as_mut_ptr());
    rows(pool, n.div_ceil(grain), &|s, e| {
        let (lo, hi) = (s * grain, (e * grain).min(n));
        f(lo, unsafe { dst.at(lo, hi - lo) });
    });
}

/// LayerNorm over one channel row (no bias), then ReLU.
fn layer_norm_relu(x: &mut [f32], w: &[f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x
        .iter()
        .map(|&v| {
            let c = v as f64 - mean;
            c * c
        })
        .sum::<f64>()
        / n;
    let inv = 1.0 / (var + EPS).sqrt();
    for (v, &g) in x.iter_mut().zip(w) {
        *v = (((*v as f64 - mean) * inv) as f32 * g).max(0.0);
    }
}

/// `h += scale · rms_w(y)` row by row.
fn add_normed(h: &mut [f32], y: &[f32], w: &[f32], scale: f32, d: usize, pool: Option<&Pool>) {
    let n = h.len() / d;
    let dst = Shared(h.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let hh = unsafe { dst.at(s * d, (e - s) * d) };
        let mut tmp = vec![0f32; d];
        for (hrow, yrow) in hh.chunks_exact_mut(d).zip(y[s * d..e * d].chunks_exact(d)) {
            rms_into(yrow, Some(w), &mut tmp);
            for (a, &b) in hrow.iter_mut().zip(&tmp) {
                *a += b * scale;
            }
        }
    });
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `x = x·σ(x)` across the pool.
fn silu_inplace(x: &mut [f32], pool: Option<&Pool>) {
    let n = x.len();
    let grain = 16384usize;
    let dst = Shared(x.as_mut_ptr());
    rows(pool, n.div_ceil(grain), &|s, e| {
        let (lo, hi) = (s * grain, (e * grain).min(n));
        let r = unsafe { dst.at(lo, hi - lo) };
        silu_slice(r);
    });
}

#[inline]
fn silu_slice(x: &mut [f32]) {
    #[allow(unused_mut)]
    let mut i = 0usize;
    #[cfg(target_arch = "aarch64")]
    // SAFETY: every lane read/written is below `x.len()`.
    unsafe {
        use core::arch::aarch64::*;
        let one = vdupq_n_f32(1.0);
        while i + 4 <= x.len() {
            let v = vld1q_f32(x.as_ptr().add(i));
            let e = crate::attention::vexpq_f32(vnegq_f32(v));
            vst1q_f32(x.as_mut_ptr().add(i), vdivq_f32(v, vaddq_f32(one, e)));
            i += 4;
        }
    }
    for v in x[i..].iter_mut() {
        *v *= sigmoid(*v);
    }
}

/// `glu`: `[n, 2d] → [n, d]`, first half times σ(second half).
fn glu(x: &mut [f32], n: usize, d: usize, pool: Option<&Pool>) -> Vec<f32> {
    let mut out = vec![0f32; n * d];
    let dst = Shared(out.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let o = unsafe { dst.at(s * d, (e - s) * d) };
        for t in s..e {
            let (a, b) = x[t * 2 * d..(t + 1) * 2 * d].split_at(d);
            for ((r, &av), &bv) in o[(t - s) * d..(t - s + 1) * d].iter_mut().zip(a).zip(b) {
                *r = av * sigmoid(bv);
            }
        }
    });
    out
}

/// `x = silu(rms_w(x))` row by row.
fn norm_silu(x: &mut [f32], w: &[f32], d: usize, pool: Option<&Pool>) {
    let n = x.len() / d;
    let dst = Shared(x.as_mut_ptr());
    rows(pool, n, &|s, e| {
        let xx = unsafe { dst.at(s * d, (e - s) * d) };
        let mut tmp = vec![0f32; d];
        for row in xx.chunks_exact_mut(d) {
            rms_into(row, Some(w), &mut tmp);
            row.copy_from_slice(&tmp);
            silu_slice(row);
        }
    });
}

// ───────────────────────────── inputs ─────────────────────────────

/// One input that holds audio: clips (mono 16 kHz) and, optionally, text
/// with one [`PLACEHOLDER`] per clip (prompt options as for text). Without
/// text the clips follow each other between BOS and EOS, space separated
/// (the processor's `"<|audio|> <|audio|>"`), unprompted.
///
/// [`crate::egemma2_mm::MediaEncoder`] is the general path (audio with
/// images and video in one input); this one is the audio-only shortcut the
/// parity test checks it against.
#[derive(Clone, Debug, Default)]
pub struct AudioInput {
    pub text: Option<TextInput>,
    pub clips: Vec<Vec<f32>>,
}

/// Token ids of an input whose clips give `n_soft[i]` soft tokens: every
/// placeholder becomes `<|audio> <|audio|>×n <audio|>`, wrapped in BOS/EOS.
pub fn input_ids(
    enc: &EmbeddingGemma2,
    tower_tokens: (u32, u32, u32),
    input: &AudioInput,
    n_soft: &[usize],
) -> Result<Vec<u32>, String> {
    let (audio, boa, eoa) = tower_tokens;
    let expand = |out: &mut Vec<u32>, n: usize| {
        out.push(boa);
        out.extend(std::iter::repeat_n(audio, n));
        out.push(eoa);
    };
    let mut ids = vec![enc.bos];
    // without text the processor lays the clips out as "<|audio|> <|audio|>"
    let bare = TextInput::plain(vec![PLACEHOLDER; n_soft.len()].join(" "));
    let t = input.text.as_ref().unwrap_or(&bare);
    let body = enc.tokenizer().encode(&enc.format(t)?);
    let found = body.iter().filter(|&&x| x == audio).count();
    if found != n_soft.len() {
        return Err(format!(
            "the text holds {found} {PLACEHOLDER} placeholder(s) for {} audio clip(s)",
            n_soft.len()
        ));
    }
    let mut k = 0usize;
    for &x in &body {
        if x == audio {
            expand(&mut ids, n_soft[k]);
            k += 1;
        } else {
            ids.push(x);
        }
    }
    ids.push(enc.eos);
    if ids.len() > enc.max_tokens {
        return Err(format!(
            "{} tokens > the model's {} token context (audio is 25 tokens a second)",
            ids.len(),
            enc.max_tokens
        ));
    }
    Ok(ids)
}

/// Embed inputs with audio: unit-length vectors of `enc.dim`, in order.
pub fn embed_audio_inputs(
    enc: &EmbeddingGemma2,
    tower: &AudioTower,
    inputs: &[AudioInput],
) -> Result<Vec<Vec<f32>>, String> {
    let toks = (tower.audio_token, tower.boa_token, tower.eoa_token);
    // ids first: a bad input fails before any compute
    let mut seqs = Vec::with_capacity(inputs.len());
    for (i, inp) in inputs.iter().enumerate() {
        if inp.clips.is_empty() && inp.text.is_none() {
            return Err(format!("input {i}: no audio and no text"));
        }
        let n: Vec<usize> = inp.clips.iter().map(|c| num_tokens(c.len())).collect();
        seqs.push(input_ids(enc, toks, inp, &n).map_err(|e| format!("input {i}: {e}"))?);
    }
    let clips: Vec<&[f32]> = inputs
        .iter()
        .flat_map(|i| i.clips.iter().map(|c| &c[..]))
        .collect();
    let t_tower = std::time::Instant::now();
    let soft = tower.soft_tokens(&clips);
    let tower_ms = t_tower.elapsed().as_secs_f64() * 1e3;
    // the tower's projections go through the text encoder's `Mat`, whose
    // profile counter the text forward reports: leave it the text's own
    crate::egemma2::prof::take(&crate::egemma2::prof::LIN);
    let t_text = std::time::Instant::now();
    let od = tower.out_dim;
    let d = enc.hidden();
    if od != d {
        return Err(format!(
            "audio soft tokens are {od}-d, the text encoder is {d}-d"
        ));
    }
    // merged rows per input
    let mut merged: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    let mut next = 0usize;
    for (inp, ids) in inputs.iter().zip(&seqs) {
        let mut x = enc.embed_rows(std::slice::from_ref(ids));
        let mut rows_it = ids
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == tower.audio_token)
            .map(|(r, _)| r);
        for _ in 0..inp.clips.len() {
            let s = &soft[next];
            next += 1;
            for tok in s.chunks_exact(od) {
                let r = rows_it.next().ok_or("audio: placeholder count mismatch")?;
                x[r * d..(r + 1) * d].copy_from_slice(tok);
            }
        }
        if rows_it.next().is_some() {
            return Err("audio: placeholder count mismatch".into());
        }
        merged.push(x);
    }
    // pack up to the context per text forward
    let mut out = Vec::with_capacity(inputs.len());
    let mut start = 0usize;
    while start < seqs.len() {
        let mut end = start;
        let mut tot = 0usize;
        while end < seqs.len()
            && (end == start || tot + seqs[end].len() <= crate::egemma2::MAX_TOKENS)
        {
            tot += seqs[end].len();
            end += 1;
        }
        let mut x = Vec::with_capacity(tot * d);
        for m in &merged[start..end] {
            x.extend_from_slice(m);
        }
        let lens: Vec<usize> = seqs[start..end].iter().map(|s| s.len()).collect();
        out.extend(enc.embed_merged(x, &lens));
        start = end;
    }
    if prof::on() {
        let (mel, sub, lin, att) = (
            prof::take(&prof::MEL),
            prof::take(&prof::SUB),
            prof::take(&prof::LIN),
            prof::take(&prof::ATT),
        );
        let secs: f64 = clips.iter().map(|c| c.len() as f64).sum::<f64>() / SAMPLE_RATE as f64;
        let toks: usize = soft.iter().map(|s| s.len() / od).sum();
        eprintln!(
            "egemma2 audio: {} clip(s), {secs:.2} s, {toks} soft tokens — tower {tower_ms:.1} ms \
             (log-mel {mel:.1}, subsampling {sub:.1}, projections {lin:.1}, attention core {att:.1}, \
             rest {:.1}); text {:.1} ms",
            clips.len(),
            tower_ms - mel - sub - lin - att,
            t_text.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_follow_the_processor() {
        // (samples, mel frames, soft tokens) from the transformers processor
        for (n, f, t) in [
            (100, 0, 0),
            (161, 1, 1),
            (321, 2, 1),
            (480, 2, 1),
            (1000, 6, 2),
            (16000, 99, 25),
            (72825, 455, 114),
            (480000, 2999, 750),
            (480001, 2999, 750),
            (560000, 2999, 750),
        ] {
            assert_eq!(num_frames(n), f, "frames of {n}");
            assert_eq!(num_tokens(n), t, "tokens of {n}");
        }
    }

    #[test]
    fn mel_bank_is_htk_triangles() {
        let m = mel_filter_bank();
        let n_freq = N_FFT / 2 + 1;
        assert_eq!(m.len(), n_freq * N_MELS);
        // DC and Nyquist fall outside every triangle
        assert!(m[..N_MELS].iter().all(|&v| v == 0.0));
        // every filter peaks at most at 1, all are non-negative
        for b in 0..N_MELS {
            let col: Vec<f64> = (0..n_freq).map(|k| m[k * N_MELS + b]).collect();
            assert!(col.iter().all(|&v| (0.0..=1.0).contains(&v)));
        }
        // the lowest band falls between bins 0 and 1 and stays empty, as in
        // the reference bank; bin 1 (31.25 Hz) splits over bands 1 and 2
        // with transformers' own weights (float32 dump of the release)
        assert!((0..n_freq).all(|k| m[k * N_MELS] == 0.0));
        assert!((m[N_MELS + 1] - 0.766_007_66).abs() < 1e-6);
        assert!((m[N_MELS + 2] - 0.233_992_32).abs() < 1e-6);
    }

    #[test]
    fn silence_is_the_mel_floor() {
        let fe = MelFrontend::new();
        let (mel, n) = fe.log_mel(&vec![0f32; 16000], None);
        assert_eq!(n, 99);
        let floor = (MEL_FLOOR).ln() as f32;
        assert!(mel.iter().all(|&v| (v - floor).abs() < 1e-6));
    }

    #[test]
    fn a_tone_lands_in_its_band() {
        let fe = MelFrontend::new();
        let f = 1000.0f64;
        let wave: Vec<f32> = (0..16000)
            .map(|i| (0.5 * (2.0 * std::f64::consts::PI * f * i as f64 / 16000.0).sin()) as f32)
            .collect();
        let (mel, n) = fe.log_mel(&wave, None);
        let row = &mel[(n / 2) * N_MELS..(n / 2 + 1) * N_MELS];
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        // band centres: 700(10^(m/2595)-1) on 130 points from 0 to mel(8000)
        let hz2mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let centre = |b: usize| {
            let m = hz2mel(8000.0) * (b + 1) as f64 / 129.0;
            700.0 * (10f64.powf(m / 2595.0) - 1.0)
        };
        assert!(
            (centre(best) - f).abs() < 60.0,
            "band {best} at {}",
            centre(best)
        );
    }

    #[test]
    fn stereo_mixes_down_and_rates_resample() {
        let wav = crate::mimo_audio::Wav {
            sample_rate: 16000,
            channels: vec![vec![1.0, 0.0, 0.5], vec![0.0, 1.0, 0.5]],
        };
        assert_eq!(to_mono_16k(&wav), vec![0.5, 0.5, 0.5]);
        let wav = crate::mimo_audio::Wav {
            sample_rate: 48000,
            channels: vec![vec![0.25; 4800]],
        };
        let m = to_mono_16k(&wav);
        assert_eq!(m.len(), 1600);
        assert!((m[800] - 0.25).abs() < 1e-4);
    }

    #[test]
    fn resampling_keeps_a_tone_and_its_length() {
        for (from, f0) in [
            (8000u32, 1000.0f64),
            (44100, 3000.0),
            (48000, 7000.0),
            (22050, 440.0),
        ] {
            let n = from as usize; // one second
            let x: Vec<f32> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * f0 * i as f64 / from as f64).sin() as f32)
                .collect();
            let y = resample_hq(&x, from, SAMPLE_RATE);
            assert_eq!(y.len(), 16000, "{from}");
            // away from the edges the tone comes out as the ideal one
            let err = (2000..14000)
                .map(|i| {
                    let want = (2.0 * std::f64::consts::PI * f0 * i as f64 / 16000.0).sin();
                    (y[i] as f64 - want).abs()
                })
                .fold(0f64, f64::max);
            assert!(err < 1e-4, "{from} Hz: max err {err}");
        }
        // a tone above the new Nyquist is removed
        let x: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f64::consts::PI * 9000.0 * i as f64 / 48000.0).sin() as f32)
            .collect();
        let y = resample_hq(&x, 48000, SAMPLE_RATE);
        let rms = (y[2000..14000]
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            / 12000.0)
            .sqrt();
        assert!(rms < 1e-4, "alias rms {rms}");
        // co-prime rates take the on-the-fly path and agree with the table path
        let x: Vec<f32> = (0..44099).map(|i| ((i % 97) as f32 / 97.0) - 0.5).collect();
        let y = resample_hq(&x, 44099, SAMPLE_RATE);
        assert_eq!(y.len(), (16000u64 * 44099).div_ceil(44099) as usize);
    }
}
