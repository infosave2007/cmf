//! MiMo-V2.6 audio input (spec milestones M8, M9 and the tower half of M10).
//!
//! The chain, in order:
//!
//! 1. [`decode_wav`]: RIFF/WAVE with PCM 8/16/24/32, IEEE float 32/64 and
//!    `WAVE_FORMAT_EXTENSIBLE`, any channel count. Samples become f32 in
//!    [−1, 1] the way ffmpeg (torchcodec, the serving decoder) scales them.
//! 2. [`resample_sinc`]: a port of torchaudio's `Resample` defaults
//!    (`sinc_interp_hann`, width 6, rolloff 0.99, gcd-reduced), including the
//!    float32 rounding of the target length. Every channel is resampled,
//!    then the channels are averaged ([`wav_to_mono_24k`]).
//! 3. [`log_mel`]: `MelSpectrogram(24000, n_fft 960, hop 240, 128 HTK mels,
//!    norm None, power 1, center reflect)` followed by `ln(max(x, 1e-7))`,
//!    as rows `[M, 128]` with `M = 1 + N/240`.
//! 4. [`AudioTokenizer`]: the MiMo audio tokenizer encoder — conv stem, 24
//!    pre-LN layers (causal, even layers windowed to 128 back), the skip
//!    from layer index 2, pooler, then a 20-level residual VQ in f32. Every
//!    6000-frame mel segment is encoded on its own (positions restart at 0).
//! 5. [`AudioEncoder`]: the LLM side — the 20 speech embedding tables summed,
//!    groups of 4 frames through a 6-layer bidirectional Qwen2, then the
//!    two-layer GELU projection to the LLM width. One output row per
//!    `<|audio_pad|>` placeholder.
//!
//! Weights are read by their source names: `audio_tokenizer.encoder.*`,
//! `audio_encoder.*` and `speech_embeddings.*`, from a companion (or a
//! single-file multimodal) CMF via [`MimoAudio::from_model`], or straight
//! from the HF checkpoint directory via [`MimoAudio::from_hf_dir`] (the
//! development loader; it reads only the tensors it needs).

use crate::dit::Proj;
use crate::pool::{Pool, SendMut};
use cortiq_core::CmfModel;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// `<|audio_pad|>`: one per audio embedding row.
pub const AUDIO_PAD_ID: u32 = 151669;
/// `<|mimo_audio_start|>`.
pub const AUDIO_START_ID: u32 = 151673;
/// `<|mimo_audio_end|>`.
pub const AUDIO_END_ID: u32 = 151674;

/// The tokenizer's input rate.
pub const SAMPLE_RATE: u32 = 24_000;
/// STFT size and window length.
pub const N_FFT: usize = 960;
/// STFT hop.
pub const HOP: usize = 240;
/// Mel bands.
pub const N_MELS: usize = 128;
/// Mel frames per independently encoded tokenizer segment.
pub const SEGMENT_FRAMES: usize = 6000;
/// The mel floor before the natural log.
pub const MEL_FLOOR: f64 = 1e-7;

// ───────────────────────────── WAV decode ─────────────────────────────

/// A decoded WAV file: one f32 vector per channel, all the same length.
#[derive(Clone, Debug)]
pub struct Wav {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

impl Wav {
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleKind {
    /// Integer PCM; 8-bit is unsigned, wider is signed two's complement.
    Pcm,
    /// IEEE float.
    Float,
}

/// Decode a RIFF/WAVE byte buffer.
///
/// Scaling follows ffmpeg's sample-format conversion (what torchcodec feeds
/// the reference processor): unsigned 8-bit is `(x − 128) / 128`, signed
/// n-bit is `x / 2^(n−1)`; float samples pass through unchanged. For
/// `WAVE_FORMAT_EXTENSIBLE` the container width (`block_align / channels`)
/// sets the scale, which equals the valid-bit scale for left-justified
/// samples. A `data` chunk whose declared size runs past the end of the
/// buffer (streamed writers put 0xFFFFFFFF there) is read to the end.
pub fn decode_wav(bytes: &[u8]) -> Result<Wav, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("audio: not a RIFF/WAVE file (only WAV input is supported)".into());
    }
    let le16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let le32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let mut fmt: Option<(SampleKind, usize, u32, usize)> = None; // kind, channels, rate, container bytes
    let mut data: Option<&[u8]> = None;
    let mut pos = 12usize;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = le32(bytes, pos + 4) as usize;
        let start = pos + 8;
        let end = start.saturating_add(size).min(bytes.len());
        let body = &bytes[start..end];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err("audio: WAV fmt chunk is shorter than 16 bytes".into());
                }
                let mut tag = le16(body, 0);
                let channels = le16(body, 2) as usize;
                let rate = le32(body, 4);
                let block_align = le16(body, 12) as usize;
                let bits = le16(body, 14) as usize;
                if tag == 0xFFFE {
                    if body.len() < 40 {
                        return Err("audio: WAVE_FORMAT_EXTENSIBLE fmt chunk is truncated".into());
                    }
                    // The sub-format GUID starts with the plain format tag.
                    tag = le16(body, 24);
                }
                let kind = match tag {
                    1 => SampleKind::Pcm,
                    3 => SampleKind::Float,
                    other => {
                        return Err(format!(
                            "audio: WAV format tag {other:#06x} is not supported (PCM and IEEE float only)"
                        ));
                    }
                };
                if channels == 0 {
                    return Err("audio: WAV declares zero channels".into());
                }
                if rate == 0 {
                    return Err("audio: WAV declares a zero sample rate".into());
                }
                let container = if block_align > 0 && block_align % channels == 0 {
                    block_align / channels
                } else {
                    bits.div_ceil(8)
                };
                let ok = match kind {
                    SampleKind::Pcm => matches!(container, 1..=4),
                    SampleKind::Float => matches!(container, 4 | 8),
                };
                if !ok {
                    return Err(format!(
                        "audio: WAV {kind:?} with {container}-byte samples ({bits} bits) is not supported"
                    ));
                }
                fmt = Some((kind, channels, rate, container));
            }
            b"data" => {
                data = Some(body);
                if fmt.is_some() {
                    break;
                }
            }
            _ => {}
        }
        // Chunks are word aligned: an odd size carries one pad byte.
        pos = start.saturating_add(size).saturating_add(size & 1);
    }
    let (kind, nch, rate, cb) = fmt.ok_or("audio: WAV has no fmt chunk")?;
    let data = data.ok_or("audio: WAV has no data chunk")?;
    let frame = nch * cb;
    let frames = data.len() / frame;
    let mut channels = vec![Vec::with_capacity(frames); nch];
    for f in 0..frames {
        for (c, ch) in channels.iter_mut().enumerate() {
            let o = f * frame + c * cb;
            let s = &data[o..o + cb];
            let v = match (kind, cb) {
                (SampleKind::Pcm, 1) => (s[0] as f32 - 128.0) / 128.0,
                (SampleKind::Pcm, 2) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0,
                (SampleKind::Pcm, 3) => {
                    let v = i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8;
                    v as f32 / 8_388_608.0
                }
                (SampleKind::Pcm, 4) => {
                    (i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0) as f32
                }
                (SampleKind::Float, 4) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                (SampleKind::Float, 8) => {
                    f64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]) as f32
                }
                _ => unreachable!("container width validated above"),
            };
            ch.push(v);
        }
    }
    Ok(Wav {
        sample_rate: rate,
        channels,
    })
}

// ───────────────────────────── resampling ─────────────────────────────

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// torchaudio `_get_sinc_resample_kernel` with its defaults
/// (`lowpass_filter_width = 6`, `rolloff = 0.99`, `sinc_interp_hann`,
/// `dtype = None`): the kernel is built in f64 and stored as f32. `orig` and
/// `new` are already gcd-reduced. Returns `(kernel [new][2·width + orig],
/// width)`.
///
/// One detail is kept on purpose: the output phase `-j / new` is a float32
/// division in torch (`torch.arange(0, -new, -1)` is int64 and true
/// division promotes it to the default dtype) before it meets the f64 tap
/// grid.
pub fn sinc_resample_kernel(orig: usize, new: usize) -> (Vec<f32>, usize) {
    const WIDTH: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    let base_freq = (orig.min(new) as f64) * ROLLOFF;
    let width = (WIDTH * orig as f64 / base_freq).ceil() as usize;
    let taps = 2 * width + orig;
    let scale = base_freq / orig as f64;
    let mut kernel = vec![0f32; new * taps];
    for j in 0..new {
        let phase = (-(j as f32) / new as f32) as f64;
        for i in 0..taps {
            let idx = (i as f64 - width as f64) / orig as f64;
            let mut t = (phase + idx) * base_freq;
            t = t.clamp(-WIDTH, WIDTH);
            let window = {
                let c = (t * std::f64::consts::PI / WIDTH / 2.0).cos();
                c * c
            };
            let t = t * std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            kernel[j * taps + i] = (sinc * (window * scale)) as f32;
        }
    }
    (kernel, width)
}

/// torchaudio's `Resample(orig, new)` output length: the f64 quotient is
/// rounded to float32 (`torch.as_tensor` of a Python float) before the
/// ceiling, so a long input can come out one sample shorter than the exact
/// `⌈new·N/orig⌉`. This returns what torchaudio returns.
pub fn resampled_len(n: usize, orig: u32, new: u32) -> usize {
    if orig == new {
        return n;
    }
    let g = gcd(orig as u64, new as u64);
    let (o, nw) = ((orig as u64 / g) as f64, (new as u64 / g) as f64);
    ((nw * n as f64 / o) as f32).ceil() as usize
}

/// Resample one channel from `orig` Hz to `new` Hz exactly as torchaudio's
/// `Resample` does: zero padding of `width` on the left and `width + orig`
/// on the right, a strided polyphase convolution, truncated to
/// [`resampled_len`]. Accumulation is f64.
pub fn resample_sinc(x: &[f32], orig: u32, new: u32) -> Vec<f32> {
    if orig == new {
        return x.to_vec();
    }
    let g = gcd(orig as u64, new as u64);
    let o = (orig as u64 / g) as usize;
    let nw = (new as u64 / g) as usize;
    let (kernel, width) = sinc_resample_kernel(o, nw);
    let taps = 2 * width + o;
    let len = x.len();
    let frames = len / o + 1;
    let target = resampled_len(len, orig, new).min(frames * nw);
    let mut out = vec![0f32; target];
    // x_pad[p] = x[p − width] inside the signal, 0 outside.
    for (n_out, dst) in out.iter_mut().enumerate() {
        let (f, j) = (n_out / nw, n_out % nw);
        let krow = &kernel[j * taps..(j + 1) * taps];
        let base = (f * o) as isize - width as isize;
        let mut acc = 0f64;
        for (m, &kv) in krow.iter().enumerate() {
            let p = base + m as isize;
            if p >= 0 && (p as usize) < len {
                acc += kv as f64 * x[p as usize] as f64;
            }
        }
        *dst = acc as f32;
    }
    out
}

/// Resample every channel to 24 kHz, then average the channels (the
/// reference resamples `[C, N]` first and takes `mean(dim=0)` after).
pub fn wav_to_mono_24k(wav: &Wav) -> Result<Vec<f32>, String> {
    if wav.channels.is_empty() {
        return Err("audio: WAV has no channels".into());
    }
    let chans: Vec<Vec<f32>> = wav
        .channels
        .iter()
        .map(|c| resample_sinc(c, wav.sample_rate, SAMPLE_RATE))
        .collect();
    if chans.len() == 1 {
        return Ok(chans.into_iter().next().unwrap());
    }
    let n = chans[0].len();
    let inv = chans.len() as f32;
    Ok((0..n)
        .map(|i| chans.iter().map(|c| c[i]).sum::<f32>() / inv)
        .collect())
}

// ───────────────────────────── log-mel ─────────────────────────────

/// A mixed-radix complex FFT in f64 for a fixed size (factors 2, 3, 5, …).
struct Fft {
    n: usize,
    factors: Vec<usize>,
    /// `e^{−2πi·k/n}` for k in 0..n.
    tw: Vec<(f64, f64)>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let mut factors = Vec::new();
        let mut m = n;
        for p in [4usize, 2, 3, 5] {
            while m % p == 0 {
                factors.push(p);
                m /= p;
            }
        }
        let mut p = 7;
        while m > 1 {
            while m % p == 0 {
                factors.push(p);
                m /= p;
            }
            p += 2;
        }
        let tw = (0..n)
            .map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (a.cos(), a.sin())
            })
            .collect();
        Self { n, factors, tw }
    }

    /// `out[k] = Σ_t x[t]·e^{−2πi·kt/n}` for a real input of length n.
    fn forward_real(&self, x: &[f64], out: &mut [(f64, f64)]) {
        debug_assert_eq!(x.len(), self.n);
        debug_assert_eq!(out.len(), self.n);
        let cx: Vec<(f64, f64)> = x.iter().map(|&v| (v, 0.0)).collect();
        self.rec(&cx, 0, 1, out, self.n, 0);
    }

    /// Decimation in time: `out` (length `n`) receives the DFT of
    /// `input[off + k·stride]`, k in 0..n.
    fn rec(
        &self,
        input: &[(f64, f64)],
        off: usize,
        stride: usize,
        out: &mut [(f64, f64)],
        n: usize,
        level: usize,
    ) {
        if n == 1 {
            out[0] = input[off];
            return;
        }
        let p = self.factors[level];
        let m = n / p;
        for r in 0..p {
            self.rec(
                input,
                off + r * stride,
                stride * p,
                &mut out[r * m..(r + 1) * m],
                m,
                level + 1,
            );
        }
        // X[k + m·q] = Σ_r W_n^{r(k+mq)} S_r[k]; the p inputs S_r[k] sit at
        // out[r·m + k] and the p outputs at out[k + m·q] — the same slots,
        // so each k is combined in place through a small buffer.
        let big = self.n;
        let step_n = big / n;
        let step_p = big / p;
        let mut t = [(0f64, 0f64); 16];
        let mut tmp = vec![(0f64, 0f64); if p > 16 { p } else { 0 }];
        let buf: &mut [(f64, f64)] = if p > 16 { &mut tmp } else { &mut t[..p] };
        for k in 0..m {
            for r in 0..p {
                let s = out[r * m + k];
                let w = self.tw[(r * k * step_n) % big];
                buf[r] = (s.0 * w.0 - s.1 * w.1, s.0 * w.1 + s.1 * w.0);
            }
            for q in 0..p {
                let (mut re, mut im) = (0f64, 0f64);
                for (r, b) in buf.iter().enumerate() {
                    let w = self.tw[(r * q * step_p) % big];
                    re += b.0 * w.0 - b.1 * w.1;
                    im += b.0 * w.1 + b.1 * w.0;
                }
                out[k + m * q] = (re, im);
            }
        }
    }
}

/// torchaudio `melscale_fbanks(n_freqs, 0, sr/2, n_mels, sr, norm=None,
/// mel_scale="htk")`, evaluated in f64. Row-major `[n_freqs][n_mels]`.
pub fn mel_filterbank(n_freqs: usize, n_mels: usize, sample_rate: u32) -> Vec<f64> {
    let f_max = (sample_rate / 2) as f64;
    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
    let (m_min, m_max) = (hz_to_mel(0.0), hz_to_mel(f_max));
    let f_pts: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(m_min + (m_max - m_min) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let f_diff: Vec<f64> = f_pts.windows(2).map(|w| w[1] - w[0]).collect();
    let mut fb = vec![0f64; n_freqs * n_mels];
    for k in 0..n_freqs {
        let freq = f_max * k as f64 / (n_freqs - 1) as f64;
        for m in 0..n_mels {
            let down = -(f_pts[m] - freq) / f_diff[m];
            let up = (f_pts[m + 2] - freq) / f_diff[m + 1];
            fb[k * n_mels + m] = down.min(up).max(0.0);
        }
    }
    fb
}

/// Entries of `torch.hann_window(960)` (torch 2.14, x86 AVX-512 build) whose
/// float32 bits differ from the correctly rounded evaluation of ATen's
/// formula — its vectorized `cos` is 1 ulp off there. Quiet mel bands are
/// ill conditioned: these 27 one-ulp differences alone move a near-empty
/// top band of a resampled clip by 0.12 in log, so the stored window is
/// reproduced bit for bit.
const TORCH_HANN_960_FIXUPS: [(usize, u32); 27] = [
    (30, 0x3C1D6820),
    (42, 0x3C99C880),
    (70, 0x3D533468),
    (87, 0x3DA191D0),
    (99, 0x3DCF8B30),
    (109, 0x3DF9B6B0),
    (125, 0x3E220032),
    (166, 0x3E88CD7D),
    (172, 0x3E91CA07),
    (315, 0x3F3C56BC),
    (331, 0x3F47CEDC),
    (338, 0x3F4C95E8),
    (384, 0x3F678DDE),
    (466, 0x3F7F7689),
    (468, 0x3F7F9AFC),
    (522, 0x3F7B31BC),
    (558, 0x3F6FADF2),
    (582, 0x3F648543),
    (639, 0x3F40B962),
    (661, 0x3F30355E),
    (695, 0x3F14D9C0),
    (857, 0x3DE00070),
    (859, 0x3DD7B41C),
    (866, 0x3DBBC250),
    (871, 0x3DA8DEB4),
    (924, 0x3C625860),
    (940, 0x3B8C2B00),
];

/// `torch.hann_window(960)` (periodic, float32) as ATen builds it —
/// `arange · f32(2π/N)`, cos, `· −0.5 + 0.5`, all in float32 — plus the
/// measured 1-ulp fixups of torch's vectorized cos.
fn torch_hann_960() -> Vec<f64> {
    let step = (2.0 * std::f64::consts::PI / N_FFT as f64) as f32;
    let mut w: Vec<f32> = (0..N_FFT)
        .map(|i| {
            let c = ((i as f32 * step) as f64).cos() as f32;
            c * -0.5f32 + 0.5f32
        })
        .collect();
    for (i, bits) in TORCH_HANN_960_FIXUPS {
        w[i] = f32::from_bits(bits);
    }
    w.into_iter().map(f64::from).collect()
}

/// Mel frames for a 24 kHz waveform of `n` samples (`center=True`).
pub fn mel_frames(n: usize) -> usize {
    1 + n / HOP
}

/// Log-mel features of a 24 kHz mono waveform, row-major `[M, 128]`.
///
/// The reflect pad of 480 needs more than 480 samples, as in torch.
pub fn log_mel(wave: &[f32], pool: Option<&Pool>) -> Result<(Vec<f32>, usize), String> {
    let n = wave.len();
    let pad = N_FFT / 2;
    if n <= pad {
        return Err(format!(
            "audio: {n} samples at 24 kHz is too short (the STFT reflect pad needs more than {pad})"
        ));
    }
    let frames = mel_frames(n);
    let n_freqs = N_FFT / 2 + 1;
    let fb = mel_filterbank(n_freqs, N_MELS, SAMPLE_RATE);
    // Each mel band touches a short run of bins; keep only those.
    let bands: Vec<(usize, Vec<f64>)> = (0..N_MELS)
        .map(|m| {
            let nz: Vec<usize> = (0..n_freqs)
                .filter(|&k| fb[k * N_MELS + m] != 0.0)
                .collect();
            match (nz.first(), nz.last()) {
                (Some(&a), Some(&b)) => (a, (a..=b).map(|k| fb[k * N_MELS + m]).collect()),
                _ => (0, Vec::new()),
            }
        })
        .collect();
    let window = torch_hann_960();
    let reflect = |p: isize| -> f32 {
        let last = n as isize - 1;
        let mut i = p - pad as isize;
        if i < 0 {
            i = -i;
        }
        if i > last {
            i = 2 * last - i;
        }
        wave[i as usize]
    };
    let fft = Fft::new(N_FFT);
    let mut out = vec![0f32; frames * N_MELS];
    let dst = SendMut::new(out.as_mut_ptr());
    let run = |start: usize, end: usize| {
        let mut buf = vec![0f64; N_FFT];
        let mut spec = vec![(0f64, 0f64); N_FFT];
        let mut mag = vec![0f64; n_freqs];
        for t in start..end {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = reflect((t * HOP + i) as isize) as f64 * window[i];
            }
            fft.forward_real(&buf, &mut spec);
            for k in 0..n_freqs {
                mag[k] = (spec[k].0 * spec[k].0 + spec[k].1 * spec[k].1).sqrt();
            }
            for (m, (k0, w)) in bands.iter().enumerate() {
                let mut acc = 0f64;
                for (i, wv) in w.iter().enumerate() {
                    acc += wv * mag[k0 + i];
                }
                unsafe { *dst.at(t * N_MELS + m) = acc.max(MEL_FLOOR).ln() as f32 };
            }
        }
    };
    match pool {
        Some(p) if frames >= 64 => p.run_rows(frames, &run),
        _ => run(0, frames),
    }
    Ok((out, frames))
}

// ───────────────────────────── token counts ─────────────────────────────

/// Mel-frame lengths of the independently encoded segments.
pub fn segment_lengths(mel_frames: usize) -> Vec<usize> {
    let mut v = vec![SEGMENT_FRAMES; mel_frames / SEGMENT_FRAMES];
    if mel_frames % SEGMENT_FRAMES > 0 {
        v.push(mel_frames % SEGMENT_FRAMES);
    }
    v
}

/// Tokenizer codes (25 Hz frames) for one segment of `m` mel frames:
/// `⌈⌈m/2⌉/2⌉` (conv2 stride 2, then the pooler of 2).
pub fn segment_codes(m: usize) -> usize {
    m.div_ceil(2).div_ceil(2)
}

/// Codes for a whole clip: the segments' codes concatenated.
pub fn codes_for_mel(mel_frames: usize) -> usize {
    segment_lengths(mel_frames)
        .into_iter()
        .map(segment_codes)
        .sum()
}

/// `<|audio_pad|>` count for a clip of `mel_frames` (the processor's
/// `compute_audio_token_len`: `⌈⌈⌈M/2⌉/2⌉/4⌉`). Equal to the number of
/// 4-frame groups the encoder produces, since 6000 is divisible by 4.
pub fn audio_token_count(mel_frames: usize, group: usize) -> usize {
    mel_frames.div_ceil(2).div_ceil(2).div_ceil(group)
}

/// The placeholder id run for one clip of `k` embedding rows.
pub fn audio_placeholder_ids(k: usize) -> Vec<u32> {
    let mut v = Vec::with_capacity(k + 2);
    v.push(AUDIO_START_ID);
    v.extend(std::iter::repeat_n(AUDIO_PAD_ID, k));
    v.push(AUDIO_END_ID);
    v
}

/// Expand each `<|mimo_audio_start|><|audio_pad|><|mimo_audio_end|>`
/// triple, in order, to `counts[i]` pads. A count/triple mismatch, or a pad
/// outside a triple, is an error.
pub fn expand_audio_placeholders(ids: &[u32], counts: &[usize]) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut used = 0usize;
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] == AUDIO_START_ID
            && ids.get(i + 1) == Some(&AUDIO_PAD_ID)
            && ids.get(i + 2) == Some(&AUDIO_END_ID)
        {
            let k = *counts.get(used).ok_or_else(|| {
                format!(
                    "audio: the prompt has more audio placeholders than the {} audio input(s)",
                    counts.len()
                )
            })?;
            out.extend(audio_placeholder_ids(k));
            used += 1;
            i += 3;
            continue;
        }
        if ids[i] == AUDIO_PAD_ID {
            return Err(format!(
                "audio: <|audio_pad|> at position {i} is not inside a <|mimo_audio_start|>…<|mimo_audio_end|> triple"
            ));
        }
        out.push(ids[i]);
        i += 1;
    }
    if used != counts.len() {
        return Err(format!(
            "audio: {} audio input(s) but {used} placeholder(s) in the prompt",
            counts.len()
        ));
    }
    Ok(out)
}

// ───────────────────────────── numerics ─────────────────────────────

/// erfc by the Numerical Recipes rational form (1.2e-7 relative
/// everywhere, including the far tail where `1 + erf` would cancel).
fn erfc(x: f64) -> f64 {
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
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// `nn.GELU()` (erf form) written as `x/2 · erfc(−x/√2)`, which keeps
/// relative accuracy for negative inputs.
#[inline]
fn gelu(v: f32) -> f32 {
    (0.5 * v as f64 * erfc(-(v as f64) * std::f64::consts::FRAC_1_SQRT_2)) as f32
}

fn gelu_inplace(x: &mut [f32]) {
    for v in x {
        *v = gelu(*v);
    }
}

fn layer_norm_rows(x: &[f32], w: &[f32], b: &[f32], eps: f64, dst: &mut [f32]) {
    let d = w.len();
    for (xr, dr) in x.chunks_exact(d).zip(dst.chunks_exact_mut(d)) {
        let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
        let var = xr.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..d {
            dr[i] = ((xr[i] as f64 - mean) * inv * w[i] as f64 + b[i] as f64) as f32;
        }
    }
}

/// Qwen2 RMSNorm: `w · (x · rsqrt(mean(x²) + eps))`.
fn rms_norm_rows(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let d = w.len();
    for (xr, dr) in x.chunks_exact(d).zip(dst.chunks_exact_mut(d)) {
        let ms = xr.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..d {
            dr[i] = w[i] * ((xr[i] as f64 * inv) as f32);
        }
    }
}

fn add_bias(x: &mut [f32], b: &[f32]) {
    for row in x.chunks_exact_mut(b.len()) {
        for (v, bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
}

fn add_into(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// RoPE tables as the references build them: `inv_freq` in float32
/// (`1 / base^(arange(0, d, 2)/d)`), `freqs = pos · inv_freq` as one f32
/// product, cos/sin of that f32 angle. Returns `(cos, sin)`, each
/// `[positions][head_dim/2]` (the full table is `[f | f]`).
fn rope_tables(base: f64, head_dim: usize, positions: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| {
            let e = (2 * i) as f32 / head_dim as f32;
            1.0f32 / ((base as f32 as f64).powf(e as f64) as f32)
        })
        .collect();
    let mut cos = vec![0f32; positions * half];
    let mut sin = vec![0f32; positions * half];
    for p in 0..positions {
        for i in 0..half {
            let a = (p as f32 * inv[i]) as f64;
            cos[p * half + i] = a.cos() as f32;
            sin[p * half + i] = a.sin() as f32;
        }
    }
    (cos, sin)
}

/// In-place rotate_half RoPE on `x [rows][heads·hd]`, row r at position
/// `pos(r)`.
fn apply_rope(
    x: &mut [f32],
    heads: usize,
    hd: usize,
    cos: &[f32],
    sin: &[f32],
    pos: impl Fn(usize) -> usize,
) {
    let half = hd / 2;
    let width = heads * hd;
    for (r, row) in x.chunks_exact_mut(width).enumerate() {
        let p = pos(r);
        let (c, s) = (
            &cos[p * half..(p + 1) * half],
            &sin[p * half..(p + 1) * half],
        );
        for h in 0..heads {
            let v = &mut row[h * hd..(h + 1) * hd];
            for i in 0..half {
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * c[i] - b * s[i];
                v[i + half] = b * c[i] + a * s[i];
            }
        }
    }
}

/// f32 dot product in eight independent lanes (vectorizes; the lane sums
/// are added pairwise at the end).
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// f64 dot product of f32 vectors in four lanes.
#[inline]
fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    let mut acc = [0f64; 4];
    let (ca, cb) = (a.chunks_exact(4), b.chunks_exact(4));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..4 {
            acc[l] += x[l] as f64 * y[l] as f64;
        }
    }
    let mut s = (acc[0] + acc[2]) + (acc[1] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += *x as f64 * *y as f64;
    }
    s
}

/// Scaled-dot-product attention over `l` rows of `[heads·hd]` q/k/v, with
/// row `i` attending keys `range(i) = [lo, hi)`. Output `[l][heads·hd]`.
#[allow(clippy::too_many_arguments)]
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    l: usize,
    heads: usize,
    hd: usize,
    range: &(dyn Fn(usize) -> (usize, usize) + Sync),
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    let width = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let dst = SendMut::new(out.as_mut_ptr());
    let run = |start: usize, end: usize| {
        let mut sc: Vec<f32> = Vec::new();
        let mut acc = vec![0f32; hd];
        for unit in start..end {
            let (i, h) = (unit / heads, unit % heads);
            let (lo, hi) = range(i);
            let qi = &q[i * width + h * hd..i * width + (h + 1) * hd];
            sc.clear();
            let mut mx = f32::NEG_INFINITY;
            for j in lo..hi {
                let kj = &k[j * width + h * hd..j * width + (h + 1) * hd];
                let s = dot8(qi, kj) * scale;
                mx = mx.max(s);
                sc.push(s);
            }
            let mut sum = 0f64;
            for s in sc.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s as f64;
            }
            acc.iter_mut().for_each(|a| *a = 0.0);
            for (jj, &p) in sc.iter().enumerate() {
                let j = lo + jj;
                let vj = &v[j * width + h * hd..j * width + (h + 1) * hd];
                for t in 0..hd {
                    acc[t] += p * vj[t];
                }
            }
            let inv = (1.0 / sum) as f32;
            for t in 0..hd {
                unsafe { *dst.at(i * width + h * hd + t) = acc[t] * inv };
            }
        }
    };
    let units = l * heads;
    match pool {
        Some(p) if units >= 64 => p.run_rows(units, &run),
        _ => run(0, units),
    }
}

/// A projection with its source name (the GPTQ Hessian hook keys on it).
struct Mat {
    p: Proj,
    name: String,
}

/// `y [b][rows] = x [b][cols] · Wᵀ (+ bias)`.
///
/// Under `gptq_capture`, exact (f32) weights report their inputs here;
/// mapped quantized ones report from `QTensor::matmat` itself.
fn linear(m: &Mat, bias: Option<&[f32]>, x: &[f32], b: usize, pool: Option<&Pool>) -> Vec<f32> {
    let p = &m.p;
    if b > 0 && crate::gptq_capture::capturing() && matches!(p, Proj::F32 { .. }) {
        crate::gptq_capture::accumulate(&m.name, x, b, p.cols());
    }
    let mut y = vec![0f32; b * p.rows()];
    if b > 0 {
        p.matmat(x, b, &mut y, pool);
    }
    if let Some(bias) = bias {
        add_bias(&mut y, bias);
    }
    y
}

// ───────────────────────────── weight sources ─────────────────────────────

/// Where tower tensors come from. Names are the canonical source names
/// (`audio_tokenizer.encoder.*`, `audio_encoder.*`, `speech_embeddings.*`).
trait WeightSource {
    fn has(&self, name: &str) -> bool;
    /// Any tensor as f32 plus its shape.
    fn dense(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String>;
    /// A matrix `[out, in]` for the batched projection path. `cols` reshapes
    /// a rank-3 conv kernel `[out, in, k]` to `[out, in·k]`.
    fn proj(&mut self, name: &str) -> Result<Proj, String>;
    fn config_blob(&self, name: &str) -> Option<Vec<u8>>;
}

struct CmfSource(Arc<CmfModel>);

impl WeightSource for CmfSource {
    fn has(&self, name: &str) -> bool {
        self.0.tensor_index(name).is_some()
    }
    fn dense(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let idx = self
            .0
            .tensor_index(name)
            .ok_or_else(|| format!("mimo audio: missing tensor '{name}'"))?;
        let entry = &self.0.tensors[idx];
        let mut dst = vec![0f32; entry.n_elems()];
        cortiq_core::quant::dequant_tensor(entry, self.0.entry_bytes(entry), &mut dst)
            .map_err(|e| format!("mimo audio: tensor '{name}': {e}"))?;
        Ok((dst, entry.shape.clone()))
    }
    /// Quantized tower matrices are dequantized to f32 at load (about 2 GB
    /// for the audio towers) and run through the f32 GEMM. Left mapped,
    /// `QTensor::matmat`'s host arm quantizes the activations to int8,
    /// which the tokenizer's codes do not survive: with q8_2f weights the
    /// level-0 codes agreed with the exact tower on 95.2–96.3 % of frames
    /// that way against 96.5–98.6 % with f32 activations, and the encoder
    /// rows' mean cosine was 0.978–0.985 against 0.9994 (3 clips).
    /// `CMF_MIMO_AUDIO_MAPPED=1` keeps them mapped (less RAM, int8 host
    /// activations).
    fn proj(&mut self, name: &str) -> Result<Proj, String> {
        let entry = self
            .0
            .tensor(name)
            .ok_or_else(|| format!("mimo audio: missing tensor '{name}'"))?;
        if entry.shape.len() == 2 {
            let p = Proj::from_model(&self.0, name)?;
            let keep_mapped = std::env::var("CMF_MIMO_AUDIO_MAPPED").is_ok_and(|v| v == "1");
            if matches!(p, Proj::F32 { .. }) || keep_mapped {
                return Ok(p);
            }
        }
        let (w, shape) = self.dense(name)?;
        let cols = shape[1..].iter().product::<usize>();
        Ok(Proj::f32(w, cols))
    }
    fn config_blob(&self, name: &str) -> Option<Vec<u8>> {
        let e = self.0.tensor(name)?;
        Some(self.0.entry_bytes(e).to_vec())
    }
}

/// The development loader: tensors read straight from the HF checkpoint
/// (BF16/F16/F32 → f32, exact), only those the audio towers need.
struct HfSource {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
    blobs: HashMap<String, Vec<u8>>,
}

/// Read the named tensors of one safetensors file without touching the
/// rest of it. `rename(src_name) -> Some(canonical)` selects and renames.
fn read_safetensors_selected(
    path: &Path,
    rename: &dyn Fn(&str) -> Option<String>,
    out: &mut Vec<RawTensor>,
) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let hlen = u64::from_le_bytes(len8) as usize;
    let mut hbytes = vec![0u8; hlen];
    f.read_exact(&mut hbytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let header: Value = serde_json::from_slice(&hbytes)
        .map_err(|e| format!("{}: safetensors header: {e}", path.display()))?;
    let obj = header
        .as_object()
        .ok_or_else(|| format!("{}: safetensors header is not an object", path.display()))?;
    let base = 8 + hlen as u64;
    let mut picks: Vec<(u64, u64, String, String, Vec<usize>)> = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let Some(canon) = rename(name) else { continue };
        let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        let offs = meta["data_offsets"]
            .as_array()
            .ok_or_else(|| format!("{name}: no data_offsets"))?;
        let (s, e) = (offs[0].as_u64().unwrap_or(0), offs[1].as_u64().unwrap_or(0));
        picks.push((s, e, canon, dtype, shape));
    }
    picks.sort_by_key(|p| p.0);
    for (s, e, name, dtype, shape) in picks {
        let mut bytes = vec![0u8; (e - s) as usize];
        f.seek(SeekFrom::Start(base + s))
            .and_then(|_| f.read_exact(&mut bytes))
            .map_err(|err| format!("{}: {name}: {err}", path.display()))?;
        out.push(RawTensor {
            name,
            dtype,
            shape,
            bytes,
        });
    }
    Ok(())
}

/// One tensor as stored in a safetensors file, under its canonical name.
#[derive(Clone, Debug)]
pub struct RawTensor {
    pub name: String,
    /// safetensors dtype string: "BF16", "F16" or "F32".
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl RawTensor {
    pub fn to_f32(&self) -> Result<Vec<f32>, String> {
        let data: Vec<f32> = match self.dtype.as_str() {
            "F32" => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => self
                .bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            "F16" => self
                .bytes
                .chunks_exact(2)
                .map(|c| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            other => {
                return Err(format!(
                    "{}: unsupported safetensors dtype {other}",
                    self.name
                ));
            }
        };
        if data.len() != self.shape.iter().product::<usize>().max(1) {
            return Err(format!(
                "{}: {} values for shape {:?}",
                self.name,
                data.len(),
                self.shape
            ));
        }
        Ok(data)
    }
}

/// Tokenizer tensors the encoder path does not use (EMA state).
fn tokenizer_tensor_unused(name: &str) -> bool {
    name.ends_with("._codebook.cluster_size")
        || name.ends_with("._codebook.embed_avg")
        || name.ends_with("._codebook.inited")
}

/// Development helper: every audio-tower tensor of an HF MiMo-V2.6
/// checkpoint directory under its canonical name (`audio_tokenizer.` is
/// prefixed to the tokenizer file's `encoder.*`; its decoder and EMA state
/// are skipped), plus the config blobs `mm.config_json` and
/// `audio_tokenizer.config_json` when the files exist.
pub fn read_hf_audio_tensors(
    dir: &Path,
) -> Result<(Vec<RawTensor>, Vec<(String, Vec<u8>)>), String> {
    let mut tensors = Vec::new();
    let at_dir = dir.join("audio_tokenizer");
    read_safetensors_selected(
        &at_dir.join("model.safetensors"),
        &|n| {
            (n.starts_with("encoder.") && !tokenizer_tensor_unused(n))
                .then(|| format!("audio_tokenizer.{n}"))
        },
        &mut tensors,
    )?;
    // LLM-side audio tensors: whichever shards the index names.
    let index_path = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_slice(
        &std::fs::read(&index_path).map_err(|e| format!("{}: {e}", index_path.display()))?,
    )
    .map_err(|e| format!("{}: {e}", index_path.display()))?;
    let map = index["weight_map"]
        .as_object()
        .ok_or("model.safetensors.index.json: no weight_map")?;
    let keep = |k: &str| k.starts_with("audio_encoder.") || k.starts_with("speech_embeddings.");
    let mut files: Vec<String> = map
        .iter()
        .filter(|(k, _)| keep(k))
        .filter_map(|(_, v)| v.as_str().map(str::to_string))
        .collect();
    files.sort();
    files.dedup();
    for file in files {
        read_safetensors_selected(
            &dir.join(&file),
            &|n| keep(n).then(|| n.to_string()),
            &mut tensors,
        )?;
    }
    let mut blobs = Vec::new();
    for (blob, file) in [
        ("mm.config_json", dir.join("config.json")),
        ("audio_tokenizer.config_json", at_dir.join("config.json")),
    ] {
        if let Ok(b) = std::fs::read(&file) {
            blobs.push((blob.to_string(), b));
        }
    }
    Ok((tensors, blobs))
}

impl HfSource {
    fn open(dir: &Path) -> Result<Self, String> {
        let (raw, blobs) = read_hf_audio_tensors(dir)?;
        let mut tensors = HashMap::with_capacity(raw.len());
        for t in raw {
            let data = t.to_f32()?;
            tensors.insert(t.name, (data, t.shape));
        }
        Ok(Self {
            tensors,
            blobs: blobs.into_iter().collect(),
        })
    }
}

impl WeightSource for HfSource {
    fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    fn dense(&mut self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.tensors
            .remove(name)
            .ok_or_else(|| format!("mimo audio: missing tensor '{name}'"))
    }
    fn proj(&mut self, name: &str) -> Result<Proj, String> {
        let (w, shape) = self.dense(name)?;
        if shape.len() < 2 {
            return Err(format!("mimo audio: '{name}' is not a matrix: {shape:?}"));
        }
        let cols = shape[1..].iter().product::<usize>();
        Ok(Proj::f32(w, cols))
    }
    fn config_blob(&self, name: &str) -> Option<Vec<u8>> {
        self.blobs.get(name).cloned()
    }
}

fn take_vec(src: &mut dyn WeightSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let (v, shape) = src.dense(name)?;
    if v.len() != len {
        return Err(format!(
            "mimo audio: '{name}' has shape {shape:?}, expected {len} values"
        ));
    }
    Ok(v)
}

fn take_proj(
    src: &mut dyn WeightSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Mat, String> {
    let p = src.proj(name)?;
    if p.rows() != rows || p.cols() != cols {
        return Err(format!(
            "mimo audio: '{name}' is [{}, {}], expected [{rows}, {cols}]",
            p.rows(),
            p.cols()
        ));
    }
    Ok(Mat {
        p,
        name: name.to_string(),
    })
}

// ───────────────────────────── configs ─────────────────────────────

/// The audio tokenizer encoder hyper-parameters (`audio_tokenizer/config.json`).
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    pub d_model: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub n_mels: usize,
    pub skip_layer_id: Option<usize>,
    pub causal: bool,
    /// Band half-width on windowed layers (`encoder_attn_window_size[0]`),
    /// 0 = none.
    pub window: usize,
    pub hybrid: bool,
    pub swa_per_block: usize,
    pub rope_theta: f64,
    pub codebook_sizes: Vec<usize>,
    pub ln_eps: f64,
}

impl TokenizerConfig {
    /// The pinned MiMo-Audio-Tokenizer values.
    pub fn mimo_default() -> Self {
        let mut books = vec![1024, 1024, 256];
        books.extend(std::iter::repeat_n(128, 17));
        Self {
            d_model: 1024,
            layers: 24,
            heads: 16,
            ffn: 4096,
            n_mels: 128,
            skip_layer_id: Some(3),
            causal: true,
            window: 128,
            hybrid: true,
            swa_per_block: 2,
            rope_theta: 10_000.0,
            codebook_sizes: books,
            ln_eps: 1e-5,
        }
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let v: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("audio_tokenizer config: {e}"))?;
        let d = Self::mimo_default();
        let us = |k: &str, dflt: usize| {
            v.get(k)
                .and_then(Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let bl = |k: &str, dflt: bool| v.get(k).and_then(Value::as_bool).unwrap_or(dflt);
        // Only the geometry this implementation computes is accepted.
        for (k, want) in [
            ("kernel_size", 3u64),
            ("stride_size", 2),
            ("avg_pooler", 2),
            ("sampling_rate", SAMPLE_RATE as u64),
            ("hop_length", HOP as u64),
            ("nfft", N_FFT as u64),
            ("window_size", N_FFT as u64),
        ] {
            if let Some(got) = v.get(k).and_then(Value::as_u64) {
                if got != want {
                    return Err(format!(
                        "audio_tokenizer config: {k} = {got}, only {want} is supported"
                    ));
                }
            }
        }
        if bl("scale_embedding", false) {
            return Err("audio_tokenizer config: scale_embedding = true is not supported".into());
        }
        if let Some(ln) = v.get("ln_type").and_then(Value::as_str) {
            if ln != "LayerNorm" {
                return Err(format!(
                    "audio_tokenizer config: ln_type {ln} is not supported"
                ));
            }
        }
        let n_q = us("num_quantizers", 20);
        let mut books: Vec<usize> = match v.get("codebook_size") {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(Value::as_u64)
                .map(|x| x as usize)
                .collect(),
            Some(x) => vec![x.as_u64().unwrap_or(1024) as usize],
            None => d.codebook_sizes.clone(),
        };
        if books.is_empty() {
            return Err("audio_tokenizer config: empty codebook_size".into());
        }
        while books.len() < n_q {
            books.push(*books.last().unwrap());
        }
        books.truncate(n_q);
        let window = v
            .get("encoder_attn_window_size")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_i64)
            .map_or(d.window as i64, |x| x)
            .max(0) as usize;
        Ok(Self {
            d_model: us("d_model", d.d_model),
            layers: us("encoder_layers", d.layers),
            heads: us("encoder_attention_heads", d.heads),
            ffn: us("encoder_ffn_dim", d.ffn),
            n_mels: us("n_mels", d.n_mels),
            skip_layer_id: match v.get("encoder_skip_layer_id") {
                Some(Value::Null) => None,
                Some(x) => x.as_u64().map(|x| x as usize),
                None => d.skip_layer_id,
            },
            causal: bl("encoder_causal", d.causal),
            window,
            hybrid: bl("hybrid_attention", d.hybrid),
            swa_per_block: us("swa_per_block", d.swa_per_block).max(1),
            rope_theta: v
                .get("rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(d.rope_theta),
            codebook_sizes: books,
            ln_eps: d.ln_eps,
        })
    }

    /// Whether layer `i` carries the band window (HF: hybrid → the first
    /// `swa_per_block − 1` of every `swa_per_block` layers; else all).
    pub fn windowed(&self, i: usize) -> bool {
        self.window > 0 && (!self.hybrid || i % self.swa_per_block < self.swa_per_block - 1)
    }
}

/// The LLM-side audio encoder hyper-parameters (`config.json` `audio_config`).
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub channels: usize,
    pub group: usize,
    pub dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rope_theta: f64,
    pub out_dim: usize,
    pub vocab: usize,
    pub bidirectional: bool,
    pub post_norm: bool,
    pub projection_layers: usize,
    pub rms_eps: f64,
}

impl EncoderConfig {
    pub fn mimo_default() -> Self {
        Self {
            channels: 20,
            group: 4,
            dim: 1024,
            layers: 6,
            heads: 16,
            head_dim: 64,
            ffn: 4096,
            rope_theta: 640_000.0,
            out_dim: 4096,
            vocab: 1280,
            bidirectional: true,
            post_norm: true,
            projection_layers: 2,
            rms_eps: 1e-6,
        }
    }

    /// Parse the `audio_config` object of the main `config.json`.
    pub fn from_main_config(bytes: &[u8]) -> Result<Self, String> {
        let v: Value =
            serde_json::from_slice(bytes).map_err(|e| format!("mimo config.json: {e}"))?;
        let a = v
            .get("audio_config")
            .filter(|a| a.is_object())
            .ok_or("mimo config.json: no audio_config")?;
        let d = Self::mimo_default();
        let us = |k: &str, dflt: usize| {
            a.get(k)
                .and_then(Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let first_num = |k: &str, dflt: usize| -> usize {
            match a.get(k) {
                Some(Value::String(s)) => s
                    .split('-')
                    .next()
                    .and_then(|x| x.trim().parse().ok())
                    .unwrap_or(dflt),
                Some(x) => x.as_u64().map_or(dflt, |x| x as usize),
                None => dflt,
            }
        };
        if let Some(Value::String(s)) = a.get("speech_vocab_size") {
            if s.contains('-') {
                return Err(format!(
                    "mimo audio_config: per-channel speech_vocab_size '{s}' is not supported"
                ));
            }
        }
        let prf = a
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if (prf - 1.0).abs() > 1e-9 {
            return Err(format!(
                "mimo audio_config: partial_rotary_factor {prf} is not supported"
            ));
        }
        let dim = us("input_local_dim", d.dim);
        let heads = us("input_local_attn_heads", d.heads);
        Ok(Self {
            channels: us("audio_channels", d.channels),
            group: us("group_size", d.group),
            dim,
            layers: us("input_local_layers", d.layers),
            heads,
            head_dim: dim / heads.max(1),
            ffn: us("input_local_intermediate_size", d.ffn),
            rope_theta: a
                .get("rope_theta")
                .and_then(Value::as_f64)
                .unwrap_or(d.rope_theta),
            out_dim: us("out_hidden_size", d.out_dim),
            vocab: first_num("speech_vocab_size", d.vocab),
            bidirectional: a
                .get("input_full_attention")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            post_norm: a
                .get("add_post_norm")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            projection_layers: us("projection_layers", d.projection_layers),
            rms_eps: d.rms_eps,
        })
    }
}

// ───────────────────────────── tokenizer ─────────────────────────────

struct TokLayer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    q: Mat,
    q_b: Vec<f32>,
    k: Mat,
    v: Mat,
    v_b: Vec<f32>,
    o: Mat,
    o_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc1: Mat,
    fc1_b: Vec<f32>,
    fc2: Mat,
    fc2_b: Vec<f32>,
}

/// The MiMo audio tokenizer encoder and its residual VQ.
pub struct AudioTokenizer {
    pub cfg: TokenizerConfig,
    conv1: Mat, // [d, n_mels·3]
    conv1_b: Vec<f32>,
    conv2: Mat, // [d, d·3]
    conv2_b: Vec<f32>,
    layers: Vec<TokLayer>,
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    pool_w: Mat, // [d, d·2], no bias
    pool_ln_w: Vec<f32>,
    pool_ln_b: Vec<f32>,
    /// Codebooks as stored (f32), `[size][d]` each.
    books: Vec<Vec<f32>>,
    /// The same codebooks rounded to bf16 (what a bf16 serving load holds).
    books_bf16: Vec<Vec<f32>>,
}

fn bf16_round(x: f32) -> f32 {
    let b = x.to_bits();
    if x.is_nan() {
        return x;
    }
    let lsb = (b >> 16) & 1;
    let r = b.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    f32::from_bits(r)
}

impl AudioTokenizer {
    fn load(src: &mut dyn WeightSource, cfg: TokenizerConfig) -> Result<Self, String> {
        let p = "audio_tokenizer.encoder";
        let d = cfg.d_model;
        if d % cfg.heads != 0 || (d / cfg.heads) % 2 != 0 {
            return Err(format!(
                "audio tokenizer: d_model {d} / heads {} is not an even head width",
                cfg.heads
            ));
        }
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("{p}.layers.{i}");
            if src.has(&format!("{l}.self_attn.k_proj.bias")) {
                return Err(format!(
                    "audio tokenizer: unexpected {l}.self_attn.k_proj.bias (k has no bias)"
                ));
            }
            layers.push(TokLayer {
                ln1_w: take_vec(src, &format!("{l}.self_attn_layer_norm.weight"), d)?,
                ln1_b: take_vec(src, &format!("{l}.self_attn_layer_norm.bias"), d)?,
                q: take_proj(src, &format!("{l}.self_attn.q_proj.weight"), d, d)?,
                q_b: take_vec(src, &format!("{l}.self_attn.q_proj.bias"), d)?,
                k: take_proj(src, &format!("{l}.self_attn.k_proj.weight"), d, d)?,
                v: take_proj(src, &format!("{l}.self_attn.v_proj.weight"), d, d)?,
                v_b: take_vec(src, &format!("{l}.self_attn.v_proj.bias"), d)?,
                o: take_proj(src, &format!("{l}.self_attn.out_proj.weight"), d, d)?,
                o_b: take_vec(src, &format!("{l}.self_attn.out_proj.bias"), d)?,
                ln2_w: take_vec(src, &format!("{l}.final_layer_norm.weight"), d)?,
                ln2_b: take_vec(src, &format!("{l}.final_layer_norm.bias"), d)?,
                fc1: take_proj(src, &format!("{l}.fc1.weight"), cfg.ffn, d)?,
                fc1_b: take_vec(src, &format!("{l}.fc1.bias"), cfg.ffn)?,
                fc2: take_proj(src, &format!("{l}.fc2.weight"), d, cfg.ffn)?,
                fc2_b: take_vec(src, &format!("{l}.fc2.bias"), d)?,
            });
        }
        let mut books = Vec::with_capacity(cfg.codebook_sizes.len());
        for (i, &size) in cfg.codebook_sizes.iter().enumerate() {
            books.push(take_vec(
                src,
                &format!("{p}.quantizer.vq.layers.{i}._codebook.embed"),
                size * d,
            )?);
        }
        let books_bf16 = books
            .iter()
            .map(|b| b.iter().map(|&v| bf16_round(v)).collect())
            .collect();
        Ok(Self {
            conv1: take_proj(src, &format!("{p}.conv1.weight"), d, cfg.n_mels * 3)?,
            conv1_b: take_vec(src, &format!("{p}.conv1.bias"), d)?,
            conv2: take_proj(src, &format!("{p}.conv2.weight"), d, d * 3)?,
            conv2_b: take_vec(src, &format!("{p}.conv2.bias"), d)?,
            layers,
            ln_w: take_vec(src, &format!("{p}.layer_norm.weight"), d)?,
            ln_b: take_vec(src, &format!("{p}.layer_norm.bias"), d)?,
            pool_w: take_proj(src, &format!("{p}.down_sample_layer.0.weight"), d, d * 2)?,
            pool_ln_w: take_vec(src, &format!("{p}.down_sample_norm.weight"), d)?,
            pool_ln_b: take_vec(src, &format!("{p}.down_sample_norm.bias"), d)?,
            books,
            books_bf16,
            cfg,
        })
    }

    /// Pre-RVQ features of ONE segment of `m` mel frames (`mel [m][n_mels]`):
    /// rows `[⌈⌈m/2⌉/2⌉][d]` after `down_sample_norm`.
    pub fn features(&self, mel: &[f32], m: usize, pool: Option<&Pool>) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let (d, nm) = (cfg.d_model, cfg.n_mels);
        if m == 0 || mel.len() != m * nm {
            return Err(format!(
                "audio tokenizer: mel has {} values for {m} frames",
                mel.len()
            ));
        }
        // conv1: k3 p1 stride 1, im2col row t = [x[t+k−1][c]] in (c, k) order.
        let mut col = vec![0f32; m * nm * 3];
        for t in 0..m {
            for k in 0..3 {
                let s = t as isize + k as isize - 1;
                if s < 0 || s >= m as isize {
                    continue;
                }
                let src = &mel[s as usize * nm..(s as usize + 1) * nm];
                let row = &mut col[t * nm * 3..(t + 1) * nm * 3];
                for c in 0..nm {
                    row[c * 3 + k] = src[c];
                }
            }
        }
        let mut h1 = linear(&self.conv1, Some(&self.conv1_b), &col, m, pool);
        gelu_inplace(&mut h1);
        // conv2: k3 p1 stride 2.
        let l1 = m.div_ceil(2);
        let mut col = vec![0f32; l1 * d * 3];
        for t in 0..l1 {
            for k in 0..3 {
                let s = 2 * t as isize + k as isize - 1;
                if s < 0 || s >= m as isize {
                    continue;
                }
                let src = &h1[s as usize * d..(s as usize + 1) * d];
                let row = &mut col[t * d * 3..(t + 1) * d * 3];
                for c in 0..d {
                    row[c * 3 + k] = src[c];
                }
            }
        }
        drop(h1);
        let mut h = linear(&self.conv2, Some(&self.conv2_b), &col, l1, pool);
        drop(col);
        gelu_inplace(&mut h);

        let heads = cfg.heads;
        let hd = d / heads;
        let (cos, sin) = rope_tables(cfg.rope_theta, hd, l1);
        let mut skip: Option<Vec<f32>> = None;
        let mut n = vec![0f32; l1 * d];
        let mut att = vec![0f32; l1 * d];
        for (li, layer) in self.layers.iter().enumerate() {
            layer_norm_rows(&h, &layer.ln1_w, &layer.ln1_b, cfg.ln_eps, &mut n);
            let mut q = linear(&layer.q, Some(&layer.q_b), &n, l1, pool);
            let mut k = linear(&layer.k, None, &n, l1, pool);
            let v = linear(&layer.v, Some(&layer.v_b), &n, l1, pool);
            apply_rope(&mut q, heads, hd, &cos, &sin, |r| r);
            apply_rope(&mut k, heads, hd, &cos, &sin, |r| r);
            let causal = cfg.causal;
            let window = if cfg.windowed(li) { cfg.window } else { 0 };
            let range = move |i: usize| -> (usize, usize) {
                let hi = if causal { i + 1 } else { l1 };
                let (lo, hi) = if window > 0 {
                    (i.saturating_sub(window), hi.min(i + window + 1))
                } else {
                    (0, hi)
                };
                (lo, hi)
            };
            attention(&q, &k, &v, l1, heads, hd, &range, &mut att, pool);
            let o = linear(&layer.o, Some(&layer.o_b), &att, l1, pool);
            add_into(&mut h, &o);
            layer_norm_rows(&h, &layer.ln2_w, &layer.ln2_b, cfg.ln_eps, &mut n);
            let mut f = linear(&layer.fc1, Some(&layer.fc1_b), &n, l1, pool);
            gelu_inplace(&mut f);
            let f2 = linear(&layer.fc2, Some(&layer.fc2_b), &f, l1, pool);
            add_into(&mut h, &f2);
            if cfg.skip_layer_id.is_some_and(|s| s >= 1 && li == s - 1) {
                skip = Some(h.clone());
            }
        }
        if let Some(s) = skip {
            add_into(&mut h, &s);
        }
        layer_norm_rows(&h, &self.ln_w, &self.ln_b, cfg.ln_eps, &mut n);
        // Pooler: zero-pad to even, conv k2 s2 (no bias), GELU, LayerNorm.
        let l2 = l1.div_ceil(2);
        let mut col = vec![0f32; l2 * d * 2];
        for t in 0..l2 {
            for k in 0..2 {
                let s = 2 * t + k;
                if s >= l1 {
                    continue;
                }
                let src = &n[s * d..(s + 1) * d];
                let row = &mut col[t * d * 2..(t + 1) * d * 2];
                for c in 0..d {
                    row[c * 2 + k] = src[c];
                }
            }
        }
        let mut p = linear(&self.pool_w, None, &col, l2, pool);
        gelu_inplace(&mut p);
        let mut out = vec![0f32; l2 * d];
        layer_norm_rows(&p, &self.pool_ln_w, &self.pool_ln_b, cfg.ln_eps, &mut out);
        Ok(out)
    }

    /// Residual VQ in f32: per level, the nearest codeword by
    /// `‖E‖² − 2 r·E` (the row's ‖r‖² does not move the argmin), then
    /// `r −= E[idx]`. `bf16_books` selects the bf16-rounded codebooks a
    /// bf16 serving load holds. Returns `[rows][levels]`.
    pub fn quantize(
        &self,
        feats: &[f32],
        rows: usize,
        bf16_books: bool,
        pool: Option<&Pool>,
    ) -> Vec<u32> {
        let d = self.cfg.d_model;
        let books = if bf16_books {
            &self.books_bf16
        } else {
            &self.books
        };
        let nq = books.len();
        let norms: Vec<Vec<f64>> = books
            .iter()
            .map(|b| {
                b.chunks_exact(d)
                    .map(|e| e.iter().map(|&v| v as f64 * v as f64).sum())
                    .collect()
            })
            .collect();
        let mut codes = vec![0u32; rows * nq];
        let dst = crate::pool::SendMutT::new(codes.as_mut_ptr());
        let run = |start: usize, end: usize| {
            let mut r = vec![0f32; d];
            for row in start..end {
                r.copy_from_slice(&feats[row * d..(row + 1) * d]);
                for (lvl, book) in books.iter().enumerate() {
                    let mut best = (f64::INFINITY, 0usize);
                    for (ci, e) in book.chunks_exact(d).enumerate() {
                        let dist = norms[lvl][ci] - 2.0 * dot_f64(&r, e);
                        if dist < best.0 {
                            best = (dist, ci);
                        }
                    }
                    let e = &book[best.1 * d..(best.1 + 1) * d];
                    for t in 0..d {
                        r[t] -= e[t];
                    }
                    unsafe { *dst.at(row * nq + lvl) = best.1 as u32 };
                }
            }
        };
        match pool {
            Some(p) if rows >= 8 => p.run_rows(rows, &run),
            _ => run(0, rows),
        }
        codes
    }
}

// ───────────────────────────── LLM-side encoder ─────────────────────────────

struct LocalLayer {
    ln1: Vec<f32>,
    q: Mat,
    q_b: Vec<f32>,
    k: Mat,
    k_b: Vec<f32>,
    v: Mat,
    v_b: Vec<f32>,
    o: Mat,
    ln2: Vec<f32>,
    gate: Mat,
    up: Mat,
    down: Mat,
}

/// Speech embeddings + the local Qwen2 transformer + the projection.
pub struct AudioEncoder {
    pub cfg: EncoderConfig,
    /// `[channel][vocab·dim]`.
    speech_emb: Vec<Vec<f32>>,
    layers: Vec<LocalLayer>,
    norm: Option<Vec<f32>>,
    proj0: Mat,
    proj2: Option<Mat>,
}

impl AudioEncoder {
    fn load(src: &mut dyn WeightSource, cfg: EncoderConfig) -> Result<Self, String> {
        let (d, f) = (cfg.dim, cfg.ffn);
        let qd = cfg.heads * cfg.head_dim;
        let mut speech_emb = Vec::with_capacity(cfg.channels);
        for c in 0..cfg.channels {
            speech_emb.push(take_vec(
                src,
                &format!("speech_embeddings.{c}.weight"),
                cfg.vocab * d,
            )?);
        }
        let p = "audio_encoder.input_local_transformer";
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let l = format!("{p}.layers.{i}");
            layers.push(LocalLayer {
                ln1: take_vec(src, &format!("{l}.input_layernorm.weight"), d)?,
                q: take_proj(src, &format!("{l}.self_attn.q_proj.weight"), qd, d)?,
                q_b: take_vec(src, &format!("{l}.self_attn.q_proj.bias"), qd)?,
                k: take_proj(src, &format!("{l}.self_attn.k_proj.weight"), qd, d)?,
                k_b: take_vec(src, &format!("{l}.self_attn.k_proj.bias"), qd)?,
                v: take_proj(src, &format!("{l}.self_attn.v_proj.weight"), qd, d)?,
                v_b: take_vec(src, &format!("{l}.self_attn.v_proj.bias"), qd)?,
                o: take_proj(src, &format!("{l}.self_attn.o_proj.weight"), d, qd)?,
                ln2: take_vec(src, &format!("{l}.post_attention_layernorm.weight"), d)?,
                gate: take_proj(src, &format!("{l}.mlp.gate_proj.weight"), f, d)?,
                up: take_proj(src, &format!("{l}.mlp.up_proj.weight"), f, d)?,
                down: take_proj(src, &format!("{l}.mlp.down_proj.weight"), d, f)?,
            });
        }
        let norm = if cfg.post_norm {
            Some(take_vec(src, &format!("{p}.norm.weight"), d)?)
        } else {
            None
        };
        let flat = d * cfg.group;
        let (proj0, proj2) = match cfg.projection_layers {
            2 => (
                take_proj(src, "audio_encoder.projection.mlp.0.weight", flat * 4, flat)?,
                Some(take_proj(
                    src,
                    "audio_encoder.projection.mlp.2.weight",
                    cfg.out_dim,
                    flat * 4,
                )?),
            ),
            1 => (
                take_proj(src, "audio_encoder.projection.weight", cfg.out_dim, flat)?,
                None,
            ),
            n => {
                return Err(format!(
                    "mimo audio_config: projection_layers = {n} is not supported"
                ));
            }
        };
        Ok(Self {
            cfg,
            speech_emb,
            layers,
            norm,
            proj0,
            proj2,
        })
    }

    /// Codes `[t][cols]` (cols ≥ channels; extra columns are ignored) →
    /// grouped `[G][group][channels]`, the tail padded by repeating the last
    /// row.
    pub fn group_codes(
        &self,
        codes: &[u32],
        t: usize,
        cols: usize,
    ) -> Result<(Vec<u32>, usize), String> {
        let (c, g) = (self.cfg.channels, self.cfg.group);
        if cols < c {
            return Err(format!("audio codes have {cols} channels, need {c}"));
        }
        if t == 0 || codes.len() != t * cols {
            return Err(format!(
                "audio codes: {} values for {t}×{cols}",
                codes.len()
            ));
        }
        let padded = t.div_ceil(g) * g;
        let mut out = Vec::with_capacity(padded * c);
        for r in 0..padded {
            let src = r.min(t - 1);
            out.extend_from_slice(&codes[src * cols..src * cols + c]);
        }
        Ok((out, padded / g))
    }

    /// Codes `[t][cols]` → LLM rows `[G][out_dim]`, `G = ⌈t/group⌉`.
    pub fn forward(
        &self,
        codes: &[u32],
        t: usize,
        cols: usize,
        pool: Option<&Pool>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.cfg;
        let (grouped, groups) = self.group_codes(codes, t, cols)?;
        let (d, c, g) = (cfg.dim, cfg.channels, cfg.group);
        let rows = groups * g;
        // Speech embeddings: Σ over channels, channel order, in f32.
        let mut x = vec![0f32; rows * d];
        for r in 0..rows {
            let dst = &mut x[r * d..(r + 1) * d];
            for ch in 0..c {
                let id = grouped[r * c + ch] as usize;
                if id >= cfg.vocab {
                    return Err(format!(
                        "audio code {id} in channel {ch} is outside the {} embedding rows",
                        cfg.vocab
                    ));
                }
                let e = &self.speech_emb[ch][id * d..(id + 1) * d];
                for (a, b) in dst.iter_mut().zip(e) {
                    *a += b;
                }
            }
        }
        let (heads, hd) = (cfg.heads, cfg.head_dim);
        let (cos, sin) = rope_tables(cfg.rope_theta, hd, g);
        let mut n = vec![0f32; rows * d];
        let mut att = vec![0f32; rows * heads * hd];
        let bidir = cfg.bidirectional;
        let range = move |i: usize| -> (usize, usize) {
            let base = i / g * g;
            (base, if bidir { base + g } else { i + 1 })
        };
        for layer in &self.layers {
            rms_norm_rows(&x, &layer.ln1, cfg.rms_eps, &mut n);
            let mut q = linear(&layer.q, Some(&layer.q_b), &n, rows, pool);
            let mut k = linear(&layer.k, Some(&layer.k_b), &n, rows, pool);
            let v = linear(&layer.v, Some(&layer.v_b), &n, rows, pool);
            apply_rope(&mut q, heads, hd, &cos, &sin, |r| r % g);
            apply_rope(&mut k, heads, hd, &cos, &sin, |r| r % g);
            attention(&q, &k, &v, rows, heads, hd, &range, &mut att, pool);
            let o = linear(&layer.o, None, &att, rows, pool);
            add_into(&mut x, &o);
            rms_norm_rows(&x, &layer.ln2, cfg.rms_eps, &mut n);
            let mut gt = linear(&layer.gate, None, &n, rows, pool);
            let up = linear(&layer.up, None, &n, rows, pool);
            for (a, b) in gt.iter_mut().zip(&up) {
                let s = *a / (1.0 + (-*a).exp());
                *a = s * b;
            }
            let dn = linear(&layer.down, None, &gt, rows, pool);
            add_into(&mut x, &dn);
        }
        if let Some(w) = &self.norm {
            rms_norm_rows(&x, w, cfg.rms_eps, &mut n);
            std::mem::swap(&mut x, &mut n);
        }
        // [G, group·dim] is exactly the row-major [rows, dim] buffer.
        let mut hdn = linear(&self.proj0, None, &x, groups, pool);
        match &self.proj2 {
            Some(p2) => {
                gelu_inplace(&mut hdn);
                Ok(linear(p2, None, &hdn, groups, pool))
            }
            None => Ok(hdn),
        }
    }
}

// ───────────────────────────── the whole tower ─────────────────────────────

/// Tokenizer codes of one clip.
#[derive(Clone, Debug)]
pub struct AudioCodes {
    /// 25 Hz frames.
    pub frames: usize,
    /// Codes per frame (the RVQ levels).
    pub levels: usize,
    /// Row-major `[frames][levels]`.
    pub codes: Vec<u32>,
}

/// One clip ready for injection: `rows [n_tokens][out_dim]`.
#[derive(Clone, Debug)]
pub struct AudioEmbeds {
    pub n_tokens: usize,
    pub dim: usize,
    pub rows: Vec<f32>,
}

/// The MiMo audio towers.
pub struct MimoAudio {
    pub tokenizer: AudioTokenizer,
    pub encoder: AudioEncoder,
    /// Quantize against bf16-rounded codebooks, as the bf16 serving stacks
    /// do (default on; `CMF_MIMO_AUDIO_BF16_BOOKS=0` turns it off).
    pub bf16_codebooks: bool,
    pool: Option<Arc<Pool>>,
}

impl MimoAudio {
    /// Whether a CMF carries the audio towers (a companion or a single-file
    /// multimodal container).
    pub fn present_in(model: &CmfModel) -> bool {
        model
            .tensor_index("audio_tokenizer.encoder.conv1.weight")
            .is_some()
            && model
                .tensor_index("audio_encoder.projection.mlp.0.weight")
                .is_some()
    }

    fn configs(src: &dyn WeightSource) -> Result<(TokenizerConfig, EncoderConfig), String> {
        let tok = match src.config_blob("audio_tokenizer.config_json") {
            Some(b) => TokenizerConfig::from_json(&b)?,
            None => TokenizerConfig::mimo_default(),
        };
        let enc = match src.config_blob("mm.config_json") {
            Some(b) => EncoderConfig::from_main_config(&b)?,
            None => EncoderConfig::mimo_default(),
        };
        if tok.codebook_sizes.len() < enc.channels {
            return Err(format!(
                "mimo audio: {} RVQ levels but {} speech embedding channels",
                tok.codebook_sizes.len(),
                enc.channels
            ));
        }
        if let Some(&big) = tok.codebook_sizes.iter().max() {
            if big > enc.vocab {
                return Err(format!(
                    "mimo audio: codebook of {big} entries exceeds the {} speech embedding rows",
                    enc.vocab
                ));
            }
        }
        Ok((tok, enc))
    }

    fn build(mut src: Box<dyn WeightSource>) -> Result<Self, String> {
        let (tc, ec) = Self::configs(src.as_ref())?;
        let tokenizer = AudioTokenizer::load(src.as_mut(), tc)?;
        let encoder = AudioEncoder::load(src.as_mut(), ec)?;
        let bf16_codebooks = std::env::var("CMF_MIMO_AUDIO_BF16_BOOKS")
            .map(|v| v != "0")
            .unwrap_or(true);
        Ok(Self {
            tokenizer,
            encoder,
            bf16_codebooks,
            pool: Pool::from_env(),
        })
    }

    /// Load from a CMF holding the source tensor names (the `<stem>.mm.cmf`
    /// companion or a single-file multimodal container).
    pub fn from_model(model: &Arc<CmfModel>) -> Result<Self, String> {
        Self::build(Box::new(CmfSource(model.clone())))
    }

    /// Development loader: read the tensors straight from the HF checkpoint
    /// directory (`config.json`, `model.safetensors.index.json` and its
    /// shards, `audio_tokenizer/`), exact f32 from BF16.
    pub fn from_hf_dir(dir: &Path) -> Result<Self, String> {
        Self::build(Box::new(HfSource::open(dir)?))
    }

    pub fn pool(&self) -> Option<&Pool> {
        self.pool.as_deref()
    }

    /// WAV bytes → 24 kHz mono → log-mel `[M][128]`.
    pub fn wav_to_mel(&self, bytes: &[u8]) -> Result<(Vec<f32>, usize), String> {
        let wav = decode_wav(bytes)?;
        let mono = wav_to_mono_24k(&wav)?;
        log_mel(&mono, self.pool())
    }

    /// Pre-RVQ features of every segment, concatenated `[codes][d]`.
    pub fn features(&self, mel: &[f32], m: usize) -> Result<Vec<f32>, String> {
        let nm = self.tokenizer.cfg.n_mels;
        let mut out = Vec::new();
        let mut start = 0usize;
        for seg in segment_lengths(m) {
            out.extend(self.tokenizer.features(
                &mel[start * nm..(start + seg) * nm],
                seg,
                self.pool(),
            )?);
            start += seg;
        }
        Ok(out)
    }

    /// Log-mel → tokenizer codes, each 6000-frame segment on its own.
    pub fn encode_mel(&self, mel: &[f32], m: usize) -> Result<AudioCodes, String> {
        let d = self.tokenizer.cfg.d_model;
        let feats = self.features(mel, m)?;
        let frames = feats.len() / d;
        debug_assert_eq!(frames, codes_for_mel(m));
        let codes = self
            .tokenizer
            .quantize(&feats, frames, self.bf16_codebooks, self.pool());
        Ok(AudioCodes {
            frames,
            levels: self.tokenizer.books.len(),
            codes,
        })
    }

    /// Codes → LLM embedding rows.
    pub fn embed_codes(&self, codes: &AudioCodes) -> Result<AudioEmbeds, String> {
        let rows = self
            .encoder
            .forward(&codes.codes, codes.frames, codes.levels, self.pool())?;
        let dim = self.encoder.cfg.out_dim;
        Ok(AudioEmbeds {
            n_tokens: rows.len() / dim,
            dim,
            rows,
        })
    }

    /// WAV bytes → LLM embedding rows (one per `<|audio_pad|>`).
    pub fn embed_wav(&self, bytes: &[u8]) -> Result<AudioEmbeds, String> {
        let (mel, m) = self.wav_to_mel(bytes)?;
        let codes = self.encode_mel(&mel, m)?;
        let emb = self.embed_codes(&codes)?;
        let k = audio_token_count(m, self.encoder.cfg.group);
        if emb.n_tokens != k {
            return Err(format!(
                "mimo audio: encoder produced {} rows but the placeholder count is {k}",
                emb.n_tokens
            ));
        }
        Ok(emb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_bytes(
        tag: u16,
        channels: u16,
        rate: u32,
        bits: u16,
        extensible: bool,
        data: &[u8],
    ) -> Vec<u8> {
        let block = channels * bits.div_ceil(8);
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&(if extensible { 0xFFFEu16 } else { tag }).to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * block as u32).to_le_bytes());
        fmt.extend_from_slice(&block.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        if extensible {
            fmt.extend_from_slice(&22u16.to_le_bytes());
            fmt.extend_from_slice(&bits.to_le_bytes());
            fmt.extend_from_slice(&0u32.to_le_bytes());
            fmt.extend_from_slice(&tag.to_le_bytes());
            fmt.extend_from_slice(&[0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71]);
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(b"WAVE");
        // An odd-sized junk chunk exercises the pad byte.
        out.extend_from_slice(b"junk");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&[1, 2, 3, 0]);
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        out.extend_from_slice(&fmt);
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
        let riff = (out.len() - 8) as u32;
        out[4..8].copy_from_slice(&riff.to_le_bytes());
        out
    }

    #[test]
    fn wav_decodes_every_supported_layout() {
        // Two stereo frames: (min, max-ish) then (0, −half).
        let pcm16: Vec<u8> = [-32768i16, 32767, 0, -16384]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = decode_wav(&wav_bytes(1, 2, 16000, 16, false, &pcm16)).unwrap();
        assert_eq!(w.sample_rate, 16000);
        assert_eq!(
            w.channels,
            vec![vec![-1.0, 0.0], vec![32767.0 / 32768.0, -0.5]]
        );

        let u8d = [0u8, 128, 255];
        let w = decode_wav(&wav_bytes(1, 1, 8000, 8, false, &u8d)).unwrap();
        assert_eq!(w.channels[0], vec![-1.0, 0.0, 127.0 / 128.0]);

        let s24: Vec<u8> = [-8_388_608i32, 4_194_304, -1]
            .iter()
            .flat_map(|v| v.to_le_bytes()[..3].to_vec())
            .collect();
        for ext in [false, true] {
            let w = decode_wav(&wav_bytes(1, 1, 44100, 24, ext, &s24)).unwrap();
            assert_eq!(w.channels[0], vec![-1.0, 0.5, -1.0 / 8_388_608.0]);
        }

        let s32: Vec<u8> = [i32::MIN, 1 << 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w = decode_wav(&wav_bytes(1, 1, 48000, 32, false, &s32)).unwrap();
        assert_eq!(w.channels[0], vec![-1.0, 0.5]);

        let f32d: Vec<u8> = [0.25f32, -0.75]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        for ext in [false, true] {
            let w = decode_wav(&wav_bytes(3, 1, 22050, 32, ext, &f32d)).unwrap();
            assert_eq!(w.channels[0], vec![0.25, -0.75]);
        }
        let f64d: Vec<u8> = [0.125f64].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            decode_wav(&wav_bytes(3, 1, 24000, 64, false, &f64d))
                .unwrap()
                .channels[0],
            vec![0.125]
        );

        // A-law is refused, not misread.
        assert!(decode_wav(&wav_bytes(6, 1, 8000, 8, false, &[0])).is_err());
        assert!(decode_wav(b"OggS....").is_err());
    }

    #[test]
    fn wav_data_size_past_the_end_reads_to_the_end() {
        let pcm16: Vec<u8> = [1000i16, -1000, 5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut b = wav_bytes(1, 1, 24000, 16, false, &pcm16);
        let n = b.len();
        // data size field sits 4 + data bytes before the end.
        b[n - pcm16.len() - 4..n - pcm16.len()].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(decode_wav(&b).unwrap().channels[0].len(), 3);
    }

    #[test]
    fn resample_kernel_widths_match_torchaudio() {
        // (orig, new) → width, from torchaudio's formula.
        for (o, n, w) in [
            (2usize, 3usize, 7usize),
            (147, 80, 12),
            (147, 160, 7),
            (2, 1, 13),
        ] {
            let (k, width) = sinc_resample_kernel(o, n);
            assert_eq!(width, w, "{o}->{n}");
            assert_eq!(k.len(), n * (2 * w + o));
        }
    }

    #[test]
    fn resample_lengths_follow_torchaudio_rounding() {
        assert_eq!(resampled_len(16000, 16000, 24000), 24000);
        assert_eq!(resampled_len(16001, 16000, 24000), 24002);
        assert_eq!(resampled_len(44100, 44100, 24000), 24000);
        assert_eq!(resample_sinc(&vec![0.0; 16001], 16000, 24000).len(), 24002);
        // 80·1837421/147 = 999957 + 1/147: float32 (ulp 1/16 there) drops
        // the fraction, so torchaudio keeps one sample fewer than the exact
        // ceiling.
        let n = 1_837_421usize;
        assert_eq!((80 * n) % 147, 1);
        assert_eq!((80 * n).div_ceil(147), 999_958);
        assert_eq!(resampled_len(n, 44100, 24000), 999_957);
        // DC passes through the interior at the filter's unit gain × rolloff.
        let y = resample_sinc(&vec![1.0; 4000], 16000, 24000);
        assert!((y[3000] - 1.0).abs() < 1e-2, "{}", y[3000]);
    }

    #[test]
    fn fft_matches_a_direct_dft() {
        let n = 960;
        let x: Vec<f64> = (0..n)
            .map(|i| ((i * 7919) % 97) as f64 / 97.0 - 0.5)
            .collect();
        let mut out = vec![(0.0, 0.0); n];
        Fft::new(n).forward_real(&x, &mut out);
        for k in [0usize, 1, 37, 240, 480, 959] {
            let (mut re, mut im) = (0.0, 0.0);
            for (t, v) in x.iter().enumerate() {
                let a = -2.0 * std::f64::consts::PI * (k * t % n) as f64 / n as f64;
                re += v * a.cos();
                im += v * a.sin();
            }
            assert!(
                (re - out[k].0).abs() < 1e-9 && (im - out[k].1).abs() < 1e-9,
                "bin {k}"
            );
        }
    }

    #[test]
    fn hann_fixups_are_single_ulp_corrections() {
        let step = (2.0 * std::f64::consts::PI / N_FFT as f64) as f32;
        // Each fixup is torch's cos landing one ulp away from the correctly
        // rounded one (the window value itself can move by many ulps near
        // zero, where `0.5 − 0.5·c` cancels).
        for (i, bits) in TORCH_HANN_960_FIXUPS {
            let c = ((i as f32 * step) as f64).cos() as f32;
            let win = |c: f32| (c * -0.5f32 + 0.5f32).to_bits();
            assert_ne!(win(c), bits, "index {i} needs no fixup");
            let up = f32::from_bits(c.to_bits() + 1);
            let dn = f32::from_bits(c.to_bits() - 1);
            assert!(
                win(up) == bits || win(dn) == bits,
                "index {i}: not a one-ulp cos difference"
            );
        }
        let w = torch_hann_960();
        assert_eq!(w[0], 0.0);
        assert_eq!(w[480], 1.0);
    }

    #[test]
    fn mel_shape_and_short_input() {
        assert!(log_mel(&vec![0.0; 480], None).is_err());
        let (mel, m) = log_mel(&vec![0.0; 481], None).unwrap();
        assert_eq!((m, mel.len()), (3, 3 * N_MELS));
        // Silence floors at ln(1e-7).
        assert!(
            mel.iter()
                .all(|&v| (v - (1e-7f64).ln() as f32).abs() < 1e-6)
        );
        assert_eq!(mel_frames(120_000), 501);
        // Filterbank: every band is non-empty (torchaudio warns otherwise).
        let fb = mel_filterbank(481, 128, 24000);
        for m in 0..128 {
            assert!((0..481).any(|k| fb[k * 128 + m] > 0.0), "band {m}");
        }
    }

    #[test]
    fn token_counts_and_segments() {
        // Spec §2.5 worked examples.
        assert_eq!(audio_token_count(501, 4), 32);
        assert_eq!(codes_for_mel(501), 126);
        assert_eq!(codes_for_mel(6501), 1626);
        assert_eq!(audio_token_count(6501, 4), 407);
        assert_eq!(segment_lengths(6501), vec![6000, 501]);
        // G10.2: the placeholder count equals the encoder's group count for
        // every length from 1 s to 65 s at 24 kHz (and beyond a segment edge).
        for n in (24_000..=65 * 24_000).step_by(240 * 7) {
            let m = mel_frames(n);
            assert_eq!(
                codes_for_mel(m).div_ceil(4),
                audio_token_count(m, 4),
                "M={m}"
            );
        }
        for m in 5990..6020 {
            assert_eq!(
                codes_for_mel(m).div_ceil(4),
                audio_token_count(m, 4),
                "M={m}"
            );
        }
    }

    #[test]
    fn placeholder_expansion() {
        let ids = [
            1,
            AUDIO_START_ID,
            AUDIO_PAD_ID,
            AUDIO_END_ID,
            2,
            AUDIO_START_ID,
            AUDIO_PAD_ID,
            AUDIO_END_ID,
        ];
        let out = expand_audio_placeholders(&ids, &[2, 3]).unwrap();
        assert_eq!(
            out,
            vec![
                1,
                AUDIO_START_ID,
                AUDIO_PAD_ID,
                AUDIO_PAD_ID,
                AUDIO_END_ID,
                2,
                AUDIO_START_ID,
                AUDIO_PAD_ID,
                AUDIO_PAD_ID,
                AUDIO_PAD_ID,
                AUDIO_END_ID
            ]
        );
        assert!(expand_audio_placeholders(&ids, &[2]).is_err());
        assert!(expand_audio_placeholders(&ids, &[2, 3, 4]).is_err());
        assert!(expand_audio_placeholders(&[AUDIO_PAD_ID], &[]).is_err());
    }

    #[test]
    fn gelu_and_bf16_rounding() {
        // torch: F.gelu(tensor([-3., -0.5, 0.5, 3.])) (erf form).
        let want = [-0.004_049_71f32, -0.154_268_5, 0.345_731_5, 2.995_950_3];
        for (x, w) in [-3.0f32, -0.5, 0.5, 3.0].iter().zip(want) {
            assert!((gelu(*x) - w).abs() < 2e-6, "{x}: {} vs {w}", gelu(*x));
        }
        assert_eq!(bf16_round(1.0), 1.0);
        assert_eq!(bf16_round(f32::from_bits(0x3F80_8000)), 1.0); // tie → even
        assert_eq!(
            bf16_round(f32::from_bits(0x3F81_8000)),
            f32::from_bits(0x3F82_0000)
        );
    }
}
