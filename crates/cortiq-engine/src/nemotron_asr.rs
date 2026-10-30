//! NVIDIA Nemotron 3.5 Transcribe (30B-A4B) on CMF tensors: the audio
//! perception module and the transcription loop around the Nemotron-H
//! decoder (`crate::nemotron_h`).
//!
//! Audio path, as the Transformers reference (`Nemotron3_5TranscribeAudioModel`):
//!
//! 1. log-mel front end: pre-emphasis 0.97, centered 512-point STFT with the
//!    checkpoint's 400-tap window, power spectrum, the checkpoint's 128-bin
//!    filter bank, `ln(x + 2⁻²⁴)`; frames past the audio are zero and the
//!    spectrogram is padded to a multiple of the 8× subsampling;
//! 2. ASR encoder (transformer, 1280 wide, 32 layers, per-head LayerNorm on
//!    q/k, RoPE on half of each head) over per-feature normalized frames,
//!    in chunks of 500 encoder frames with 50 frames of context each side;
//! 3. Sortformer diarization (transformer 512 wide, 31 layers) over the raw
//!    frames with the arrival-order speaker cache and FIFO of the offline
//!    mode; its thresholded speaker activities, layer-normed, select a fixed
//!    sinusoidal speaker embedding added to the layer-normed ASR states;
//! 4. a linear projector to the language model width.
//!
//! The audio module computes in f32 (as the reference keeps it in fp32);
//! the projected embeddings are rounded to bf16 before they replace the
//! `<|audio|>` placeholders, as the reference scatters them.

use crate::fcd_ops::gemm_nt;
use crate::nemotron_h::{NemotronH, SendPtr, dot, par_rows, tensor_f32};
use crate::pool::Pool;
use crate::tokenizer::Tokenizer;
use cortiq_core::CmfModel;
use rustfft::{FftPlanner, num_complex::Complex};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub const ARCH_NAME: &str = "nemotron3_5_transcribe";

const SR: usize = 16_000;
const HOP: usize = 160;
const NFFT: usize = 512;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24

struct Ln {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Ln {
    fn load(m: &CmfModel, p: &str) -> Result<Self, String> {
        Ok(Self {
            w: tensor_f32(m, &format!("{p}.weight"))?,
            b: tensor_f32(m, &format!("{p}.bias"))?,
        })
    }
    fn apply(&self, x: &mut [f32]) {
        let n = x.len();
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        for i in 0..n {
            x[i] = (x[i] - mean) * inv * self.w[i] + self.b[i];
        }
    }
    fn apply_rows(&self, x: &mut [f32]) {
        for r in x.chunks_exact_mut(self.w.len()) {
            self.apply(r);
        }
    }
}

/// A dense f32 linear layer (`[out, in]` weights, optional bias).
struct Lin {
    w: Vec<f32>,
    b: Option<Vec<f32>>,
    out: usize,
    inp: usize,
}

impl Lin {
    fn load(m: &CmfModel, p: &str, bias: bool) -> Result<Self, String> {
        let name = format!("{p}.weight");
        let e = m
            .tensor(&name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if e.shape.len() != 2 {
            return Err(format!("{name}: expected 2-D, got {:?}", e.shape));
        }
        Ok(Self {
            w: tensor_f32(m, &name)?,
            b: if bias {
                Some(tensor_f32(m, &format!("{p}.bias"))?)
            } else {
                None
            },
            out: e.shape[0],
            inp: e.shape[1],
        })
    }
    fn apply(&self, x: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        let mut y = vec![0.0f32; n * self.out];
        gemm_nt(x, &self.w, &mut y, n, self.inp, self.out, pool);
        if let Some(b) = &self.b {
            for r in y.chunks_exact_mut(self.out) {
                for (v, bb) in r.iter_mut().zip(b) {
                    *v += bb;
                }
            }
        }
        y
    }
}

struct EncLayer {
    n1: Ln,
    qkv: Lin,
    qn: Option<Ln>,
    kn: Option<Ln>,
    out: Lin,
    n2: Ln,
    fc1: Lin,
    fc2: Lin,
}

/// NeMo `TransformerEncoder` with feature-stacking subsampling, pre-norm
/// blocks, GELU MLP and RoPE on the first `rot` dims of every head.
struct Encoder {
    proj: Lin,
    embed_norm: Ln,
    layers: Vec<EncLayer>,
    final_norm: Ln,
    d: usize,
    heads: usize,
    hd: usize,
    rot: usize,
    stack: usize,
    /// The same tower on the wgpu device (bf16 weights, f32 arithmetic).
    #[cfg(feature = "gpu")]
    dev: Option<crate::gpu_wgpu::nemotron_audio::AudioEncoder>,
}

impl Encoder {
    #[cfg(feature = "gpu")]
    fn build_dev(
        &self,
        m: &CmfModel,
        p: &str,
    ) -> Option<crate::gpu_wgpu::nemotron_audio::AudioEncoder> {
        use crate::gpu_wgpu::nemotron_audio::{AeDesc, AeLayerDesc, AudioEncoder};
        if std::env::var("CMF_NEMO_AUDIO_GPU").as_deref() == Ok("0") || !crate::gpu::wgpu_active() {
            return None;
        }
        let bf16 = |name: String| -> Option<&[u8]> {
            let e = m.tensor(&name)?;
            (e.dtype == cortiq_core::TensorDtype::Bf16).then(|| m.entry_bytes(e))
        };
        let wb = |ln: &Ln| -> Vec<f32> { ln.w.iter().chain(&ln.b).copied().collect() };
        let mut layers = Vec::with_capacity(self.layers.len());
        for (i, l) in self.layers.iter().enumerate() {
            let q = format!("{p}.layers.{i}");
            layers.push(AeLayerDesc {
                ln1: wb(&l.n1),
                qkv: bf16(format!("{q}.attn.w_qkv.weight"))?,
                qn: l.qn.as_ref().map(wb),
                kn: l.kn.as_ref().map(wb),
                out: bf16(format!("{q}.attn.out_proj.weight"))?,
                out_b: l.out.b.clone()?,
                ln2: wb(&l.n2),
                fc1: bf16(format!("{q}.ffn.net.0.weight"))?,
                fc1_b: l.fc1.b.clone()?,
                fc2: bf16(format!("{q}.ffn.net.3.weight"))?,
                fc2_b: l.fc2.b.clone()?,
            });
        }
        AudioEncoder::build(&AeDesc {
            d: self.d,
            heads: self.heads,
            rot: self.rot,
            inter: self.layers[0].fc1.out,
            stack_in: self.proj.inp,
            proj: bf16(format!("{p}.pre_encode.proj.weight"))?,
            embed_norm: wb(&self.embed_norm),
            final_norm: wb(&self.final_norm),
            layers,
        })
    }

    fn load(
        m: &CmfModel,
        p: &str,
        heads: usize,
        rot_frac: f64,
        stack: usize,
    ) -> Result<Self, String> {
        let proj = Lin::load(m, &format!("{p}.pre_encode.proj"), false)?;
        let d = proj.out;
        let hd = d / heads;
        let mut layers = Vec::new();
        for i in 0.. {
            let l = format!("{p}.layers.{i}");
            if m.tensor(&format!("{l}.norm1.weight")).is_none() {
                break;
            }
            let qk = m.tensor(&format!("{l}.attn.q_norm.weight")).is_some();
            layers.push(EncLayer {
                n1: Ln::load(m, &format!("{l}.norm1"))?,
                qkv: Lin::load(m, &format!("{l}.attn.w_qkv"), false)?,
                qn: if qk {
                    Some(Ln::load(m, &format!("{l}.attn.q_norm"))?)
                } else {
                    None
                },
                kn: if qk {
                    Some(Ln::load(m, &format!("{l}.attn.k_norm"))?)
                } else {
                    None
                },
                out: Lin::load(m, &format!("{l}.attn.out_proj"), true)?,
                n2: Ln::load(m, &format!("{l}.norm2"))?,
                fc1: Lin::load(m, &format!("{l}.ffn.net.0"), true)?,
                fc2: Lin::load(m, &format!("{l}.ffn.net.3"), true)?,
            });
        }
        if layers.is_empty() {
            return Err(format!("{p}: no encoder layers"));
        }
        #[allow(unused_mut)]
        let mut enc = Self {
            embed_norm: Ln::load(m, &format!("{p}.embed_norm"))?,
            final_norm: Ln::load(m, &format!("{p}.final_norm"))?,
            proj,
            layers,
            d,
            heads,
            hd,
            rot: (hd as f64 * rot_frac) as usize,
            stack,
            #[cfg(feature = "gpu")]
            dev: None,
        };
        #[cfg(feature = "gpu")]
        {
            enc.dev = enc.build_dev(m, p);
            // the device holds the tower: release the host f32 copy of its
            // blocks (2.9 GB for both towers — what decides whether a 24 GB
            // Mac fits); the small stacking projection stays (the
            // diarization embeds every frame on the host)
            if enc.dev.is_some() && std::env::var("CMF_NEMO_KEEP_HOST").as_deref() != Ok("1") {
                for l in &mut enc.layers {
                    for lin in [&mut l.qkv, &mut l.out, &mut l.fc1, &mut l.fc2] {
                        lin.w = Vec::new();
                    }
                }
            }
        }
        Ok(enc)
    }

    /// Stacked frames `[n·stack · n_mel]` → encoder states `[n · d]`.
    fn encode_frames(
        &self,
        feats: &[f32],
        n: usize,
        valid: &[bool],
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(dev) = &self.dev
            && let Some(h) = dev.run(feats, n, false, valid)
        {
            return h;
        }
        assert!(
            !self.layers[0].qkv.w.is_empty(),
            "audio encoder: the device run failed and the host weights were released (CMF_NEMO_KEEP_HOST=1 keeps them)"
        );
        let x = self.embed(feats, n * self.stack, pool);
        self.run(&x, n, valid, pool)
    }

    /// Embedded rows `[n · d]` → encoder states `[n · d]`.
    fn encode_rows(&self, x: &[f32], n: usize, valid: &[bool], pool: Option<&Pool>) -> Vec<f32> {
        #[cfg(feature = "gpu")]
        if let Some(dev) = &self.dev
            && let Some(h) = dev.run(x, n, true, valid)
        {
            return h;
        }
        assert!(
            !self.layers[0].qkv.w.is_empty(),
            "audio encoder: the device run failed and the host weights were released (CMF_NEMO_KEEP_HOST=1 keeps them)"
        );
        self.run(x, n, valid, pool)
    }

    /// Feature stacking: `stack` consecutive frames → one projected row.
    fn embed(&self, feats: &[f32], frames: usize, pool: Option<&Pool>) -> Vec<f32> {
        debug_assert_eq!(frames % self.stack, 0);
        self.proj.apply(feats, frames / self.stack, pool)
    }

    /// The transformer over `n` embedded rows; `valid[t]` masks keys.
    fn run(&self, x: &[f32], n: usize, valid: &[bool], pool: Option<&Pool>) -> Vec<f32> {
        let d = self.d;
        let mut h = x.to_vec();
        self.embed_norm.apply_rows(&mut h);
        let (cos, sin) = rope_table(n, self.rot);
        for l in &self.layers {
            let mut a = h.clone();
            l.n1.apply_rows(&mut a);
            let mut qkv = l.qkv.apply(&a, n, pool);
            for t in 0..n {
                let row = &mut qkv[t * 3 * d..(t + 1) * 3 * d];
                for hh in 0..self.heads {
                    let q = &mut row[hh * self.hd..(hh + 1) * self.hd];
                    if let Some(qn) = &l.qn {
                        qn.apply(q);
                    }
                    rope(
                        q,
                        &cos[t * self.rot / 2..],
                        &sin[t * self.rot / 2..],
                        self.rot,
                    );
                }
                for hh in 0..self.heads {
                    let k = &mut row[d + hh * self.hd..d + (hh + 1) * self.hd];
                    if let Some(kn) = &l.kn {
                        kn.apply(k);
                    }
                    rope(
                        k,
                        &cos[t * self.rot / 2..],
                        &sin[t * self.rot / 2..],
                        self.rot,
                    );
                }
            }
            let o = bidir_attention(&qkv, n, self.heads, self.hd, valid, pool);
            let o = l.out.apply(&o, n, pool);
            for (v, w) in h.iter_mut().zip(&o) {
                *v += w;
            }
            let mut a = h.clone();
            l.n2.apply_rows(&mut a);
            let mut f = l.fc1.apply(&a, n, pool);
            for v in f.iter_mut() {
                *v = gelu_erf(*v);
            }
            let f = l.fc2.apply(&f, n, pool);
            for (v, w) in h.iter_mut().zip(&f) {
                *v += w;
            }
        }
        self.final_norm.apply_rows(&mut h);
        h
    }
}

/// RoPE angles `[n · rot/2]` (positions from 0, θ = 10000).
fn rope_table(n: usize, rot: usize) -> (Vec<f32>, Vec<f32>) {
    let half = rot / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / rot as f32))
        .collect();
    let mut cos = vec![0.0; n * half];
    let mut sin = vec![0.0; n * half];
    for t in 0..n {
        for i in 0..half {
            let a = t as f32 * inv[i];
            cos[t * half + i] = a.cos();
            sin[t * half + i] = a.sin();
        }
    }
    (cos, sin)
}

/// NeoX rotate-half over the first `rot` dims.
fn rope(x: &mut [f32], cos: &[f32], sin: &[f32], rot: usize) {
    let half = rot / 2;
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        x[i] = a * cos[i] - b * sin[i];
        x[i + half] = b * cos[i] + a * sin[i];
    }
}

/// Bidirectional attention over the packed `[q | k | v]` rows; keys with
/// `valid[t] == false` are masked. Output `[n · heads·hd]`.
fn bidir_attention(
    qkv: &[f32],
    n: usize,
    heads: usize,
    hd: usize,
    valid: &[bool],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut o = vec![0.0f32; n * d];
    let o_addr = SendPtr(o.as_mut_ptr());
    let run = |r0: usize, r1: usize| {
        let o = o_addr;
        let mut sc = vec![0.0f32; n];
        for r in r0..r1 {
            let (hh, t) = (r / n, r % n);
            let q = &qkv[t * 3 * d + hh * hd..t * 3 * d + (hh + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..n {
                if !valid[j] {
                    sc[j] = f32::NEG_INFINITY;
                    continue;
                }
                let k = &qkv[j * 3 * d + d + hh * hd..j * 3 * d + d + (hh + 1) * hd];
                let s = dot(q, k) * scale;
                sc[j] = s;
                mx = mx.max(s);
            }
            let mut den = 0.0f32;
            for s in sc.iter_mut() {
                *s = if s.is_finite() { (*s - mx).exp() } else { 0.0 };
                den += *s;
            }
            let out = unsafe { std::slice::from_raw_parts_mut(o.0.add(t * d + hh * hd), hd) };
            out.fill(0.0);
            for j in 0..n {
                if sc[j] == 0.0 {
                    continue;
                }
                let p = sc[j] / den;
                let v = &qkv[j * 3 * d + 2 * d + hh * hd..j * 3 * d + 2 * d + (hh + 1) * hd];
                for (a, &b) in out.iter_mut().zip(v) {
                    *a += p * b;
                }
            }
        }
    };
    par_rows(pool, heads * n, &run);
    o
}

fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + crate::inference::erf_f32(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Offline-mode settings of the Sortformer speaker cache.
#[derive(Clone, Debug)]
struct SpkCfg {
    chunk: usize,
    left: usize,
    right: usize,
    fifo_len: usize,
    update_period: usize,
    cache_len: usize,
    n_spk: usize,
    sil_frames: usize,
    pred_thr: f32,
    latest_boost: f32,
    min_pos: usize,
    n_strong: usize,
    n_weak: usize,
    activity_thr: f32,
}

impl SpkCfg {
    fn from_config(audio: &serde_json::Value) -> Self {
        let dz = &audio["diarization_config"];
        let sc = &dz["streaming_config"];
        let u = |v: &serde_json::Value, k: &str, d: u64| {
            v.get(k).and_then(|x| x.as_u64()).unwrap_or(d) as usize
        };
        let f =
            |v: &serde_json::Value, k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let n_spk = u(sc, "num_speakers", 8);
        let cache_len = u(sc, "speaker_cache_length", 188);
        let sil = u(sc, "speaker_cache_silence_frames_per_speaker", 1);
        let budget = (cache_len / n_spk - sil) as f64;
        Self {
            chunk: u(dz, "chunk_length", 500),
            left: u(dz, "chunk_left_context", 50),
            right: u(dz, "chunk_right_context", 50),
            fifo_len: u(dz, "fifo_length", 40),
            update_period: u(dz, "speaker_cache_update_period", 300),
            cache_len,
            n_spk,
            sil_frames: sil,
            pred_thr: f(sc, "prediction_score_threshold", 0.25) as f32,
            latest_boost: f(sc, "latest_frames_score_boost", 0.05) as f32,
            min_pos: (budget * f(sc, "min_positive_scores_rate", 0.5)).floor() as usize,
            n_strong: (budget * f(sc, "strong_boost_rate", 0.75)).floor() as usize,
            n_weak: (budget * f(sc, "weak_boost_rate", 1.5)).floor() as usize,
            activity_thr: audio
                .get("speaker_activity_threshold")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.5) as f32,
        }
    }
}

/// The arrival-order speaker cache and FIFO of the Sortformer offline mode
/// (`Nemotron3DiarizationSpeakerCache`), batch 1.
struct SpkCache {
    embeds: Vec<f32>,
    probs: Vec<f32>,
    fifo: Vec<f32>,
    compressed: bool,
}

impl SpkCache {
    fn n_cache(&self, d: usize) -> usize {
        self.embeds.len() / d
    }
    fn n_fifo(&self, d: usize) -> usize {
        self.fifo.len() / d
    }

    #[allow(clippy::too_many_arguments)]
    fn update(
        &mut self,
        c: &SpkCfg,
        d: usize,
        input: &[f32],
        logits: &[f32],
        mask: &[bool],
        silence: &[f32],
        n_chunk: usize,
        lookback: usize,
    ) {
        let ns = c.n_spk;
        let n_in = input.len() / d;
        // speaker probabilities at the encoder rate, zero on padding
        let sub = logits.len() / (n_in * ns);
        let mut probs = vec![0.0f32; n_in * ns];
        for t in 0..n_in {
            if !mask[t] {
                continue;
            }
            for s in 0..ns {
                let mut acc = 0.0f32;
                for j in 0..sub {
                    acc += sigmoid(logits[(t * sub + j) * ns + s]);
                }
                probs[t * ns + s] = acc / sub as f32;
            }
        }
        let (nc, nf) = (self.n_cache(d), self.n_fifo(d));
        let start = nc + nf + lookback;
        let mut fifo = self.fifo.clone();
        fifo.extend_from_slice(&input[start * d..(start + n_chunk) * d]);
        let len = fifo.len() / d;
        let popped = if len <= c.fifo_len {
            0
        } else {
            c.update_period.max(len - c.fifo_len).min(len)
        };
        if popped > 0 {
            let mut fifo_probs = probs[nc * ns..(nc + nf) * ns].to_vec();
            fifo_probs.extend_from_slice(&probs[start * ns..(start + n_chunk) * ns]);
            let mut cache_probs = if self.compressed {
                self.probs.clone()
            } else {
                probs[..nc * ns].to_vec()
            };
            let mut cache_embeds = self.embeds.clone();
            cache_embeds.extend_from_slice(&fifo[..popped * d]);
            cache_probs.extend_from_slice(&fifo_probs[..popped * ns]);
            fifo.drain(..popped * d);
            if cache_embeds.len() / d > c.cache_len {
                let (e, p) = compress(c, d, &cache_embeds, &cache_probs, silence);
                cache_embeds = e;
                cache_probs = p;
                self.compressed = true;
            }
            self.embeds = cache_embeds;
            self.probs = cache_probs;
        }
        self.fifo = fifo;
    }
}

/// `_get_frame_scores`: per speaker log-odds against the other speakers,
/// −∞ for non-speech frames and, where a speaker has enough positive
/// frames, for its non-positive speech frames.
fn frame_scores(c: &SpkCfg, probs: &[f32], n: usize) -> Vec<f32> {
    let ns = c.n_spk;
    let thr = c.pred_thr;
    let mut s = vec![0.0f32; n * ns];
    for t in 0..n {
        let p = &probs[t * ns..(t + 1) * ns];
        let lc: Vec<f32> = p.iter().map(|&x| (1.0 - x).max(thr).ln()).collect();
        let lcs: f32 = lc.iter().sum();
        for k in 0..ns {
            let v = p[k].max(thr).ln() - lc[k] + lcs - 0.5f32.ln();
            s[t * ns + k] = if p[k] > 0.5 { v } else { f32::NEG_INFINITY };
        }
    }
    for k in 0..ns {
        let positive = (0..n).filter(|&t| s[t * ns + k] > 0.0).count();
        if positive >= c.min_pos {
            for t in 0..n {
                let v = s[t * ns + k];
                if probs[t * ns + k] > 0.5 && v <= 0.0 {
                    s[t * ns + k] = f32::NEG_INFINITY;
                }
            }
        }
    }
    s
}

/// Indices of the `k` largest values (torch.topk; ties toward lower index).
fn topk_idx(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| {
        v[b].partial_cmp(&v[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}

/// `_compress`: keep `cache_len` frames grouped by speaker, in arrival
/// order within a speaker, with `sil_frames` silence slots per speaker.
fn compress(
    c: &SpkCfg,
    d: usize,
    embeds: &[f32],
    probs: &[f32],
    silence: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let ns = c.n_spk;
    let n = embeds.len() / d;
    let mut s = frame_scores(c, probs, n);
    for t in c.cache_len..n {
        for k in 0..ns {
            s[t * ns + k] += c.latest_boost;
        }
    }
    for (count, boost) in [(c.n_strong, -2.0 * 0.5f32.ln()), (c.n_weak, -(0.5f32.ln()))] {
        for k in 0..ns {
            let col: Vec<f32> = (0..n).map(|t| s[t * ns + k]).collect();
            for t in topk_idx(&col, count) {
                s[t * ns + k] += boost;
            }
        }
    }
    let scored = n + c.sil_frames;
    // speaker-major flat scores; silence rows score +∞
    let mut flat = vec![f32::INFINITY; ns * scored];
    for k in 0..ns {
        for t in 0..n {
            flat[k * scored + t] = s[t * ns + k];
        }
    }
    let sentinel = scored * ns;
    let mut picked: Vec<usize> = topk_idx(&flat, c.cache_len)
        .into_iter()
        .map(|i| {
            if flat[i] == f32::NEG_INFINITY {
                sentinel
            } else {
                i
            }
        })
        .collect();
    picked.sort_unstable();
    let mut out_e = Vec::with_capacity(c.cache_len * d);
    let mut out_p = Vec::with_capacity(c.cache_len * ns);
    for i in picked {
        let f = if i == sentinel {
            n
        } else {
            (i % scored).min(n)
        };
        if f < n {
            out_e.extend_from_slice(&embeds[f * d..(f + 1) * d]);
            out_p.extend_from_slice(&probs[f * ns..(f + 1) * ns]);
        } else {
            out_e.extend_from_slice(silence);
            out_p.extend(std::iter::repeat_n(0.0, ns));
        }
    }
    (out_e, out_p)
}

/// The perception module: mel front end, ASR encoder, diarization, projector.
pub struct Audio {
    fb: Vec<f32>,
    window: Vec<f32>,
    n_mel: usize,
    asr: Encoder,
    diar: Encoder,
    diar_proj: Lin,
    up_w: Vec<f32>,
    up_b: Vec<f32>,
    cls_dense: Lin,
    cls_out: Lin,
    silence: Vec<f32>,
    asr_norm: Ln,
    spk_norm: Ln,
    spk_table: Vec<f32>,
    proj: Lin,
    spk: SpkCfg,
}

/// What the audio module hands the decoder (and the parity tools).
pub struct AudioOut {
    /// log-mel `[frames · n_mel]`, frames padded to a multiple of 8
    pub feats: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
    /// ASR encoder states before `asr_norm` `[T · 1280]`
    pub asr_hidden: Vec<f32>,
    /// diarization logits `[frames · speakers]`
    pub speaker_logits: Vec<f32>,
    /// projected embeddings of the valid audio tokens `[tokens · hidden]`
    pub embeds: Vec<f32>,
    pub tokens: usize,
}

impl Audio {
    pub fn load(m: &Arc<CmfModel>, audio_cfg: &serde_json::Value) -> Result<Self, String> {
        let fb = tensor_f32(m, "perception.preprocessor.featurizer.fb")?;
        let window = tensor_f32(m, "perception.preprocessor.featurizer.window")?;
        let n_mel = fb.len() / (NFFT / 2 + 1);
        let asr_cfg = &audio_cfg["asr_config"];
        let dz = &audio_cfg["diarization_config"]["audio_config"];
        let heads = |v: &serde_json::Value, d| {
            v.get("num_attention_heads")
                .and_then(|x| x.as_u64())
                .unwrap_or(d) as usize
        };
        let rot = |v: &serde_json::Value, d| {
            v.get("rope_parameters")
                .and_then(|r| r.get("partial_rotary_factor"))
                .and_then(|x| x.as_f64())
                .unwrap_or(d)
        };
        let stack = asr_cfg
            .get("subsampling_factor")
            .and_then(|x| x.as_u64())
            .unwrap_or(8) as usize;
        let asr = Encoder::load(
            m,
            "perception.encoder.asr_encoder",
            heads(asr_cfg, 10),
            rot(asr_cfg, 0.5),
            stack,
        )?;
        let diar = Encoder::load(
            m,
            "perception.encoder.diarization_model.encoder",
            heads(dz, 8),
            rot(dz, 1.0),
            stack,
        )?;
        let sm = "perception.encoder.diarization_model.sortformer_modules";
        let spk = SpkCfg::from_config(audio_cfg);
        let d = asr.d;
        // fixed sinusoidal speaker table [n_spk · d]
        let mut spk_table = vec![0.0f32; spk.n_spk * d];
        for s in 0..spk.n_spk {
            for i in 0..d / 2 {
                let inv = ((2 * i) as f32 * -(10000f32.ln() / d as f32)).exp();
                spk_table[s * d + 2 * i] = (s as f32 * inv).sin();
                spk_table[s * d + 2 * i + 1] = (s as f32 * inv).cos();
            }
        }
        Ok(Self {
            fb,
            window,
            n_mel,
            asr,
            diar,
            diar_proj: Lin::load(m, &format!("{sm}.encoder_proj"), true)?,
            // conv taps [cout, cin, 3] → GEMM layout [cout, 3·cin]
            up_w: {
                let name = format!("{sm}.subpixel_upsample.weight");
                let w = tensor_f32(m, &name)?;
                let shape = &m.tensor(&name).ok_or("upsampler")?.shape;
                if shape.len() != 3 || shape[2] != 3 {
                    return Err(format!("{name}: shape {shape:?}, expected [cout, cin, 3]"));
                }
                let (cout, cin) = (shape[0], shape[1]);
                let mut g = vec![0.0f32; w.len()];
                for o in 0..cout {
                    for c in 0..cin {
                        for j in 0..3 {
                            g[o * 3 * cin + j * cin + c] = w[(o * cin + c) * 3 + j];
                        }
                    }
                }
                g
            },
            up_b: tensor_f32(m, &format!("{sm}.subpixel_upsample.bias"))?,
            cls_dense: Lin::load(m, &format!("{sm}.first_hidden_to_hidden"), true)?,
            cls_out: Lin::load(m, &format!("{sm}.single_hidden_to_spks"), true)?,
            silence: tensor_f32(m, &format!("{sm}.learnable_sil_emb"))?,
            asr_norm: Ln::load(m, "perception.encoder.asr_norm")?,
            spk_norm: Ln::load(m, "perception.encoder.diar_norm")?,
            spk_table,
            proj: Lin::load(m, "perception.proj", true)?,
            spk,
        })
    }

    /// Log-mel features, time-major `[frames · n_mel]` padded to a
    /// multiple of the stacking factor, and the number of valid frames.
    pub fn log_mel(&self, x: &[f32]) -> (Vec<f32>, usize, usize) {
        let len = x.len();
        let mut y = vec![0.0f32; len];
        if len > 0 {
            y[0] = x[0];
            for i in 1..len {
                y[i] = x[i] - 0.97 * x[i - 1];
            }
        }
        let half = NFFT / 2;
        let mut padded = vec![0.0f32; len + NFFT];
        padded[half..half + len].copy_from_slice(&y);
        let n_frames = 1 + len / HOP;
        let valid = len / HOP;
        let frames = n_frames.div_ceil(self.asr.stack) * self.asr.stack;
        let wl = self.window.len();
        let off = (NFFT - wl) / 2;
        let bins = half + 1;
        let mut feats = vec![0.0f32; frames * self.n_mel];
        let fft = FftPlanner::<f64>::new().plan_fft_forward(NFFT);
        let mut buf = vec![Complex::new(0.0f64, 0.0); NFFT];
        let mut power = vec![0.0f32; bins];
        for t in 0..valid {
            for (i, c) in buf.iter_mut().enumerate() {
                let w = if i >= off && i < off + wl {
                    self.window[i - off]
                } else {
                    0.0
                };
                *c = Complex::new((padded[t * HOP + i] * w) as f64, 0.0);
            }
            fft.process(&mut buf);
            for k in 0..bins {
                let (re, im) = (buf[k].re as f32, buf[k].im as f32);
                let mag = (re * re + im * im).sqrt();
                power[k] = mag * mag;
            }
            let row = &mut feats[t * self.n_mel..(t + 1) * self.n_mel];
            for (mi, v) in row.iter_mut().enumerate() {
                let f = &self.fb[mi * bins..(mi + 1) * bins];
                *v = (dot(f, &power) + LOG_GUARD).ln();
            }
        }
        (feats, frames, valid)
    }

    /// Per-feature normalization over the valid frames (unbiased variance,
    /// epsilon on the standard deviation); padding frames stay zero.
    fn normalize(&self, feats: &[f32], frames: usize, valid: usize) -> Vec<f32> {
        let nm = self.n_mel;
        let mut out = vec![0.0f32; frames * nm];
        for m in 0..nm {
            let mean = (0..valid).map(|t| feats[t * nm + m]).sum::<f32>() / valid.max(1) as f32;
            let var = (0..valid)
                .map(|t| (feats[t * nm + m] - mean).powi(2))
                .sum::<f32>()
                / (valid.max(2) - 1) as f32;
            let std = var.sqrt() + 1e-5;
            for t in 0..valid {
                out[t * nm + m] = (feats[t * nm + m] - mean) / std;
            }
        }
        out
    }

    /// `_get_chunks`: (context range, chunk range within the context).
    fn chunks(&self, n: usize) -> Vec<((usize, usize), (usize, usize))> {
        let c = &self.spk;
        let mut out = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + c.chunk).min(n);
            let cs = start.saturating_sub(c.left);
            let ce = (end + c.right).min(n);
            out.push(((cs, ce), (start - cs, end - cs)));
            start += c.chunk;
        }
        out
    }

    /// Diarization logits `[frames · n_spk]` (offline mode).
    fn diarize(&self, feats: &[f32], frames: usize, valid: usize, pool: Option<&Pool>) -> Vec<f32> {
        let c = &self.spk;
        let ns = c.n_spk;
        let sub = self.diar.stack;
        let d = self.diar.d;
        let embeds = self.diar.embed(feats, frames, pool);
        let n_emb = frames / sub;
        let emask: Vec<bool> = (0..n_emb).map(|t| t * sub < valid).collect();
        let mut cache = SpkCache {
            embeds: Vec::new(),
            probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
        };
        let mut logits_all = Vec::with_capacity(frames * ns);
        let mut start = 0;
        while start < n_emb {
            let end = (start + c.chunk).min(n_emb);
            let n_chunk = end - start;
            let cs = start.saturating_sub(c.left);
            let lookback = start - cs;
            let ce = (end + c.right).min(n_emb);
            let mut input = cache.embeds.clone();
            input.extend_from_slice(&cache.fifo);
            let cached = input.len() / d;
            input.extend_from_slice(&embeds[cs * d..ce * d]);
            let n_in = input.len() / d;
            let mut mask = vec![true; cached];
            mask.extend_from_slice(&emask[cs..ce]);
            let h = self.diar.encode_rows(&input, n_in, &mask, pool);
            let h = self.diar_proj.apply(&h, n_in, pool);
            let hu = self.upsample(&h, n_in, pool);
            // classifier: dense(relu(h)) → out(relu(·))
            let mut a: Vec<f32> = hu.iter().map(|v| v.max(0.0)).collect();
            a = self.cls_dense.apply(&a, n_in * sub, pool);
            for v in a.iter_mut() {
                *v = v.max(0.0);
            }
            let logits = self.cls_out.apply(&a, n_in * sub, pool);
            cache.update(
                c,
                d,
                &input,
                &logits,
                &mask,
                &self.silence,
                n_chunk,
                lookback,
            );
            let s0 = (cached + lookback) * sub;
            logits_all.extend_from_slice(&logits[s0 * ns..(s0 + n_chunk * sub) * ns]);
            start += c.chunk;
        }
        logits_all.truncate(frames * ns);
        logits_all
    }

    /// Sub-pixel upsampler: conv1d (k3, pad 1) to `stack ×` channels, then
    /// every encoder frame becomes `stack` frames.
    fn upsample(&self, h: &[f32], n: usize, pool: Option<&Pool>) -> Vec<f32> {
        // as a GEMM: row t = [h(t−1) | h(t) | h(t+1)] (zero padded) against
        // the taps laid out `[cout, 3·cin]` (`up_w` at load)
        let cin = self.diar_proj.out;
        let cout = self.up_b.len();
        let mut x3 = vec![0.0f32; n * 3 * cin];
        for t in 0..n {
            for j in 0..3 {
                let src = t as isize + j as isize - 1;
                if src >= 0 && src < n as isize {
                    let s = src as usize;
                    x3[t * 3 * cin + j * cin..t * 3 * cin + (j + 1) * cin]
                        .copy_from_slice(&h[s * cin..(s + 1) * cin]);
                }
            }
        }
        let mut y = vec![0.0f32; n * cout];
        gemm_nt(&x3, &self.up_w, &mut y, n, 3 * cin, cout, pool);
        for r in y.chunks_exact_mut(cout) {
            for (v, b) in r.iter_mut().zip(&self.up_b) {
                *v += b;
            }
        }
        y
    }

    pub fn run(&self, samples: &[f32], pool: Option<&Pool>) -> AudioOut {
        let prof = std::env::var("CMF_NEMO_PROF").as_deref() == Ok("1");
        let t0 = Instant::now();
        let (feats, frames, valid) = self.log_mel(samples);
        let t_mel = t0.elapsed();
        let norm = self.normalize(&feats, frames, valid);
        let stack = self.asr.stack;
        let t_enc = frames / stack;
        let d = self.asr.d;
        let mut hidden = vec![0.0f32; t_enc * d];
        for ((cs, ce), (a, b)) in self.chunks(t_enc) {
            let valid_mask: Vec<bool> = (cs..ce).map(|t| t * stack < valid).collect();
            let h = self.asr.encode_frames(
                &norm[cs * stack * self.n_mel..ce * stack * self.n_mel],
                ce - cs,
                &valid_mask,
                pool,
            );
            hidden[(cs + a) * d..(cs + b) * d].copy_from_slice(&h[a * d..b * d]);
        }
        let t_asr = t0.elapsed();
        let speaker_logits = self.diarize(&feats, frames, valid, pool);
        let t_diar = t0.elapsed();
        let ns = self.spk.n_spk;
        let mut joined = hidden.clone();
        self.asr_norm.apply_rows(&mut joined);
        for t in 0..t_enc {
            let mut act = vec![0.0f32; ns];
            for (s, a) in act.iter_mut().enumerate() {
                let mut p = 0.0f32;
                for j in 0..stack {
                    p += sigmoid(speaker_logits[(t * stack + j) * ns + s]);
                }
                *a = if p / stack as f32 > self.spk.activity_thr {
                    1.0
                } else {
                    0.0
                };
            }
            self.spk_norm.apply(&mut act);
            let row = &mut joined[t * d..(t + 1) * d];
            for (s, &a) in act.iter().enumerate() {
                for (v, &e) in row.iter_mut().zip(&self.spk_table[s * d..(s + 1) * d]) {
                    *v += a * e;
                }
            }
        }
        let tokens = valid.div_ceil(stack);
        let mut embeds = self.proj.apply(&joined[..tokens * d], tokens, pool);
        for v in embeds.iter_mut() {
            *v = round_bf16(*v);
        }
        if prof {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "nemotron audio: {frames} frames, mel {:.1} ms, asr encoder {:.1} ms, diarization {:.1} ms, join+projector {:.1} ms",
                ms(t_mel),
                ms(t_asr - t_mel),
                ms(t_diar - t_asr),
                ms(t0.elapsed() - t_diar)
            );
        }
        AudioOut {
            feats,
            frames,
            valid_frames: valid,
            asr_hidden: hidden,
            speaker_logits,
            embeds,
            tokens,
        }
    }
}

fn round_bf16(x: f32) -> f32 {
    let b = x.to_bits();
    let r = b.wrapping_add(0x7fff + ((b >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// The full model: perception module + Nemotron-H decoder + tokenizer.
pub struct Transcriber {
    pub audio: Audio,
    pub llm: NemotronH,
    pub tok: Tokenizer,
    pub audio_token: u32,
    pub eos: Vec<u32>,
    pub default_prompt: String,
}

/// A transcription and its timings.
pub struct Transcript {
    pub text: String,
    pub ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub audio_s: f64,
    pub encode_s: f64,
    pub prefill_s: f64,
    pub decode_s: f64,
}

impl Transcriber {
    pub fn load(model: &Arc<CmfModel>) -> Result<Self, String> {
        if model.header.arch.arch_name != ARCH_NAME {
            return Err(format!(
                "{} is not a Nemotron 3.5 Transcribe CMF",
                model.path.display()
            ));
        }
        let prov = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("nemotron_transcribe"))
            .ok_or("CMF is missing the nemotron_transcribe provenance")?;
        let llm = NemotronH::load(model, &prov["llm_config"])?;
        let audio = Audio::load(model, &prov["audio_config"])?;
        let vocab = model
            .vocab
            .as_deref()
            .ok_or("CMF has no embedded tokenizer.json")?;
        let tok = Tokenizer::from_bytes(vocab).map_err(|e| format!("tokenizer: {e}"))?;
        let audio_token = prov
            .get("audio_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(131072) as u32;
        let mut eos: Vec<u32> = prov["generation_config"]["eos_token_id"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as u32)
                    .collect()
            })
            .unwrap_or_default();
        if let Some(id) = tok.token_to_id("<|im_end|>")
            && !eos.contains(&id)
        {
            eos.push(id);
        }
        // Build the device decode frame now (wgpu only), so the upload is
        // load time rather than the first transcription's decode time.
        let _ = llm.frame_seed(&llm.new_state());
        let default_prompt = prov["processor"]
            .get("default_transcription_prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Transcribe the following audio.")
            .to_string();
        Ok(Self {
            audio,
            llm,
            tok,
            audio_token,
            eos,
            default_prompt,
        })
    }

    /// Token ids of the chat prompt with `n_audio` placeholders, as the
    /// reference chat template renders a user turn of `prompt` + audio
    /// (empty system turn, thinking off).
    pub fn prompt_ids(&self, prompt: &str, n_audio: usize, think: bool) -> Vec<u32> {
        let head = format!("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{prompt} ");
        let tail = if think {
            "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        } else {
            "<|im_end|>\n<|im_start|>assistant\n<think></think>"
        };
        let mut ids = self.tok.encode(&head);
        ids.extend(std::iter::repeat_n(self.audio_token, n_audio));
        ids.extend(self.tok.encode(tail));
        ids
    }

    /// Input rows for `ids`: token embeddings, audio rows at the placeholders.
    pub fn input_rows(&self, ids: &[u32], audio: &[f32]) -> Result<Vec<f32>, String> {
        let hd = self.llm.g.hidden;
        let mut rows = vec![0.0f32; ids.len() * hd];
        let mut ai = 0;
        for (i, &id) in ids.iter().enumerate() {
            let dst = &mut rows[i * hd..(i + 1) * hd];
            if id == self.audio_token {
                let src = audio
                    .get(ai * hd..(ai + 1) * hd)
                    .ok_or("more audio placeholders than audio tokens")?;
                dst.copy_from_slice(src);
                ai += 1;
            } else {
                self.llm.embed_row(id, dst);
            }
        }
        if ai * hd != audio.len() {
            return Err(format!(
                "{ai} audio placeholders for {} audio tokens",
                audio.len() / hd
            ));
        }
        Ok(rows)
    }

    pub fn transcribe(
        &self,
        samples: &[f32],
        prompt: Option<&str>,
        max_new_tokens: usize,
        pool: Option<&Pool>,
        mut on_token: impl FnMut(&str),
    ) -> Result<Transcript, String> {
        let t0 = Instant::now();
        let a = self.audio.run(samples, pool);
        let encode_s = t0.elapsed().as_secs_f64();
        let ids = self.prompt_ids(prompt.unwrap_or(&self.default_prompt), a.tokens, false);
        let rows = self.input_rows(&ids, &a.embeds)?;
        // Greedy decode. On a wgpu device the decoder runs in the resident
        // frame: `FRAME_STEPS` tokens ride one submit and only their ids come
        // back. The prompt is prefilled through the frame position by
        // position (`CMF_NEMO_PREFILL=frame`) or on the host / per-op
        // batched kernels (`host`, default) and its state handed over.
        const FRAME_STEPS: usize = 16;
        let frame_prefill = std::env::var("CMF_NEMO_PREFILL").as_deref() == Ok("frame");
        let t1 = Instant::now();
        let mut st = self.llm.new_state();
        let mut ready = std::collections::VecDeque::new();
        let mut on_device = false;
        if frame_prefill && max_new_tokens > 1 {
            if let Some(first) = self.llm.frame_prefill(&rows, 1) {
                ready.extend(first);
                on_device = true;
            }
        }
        if !on_device {
            let logits = self.llm.forward(&mut st, &rows, pool);
            ready.push_back(argmax(&logits[..self.llm.g.vocab.min(logits.len())]));
            on_device = max_new_tokens > 1 && self.llm.frame_seed(&st);
        }
        let prefill_s = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let mut out = Vec::new();
        let hd = self.llm.g.hidden;
        let mut row = vec![0.0f32; hd];
        let mut printed = 0usize;
        while let Some(tok) = ready.pop_front() {
            if self.eos.contains(&tok) || out.len() >= max_new_tokens {
                break;
            }
            out.push(tok);
            let text = self.tok.decode(&out);
            if text.len() > printed && text.is_char_boundary(printed) {
                on_token(&text[printed..]);
                printed = text.len();
            }
            if !ready.is_empty() || out.len() >= max_new_tokens {
                continue;
            }
            // every ready token but the last was already consumed on the device
            self.llm.embed_row(tok, &mut row);
            if on_device {
                let k = (max_new_tokens - out.len()).min(FRAME_STEPS);
                ready.extend(
                    self.llm
                        .frame_greedy(&row, k)
                        .ok_or("device decode frame failed")?,
                );
            } else {
                let logits = self.llm.forward(&mut st, &row, pool);
                ready.push_back(argmax(&logits[..self.llm.g.vocab.min(logits.len())]));
            }
        }
        let decode_s = t2.elapsed().as_secs_f64();
        Ok(Transcript {
            text: self.tok.decode(&out).trim().to_string(),
            prompt_tokens: ids.len(),
            audio_tokens: a.tokens,
            ids: out,
            audio_s: samples.len() as f64 / SR as f64,
            encode_s,
            prefill_s,
            decode_s,
        })
    }
}

impl Transcriber {
    /// Parity dump against the reference: writes the audio-module tensors,
    /// the prompt ids and the teacher-forced logits of `forced` (the
    /// reference transcript ids) into `dir` as raw little-endian f32 / u32
    /// files plus `meta.json` with the shapes.
    pub fn dump_parity(
        &self,
        samples: &[f32],
        prompt: Option<&str>,
        forced: &[u32],
        dir: &Path,
        pool: Option<&Pool>,
    ) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let put = |name: &str, v: &[f32]| -> Result<(), String> {
            let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join(name), b).map_err(|e| format!("{name}: {e}"))
        };
        let a = self.audio.run(samples, pool);
        let ids = self.prompt_ids(prompt.unwrap_or(&self.default_prompt), a.tokens, false);
        put("feats.f32", &a.feats)?;
        put("asr_hidden.f32", &a.asr_hidden)?;
        put("speaker_logits.f32", &a.speaker_logits)?;
        put("embeds.f32", &a.embeds)?;
        let idb: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(dir.join("input_ids.u32"), idb).map_err(|e| e.to_string())?;
        if forced.is_empty() {
            let meta = serde_json::json!({
                "frames": a.frames, "valid_frames": a.valid_frames, "n_mel": self.audio.n_mel,
                "enc_frames": a.asr_hidden.len() / self.audio.asr.d, "enc_d": self.audio.asr.d,
                "tokens": a.tokens, "hidden": self.llm.g.hidden, "prompt_len": ids.len(), "forced": 0,
            });
            return std::fs::write(dir.join("meta.json"), meta.to_string())
                .map_err(|e| e.to_string());
        }
        let rows = self.input_rows(&ids, &a.embeds)?;
        let mut st = self.llm.new_state();
        let t0 = Instant::now();
        let mut logits = self.llm.forward(&mut st, &rows, pool);
        let prefill_s = t0.elapsed().as_secs_f64();
        let vocab = self.llm.g.vocab;
        let mut all = Vec::with_capacity(forced.len() * vocab);
        let mut row = vec![0.0f32; self.llm.g.hidden];
        let t1 = Instant::now();
        for (i, &id) in forced.iter().enumerate() {
            all.extend_from_slice(&logits[..vocab]);
            if i + 1 < forced.len() {
                self.llm.embed_row(id, &mut row);
                logits = self.llm.forward(&mut st, &row, pool);
            }
        }
        let decode_s = t1.elapsed().as_secs_f64();
        put("logits.f32", &all)?;
        let meta = serde_json::json!({
            "frames": a.frames, "valid_frames": a.valid_frames, "n_mel": self.audio.n_mel,
            "enc_frames": a.asr_hidden.len() / self.audio.asr.d, "enc_d": self.audio.asr.d,
            "tokens": a.tokens, "hidden": self.llm.g.hidden, "prompt_len": ids.len(),
            "forced": forced.len(), "vocab": vocab, "prefill_s": prefill_s, "decode_s": decode_s,
        });
        std::fs::write(dir.join("meta.json"), meta.to_string()).map_err(|e| e.to_string())
    }
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best as u32
}

/// Read a WAV file as 16 kHz mono f32 (resampled when needed).
pub fn read_audio(path: &Path) -> Result<Vec<f32>, String> {
    let (mono, rate) = crate::whisper::read_wav_samples(path, false)?;
    Ok(if rate == SR {
        mono
    } else {
        crate::whisper::resample_sinc(&mono, rate, SR)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_rotates_the_first_half_pairs() {
        let (c, s) = rope_table(3, 4);
        let mut x = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        rope(&mut x, &c[2 * 2..], &s[2 * 2..], 4);
        // position 2, pair (0,2) at θ=2, pair (1,3) at θ=2/100
        let (a0, a1) = (2.0f32, 2.0f32 / 100.0);
        assert!((x[0] - (1.0 * a0.cos() - 3.0 * a0.sin())).abs() < 1e-6);
        assert!((x[2] - (3.0 * a0.cos() + 1.0 * a0.sin())).abs() < 1e-6);
        assert!((x[1] - (2.0 * a1.cos() - 4.0 * a1.sin())).abs() < 1e-6);
        assert_eq!(x[4], 5.0);
    }

    #[test]
    fn bf16_rounding_is_nearest_even() {
        assert_eq!(round_bf16(1.0), 1.0);
        let x = f32::from_bits(0x3f80_8000); // halfway, even below
        assert_eq!(round_bf16(x).to_bits(), 0x3f80_0000);
        let y = f32::from_bits(0x3f81_8000); // halfway, odd below
        assert_eq!(round_bf16(y).to_bits(), 0x3f82_0000);
    }

    fn cfg() -> SpkCfg {
        SpkCfg {
            chunk: 500,
            left: 50,
            right: 50,
            fifo_len: 4,
            update_period: 3,
            cache_len: 16,
            n_spk: 2,
            sil_frames: 1,
            pred_thr: 0.25,
            latest_boost: 0.05,
            min_pos: 3,
            n_strong: 5,
            n_weak: 10,
            activity_thr: 0.5,
        }
    }

    #[test]
    fn compress_keeps_cache_length_with_silence_slots() {
        let c = cfg();
        let d = 3;
        let n = 20;
        let embeds: Vec<f32> = (0..n * d).map(|i| i as f32).collect();
        let probs: Vec<f32> = (0..n)
            .flat_map(|t| if t < 10 { [0.9f32, 0.1] } else { [0.2, 0.8] })
            .collect();
        let sil = [-1.0f32; 3];
        let (e, p) = compress(&c, d, &embeds, &probs, &sil);
        assert_eq!(e.len(), c.cache_len * d);
        assert_eq!(p.len(), c.cache_len * c.n_spk);
        // two silence slots (one per speaker) ride at the end of each speaker's block
        assert_eq!(e.chunks_exact(d).filter(|r| r[0] == -1.0).count(), 2);
    }

    #[test]
    fn fifo_pops_at_least_the_update_period() {
        let c = cfg();
        let d = 1;
        let mut cache = SpkCache {
            embeds: vec![],
            probs: vec![],
            fifo: vec![],
            compressed: false,
        };
        let input: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let logits = vec![0.0f32; 6 * 2 * 2];
        let mask = vec![true; 6];
        cache.update(&c, d, &input, &logits, &mask, &[0.0], 6, 0);
        // 6 frames > fifo 4: pop max(3, 2) = 3 into the cache
        assert_eq!(cache.embeds, vec![0.0, 1.0, 2.0]);
        assert_eq!(cache.fifo, vec![3.0, 4.0, 5.0]);
    }
}
