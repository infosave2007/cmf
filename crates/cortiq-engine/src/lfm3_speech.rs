//! LFM3-Audio / LFM3-Omni speech output. After `<|audio_start|>` every step
//! of the language model yields one speech frame: the depth decoder (a
//! three-layer bidirectional LFM3 over the frame's 12 codebook slots) reads
//! the step's hidden state and reveals the codebooks in three MaskGIT
//! passes of four; the frame's codebook embeddings, summed, are the next
//! step's input; a frame holding the end-of-audio code ends the speech.
//! LiquidCodec's decoder turns the frames into 24 kHz audio: FSQ codes →
//! 48 latents → 256 → 512, six steps a frame through an eight-layer LFM3
//! (short convolutions, attention causal within 30 steps), log-magnitude
//! and phase of a 1280-point spectrum, inverse STFT with hop 320.
//! Everything here runs in f32 on the host from the f16 tensors the
//! converter keeps (`depth_decoder.*`, `codebook_embedding`, `codec.*`).

use crate::lfm3_vl::{Dense, add_bias, load_proj, load_vec};
use crate::pipeline::Pipeline;
use crate::pool::Pool;
use crate::sampler::SplitMix64;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Codebooks of a speech frame and codes of a codebook (the last one is
/// the end-of-audio code).
pub const N_CODEBOOKS: usize = 12;
pub const CODEBOOK_SIZE: usize = 2017;
pub const AUDIO_EOS: u32 = 2016;
/// Codebooks revealed by each MaskGIT pass.
const MASKGIT: [usize; 3] = [4, 4, 4];
/// Samples a frame decodes to at 24 kHz.
pub const FRAME_SAMPLES: usize = 1920;

fn tensor_rows(model: &CmfModel, name: &str) -> Option<usize> {
    model.tensors.iter().find(|t| t.name == name).and_then(|t| t.shape.first().copied())
}

fn has_tensor(model: &CmfModel, name: &str) -> bool {
    model.tensors.iter().any(|t| t.name == name)
}

// ───────────────────────────── an LFM stack ─────────────────────────────

enum Mixer {
    Conv { in_proj: Dense, taps: Vec<f32>, out_proj: Dense },
    Attn { q: Dense, k: Dense, v: Dense, o: Dense, q_norm: Vec<f32>, k_norm: Vec<f32>, gate: Option<Dense> },
}

struct Layer {
    op_norm: Vec<f32>,
    post_op: Option<Vec<f32>>,
    ffn_norm: Vec<f32>,
    post_ffn: Option<Vec<f32>>,
    mixer: Mixer,
    w1: Dense,
    w3: Dense,
    w2: Dense,
}

/// An LFM3 layer stack in Liquid's names (`operator_norm`, `conv.*`,
/// `self_attn.*`, `feed_forward.w1..3`, the optional sandwich norms
/// `post_operator_norm` / `post_ffn_norm`, the optional per-head sigmoid
/// gate `self_attn.gate_proj`), final `embedding_norm`.
struct Stack {
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    d: usize,
    heads: usize,
    kv: usize,
    hd: usize,
    conv_k: usize,
    eps: f64,
    inv: Vec<f32>,
    /// Some(w): causal within w positions; None: bidirectional.
    window: Option<usize>,
}

fn rms(x: &[f32], w: &[f32], eps: f64, out: &mut [f32]) {
    let d = w.len();
    for (src, dst) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let ss = src.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64;
        let inv = 1.0 / (ss + eps).sqrt();
        for ((o, &v), &g) in dst.iter_mut().zip(src).zip(w) {
            *o = (v as f64 * inv) as f32 * g;
        }
    }
}

impl Stack {
    #[allow(clippy::too_many_arguments)]
    fn load(
        model: &Arc<CmfModel>,
        prefix: &str,
        d: usize,
        heads: usize,
        kv: usize,
        eps: f64,
        theta: f64,
        window: Option<usize>,
    ) -> Result<Self, String> {
        let hd = d / heads;
        let mut layers = Vec::new();
        let mut conv_k = 1;
        for i in 0.. {
            let p = format!("{prefix}.layers.{i}");
            let conv = has_tensor(model, &format!("{p}.conv.in_proj.weight"));
            let attn = has_tensor(model, &format!("{p}.self_attn.q_proj.weight"));
            if !conv && !attn {
                break;
            }
            let v = |n: &str, len: usize| load_vec(model, &format!("{p}.{n}"), len);
            let opt = |n: &str| {
                has_tensor(model, &format!("{p}.{n}")).then(|| v(n, d)).transpose()
            };
            let mixer = if conv {
                let taps_len = model
                    .tensors
                    .iter()
                    .find(|t| t.name == format!("{p}.conv.conv.weight"))
                    .map(|t| t.shape.iter().product::<usize>())
                    .ok_or("lfm stack: conv taps missing")?;
                conv_k = taps_len / d;
                Mixer::Conv {
                    in_proj: load_proj(model, &format!("{p}.conv.in_proj.weight"), 3 * d, d)?,
                    taps: v("conv.conv.weight", taps_len)?,
                    out_proj: load_proj(model, &format!("{p}.conv.out_proj.weight"), d, d)?,
                }
            } else {
                let gate_name = format!("{p}.self_attn.gate_proj.weight");
                Mixer::Attn {
                    q: load_proj(model, &format!("{p}.self_attn.q_proj.weight"), heads * hd, d)?,
                    k: load_proj(model, &format!("{p}.self_attn.k_proj.weight"), kv * hd, d)?,
                    v: load_proj(model, &format!("{p}.self_attn.v_proj.weight"), kv * hd, d)?,
                    o: load_proj(model, &format!("{p}.self_attn.o_proj.weight"), d, heads * hd)?,
                    q_norm: v("self_attn.q_norm.weight", hd)?,
                    k_norm: v("self_attn.k_norm.weight", hd)?,
                    gate: has_tensor(model, &gate_name)
                        .then(|| load_proj(model, &gate_name, heads, d))
                        .transpose()?,
                }
            };
            let inter = tensor_rows(model, &format!("{p}.feed_forward.w1.weight")).ok_or("lfm stack: FFN missing")?;
            layers.push(Layer {
                op_norm: v("operator_norm.weight", d)?,
                post_op: opt("post_operator_norm.weight")?,
                ffn_norm: v("ffn_norm.weight", d)?,
                post_ffn: opt("post_ffn_norm.weight")?,
                mixer,
                w1: load_proj(model, &format!("{p}.feed_forward.w1.weight"), inter, d)?,
                w3: load_proj(model, &format!("{p}.feed_forward.w3.weight"), inter, d)?,
                w2: load_proj(model, &format!("{p}.feed_forward.w2.weight"), d, inter)?,
            });
        }
        if layers.is_empty() {
            return Err(format!("lfm stack: no layers under {prefix}"));
        }
        let inv = (0..hd / 2)
            .map(|i| (1.0 / theta.powf((2 * i) as f64 / hd as f64)) as f32)
            .collect();
        Ok(Self {
            layers,
            final_norm: load_vec(model, &format!("{prefix}.embedding_norm.weight"), d)?,
            d,
            heads,
            kv,
            hd,
            conv_k,
            eps,
            inv,
            window,
        })
    }

    /// `x` `[n, d]` through every layer and the final norm, in place.
    fn forward(&self, x: &mut [f32], pool: Option<&Pool>) {
        let (d, hd) = (self.d, self.hd);
        let n = x.len() / d;
        let mut xn = vec![0f32; n * d];
        let mut mix = vec![0f32; n * d];
        let mut tmp = vec![0f32; n * d];
        for l in &self.layers {
            rms(x, &l.op_norm, self.eps, &mut xn);
            match &l.mixer {
                Mixer::Conv { in_proj, taps, out_proj } => {
                    let k = self.conv_k;
                    let mut bcx = vec![0f32; n * 3 * d];
                    in_proj.matmat(&xn, n, &mut bcx, pool);
                    let mut y = vec![0f32; n * d];
                    for t in 0..n {
                        for j in 0..d {
                            let mut acc = 0f32;
                            for tap in 0..k {
                                let back = k - 1 - tap;
                                if t >= back {
                                    let r = &bcx[(t - back) * 3 * d..];
                                    acc += taps[j * k + tap] * (r[j] * r[2 * d + j]);
                                }
                            }
                            y[t * d + j] = bcx[t * 3 * d + d + j] * acc;
                        }
                    }
                    out_proj.matmat(&y, n, &mut mix, pool);
                }
                Mixer::Attn { q, k, v, o, q_norm, k_norm, gate } => {
                    let (qd, kvd) = (self.heads * hd, self.kv * hd);
                    let mut qb = vec![0f32; n * qd];
                    let mut kb = vec![0f32; n * kvd];
                    let mut vb = vec![0f32; n * kvd];
                    q.matmat(&xn, n, &mut qb, pool);
                    k.matmat(&xn, n, &mut kb, pool);
                    v.matmat(&xn, n, &mut vb, pool);
                    let mut head = vec![0f32; hd];
                    for (buf, heads, w) in [(&mut qb, self.heads, q_norm), (&mut kb, self.kv, k_norm)] {
                        for t in 0..n {
                            for hh in 0..heads {
                                let hv = &mut buf[(t * heads + hh) * hd..][..hd];
                                rms(hv, w, self.eps, &mut head);
                                hv.copy_from_slice(&head);
                                crate::attention::rope_rotate_scaled(hv, t, &self.inv, 1.0);
                            }
                        }
                    }
                    let group = self.heads / self.kv.max(1);
                    let scale = (hd as f32).powf(-0.5);
                    let mut ao = vec![0f32; n * qd];
                    let mut sc = Vec::new();
                    for t in 0..n {
                        let (lo, hi) = match self.window {
                            Some(w) => ((t + 1).saturating_sub(w), t + 1),
                            None => (0, n),
                        };
                        for hh in 0..self.heads {
                            let g = hh / group;
                            let qv = &qb[(t * self.heads + hh) * hd..][..hd];
                            sc.clear();
                            for j in lo..hi {
                                let kv = &kb[(j * self.kv + g) * hd..][..hd];
                                sc.push(qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale);
                            }
                            let mx = sc.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
                            let mut sum = 0f32;
                            for s in sc.iter_mut() {
                                *s = (*s - mx).exp();
                                sum += *s;
                            }
                            let out = &mut ao[(t * self.heads + hh) * hd..][..hd];
                            for (j, &p) in (lo..hi).zip(sc.iter()) {
                                let vv = &vb[(j * self.kv + g) * hd..][..hd];
                                for (od, &x) in out.iter_mut().zip(vv) {
                                    *od += p / sum * x;
                                }
                            }
                        }
                    }
                    if let Some(g) = gate {
                        let mut raw = vec![0f32; n * self.heads];
                        g.matmat(&xn, n, &mut raw, pool);
                        for t in 0..n {
                            for hh in 0..self.heads {
                                let s = 1.0 / (1.0 + (-raw[t * self.heads + hh]).exp());
                                for a in &mut ao[(t * self.heads + hh) * hd..][..hd] {
                                    *a *= s;
                                }
                            }
                        }
                    }
                    o.matmat(&ao, n, &mut mix, pool);
                }
            }
            if let Some(w) = &l.post_op {
                rms(&mix, w, self.eps, &mut tmp);
                std::mem::swap(&mut mix, &mut tmp);
            }
            for (a, &b) in x.iter_mut().zip(&mix) {
                *a += b;
            }
            rms(x, &l.ffn_norm, self.eps, &mut xn);
            let inter = l.w1.rows;
            let mut g = vec![0f32; n * inter];
            let mut u = vec![0f32; n * inter];
            l.w1.matmat(&xn, n, &mut g, pool);
            l.w3.matmat(&xn, n, &mut u, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = *a / (1.0 + (-*a).exp()) * b;
            }
            l.w2.matmat(&g, n, &mut mix, pool);
            if let Some(w) = &l.post_ffn {
                rms(&mix, w, self.eps, &mut tmp);
                std::mem::swap(&mut mix, &mut tmp);
            }
            for (a, &b) in x.iter_mut().zip(&mix) {
                *a += b;
            }
        }
        rms(x, &self.final_norm, self.eps, &mut xn);
        x.copy_from_slice(&xn);
    }
}

// ───────────────────────────── depth decoder ─────────────────────────────

/// Audio sampling: Liquid's recommended `audio_temperature` 1.0,
/// `audio_top_k` 128, repetition penalty 1.1 over the last 4096 frames.
#[derive(Clone, Debug)]
pub struct AudioSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    /// Reveal the codebooks in order 0..11 instead of a random order (the
    /// reference harness's fixed order).
    pub fixed_order: bool,
}

impl Default for AudioSampling {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 128, repetition_penalty: 1.1, repetition_window: 4096, fixed_order: false }
    }
}

impl AudioSampling {
    /// The release's own `audio_*` settings (its generation config, kept by
    /// the converter in `provenance.lfm3_audio.generation`): LFM3-Omni
    /// samples at 0.4 with a 1.2 penalty over 64 frames and no top-k.
    pub fn from_model(model: &CmfModel) -> Self {
        let mut s = Self::default();
        let g = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("lfm3_audio"))
            .and_then(|a| a.get("generation"));
        if let Some(g) = g {
            if let Some(t) = g.get("audio_temperature").and_then(|v| v.as_f64()) {
                s.temperature = t as f32;
            }
            match g.get("audio_top_k") {
                Some(v) if v.is_null() => s.top_k = 0,
                Some(v) => s.top_k = v.as_u64().map_or(s.top_k, |k| k as usize),
                None => {}
            }
            if let Some(r) = g.get("audio_repetition_penalty").and_then(|v| v.as_f64()) {
                s.repetition_penalty = r as f32;
            }
            if let Some(w) = g.get("audio_repetition_window").and_then(|v| v.as_u64()) {
                s.repetition_window = w as usize;
            }
        }
        s
    }
}

/// The depth decoder and the language model's speech-frame embedding.
pub struct DepthDecoder {
    linear: (Dense, Vec<f32>),
    mask: Vec<f32>,
    /// `[12][2017][d]` code embeddings (also the tied output heads).
    emb: Vec<f32>,
    emb_norm: Vec<f32>,
    stack: Stack,
    d: usize,
    /// `[12·2017][text_hidden]`, the frame embedding of the language model.
    codebook: Vec<f32>,
    text_hidden: usize,
    pool: Option<Arc<Pool>>,
}

impl DepthDecoder {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let text_hidden = model
            .tensors
            .iter()
            .find(|t| t.name == "codebook_embedding.weight")
            .and_then(|t| t.shape.get(1).copied())
            .ok_or("lfm3_speech: codebook_embedding missing")?;
        let d = tensor_rows(model, "depth_decoder.mask_embedding").ok_or("lfm3_speech: mask_embedding missing")?;
        let mut emb = Vec::with_capacity(N_CODEBOOKS * CODEBOOK_SIZE * d);
        let mut emb_norm = Vec::with_capacity(N_CODEBOOKS * d);
        for i in 0..N_CODEBOOKS {
            let p = format!("depth_decoder.depth_embeddings.{i}");
            emb.extend(load_vec(model, &format!("{p}.embedding.weight"), CODEBOOK_SIZE * d)?);
            emb_norm.extend(load_vec(model, &format!("{p}.embedding_norm.weight"), d)?);
        }
        // Geometry of the depth model: 32-wide heads (its config).
        let q_rows = tensor_rows(model, "depth_decoder.model.layers.0.self_attn.q_proj.weight").unwrap_or(d);
        let k_rows = tensor_rows(model, "depth_decoder.model.layers.0.self_attn.k_proj.weight").unwrap_or(d);
        let hd = model
            .tensors
            .iter()
            .find(|t| t.name == "depth_decoder.model.layers.0.self_attn.q_norm.weight")
            .and_then(|t| t.shape.first().copied())
            .ok_or("lfm3_speech: depth q_norm missing")?;
        let stack = Stack::load(model, "depth_decoder.model", d, q_rows / hd, k_rows / hd, 1e-5, 1e6, None)?;
        Ok(Self {
            linear: (
                load_proj(model, "depth_decoder.depth_linear.weight", N_CODEBOOKS * d, text_hidden)?,
                load_vec(model, "depth_decoder.depth_linear.bias", N_CODEBOOKS * d)?,
            ),
            mask: load_vec(model, "depth_decoder.mask_embedding", d)?,
            emb,
            emb_norm,
            stack,
            d,
            codebook: load_vec(model, "codebook_embedding.weight", N_CODEBOOKS * CODEBOOK_SIZE * text_hidden)?,
            text_hidden,
            pool,
        })
    }

    /// The frame's slots before any code: the language model's
    /// final-normed hidden state through `depth_linear`, `[12][d]`.
    fn slots(&self, h: &[f32]) -> Vec<f32> {
        let mut x = vec![0f32; N_CODEBOOKS * self.d];
        self.linear.0.matmat(h, 1, &mut x, self.pool.as_deref());
        add_bias(&mut x, &self.linear.1);
        x
    }

    /// Logits `[2017]` of the codebooks `want` (in that order) given the
    /// frame's slots, its known codes and which are still hidden.
    fn logits(
        &self,
        slots: &[f32],
        codes: &[u32; N_CODEBOOKS],
        masked: &[bool; N_CODEBOOKS],
        want: &[usize],
    ) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let d = self.d;
        let mut x = slots.to_vec();
        for i in 0..N_CODEBOOKS {
            let add = if masked[i] {
                &self.mask[..]
            } else {
                &self.emb[(i * CODEBOOK_SIZE + codes[i] as usize) * d..][..d]
            };
            for (a, &b) in x[i * d..(i + 1) * d].iter_mut().zip(add) {
                *a += b;
            }
        }
        self.stack.forward(&mut x, pool);
        let mut out = vec![0f32; want.len() * CODEBOOK_SIZE];
        let mut normed = vec![0f32; d];
        for (r, &i) in want.iter().enumerate() {
            rms(&x[i * d..(i + 1) * d], &self.emb_norm[i * d..(i + 1) * d], 1e-5, &mut normed);
            let table = &self.emb[i * CODEBOOK_SIZE * d..(i + 1) * CODEBOOK_SIZE * d];
            let row = &mut out[r * CODEBOOK_SIZE..(r + 1) * CODEBOOK_SIZE];
            crate::fcd_ops::gemm_nt_host(&normed, table, row, 1, d, CODEBOOK_SIZE, pool);
        }
        out
    }

    /// One speech frame: the codebooks revealed four at a time in a random
    /// (or fixed) order, each sampled from its logits under the repetition
    /// penalty of its own history.
    pub fn sample_frame(
        &self,
        h: &[f32],
        s: &AudioSampling,
        history: &[[u32; N_CODEBOOKS]],
        rng: &mut SplitMix64,
    ) -> [u32; N_CODEBOOKS] {
        let mut order: Vec<usize> = (0..N_CODEBOOKS).collect();
        if !s.fixed_order {
            for i in (1..N_CODEBOOKS).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
        }
        let mut codes = [0u32; N_CODEBOOKS];
        let mut masked = [true; N_CODEBOOKS];
        let hist = &history[history.len().saturating_sub(s.repetition_window)..];
        let slots = self.slots(h);
        let mut at = 0;
        for step in MASKGIT {
            let pick = &order[at..at + step];
            let logits = self.logits(&slots, &codes, &masked, pick);
            for (r, &cb) in pick.iter().enumerate() {
                let mut row = logits[r * CODEBOOK_SIZE..(r + 1) * CODEBOOK_SIZE].to_vec();
                if s.repetition_penalty != 1.0 {
                    let mut seen = vec![false; CODEBOOK_SIZE];
                    for f in hist {
                        seen[f[cb] as usize] = true;
                    }
                    for (v, &was) in row.iter_mut().zip(&seen) {
                        if was {
                            *v = if *v < 0.0 { *v * s.repetition_penalty } else { *v / s.repetition_penalty };
                        }
                    }
                }
                codes[cb] = sample(&row, s.temperature, s.top_k, rng);
            }
            for &cb in &order[at..at + step] {
                masked[cb] = false;
            }
            at += step;
        }
        codes
    }

    /// The language model's input row for a frame: its codebook
    /// embeddings, summed.
    pub fn frame_row(&self, codes: &[u32; N_CODEBOOKS]) -> Vec<f32> {
        let th = self.text_hidden;
        let mut row = vec![0f32; th];
        for (i, &c) in codes.iter().enumerate() {
            let e = &self.codebook[(i * CODEBOOK_SIZE + c as usize) * th..][..th];
            for (a, &b) in row.iter_mut().zip(e) {
                *a += b;
            }
        }
        row
    }
}

/// Greedy at temperature 0, else temperature then top-k then a draw.
fn sample(logits: &[f32], temperature: f32, top_k: usize, rng: &mut SplitMix64) -> u32 {
    if temperature <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
            .0 as u32;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    if top_k > 0 && top_k < idx.len() {
        idx.select_nth_unstable_by(top_k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
        idx.truncate(top_k);
    }
    let mx = idx.iter().map(|&i| logits[i] / temperature).fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = idx.iter().map(|&i| ((logits[i] / temperature - mx) as f64).exp()).collect();
    let total: f64 = w.iter().sum();
    let mut r = rng.next_f32() as f64 * total;
    for (&i, &wi) in idx.iter().zip(&w) {
        r -= wi;
        if r <= 0.0 {
            return i as u32;
        }
    }
    *idx.last().unwrap() as u32
}

// ───────────────────────────── codec ─────────────────────────────

/// LiquidCodec's decoder: speech frames to 24 kHz audio.
pub struct Codec {
    project_out: (Dense, Vec<f32>),
    from_q: (Dense, Vec<f32>),
    stack: Stack,
    lin: (Dense, Vec<f32>),
    window: Vec<f32>,
    pool: Option<Arc<Pool>>,
}

const FSQ_LEVELS: [u32; 4] = [8, 7, 6, 6];
const FRAME_STACK: usize = 6;
const N_FFT: usize = 1280;
const HOP: usize = 320;

impl Codec {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let qd = tensor_rows(model, "codec.quantizer.project_out.weight").ok_or("lfm3_speech: the file has no codec")?;
        let ed = tensor_rows(model, "codec.from_quantizer.weight").ok_or("lfm3_speech: codec from_quantizer missing")?;
        let lat = N_CODEBOOKS * FSQ_LEVELS.len();
        let bins2 = tensor_rows(model, "codec.decoder.lin.weight").ok_or("lfm3_speech: codec lin missing")?;
        if bins2 != 2 * (N_FFT / 2 + 1) {
            return Err("lfm3_speech: codec spectrum width differs".into());
        }
        let hd = tensor_rows(model, "codec.decoder.lfm.layers.2.self_attn.q_norm.weight").unwrap_or(64);
        let heads = tensor_rows(model, "codec.decoder.lfm.layers.2.self_attn.q_proj.weight").unwrap_or(ed) / hd;
        let kv = tensor_rows(model, "codec.decoder.lfm.layers.2.self_attn.k_proj.weight").unwrap_or(ed) / hd;
        Ok(Self {
            project_out: (
                load_proj(model, "codec.quantizer.project_out.weight", qd, lat)?,
                load_vec(model, "codec.quantizer.project_out.bias", qd)?,
            ),
            from_q: (
                load_proj(model, "codec.from_quantizer.weight", ed, qd)?,
                load_vec(model, "codec.from_quantizer.bias", ed)?,
            ),
            stack: Stack::load(model, "codec.decoder.lfm", ed, heads, kv, 1e-5, 1e6, Some(30))?,
            lin: (
                load_proj(model, "codec.decoder.lin.weight", bins2, ed)?,
                load_vec(model, "codec.decoder.lin.bias", bins2)?,
            ),
            window: load_vec(model, "codec.decoder.istft.window", N_FFT)?,
            pool,
        })
    }

    /// 24 kHz samples of `frames` (1920 a frame).
    pub fn decode(&self, frames: &[[u32; N_CODEBOOKS]]) -> Vec<f32> {
        use rustfft::{FftPlanner, num_complex::Complex};
        let pool = self.pool.as_deref();
        let nf = frames.len();
        if nf == 0 {
            return Vec::new();
        }
        let lat_w = N_CODEBOOKS * FSQ_LEVELS.len();
        let mut lat = vec![0f32; nf * lat_w];
        let basis = [1u32, 8, 56, 336];
        for (f, codes) in frames.iter().enumerate() {
            for (i, &c) in codes.iter().enumerate() {
                for (j, (&lv, &b)) in FSQ_LEVELS.iter().zip(&basis).enumerate() {
                    let half = (lv / 2) as f32;
                    lat[f * lat_w + i * 4 + j] = (((c / b) % lv) as f32 - half) / half;
                }
            }
        }
        let qd = self.project_out.0.rows;
        let mut q = vec![0f32; nf * qd];
        self.project_out.0.matmat(&lat, nf, &mut q, pool);
        add_bias(&mut q, &self.project_out.1);
        let ed = self.from_q.0.rows;
        let mut e = vec![0f32; nf * ed];
        self.from_q.0.matmat(&q, nf, &mut e, pool);
        add_bias(&mut e, &self.from_q.1);
        let n = nf * FRAME_STACK;
        let mut x = vec![0f32; n * ed];
        for t in 0..n {
            x[t * ed..(t + 1) * ed].copy_from_slice(&e[(t / FRAME_STACK) * ed..][..ed]);
        }
        self.stack.forward(&mut x, pool);
        let bins = N_FFT / 2 + 1;
        let mut spec = vec![0f32; n * 2 * bins];
        self.lin.0.matmat(&x, n, &mut spec, pool);
        add_bias(&mut spec, &self.lin.1);
        let ifft = FftPlanner::<f64>::new().plan_fft_inverse(N_FFT);
        let out_len = (n - 1) * HOP + N_FFT;
        let mut audio = vec![0f64; out_len];
        let mut env = vec![0f64; out_len];
        let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
        for t in 0..n {
            let row = &spec[t * 2 * bins..(t + 1) * 2 * bins];
            for k in 0..bins {
                let mag = (row[k] as f64).exp();
                let ph = row[bins + k] as f64;
                let v = Complex::new(mag * ph.cos(), mag * ph.sin());
                buf[k] = if k == 0 || k == N_FFT / 2 { Complex::new(v.re, 0.0) } else { v };
                if k > 0 && k < N_FFT / 2 {
                    buf[N_FFT - k] = v.conj();
                }
            }
            ifft.process(&mut buf);
            for (i, b) in buf.iter().enumerate() {
                let w = self.window[i] as f64;
                audio[t * HOP + i] += b.re / N_FFT as f64 * w;
                env[t * HOP + i] += w * w;
            }
        }
        let trim = (N_FFT - HOP) / 2;
        audio[trim..out_len - trim]
            .iter()
            .zip(&env[trim..out_len - trim])
            .map(|(&a, &w)| (a / w) as f32)
            .collect()
    }
}

// ───────────────────────────── generation ─────────────────────────────

/// A speech turn: the frames and the text tokens around them.
pub struct SpeechOut {
    pub frames: Vec<[u32; N_CODEBOOKS]>,
    pub text_ids: Vec<u32>,
}

/// Whether the file can speak (depth decoder and codec).
pub fn can_speak(model: &CmfModel) -> bool {
    has_tensor(model, "depth_decoder.depth_linear.weight") && has_tensor(model, "codec.decoder.lin.weight")
}

/// Generate after a prompt that ends with `<|audio_start|>` (speech, then
/// text until the turn ends) or `<|mixed_start|>` (Liquid's interleaved
/// turn: six text tokens, twelve speech frames, … until `<|text_end|>`,
/// then speech until the end-of-audio frame), at most `max_steps` steps.
/// Text is greedy; speech is sampled with `s`.
pub fn speak(
    pipeline: &mut Pipeline,
    depth: &DepthDecoder,
    ids: &[u32],
    rows: Option<&[f32]>,
    max_steps: usize,
    s: &AudioSampling,
    seed: u64,
) -> Result<SpeechOut, String> {
    const N_TEXT: usize = 6;
    const N_AUDIO: usize = 12;
    let id = |t: &str| pipeline.tokenizer.token_to_id(t);
    let audio_start = id("<|audio_start|>").ok_or("tokenizer lacks <|audio_start|>")?;
    let (mixed_start, text_end, im_end) = (id("<|mixed_start|>"), id("<|text_end|>"), id("<|im_end|>"));
    #[derive(PartialEq, Clone, Copy)]
    enum Mode {
        Text,
        Audio,
        Mixed,
    }
    let mut mode = match ids.last() {
        Some(&t) if t == audio_start => Mode::Audio,
        Some(&t) if Some(t) == mixed_start => Mode::Mixed,
        _ => Mode::Text,
    };
    let (mut slots_left, mut speaking, mut text_done) = (N_TEXT, false, false);
    let mut rng = SplitMix64::new(seed);
    let mut h = pipeline.speech_prefill(ids, rows)?;
    let mut pos = ids.len();
    let mut out = SpeechOut { frames: Vec::new(), text_ids: Vec::new() };
    // CMF_LFM3_SPEECH_PROF: where a turn's time goes.
    let prof = std::env::var_os("CMF_LFM3_SPEECH_PROF").is_some();
    let (mut t_depth, mut t_lm, mut t_head) = (0f64, 0f64, 0f64);
    for _ in 0..max_steps {
        let interleaving = mode == Mode::Mixed;
        let t0 = std::time::Instant::now();
        let row = if mode == Mode::Audio || (interleaving && speaking) {
            let codes = depth.sample_frame(&h, s, &out.frames, &mut rng);
            t_depth += t0.elapsed().as_secs_f64();
            if codes.contains(&AUDIO_EOS) {
                mode = Mode::Text;
                depth.frame_row(&[AUDIO_EOS; N_CODEBOOKS])
            } else {
                out.frames.push(codes);
                depth.frame_row(&codes)
            }
        } else {
            let logits = pipeline.speech_logits(&h);
            t_head += t0.elapsed().as_secs_f64();
            let t = sample(&logits, 0.0, 0, &mut rng);
            if Some(t) == text_end {
                if !interleaving {
                    break;
                }
                // The text of the interleaved turn is over; its speech runs on.
                text_done = true;
                speaking = true;
            } else if Some(t) == im_end || pipeline.tokenizer.is_eos(t) {
                break;
            } else if !interleaving && t == audio_start {
                mode = Mode::Audio;
            } else if !interleaving && Some(t) == mixed_start {
                mode = Mode::Mixed;
                (slots_left, speaking, text_done) = (N_TEXT, false, false);
            }
            out.text_ids.push(t);
            pipeline.embed_id(t)
        };
        if interleaving && !text_done {
            slots_left -= 1;
            if slots_left == 0 {
                speaking = !speaking;
                slots_left = if speaking { N_AUDIO } else { N_TEXT };
            }
        }
        let t1 = std::time::Instant::now();
        h = pipeline.speech_step(&row, pos);
        t_lm += t1.elapsed().as_secs_f64();
        pos += 1;
    }
    if prof {
        eprintln!(
            "speech: {} frames, {} text tokens | depth decoder {:.0} ms, language model {:.0} ms, text head {:.0} ms",
            out.frames.len(),
            out.text_ids.len(),
            t_depth * 1e3,
            t_lm * 1e3,
            t_head * 1e3
        );
    }
    Ok(out)
}

/// 16-bit PCM mono WAV at `rate`.
pub fn write_wav(path: &std::path::Path, samples: &[f32], rate: u32) -> std::io::Result<()> {
    use std::io::Write;
    let bytes = (samples.len() * 2) as u32;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + bytes).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&bytes.to_le_bytes())?;
    for &v in samples {
        f.write_all(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())?;
    }
    f.flush()
}
