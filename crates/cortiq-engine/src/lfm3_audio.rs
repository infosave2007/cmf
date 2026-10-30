//! LFM3-Audio / LFM3-Omni speech input. Liquid's log-mel features (25 ms
//! Hann window every 10 ms without centering, 128 Slaney mel filters,
//! log10 clamped at -12, (x + 4) / 4), the stride-8 convolutional
//! downsampler (three stride-2 3×3 convolutions over the mel × time plane:
//! 8 frames become one vector), the 17-layer LFM2 encoder (gated short
//! convolutions and attention causal within a 32-token window) and the
//! projector into the language model, in f32 on the host from the f16
//! tensors the converter keeps (`audio_encoder.*`, `audio_projector.*`).
//! The language model then reads one row per `<|audio|>` placeholder.

use crate::lfm3_vl::{Dense, add_bias, erf, load_proj, load_vec, rms_rows};
use crate::pipeline::Pipeline;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::{Value, json};
use std::sync::Arc;

/// The feature extractor and the encoder geometry, from the checkpoint's
/// config (`provenance.lfm3_audio`): the legacy LFM3-Audio layout
/// (`encoder.{dim_in, dim_conv, dim_out, sliding_window, lfm}`) or the
/// Transformers one of LFM3-Omni (`audio_encoder_config`).
pub struct Lfm3AudioConfig {
    pub n_mels: usize,
    pub channels: usize,
    pub hidden: usize,
    pub window: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// True for an attention layer, false for a short convolution.
    pub attention: Vec<bool>,
    pub conv_kernel: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub n_fft: usize,
    pub hop: usize,
    pub log_floor: f64,
    pub norm_offset: f64,
    pub norm_divisor: f64,
}

/// Whether the file carries the speech encoder.
pub fn has_audio(model: &CmfModel) -> bool {
    model
        .tensors
        .iter()
        .any(|t| t.name == "audio_encoder.downsample.conv_in.weight")
}

impl Lfm3AudioConfig {
    pub fn from_model(model: &CmfModel) -> Result<Self, String> {
        let p = model
            .header
            .provenance
            .as_ref()
            .and_then(|p| p.get("lfm3_audio"))
            .ok_or("lfm3_audio: the file has no audio config (provenance.lfm3_audio)")?;
        let cfg = p.get("config").ok_or("lfm3_audio: provenance has no config")?;
        Self::from_json(cfg, p.get("features").unwrap_or(&Value::Null))
    }

    pub fn from_json(cfg: &Value, features: &Value) -> Result<Self, String> {
        let (enc, lfm, n_mels, channels, hidden) = if let Some(e) = cfg.get("encoder") {
            let lfm = e.get("lfm").ok_or("lfm3_audio: encoder.lfm missing")?;
            (e, lfm, e.get("dim_in"), e.get("dim_conv"), e.get("dim_out"))
        } else {
            let e = cfg
                .get("audio_encoder_config")
                .ok_or("lfm3_audio: no encoder config")?;
            let lfm = e.get("text_config").ok_or("lfm3_audio: audio_encoder_config.text_config missing")?;
            (e, lfm, e.get("num_mel_bins"), e.get("conv_channels"), e.get("hidden_size"))
        };
        let u = |v: Option<&Value>, what: &str| -> Result<usize, String> {
            v.and_then(Value::as_u64)
                .map(|x| x as usize)
                .ok_or(format!("lfm3_audio: {what} missing"))
        };
        let heads = u(lfm.get("num_attention_heads"), "num_attention_heads")?;
        let hidden = u(hidden, "encoder width")?;
        let attention: Vec<bool> = lfm
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or("lfm3_audio: layer_types missing")?
            .iter()
            .map(|t| t.as_str() != Some("conv"))
            .collect();
        let rope = lfm.get("rope_parameters").unwrap_or(lfm);
        let pre = |key: &str, legacy: &str, default: f64| -> f64 {
            features
                .get(key)
                .or_else(|| cfg.get("preprocessor").and_then(|p| p.get(legacy)))
                .and_then(Value::as_f64)
                .unwrap_or(default)
        };
        let norm = cfg.get("preprocessor").and_then(|p| p.get("normalization"));
        Ok(Self {
            n_mels: u(n_mels, "mel bins")?,
            channels: u(channels, "conv channels")?,
            hidden,
            window: u(enc.get("sliding_window"), "sliding_window")?,
            heads,
            kv_heads: u(lfm.get("num_key_value_heads"), "num_key_value_heads")?,
            head_dim: hidden / heads.max(1),
            attention,
            conv_kernel: u(lfm.get("conv_L_cache"), "conv_L_cache")?,
            norm_eps: lfm.get("norm_eps").and_then(Value::as_f64).unwrap_or(1e-5),
            rope_theta: rope.get("rope_theta").and_then(Value::as_f64).unwrap_or(1e6),
            n_fft: pre("n_fft", "n_fft", 400.0) as usize,
            hop: pre("hop_length", "hop_length", 160.0) as usize,
            log_floor: pre("log_floor", "logmel_floor", -12.0),
            norm_offset: features
                .get("normalization_offset")
                .or_else(|| norm.and_then(|n| n.get("offset")))
                .and_then(Value::as_f64)
                .unwrap_or(4.0),
            norm_divisor: features
                .get("normalization_divisor")
                .or_else(|| norm.and_then(|n| n.get("divisor")))
                .and_then(Value::as_f64)
                .unwrap_or(4.0),
        })
    }
}

// ───────────────────────────── features ─────────────────────────────

/// Mel frequencies on Slaney's scale (linear below 1 kHz, log above).
fn hz_to_mel(hz: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let (min_log_hz, logstep) = (1000.0, (6.4f64).ln() / 27.0);
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_hz / f_sp + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let (min_log_hz, logstep) = (1000.0, (6.4f64).ln() / 27.0);
    let min_log_mel = min_log_hz / f_sp;
    if m < min_log_mel {
        m * f_sp
    } else {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    }
}

/// `transformers.audio_utils.mel_filter_bank(n_fft/2+1, n_mels, 0, sr/2,
/// sr, norm="slaney", mel_scale="slaney")`, row-major `[n_mels][bins]`.
fn mel_filters(n_mels: usize, n_fft: usize, sr: f64) -> Vec<f64> {
    let bins = n_fft / 2 + 1;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(sr / 2.0));
    let edges: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut f = vec![0f64; n_mels * bins];
    for m in 0..n_mels {
        let norm = 2.0 / (edges[m + 2] - edges[m]);
        for k in 0..bins {
            let hz = (sr / 2.0) * k as f64 / (bins - 1) as f64;
            let down = (hz - edges[m]) / (edges[m + 1] - edges[m]);
            let up = (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1]);
            f[m * bins + k] = down.min(up).max(0.0) * norm;
        }
    }
    f
}

/// The processor's frames for `n` samples: right-padded to whole windows.
pub fn num_frames(n: usize, cfg: &Lfm3AudioConfig) -> usize {
    let padded = if n < cfg.n_fft {
        cfg.n_fft
    } else {
        n + (cfg.hop - (n - cfg.n_fft) % cfg.hop) % cfg.hop
    };
    1 + (padded - cfg.n_fft) / cfg.hop
}

/// Log-mel features of a 16 kHz mono clip, `[n_mels][frames]` (the
/// processor's `input_features[0]`).
pub fn log_mel(wave: &[f32], cfg: &Lfm3AudioConfig) -> (Vec<f32>, usize) {
    use rustfft::{FftPlanner, num_complex::Complex};
    let (n_fft, hop, n_mels) = (cfg.n_fft, cfg.hop, cfg.n_mels);
    let frames = num_frames(wave.len(), cfg);
    let bins = n_fft / 2 + 1;
    let filters = mel_filters(n_mels, n_fft, 16_000.0);
    // torch.hann_window(n_fft) — periodic.
    let window: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
        .collect();
    let fft = FftPlanner::<f64>::new().plan_fft_forward(n_fft);
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    let mut power = vec![0f64; bins];
    let mut out = vec![0f32; n_mels * frames];
    for t in 0..frames {
        for (i, b) in buf.iter_mut().enumerate() {
            let s = wave.get(t * hop + i).copied().unwrap_or(0.0);
            *b = Complex::new(s as f64 * window[i], 0.0);
        }
        fft.process(&mut buf);
        for (p, b) in power.iter_mut().zip(&buf[..bins]) {
            *p = b.norm_sqr();
        }
        for m in 0..n_mels {
            let e: f64 = filters[m * bins..(m + 1) * bins]
                .iter()
                .zip(&power)
                .map(|(f, p)| f * p)
                .sum();
            let l = e.log10().max(cfg.log_floor);
            out[m * frames + t] = ((l + cfg.norm_offset) / cfg.norm_divisor) as f32;
        }
    }
    (out, frames)
}

// ───────────────────────────── encoder ─────────────────────────────

/// A 3×3 stride-2 convolution, padding 1, over a channel-last map
/// `[h][w][c_in]`; `groups` = c_in (depthwise) or 1. Returns
/// `[⌈h/2⌉][⌈w/2⌉][c_out]` and its height and width.
#[allow(clippy::too_many_arguments)]
fn conv3x3_s2(
    x: &[f32],
    h: usize,
    w: usize,
    c_in: usize,
    weight: &[f32],
    bias: &[f32],
    c_out: usize,
    depthwise: bool,
    pool: Option<&Pool>,
) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (h.div_ceil(2), w.div_ceil(2));
    let mut y = vec![0f32; ho * wo * c_out];
    let dst = crate::pool::SendMut::new(y.as_mut_ptr());
    let f = |lo: usize, hi: usize| {
        for oy in lo..hi {
            for ox in 0..wo {
                // SAFETY: output rows (oy) are disjoint across workers.
                let out = unsafe { std::slice::from_raw_parts_mut(dst.at((oy * wo + ox) * c_out), c_out) };
                out.copy_from_slice(bias);
                for ky in 0..3 {
                    let iy = (2 * oy + ky) as isize - 1;
                    if iy < 0 || iy >= h as isize {
                        continue;
                    }
                    for kx in 0..3 {
                        let ix = (2 * ox + kx) as isize - 1;
                        if ix < 0 || ix >= w as isize {
                            continue;
                        }
                        let src = &x[(iy as usize * w + ix as usize) * c_in..][..c_in];
                        let tap = ky * 3 + kx;
                        if depthwise {
                            for (c, o) in out.iter_mut().enumerate() {
                                *o += weight[c * 9 + tap] * src[c];
                            }
                        } else {
                            for (c, o) in out.iter_mut().enumerate() {
                                let wk = &weight[c * c_in * 9..];
                                let mut acc = 0f32;
                                for (ci, &v) in src.iter().enumerate() {
                                    acc += wk[ci * 9 + tap] * v;
                                }
                                *o += acc;
                            }
                        }
                    }
                }
            }
        }
    };
    match pool {
        Some(p) => p.run_rows(ho, &f),
        None => f(0, ho),
    }
    (y, ho, wo)
}

fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

enum Mixer {
    Conv {
        in_proj: Dense,
        /// `[channel][kernel]`, the last tap on the current position.
        taps: Vec<f32>,
        out_proj: Dense,
    },
    Attn {
        q: Dense,
        k: Dense,
        v: Dense,
        o: Dense,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

struct Layer {
    op_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    mixer: Mixer,
    w1: Dense,
    w3: Dense,
    w2: Dense,
}

/// The speech encoder and its projector.
pub struct Lfm3AudioEncoder {
    pub cfg: Lfm3AudioConfig,
    conv_in: (Vec<f32>, Vec<f32>),
    conv_mid: (Vec<f32>, Vec<f32>),
    lin_mid: (Dense, Vec<f32>),
    conv_end: (Vec<f32>, Vec<f32>),
    lin_end: (Dense, Vec<f32>),
    lin_out: (Dense, Vec<f32>),
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    ln: (Vec<f32>, Vec<f32>),
    l1: (Dense, Vec<f32>),
    l2: (Dense, Vec<f32>),
    pub text_hidden: usize,
    pool: Option<Arc<Pool>>,
}

impl Lfm3AudioEncoder {
    pub fn from_model(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let cfg = Lfm3AudioConfig::from_model(model)?;
        let (c, d) = (cfg.channels, cfg.hidden);
        let ds = |n: &str| format!("audio_encoder.downsample.{n}");
        let v = |name: String, len: usize| load_vec(model, &name, len);
        let flat = cfg.n_mels.div_ceil(8) * c;
        let mut layers = Vec::with_capacity(cfg.attention.len());
        let inter = model
            .tensors
            .iter()
            .find(|t| t.name == "audio_encoder.lfm.layers.0.feed_forward.w1.weight")
            .and_then(|t| t.shape.first().copied())
            .ok_or("lfm3_audio: encoder FFN missing")?;
        let (qd, kvd) = (cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        for (i, &attn) in cfg.attention.iter().enumerate() {
            let p = format!("audio_encoder.lfm.layers.{i}");
            let mixer = if attn {
                Mixer::Attn {
                    q: load_proj(model, &format!("{p}.self_attn.q_proj.weight"), qd, d)?,
                    k: load_proj(model, &format!("{p}.self_attn.k_proj.weight"), kvd, d)?,
                    v: load_proj(model, &format!("{p}.self_attn.v_proj.weight"), kvd, d)?,
                    o: load_proj(model, &format!("{p}.self_attn.out_proj.weight"), d, qd)?,
                    q_norm: v(format!("{p}.self_attn.q_layernorm.weight"), cfg.head_dim)?,
                    k_norm: v(format!("{p}.self_attn.k_layernorm.weight"), cfg.head_dim)?,
                }
            } else {
                Mixer::Conv {
                    in_proj: load_proj(model, &format!("{p}.conv.in_proj.weight"), 3 * d, d)?,
                    taps: v(format!("{p}.conv.conv.weight"), d * cfg.conv_kernel)?,
                    out_proj: load_proj(model, &format!("{p}.conv.out_proj.weight"), d, d)?,
                }
            };
            layers.push(Layer {
                op_norm: v(format!("{p}.operator_norm.weight"), d)?,
                ffn_norm: v(format!("{p}.ffn_norm.weight"), d)?,
                mixer,
                w1: load_proj(model, &format!("{p}.feed_forward.w1.weight"), inter, d)?,
                w3: load_proj(model, &format!("{p}.feed_forward.w3.weight"), inter, d)?,
                w2: load_proj(model, &format!("{p}.feed_forward.w2.weight"), d, inter)?,
            });
        }
        let text_hidden = model
            .tensors
            .iter()
            .find(|t| t.name == "audio_projector.linear_2.weight")
            .and_then(|t| t.shape.first().copied())
            .ok_or("lfm3_audio: projector missing")?;
        let l1_rows = model
            .tensors
            .iter()
            .find(|t| t.name == "audio_projector.linear_1.weight")
            .and_then(|t| t.shape.first().copied())
            .ok_or("lfm3_audio: projector missing")?;
        Ok(Self {
            conv_in: (v(ds("conv_in.weight"), c * 9)?, v(ds("conv_in.bias"), c)?),
            conv_mid: (v(ds("conv_mid.weight"), c * 9)?, v(ds("conv_mid.bias"), c)?),
            lin_mid: (load_proj(model, &ds("lin_mid.weight"), c, c)?, v(ds("lin_mid.bias"), c)?),
            conv_end: (v(ds("conv_end.weight"), c * 9)?, v(ds("conv_end.bias"), c)?),
            lin_end: (load_proj(model, &ds("lin_end.weight"), c, c)?, v(ds("lin_end.bias"), c)?),
            lin_out: (load_proj(model, &ds("lin_out.weight"), d, flat)?, v(ds("lin_out.bias"), d)?),
            layers,
            final_norm: v("audio_encoder.lfm.embedding_norm.weight".into(), d)?,
            ln: (
                v("audio_projector.layer_norm.weight".into(), d)?,
                v("audio_projector.layer_norm.bias".into(), d)?,
            ),
            l1: (
                load_proj(model, "audio_projector.linear_1.weight", l1_rows, d)?,
                v("audio_projector.linear_1.bias".into(), l1_rows)?,
            ),
            l2: (
                load_proj(model, "audio_projector.linear_2.weight", text_hidden, l1_rows)?,
                v("audio_projector.linear_2.bias".into(), text_hidden)?,
            ),
            text_hidden,
            cfg,
            pool,
        })
    }

    /// Encoder rows `[⌈frames/8⌉, hidden]` of log-mel features
    /// `[n_mels][frames]`, before the projector.
    pub fn encode(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let pool = self.pool.as_deref();
        if mel.len() != c.n_mels * frames || frames == 0 {
            return Err("lfm3_audio: features do not match the encoder".into());
        }
        // The (mel, time) plane, channel-last with one input channel.
        let (h, w) = (c.n_mels, frames);
        let ch = c.channels;
        let (mut x, h1, w1) = conv3x3_s2(mel, h, w, 1, &self.conv_in.0, &self.conv_in.1, ch, false, pool);
        silu_inplace(&mut x);
        let (x2, h2, w2) = conv3x3_s2(&x, h1, w1, ch, &self.conv_mid.0, &self.conv_mid.1, ch, true, pool);
        let mut x = vec![0f32; h2 * w2 * ch];
        self.lin_mid.0.matmat(&x2, h2 * w2, &mut x, pool);
        add_bias(&mut x, &self.lin_mid.1);
        silu_inplace(&mut x);
        let (x3, h3, w3) = conv3x3_s2(&x, h2, w2, ch, &self.conv_end.0, &self.conv_end.1, ch, true, pool);
        let mut x = vec![0f32; h3 * w3 * ch];
        self.lin_end.0.matmat(&x3, h3 * w3, &mut x, pool);
        add_bias(&mut x, &self.lin_end.1);
        silu_inplace(&mut x);
        // flatten(channels, mel): feature c·h3 + y of time step t.
        let n = w3;
        let flat = ch * h3;
        let mut f = vec![0f32; n * flat];
        for y in 0..h3 {
            for t in 0..n {
                let src = &x[(y * n + t) * ch..][..ch];
                for (cc, &val) in src.iter().enumerate() {
                    f[t * flat + cc * h3 + y] = val;
                }
            }
        }
        let d = c.hidden;
        let mut x = vec![0f32; n * d];
        self.lin_out.0.matmat(&f, n, &mut x, pool);
        add_bias(&mut x, &self.lin_out.1);
        let mut xn = vec![0f32; n * d];
        let mut mix = vec![0f32; n * d];
        let (hd, half) = (c.head_dim, c.head_dim / 2);
        let inv: Vec<f32> = (0..half)
            .map(|i| (1.0 / c.rope_theta.powf((2 * i) as f64 / hd as f64)) as f32)
            .collect();
        for layer in &self.layers {
            rms_rows(&x, &layer.op_norm, c.norm_eps, &mut xn, pool);
            match &layer.mixer {
                Mixer::Conv { in_proj, taps, out_proj } => {
                    let k = c.conv_kernel;
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
                Mixer::Attn { q, k, v, o, q_norm, k_norm } => {
                    let (qd, kvd) = (c.heads * hd, c.kv_heads * hd);
                    let mut qb = vec![0f32; n * qd];
                    let mut kb = vec![0f32; n * kvd];
                    let mut vb = vec![0f32; n * kvd];
                    q.matmat(&xn, n, &mut qb, pool);
                    k.matmat(&xn, n, &mut kb, pool);
                    v.matmat(&xn, n, &mut vb, pool);
                    for (buf, heads, w) in [(&mut qb, c.heads, q_norm), (&mut kb, c.kv_heads, k_norm)] {
                        let mut tmp = vec![0f32; hd];
                        for t in 0..n {
                            for hh in 0..heads {
                                let head = &mut buf[(t * heads + hh) * hd..][..hd];
                                crate::inference::rms_norm_into(
                                    head,
                                    w,
                                    c.norm_eps,
                                    cortiq_core::NormStyle::Qwen,
                                    &mut tmp,
                                );
                                head.copy_from_slice(&tmp);
                                crate::attention::rope_rotate_scaled(head, t, &inv, 1.0);
                            }
                        }
                    }
                    let group = c.heads / c.kv_heads.max(1);
                    let scale = (hd as f32).powf(-0.5);
                    let mut ao = vec![0f32; n * qd];
                    let mut sc = Vec::with_capacity(c.window);
                    for t in 0..n {
                        let lo = (t + 1).saturating_sub(c.window);
                        for hh in 0..c.heads {
                            let g = hh / group;
                            let qv = &qb[(t * c.heads + hh) * hd..][..hd];
                            sc.clear();
                            for j in lo..=t {
                                let kv = &kb[(j * c.kv_heads + g) * hd..][..hd];
                                sc.push(qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale);
                            }
                            let mx = sc.iter().fold(f32::NEG_INFINITY, |m, &s| m.max(s));
                            let mut sum = 0f32;
                            for s in sc.iter_mut() {
                                *s = (*s - mx).exp();
                                sum += *s;
                            }
                            let out = &mut ao[(t * c.heads + hh) * hd..][..hd];
                            for (j, &p) in (lo..=t).zip(sc.iter()) {
                                let vv = &vb[(j * c.kv_heads + g) * hd..][..hd];
                                for (od, &x) in out.iter_mut().zip(vv) {
                                    *od += p / sum * x;
                                }
                            }
                        }
                    }
                    o.matmat(&ao, n, &mut mix, pool);
                }
            }
            for (a, &b) in x.iter_mut().zip(&mix) {
                *a += b;
            }
            rms_rows(&x, &layer.ffn_norm, c.norm_eps, &mut xn, pool);
            let inter = layer.w1.rows;
            let mut g = vec![0f32; n * inter];
            let mut u = vec![0f32; n * inter];
            layer.w1.matmat(&xn, n, &mut g, pool);
            layer.w3.matmat(&xn, n, &mut u, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = *a / (1.0 + (-*a).exp()) * b;
            }
            layer.w2.matmat(&g, n, &mut mix, pool);
            for (a, &b) in x.iter_mut().zip(&mix) {
                *a += b;
            }
        }
        rms_rows(&x, &self.final_norm, c.norm_eps, &mut xn, pool);
        Ok(xn)
    }

    /// Projected rows `[tokens, text_hidden]` of encoder rows.
    pub fn project(&self, enc: &[f32]) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let d = self.cfg.hidden;
        let n = enc.len() / d;
        let mut x = enc.to_vec();
        for row in x.chunks_exact_mut(d) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for (j, v) in row.iter_mut().enumerate() {
                *v = ((*v as f64 - mean) * inv) as f32 * self.ln.0[j] + self.ln.1[j];
            }
        }
        let mut h1 = vec![0f32; n * self.l1.0.rows];
        self.l1.0.matmat(&x, n, &mut h1, pool);
        add_bias(&mut h1, &self.l1.1);
        for v in h1.iter_mut() {
            *v = (0.5 * *v as f64 * (1.0 + erf(*v as f64 / std::f64::consts::SQRT_2))) as f32;
        }
        let mut out = vec![0f32; n * self.text_hidden];
        self.l2.0.matmat(&h1, n, &mut out, pool);
        add_bias(&mut out, &self.l2.1);
        out
    }

    /// One clip, 16 kHz mono, to the language model's rows.
    pub fn rows(&self, wave: &[f32]) -> Result<Vec<f32>, String> {
        let (mel, frames) = log_mel(wave, &self.cfg);
        let enc = self.encode(&mel, frames)?;
        let rows = self.project(&enc);
        if !rows.iter().all(|v| v.is_finite()) {
            return Err("lfm3_audio: the encoder produced non-finite rows".into());
        }
        Ok(rows)
    }
}

// ───────────────────────────── prompts ─────────────────────────────

/// Replace every audio part of `messages` with `{"type": "audio"}` (the
/// template renders `<|audio|>`) and collect the sources in order.
pub(crate) fn normalize(messages: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut messages = messages.to_vec();
    let mut sources = Vec::new();
    for message in &mut messages {
        let Some(parts) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for part in parts {
            let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
            if matches!(typ, "audio" | "input_audio" | "audio_url")
                || part.get("input_audio").is_some()
                || part.get("audio").is_some()
            {
                let src = part
                    .get("input_audio")
                    .or_else(|| part.get("audio_url"))
                    .or_else(|| part.get("audio"))
                    .cloned()
                    .unwrap_or_else(|| part.clone());
                let src = match (&src, src.get("url")) {
                    (Value::Object(_), Some(u)) if src.get("data").is_none() => u.clone(),
                    _ => src,
                };
                sources.push(src);
                *part = json!({"type": "audio"});
            }
        }
    }
    (messages, sources)
}

/// Whether any message carries an audio part.
pub fn has_audio_parts(messages: &[Value]) -> bool {
    !normalize(messages).1.is_empty()
}

/// Render `messages`, run every clip through the encoder, expand each
/// `<|audio|>` to one placeholder per encoder token (the processor's
/// expansion) and assemble the prompt rows. Text-only prompts come back
/// without rows.
pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
) -> Result<crate::lfm3_vl::PreparedVlPrompt, String> {
    let (messages, sources) = normalize(messages);
    let tok = &pipeline.tokenizer;
    if sources.is_empty() {
        let ids = tok.try_apply_chat_template_json(&messages, tools, thinking)?;
        return Ok(crate::lfm3_vl::PreparedVlPrompt { token_ids: ids, rows: None });
    }
    let text = tok
        .render_chat_json(&messages, tools, thinking)
        .ok_or("LFM3-Audio: the chat template failed to render")?;
    let enc = Lfm3AudioEncoder::from_model(model, crate::pool::Pool::from_env())?;
    if enc.text_hidden != pipeline.hidden_size {
        return Err("LFM3-Audio: projector width differs from the text model".into());
    }
    let audio_id = tok
        .token_to_id("<|audio|>")
        .ok_or("LFM3-Audio: the tokenizer has no <|audio|> token")?;
    let dump = std::env::var_os("CMF_LFM3_AUDIO_DUMP").map(std::path::PathBuf::from);
    let mut sources = sources.into_iter();
    let mut expanded = String::with_capacity(text.len());
    let mut media_rows: Vec<f32> = Vec::new();
    let mut rest = text.as_str();
    let mut k = 0usize;
    while let Some(at) = rest.find("<|audio|>") {
        expanded.push_str(&rest[..at]);
        rest = &rest[at + "<|audio|>".len()..];
        let src = sources
            .next()
            .ok_or("LFM3-Audio: more <|audio|> placeholders than clips")?;
        let wave = crate::egemma2_audio::load_audio_source(&src)?;
        let t0 = std::time::Instant::now();
        let (mel, frames) = log_mel(&wave, &enc.cfg);
        let er = enc.encode(&mel, frames)?;
        let rows = enc.project(&er);
        if let Some(dir) = &dump {
            let write = |name: String, v: &[f32]| {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(dir.join(name), bytes).map_err(|e| format!("CMF_LFM3_AUDIO_DUMP: {e}"))
            };
            write(format!("mel_{k}.f32"), &mel)?;
            write(format!("enc_{k}.f32"), &er)?;
            write(format!("rows_{k}.f32"), &rows)?;
            eprintln!(
                "lfm3-audio: clip {k}: {} samples, {frames} frames, {} tokens, encoder {:.2} s",
                wave.len(),
                rows.len() / enc.text_hidden,
                t0.elapsed().as_secs_f64()
            );
        }
        for _ in 0..rows.len() / enc.text_hidden {
            expanded.push_str("<|audio|>");
        }
        media_rows.extend(rows);
        k += 1;
    }
    expanded.push_str(rest);
    if sources.next().is_some() {
        return Err("LFM3-Audio: the template rendered fewer placeholders than clips".into());
    }
    let ids = tok.with_bos(tok.encode(&expanded));
    let h = pipeline.hidden_size;
    let mut rows = Vec::with_capacity(ids.len() * h);
    let mut at = 0usize;
    for &id in &ids {
        if id == audio_id {
            let row = media_rows
                .get(at..at + h)
                .ok_or("LFM3-Audio: placeholder and clip row counts differ")?;
            rows.extend_from_slice(row);
            at += h;
        } else {
            rows.extend(pipeline.embed_id(id));
        }
    }
    if at != media_rows.len() {
        return Err("LFM3-Audio: unused clip rows".into());
    }
    Ok(crate::lfm3_vl::PreparedVlPrompt { token_ids: ids, rows: Some(rows) })
}

/// Token ids a text answer must not start a speech or mixed turn with
/// (`suppress_audio_out` / `suppress_mixed`).
pub fn text_only_suppress(tok: &crate::tokenizer::Tokenizer) -> Vec<u32> {
    ["<|audio_start|>", "<|mixed_start|>"]
        .iter()
        .filter_map(|t| tok.token_to_id(t))
        .collect()
}

/// [`prepare`] on the pipeline's own file.
pub fn prepare_for_pipeline(
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
) -> Result<crate::lfm3_vl::PreparedVlPrompt, String> {
    let model = pipeline
        .model
        .as_ref()
        .ok_or("LFM3-Audio clips require a file-backed pipeline")?;
    prepare(model, pipeline, messages, tools, thinking)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Lfm3AudioConfig {
        Lfm3AudioConfig {
            n_mels: 128, channels: 256, hidden: 768, window: 32, heads: 8, kv_heads: 4,
            head_dim: 96, attention: vec![], conv_kernel: 3, norm_eps: 1e-5, rope_theta: 1e6,
            n_fft: 400, hop: 160, log_floor: -12.0, norm_offset: 4.0, norm_divisor: 4.0,
        }
    }

    /// The processor's frame count: right-padded to whole hops, at least
    /// one window.
    #[test]
    fn frames_match_the_processor() {
        let c = cfg();
        assert_eq!(num_frames(100, &c), 1);
        assert_eq!(num_frames(400, &c), 1);
        assert_eq!(num_frames(401, &c), 2);
        assert_eq!(num_frames(560, &c), 2);
        assert_eq!(num_frames(70_080, &c), 436);
    }

    /// Slaney filters integrate to the band's normalized area: the first
    /// filter peaks at its centre bin and every filter is non-negative.
    #[test]
    fn slaney_filters_are_triangles() {
        let f = mel_filters(128, 400, 16_000.0);
        assert!(f.iter().all(|&v| v >= 0.0));
        assert!(f[..201].iter().any(|&v| v > 0.0));
    }
}
