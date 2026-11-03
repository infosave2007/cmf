//! The conditioning side of Kandinsky S2V: the Qwen2.5-Omni Thinker's text
//! model, its audio tower, and the CLIP text tower's pooled embedding.
//!
//! The reference runs the Thinker once over `[system, user text, audio]` and
//! reads two things from it: the last hidden state over the user's text span,
//! and the audio tower's output through a hook. The text model is causal and
//! the audio comes after the text, so the text span never sees the speech:
//! here the prefix up to the audio start runs alone and the 7B model is only
//! touched at all when a prompt has to be encoded. On a text-only prefix the
//! three MRoPE axes carry the same value, so the rotation is plain RoPE.

use crate::dit::cmf_f32;
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::s2v::{Affine, Lin, Rows, gelu, pool_rows, silu};
use crate::zimage::host_gemm;
use cortiq_core::CmfModel;
use std::sync::Arc;

fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

/// Causal or chunked scaled-dot attention, one head at a time, as two GEMMs.
/// `mask(q, k)` gives the live key range `[lo, hi)` of a query row.
fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    range: &(dyn Fn(usize) -> (usize, usize) + Sync),
    pool: Option<&Pool>,
) -> Vec<f32> {
    let hpk = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; n * heads * hd];
    let mut qh = vec![0f32; n * hd];
    let mut kh = vec![0f32; n * hd];
    let mut vt = vec![0f32; hd * n];
    let mut sc = vec![0f32; n * n];
    let mut oh = vec![0f32; n * hd];
    for hh in 0..heads {
        let kv = hh / hpk;
        for p in 0..n {
            for d in 0..hd {
                qh[p * hd + d] = q[(p * heads + hh) * hd + d] * scale;
            }
            let b = (p * kv_heads + kv) * hd;
            kh[p * hd..(p + 1) * hd].copy_from_slice(&k[b..b + hd]);
            for d in 0..hd {
                vt[d * n + p] = v[b + d];
            }
        }
        host_gemm::gemm_nt(&qh, &kh, &mut sc, n, hd, n, pool);
        let sp = Rows(sc.as_mut_ptr());
        pool_rows(pool, n, &|lo, hi| {
            for r in lo..hi {
                // SAFETY: disjoint rows.
                let row = unsafe { sp.at(r * n, n) };
                let (a, b) = range(r);
                row[..a].fill(0.0);
                row[b..].fill(0.0);
                let live = &mut row[a..b];
                let mx = live.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0f64;
                for x in live.iter_mut() {
                    *x = (*x - mx).exp();
                    den += *x as f64;
                }
                let inv = (1.0 / den) as f32;
                for x in live.iter_mut() {
                    *x *= inv;
                }
            }
        });
        host_gemm::gemm_nt(&sc, &vt, &mut oh, n, n, hd, pool);
        for p in 0..n {
            let o = (p * heads + hh) * hd;
            out[o..o + hd].copy_from_slice(&oh[p * hd..(p + 1) * hd]);
        }
    }
    out
}

// ─────────────────────────── Thinker text ───────────────────────────

struct TextLayer {
    input_norm: Vec<f32>,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    post_norm: Vec<f32>,
    gate: Lin,
    up: Lin,
    down: Lin,
}

pub struct OmniText {
    embed: QTensor,
    layers: Vec<TextLayer>,
    norm: Vec<f32>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    theta: f32,
    eps: f64,
}

impl OmniText {
    pub fn from_cmf(m: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(m.tensor_bytes("te.config_json").map_err(|e| e.to_string())?)
            .map_err(|e| format!("te.config_json: {e}"))?;
        let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
        let hidden = u("hidden_size");
        let nh = u("num_attention_heads");
        let mut layers = Vec::new();
        for l in 0..u("num_hidden_layers") {
            let p = format!("te.layers.{l}");
            layers.push(TextLayer {
                input_norm: cmf_f32(m, &format!("{p}.input_layernorm.weight"))?,
                q: Lin::load(m, &format!("{p}.self_attn.q_proj"))?,
                k: Lin::load(m, &format!("{p}.self_attn.k_proj"))?,
                v: Lin::load(m, &format!("{p}.self_attn.v_proj"))?,
                o: Lin::load(m, &format!("{p}.self_attn.o_proj"))?,
                post_norm: cmf_f32(m, &format!("{p}.post_attention_layernorm.weight"))?,
                gate: Lin::load(m, &format!("{p}.mlp.gate_proj"))?,
                up: Lin::load(m, &format!("{p}.mlp.up_proj"))?,
                down: Lin::load(m, &format!("{p}.mlp.down_proj"))?,
            });
        }
        Ok(Self {
            embed: QTensor::from_model(m, "te.embed_tokens.weight")?,
            layers,
            norm: cmf_f32(m, "te.norm.weight")?,
            pool: Pool::from_env(),
            hidden,
            nh,
            nkv: u("num_key_value_heads"),
            hd: hidden / nh,
            theta: cfg["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e6) as f32,
            eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        })
    }

    /// The last hidden state (normed, as `hidden_states[-1]` is) of `ids`.
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        let (n, hs, nh, nkv, hd) = (ids.len(), self.hidden, self.nh, self.nkv, self.hd);
        let half = hd / 2;
        let pool = self.pool.as_deref();
        let mut h = vec![0f32; n * hs];
        for (i, &id) in ids.iter().enumerate() {
            self.embed.row_f32(id as usize, &mut h[i * hs..(i + 1) * hs]);
        }
        let freqs: Vec<f32> = (0..half).map(|i| 1.0 / self.theta.powf(2.0 * i as f32 / hd as f32)).collect();
        let rot = |buf: &mut [f32], heads: usize| {
            let sr = Rows(buf.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for p in lo..hi {
                    for hh in 0..heads {
                        // SAFETY: disjoint rows.
                        let x = unsafe { sr.at((p * heads + hh) * hd, hd) };
                        for i in 0..half {
                            let (s, c) = (p as f32 * freqs[i]).sin_cos();
                            let (a, b) = (x[i], x[i + half]);
                            x[i] = a * c - b * s;
                            x[i + half] = a * s + b * c;
                        }
                    }
                }
            });
        };
        let mut xn = vec![0f32; n * hs];
        for l in &self.layers {
            for (o, s) in xn.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(s, &l.input_norm, self.eps, o);
            }
            let mut q = l.q.apply(&xn, n, pool);
            let mut k = l.k.apply(&xn, n, pool);
            let v = l.v.apply(&xn, n, pool);
            rot(&mut q, nh);
            rot(&mut k, nkv);
            let att = attend(&q, &k, &v, n, nh, nkv, hd, &|r| (0, r + 1), pool);
            let o = l.o.apply(&att, n, pool);
            for (d, &x) in h.iter_mut().zip(&o) {
                *d += x;
            }
            for (o, s) in xn.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(s, &l.post_norm, self.eps, o);
            }
            let mut g = l.gate.apply(&xn, n, pool);
            let u = l.up.apply(&xn, n, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = silu(*a) * b;
            }
            let d = l.down.apply(&g, n, pool);
            for (x, &y) in h.iter_mut().zip(&d) {
                *x += y;
            }
        }
        let mut out = vec![0f32; n * hs];
        for (o, s) in out.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
            rms_norm_into(s, &self.norm, self.eps, o);
        }
        out
    }
}

// ─────────────────────────── Audio tower ────────────────────────────

struct AudioLayer {
    sa_norm: Affine,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    fin_norm: Affine,
    fc1: Lin,
    fc2: Lin,
}

pub struct OmniAudio {
    conv1: (Vec<f32>, Vec<f32>),
    conv2: (Vec<f32>, Vec<f32>),
    layers: Vec<AudioLayer>,
    ln_post: Affine,
    proj: Lin,
    pool: Option<Arc<Pool>>,
    d: usize,
    heads: usize,
    n_window: usize,
    mel_bins: usize,
    pos: Vec<f32>,
}

/// `[out, in, 3]` conv over a `[in][t]`-major frame stack: `y[t][out]`.
fn conv1d(w: &[f32], b: &[f32], x: &[f32], t_in: usize, cin: usize, cout: usize, stride: usize, pool: Option<&Pool>) -> Vec<f32> {
    let t_out = t_in.div_ceil(stride);
    let k = 3 * cin;
    let mut pat = vec![0f32; t_out * k];
    for t in 0..t_out {
        let c = t * stride;
        for tap in 0..3usize {
            let at = c as isize + tap as isize - 1;
            if at < 0 || at as usize >= t_in {
                continue; // zero padding
            }
            let at = at as usize;
            for ci in 0..cin {
                pat[t * k + ci * 3 + tap] = x[at * cin + ci];
            }
        }
    }
    let mut y = vec![0f32; t_out * cout];
    host_gemm::gemm_nt(&pat, w, &mut y, t_out, k, cout, pool);
    for row in y.chunks_exact_mut(cout) {
        for (v, &bb) in row.iter_mut().zip(b) {
            *v += bb;
        }
    }
    y
}

impl OmniAudio {
    pub fn from_cmf(m: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(m.tensor_bytes("audio.config_json").map_err(|e| e.to_string())?)
            .map_err(|e| format!("audio.config_json: {e}"))?;
        let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
        let d = u("d_model");
        let mut layers = Vec::new();
        for l in 0..u("encoder_layers") {
            let p = format!("audio.layers.{l}");
            layers.push(AudioLayer {
                sa_norm: Affine::load(m, &format!("{p}.self_attn_layer_norm"))?,
                q: Lin::load(m, &format!("{p}.self_attn.q_proj"))?,
                k: Lin::load(m, &format!("{p}.self_attn.k_proj"))?,
                v: Lin::load(m, &format!("{p}.self_attn.v_proj"))?,
                o: Lin::load(m, &format!("{p}.self_attn.out_proj"))?,
                fin_norm: Affine::load(m, &format!("{p}.final_layer_norm"))?,
                fc1: Lin::load(m, &format!("{p}.fc1"))?,
                fc2: Lin::load(m, &format!("{p}.fc2"))?,
            });
        }
        // `SinusoidsPositionEmbedding`: f32 ladder, sin then cos.
        let max_pos = u("max_source_positions");
        let half = d / 2;
        let inc = ((10000f64).ln() / (half - 1) as f64) as f32;
        let inv: Vec<f32> = (0..half).map(|j| (-inc * j as f32).exp()).collect();
        let mut pos = vec![0f32; max_pos * d];
        for t in 0..max_pos {
            for j in 0..half {
                let a = t as f32 * inv[j];
                pos[t * d + j] = a.sin();
                pos[t * d + half + j] = a.cos();
            }
        }
        Ok(Self {
            conv1: (cmf_f32(m, "audio.conv1.weight")?, cmf_f32(m, "audio.conv1.bias")?),
            conv2: (cmf_f32(m, "audio.conv2.weight")?, cmf_f32(m, "audio.conv2.bias")?),
            layers,
            ln_post: Affine::load(m, "audio.ln_post")?,
            proj: Lin::load(m, "audio.proj")?,
            pool: Pool::from_env(),
            d,
            heads: u("encoder_attention_heads"),
            n_window: u("n_window"),
            mel_bins: u("num_mel_bins"),
            pos,
        })
    }

    /// `mel` is `[bins][frames]`. Returns `[tokens][3584]` at 25 tokens/s.
    pub fn features(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        let (d, bins) = (self.d, self.mel_bins);
        let pool = self.pool.as_deref();
        let chunk = self.n_window * 2;
        // The convolutions run per window, zero-padded at both ends and with
        // the padded tail masked after the first GELU, so a short last window
        // matches a padded batch.
        let mut embed: Vec<f32> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        while start < frames {
            let len = chunk.min(frames - start);
            let mut f = vec![0f32; len * bins];
            for t in 0..len {
                for b in 0..bins {
                    f[t * bins + b] = mel[b * frames + start + t];
                }
            }
            let mut c1 = conv1d(&self.conv1.0, &self.conv1.1, &f, len, bins, d, 1, pool);
            for v in &mut c1 {
                *v = gelu(*v);
            }
            let mut c2 = conv1d(&self.conv2.0, &self.conv2.1, &c1, len, d, d, 2, pool);
            for v in &mut c2 {
                *v = gelu(*v);
            }
            let n = c2.len() / d;
            for (t, row) in c2.chunks_exact_mut(d).enumerate() {
                for (v, &p) in row.iter_mut().zip(&self.pos[t * d..(t + 1) * d]) {
                    *v += p;
                }
            }
            spans.push((embed.len() / d, n));
            embed.extend_from_slice(&c2);
            start += len;
        }
        let total = embed.len() / d;
        let bounds: Vec<(usize, usize)> = {
            let mut b = vec![(0, 0); total];
            for &(s, n) in &spans {
                for slot in &mut b[s..s + n] {
                    *slot = (s, s + n);
                }
            }
            b
        };
        let hd = d / self.heads;
        for l in &self.layers {
            let xn = l.sa_norm.apply(&embed, total, pool);
            let q = l.q.apply(&xn, total, pool);
            let k = l.k.apply(&xn, total, pool);
            let v = l.v.apply(&xn, total, pool);
            let att = attend(&q, &k, &v, total, self.heads, self.heads, hd, &|r| bounds[r], pool);
            let o = l.o.apply(&att, total, pool);
            for (a, &b) in embed.iter_mut().zip(&o) {
                *a += b;
            }
            let xn = l.fin_norm.apply(&embed, total, pool);
            let mut hid = l.fc1.apply(&xn, total, pool);
            for v in &mut hid {
                *v = gelu(*v);
            }
            let o = l.fc2.apply(&hid, total, pool);
            for (a, &b) in embed.iter_mut().zip(&o) {
                *a += b;
            }
        }
        // Stride-2 average over the whole sequence, then the projection.
        let out_n = (total + 1) / 2;
        let mut pooled = vec![0f32; out_n * d];
        for t in 0..out_n {
            let (a, b) = (t * 2, (t * 2 + 1).min(total - 1));
            for c in 0..d {
                pooled[t * d + c] = (embed[a * d + c] + embed[b * d + c]) / 2.0;
            }
        }
        let xn = self.ln_post.apply(&pooled, out_n, pool);
        self.proj.apply(&xn, out_n, pool)
    }
}

/// `WhisperFeatureExtractor` on a mono 16 kHz wave: `[128][frames]`, `frames` =
/// samples/160. The reference pads to 300 s with zeros before the STFT, so the
/// tail frames read zeros rather than a reflection.
pub fn omni_mel(wav: &[f32]) -> (Vec<f32>, usize) {
    let frames = wav.len() / 160;
    let mut padded = wav.to_vec();
    padded.extend(std::iter::repeat_n(0f32, 400));
    (crate::whisper::log_mel(&padded, frames), frames)
}

// ─────────────────────────────── CLIP ───────────────────────────────

struct ClipLayer {
    n1: Affine,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    n2: Affine,
    fc1: Lin,
    fc2: Lin,
}

pub struct ClipText {
    tok: Vec<f32>,
    pos: Vec<f32>,
    layers: Vec<ClipLayer>,
    fin: Affine,
    pool: Option<Arc<Pool>>,
    d: usize,
    heads: usize,
}

impl ClipText {
    pub fn from_cmf(m: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(m.tensor_bytes("clip.config_json").map_err(|e| e.to_string())?)
            .map_err(|e| format!("clip.config_json: {e}"))?;
        let d = cfg["hidden_size"].as_u64().unwrap_or(0) as usize;
        let mut layers = Vec::new();
        for l in 0..cfg["num_hidden_layers"].as_u64().unwrap_or(0) as usize {
            let p = format!("clip.encoder.layers.{l}");
            layers.push(ClipLayer {
                n1: Affine::load(m, &format!("{p}.layer_norm1"))?,
                q: Lin::load(m, &format!("{p}.self_attn.q_proj"))?,
                k: Lin::load(m, &format!("{p}.self_attn.k_proj"))?,
                v: Lin::load(m, &format!("{p}.self_attn.v_proj"))?,
                o: Lin::load(m, &format!("{p}.self_attn.out_proj"))?,
                n2: Affine::load(m, &format!("{p}.layer_norm2"))?,
                fc1: Lin::load(m, &format!("{p}.mlp.fc1"))?,
                fc2: Lin::load(m, &format!("{p}.mlp.fc2"))?,
            });
        }
        Ok(Self {
            tok: cmf_f32(m, "clip.embeddings.token_embedding.weight")?,
            pos: cmf_f32(m, "clip.embeddings.position_embedding.weight")?,
            layers,
            fin: Affine::load(m, "clip.final_layer_norm")?,
            pool: Pool::from_env(),
            d,
            heads: cfg["num_attention_heads"].as_u64().unwrap_or(0) as usize,
        })
    }

    /// `ids` padded to 77; the pooled row is the first end-of-text token.
    pub fn pooled(&self, ids: &[u32], eos: u32) -> Vec<f32> {
        let (n, d) = (ids.len(), self.d);
        let pool = self.pool.as_deref();
        let mut x = vec![0f32; n * d];
        for (i, &id) in ids.iter().enumerate() {
            for c in 0..d {
                x[i * d + c] = self.tok[id as usize * d + c] + self.pos[i * d + c];
            }
        }
        let hd = d / self.heads;
        for l in &self.layers {
            let xn = l.n1.apply(&x, n, pool);
            let q = l.q.apply(&xn, n, pool);
            let k = l.k.apply(&xn, n, pool);
            let v = l.v.apply(&xn, n, pool);
            let att = attend(&q, &k, &v, n, self.heads, self.heads, hd, &|r| (0, r + 1), pool);
            let o = l.o.apply(&att, n, pool);
            for (a, &b) in x.iter_mut().zip(&o) {
                *a += b;
            }
            let xn = l.n2.apply(&x, n, pool);
            let mut hid = l.fc1.apply(&xn, n, pool);
            for v in &mut hid {
                *v *= 1.0 / (1.0 + (-1.702 * *v).exp());
            }
            let o = l.fc2.apply(&hid, n, pool);
            for (a, &b) in x.iter_mut().zip(&o) {
                *a += b;
            }
        }
        let y = self.fin.apply(&x, n, pool);
        let at = ids.iter().position(|&i| i == eos).unwrap_or(n - 1);
        y[at * d..(at + 1) * d].to_vec()
    }
}
