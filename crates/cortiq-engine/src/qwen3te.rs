//! Qwen3-VL as MiniMax-H3's prompt encoder: token ids → the
//! unnormalized hidden state after layer 50.
//!
//! The conditioning checkpoint is the 32 B model truncated at layer 50,
//! and the DiT consumes that stream directly — no final norm, no LM
//! head, and no chat template either: the H3 presentation is raw prompt
//! text with no special tokens at all.
//!
//! Same shape as the Gemma encoder next door and deliberately as plain:
//! a full-sequence causal forward, no KV cache, no sampling. Qwen3
//! specifics carried exactly — per-head RMSNorm on q and k BEFORE the
//! rotation, GQA 64/8, split-half RoPE at θ=5e6, SwiGLU, plain-w
//! RMSNorm (not Gemma's 1+w) and no embedding scale.
//!
//! Text-only. Qwen3-VL's interleaved MRoPE assigns some frequency slots
//! to the h and w axes, but a text token's three axis positions are
//! equal, so every slot sees the same angle and the interleave is a
//! no-op — which is why this can be one plain position per token. The
//! moment an image enters the presentation that stops being true.

use crate::dit::Proj;
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::CmfModel;
use std::sync::Arc;

struct Layer {
    input_norm: Vec<f32>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    q_norm: Vec<f32>, // [head_dim]
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    gate: Proj,
    up: Proj,
    down: Proj,
}

pub struct Qwen3Encoder {
    embed: QTensor,
    layers: Vec<Layer>,
    final_norm: Option<Vec<f32>>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    theta: f32,
    eps: f64,
}

fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

impl Qwen3Encoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model.tensor_bytes("te.config_json").map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("te.config_json: {e}"))?;
        let u = |k: &str, d: usize| cfg[k].as_u64().map(|v| v as usize).unwrap_or(d);
        let nl = u("num_hidden_layers", 0);
        let f32v = |n: &str| crate::dit::cmf_f32(model, n);
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let p = format!("te.layers.{l}");
            layers.push(Layer {
                input_norm: f32v(&format!("{p}.input_layernorm.weight"))?,
                q: Proj::from_model(model, &format!("{p}.self_attn.q_proj.weight"))?,
                k: Proj::from_model(model, &format!("{p}.self_attn.k_proj.weight"))?,
                v: Proj::from_model(model, &format!("{p}.self_attn.v_proj.weight"))?,
                o: Proj::from_model(model, &format!("{p}.self_attn.o_proj.weight"))?,
                q_norm: f32v(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: f32v(&format!("{p}.self_attn.k_norm.weight"))?,
                post_attn_norm: f32v(&format!("{p}.post_attention_layernorm.weight"))?,
                gate: Proj::from_model(model, &format!("{p}.mlp.gate_proj.weight"))?,
                up: Proj::from_model(model, &format!("{p}.mlp.up_proj.weight"))?,
                down: Proj::from_model(model, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        Ok(Self {
            embed: QTensor::from_model(model, "te.embed_tokens.weight")?,
            layers,
            final_norm: match cfg["final_norm"].as_bool() {
                Some(false) | None => None,
                Some(true) => Some(f32v("te.norm.weight")?),
            },
            pool: Pool::from_env(),
            hidden: u("hidden_size", 0),
            nh: u("num_attention_heads", 0),
            nkv: u("num_key_value_heads", 0),
            hd: u("head_dim", 128),
            theta: cfg["rope_theta"].as_f64().unwrap_or(5e6) as f32,
            eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
        })
    }

    /// Per-head RMSNorm then split-half RoPE over the whole head, in
    /// place across a `[n, heads·hd]` buffer.
    fn norm_rope(&self, all: &mut [f32], n: usize, heads: usize, w: &[f32]) {
        let hd = self.hd;
        let half = hd / 2;
        let pool = self.pool.as_deref();
        let ptr = SendPtr(all.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            for p in lo..hi {
                for h in 0..heads {
                    // SAFETY: workers own disjoint token ranges.
                    let x = unsafe { ptr.row((p * heads + h) * hd, hd) };
                    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hd as f64;
                    let inv = 1.0 / (ss + self.eps).sqrt();
                    for (d, &g) in x.iter_mut().zip(w) {
                        *d = (*d as f64 * inv) as f32 * g;
                    }
                    for i in 0..half {
                        let freq = 1.0 / self.theta.powf(2.0 * i as f32 / hd as f32);
                        let (s, c) = (p as f32 * freq).sin_cos();
                        let (a, b) = (x[i], x[i + half]);
                        x[i] = a * c - b * s;
                        x[i + half] = a * s + b * c;
                    }
                }
            }
        };
        match pool {
            Some(pl) => pl.run_rows(n, &work),
            None => work(0, n),
        }
    }

    /// Causal full-sequence forward. Returns `[n, hidden]` — the
    /// residual stream leaving the last layer, normed only if the
    /// config says the checkpoint has a final norm (H3's does not).
    pub fn encode(&self, ids: &[u32]) -> Vec<f32> {
        let n = ids.len();
        let hs = self.hidden;
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let hpk = nh / nkv;
        let pool = self.pool.as_deref();
        let scale = 1.0 / (hd as f32).sqrt();

        let mut h = vec![0f32; n * hs];
        for (i, &id) in ids.iter().enumerate() {
            self.embed.row_f32(id as usize, &mut h[i * hs..(i + 1) * hs]);
        }
        let mut xn = vec![0f32; n * hs];
        let mut q_all = vec![0f32; n * nh * hd];
        let mut k_all = vec![0f32; n * nkv * hd];
        let mut v_all = vec![0f32; n * nkv * hd];
        let mut attn = vec![0f32; n * nh * hd];
        let mut proj = vec![0f32; n * hs];

        for layer in &self.layers {
            for (o, src) in xn.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(src, &layer.input_norm, self.eps, o);
            }
            layer.q.matmat(&xn, n, &mut q_all, pool);
            layer.k.matmat(&xn, n, &mut k_all, pool);
            layer.v.matmat(&xn, n, &mut v_all, pool);
            self.norm_rope(&mut q_all, n, nh, &layer.q_norm);
            self.norm_rope(&mut k_all, n, nkv, &layer.k_norm);

            attn.fill(0.0);
            let ap = SendPtr(attn.as_mut_ptr());
            let heads = |lo: usize, hi: usize| {
                let mut row = vec![0f32; n];
                for hh in lo..hi {
                    let kv = hh / hpk;
                    for p in 0..n {
                        let qv = &q_all[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                        for (j, r) in row[..=p].iter_mut().enumerate() {
                            let kvv = &k_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                            *r = qv.iter().zip(kvv).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                        }
                        let mx = row[..=p].iter().cloned().fold(f32::MIN, f32::max);
                        let mut den = 0f32;
                        for r in row[..=p].iter_mut() {
                            *r = (*r - mx).exp();
                            den += *r;
                        }
                        let inv = 1.0 / den;
                        // SAFETY: workers own disjoint head ranges, and
                        // one head's slice is disjoint per token.
                        let out = unsafe { ap.row((p * nh + hh) * hd, hd) };
                        for (j, &rw) in row[..=p].iter().enumerate() {
                            let vv = &v_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                            let wgt = rw * inv;
                            for (o, &s) in out.iter_mut().zip(vv) {
                                *o += wgt * s;
                            }
                        }
                    }
                }
            };
            match pool {
                Some(pl) => pl.run_rows(nh, &heads),
                None => heads(0, nh),
            }

            layer.o.matmat(&attn, n, &mut proj, pool);
            for (d, &v) in h.iter_mut().zip(&proj) {
                *d += v;
            }

            for (o, src) in xn.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(src, &layer.post_attn_norm, self.eps, o);
            }
            let inter = layer.gate.rows();
            let mut g = vec![0f32; n * inter];
            let mut u = vec![0f32; n * inter];
            layer.gate.matmat(&xn, n, &mut g, pool);
            layer.up.matmat(&xn, n, &mut u, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = silu(*a) * b;
            }
            layer.down.matmat(&g, n, &mut proj, pool);
            for (d, &v) in h.iter_mut().zip(&proj) {
                *d += v;
            }
        }
        if let Some(w) = &self.final_norm {
            let mut out = vec![0f32; n * hs];
            for (o, src) in out.chunks_exact_mut(hs).zip(h.chunks_exact(hs)) {
                rms_norm_into(src, w, self.eps, o);
            }
            return out;
        }
        h
    }
}
