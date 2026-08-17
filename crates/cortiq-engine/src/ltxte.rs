//! The LTX-2.5 prompt encoder: Gemma-4 12B, the two aggregate projections,
//! and the embeddings connectors — everything between a prompt string and
//! the context the DiT cross-attends to.
//!
//! Three stages, all read from one `ltx-2.5-av` container:
//!
//! 1. **Gemma-4 12B** (`te.model.*`), 48 layers over a 1024-token window.
//!    Two layer kinds alternate: forty *sliding* layers (head 256, 16 query
//!    heads over 8 key heads, θ = 10⁴) and eight *full* layers every sixth
//!    (head 512, 16 query heads over one key head, θ = 10⁶ with only the
//!    first quarter of each head rotated, and the value projection **is**
//!    the key projection). Attention is unscaled — the q/k RMS-norms carry
//!    the scale — and every layer ends multiplied by its `layer_scalar`.
//! 2. **The aggregate projections** (`te.text_embedding_projection.*`).
//!    The features are not the last hidden state: all forty-nine layer
//!    outputs are RMS-normalized per token *per layer*, concatenated into
//!    188160 numbers, rescaled by `sqrt(out_dim / hidden)`, and projected —
//!    once to 4096 for video, once to 2048 for audio.
//! 3. **The connectors** (`dit.{video,audio}_embeddings_connector.*`), eight
//!    gated-attention blocks with 1-D RoPE. Padded positions are replaced by
//!    128 learnable registers tiled across the window first, which is why
//!    the DiT needs no prompt mask: after the substitution every position
//!    carries signal.

use crate::ltxdit::{Attn, Lin, Rope, gelu_tanh, rms_plain, rows};
use crate::pool::Pool;
use crate::qtensor::QTensor;
use cortiq_core::CmfModel;
use std::sync::Arc;

const EPS: f64 = 1e-6;

/// RMS normalization with a plain (not `1 + w`) learned weight — Gemma-4's,
/// unlike Gemma-2's and Gemma-3's.
fn rms(x: &[f32], w: &[f32], dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + EPS).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

/// RMS normalization with no learned weight (the value norm).
fn rms_nw(x: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + EPS).sqrt();
    for v in x.iter_mut() {
        *v = (*v as f64 * inv) as f32;
    }
}

/// Half-rotation RoPE tables: `cos`/`sin` of `[T, head_dim/2]`, applied as
/// `x·cos + rotate_half(x)·sin`.
struct Rot {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Rot {
    /// `inv_freq[j] = base^(-2j/head_dim)` for the rotated prefix, zero for
    /// the rest — a zero frequency is an identity rotation, which is how the
    /// "proportional" variant leaves three quarters of a full-attention head
    /// unrotated.
    fn build(seq: usize, head_dim: usize, base: f64, rotary: f64) -> Rot {
        let half = head_dim / 2;
        let rope_angles = (rotary * head_dim as f64 / 2.0) as usize;
        let inv: Vec<f64> = (0..half)
            .map(|j| {
                if j < rope_angles {
                    1.0 / base.powf((2 * j) as f64 / head_dim as f64)
                } else {
                    0.0
                }
            })
            .collect();
        let mut cos = vec![0f32; seq * half];
        let mut sin = vec![0f32; seq * half];
        for p in 0..seq {
            for (j, &f) in inv.iter().enumerate() {
                let a = p as f64 * f;
                cos[p * half + j] = a.cos() as f32;
                sin[p * half + j] = a.sin() as f32;
            }
        }
        Rot { cos, sin, half }
    }

    fn apply(&self, t: usize, row: &mut [f32]) {
        let h = self.half;
        let (c, s) = (&self.cos[t * h..(t + 1) * h], &self.sin[t * h..(t + 1) * h]);
        for i in 0..h {
            let (a, b) = (row[i], row[i + h]);
            row[i] = a * c[i] - b * s[i];
            row[i + h] = b * c[i] + a * s[i];
        }
    }
}

struct GemmaLayer {
    q: Lin,
    k: Lin,
    v: Option<Lin>,
    o: Lin,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    in_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    pre_ff_norm: Vec<f32>,
    post_ff_norm: Vec<f32>,
    gate: Lin,
    up: Lin,
    down: Lin,
    scalar: f32,
    head_dim: usize,
    q_heads: usize,
    kv_heads: usize,
    sliding: bool,
}

/// The full prompt encoder.
pub struct LtxTextEncoder {
    embed: QTensor,
    layers: Vec<GemmaLayer>,
    norm: Vec<f32>,
    video_agg: Lin,
    audio_agg: Lin,
    v_conn: Connector,
    a_conn: Connector,
    hidden: usize,
    embed_scale: f32,
    pub max_len: usize,
    pub bos: u32,
    pub pad: u32,
}

struct Connector {
    blocks: Vec<(Attn, Lin, Lin)>,
    registers: Vec<f32>,
    dim: usize,
    heads: usize,
    dh: usize,
    max_pos: f64,
}

fn vecf(model: &Arc<CmfModel>, name: &str) -> Result<Vec<f32>, String> {
    crate::dit::cmf_f32(model, name)
}

impl LtxTextEncoder {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<LtxTextEncoder, String> {
        let cfg_bytes = ["te.gemma_config_json", "ltx.gemma_config_json"]
            .iter()
            .find_map(|n| model.tensor(n).map(|e| model.entry_bytes(e)));
        let cfg: serde_json::Value = match cfg_bytes {
            Some(b) => serde_json::from_slice(b).map_err(|e| format!("gemma config: {e}"))?,
            None => serde_json::Value::Null,
        };
        let tc = cfg.get("text_config").cloned().unwrap_or(serde_json::Value::Null);
        let g = |k: &str, d: f64| tc.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
        let hidden = g("hidden_size", 3840.0) as usize;
        let n_layers = g("num_hidden_layers", 48.0) as usize;
        let head_dim = g("head_dim", 256.0) as usize;
        let global_head_dim = g("global_head_dim", 512.0) as usize;
        let types: Vec<String> = tc
            .get("layer_types")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|s| s.as_str().unwrap_or("sliding_attention").to_string()).collect())
            .unwrap_or_else(|| {
                // the released layout: every sixth layer is a full-attention one
                (0..n_layers)
                    .map(|i| {
                        if i % 6 == 5 { "full_attention".into() } else { "sliding_attention".into() }
                    })
                    .collect()
            });

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("te.model.layers.{i}");
            let sliding = types[i] != "full_attention";
            let hd = if sliding { head_dim } else { global_head_dim };
            let q = Lin::load(model, &format!("{p}.self_attn.q_proj"), false)?;
            let k = Lin::load(model, &format!("{p}.self_attn.k_proj"), false)?;
            let v = match model.tensor(&format!("{p}.self_attn.v_proj.weight")) {
                Some(_) => Some(Lin::load(model, &format!("{p}.self_attn.v_proj"), false)?),
                None => None,
            };
            let q_rows = model
                .tensor(&format!("{p}.self_attn.q_proj.weight"))
                .ok_or_else(|| format!("missing {p}.self_attn.q_proj.weight"))?
                .shape[0];
            let k_rows = model
                .tensor(&format!("{p}.self_attn.k_proj.weight"))
                .ok_or("missing k_proj")?
                .shape[0];
            layers.push(GemmaLayer {
                q,
                k,
                v,
                o: Lin::load(model, &format!("{p}.self_attn.o_proj"), false)?,
                q_norm: vecf(model, &format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: vecf(model, &format!("{p}.self_attn.k_norm.weight"))?,
                in_norm: vecf(model, &format!("{p}.input_layernorm.weight"))?,
                post_attn_norm: vecf(model, &format!("{p}.post_attention_layernorm.weight"))?,
                pre_ff_norm: vecf(model, &format!("{p}.pre_feedforward_layernorm.weight"))?,
                post_ff_norm: vecf(model, &format!("{p}.post_feedforward_layernorm.weight"))?,
                gate: Lin::load(model, &format!("{p}.mlp.gate_proj"), false)?,
                up: Lin::load(model, &format!("{p}.mlp.up_proj"), false)?,
                down: Lin::load(model, &format!("{p}.mlp.down_proj"), false)?,
                scalar: vecf(model, &format!("{p}.layer_scalar"))?[0],
                head_dim: hd,
                q_heads: q_rows / hd,
                kv_heads: k_rows / hd,
                sliding,
            });
        }

        let conn = |prefix: &str, dim: usize, heads: usize, dh: usize| -> Result<Connector, String> {
            let mut blocks = Vec::new();
            let mut i = 0usize;
            while model
                .tensor(&format!("{prefix}.transformer_1d_blocks.{i}.attn1.to_q.weight"))
                .is_some()
            {
                let p = format!("{prefix}.transformer_1d_blocks.{i}");
                blocks.push((
                    Attn::load(model, &format!("{p}.attn1"), heads, dh)?,
                    Lin::load(model, &format!("{p}.ff.net.0.proj"), true)?,
                    Lin::load(model, &format!("{p}.ff.net.2"), true)?,
                ));
                i += 1;
            }
            Ok(Connector {
                blocks,
                registers: vecf(model, &format!("{prefix}.learnable_registers"))?,
                dim,
                heads,
                dh,
                max_pos: 4096.0,
            })
        };

        Ok(LtxTextEncoder {
            embed: QTensor::from_model(model, "te.model.embed_tokens.weight")?,
            layers,
            norm: vecf(model, "te.model.norm.weight")?,
            video_agg: Lin::load(model, "te.text_embedding_projection.video_aggregate_embed", true)?,
            audio_agg: Lin::load(model, "te.text_embedding_projection.audio_aggregate_embed", true)?,
            v_conn: conn("dit.video_embeddings_connector", 4096, 32, 128)?,
            a_conn: conn("dit.audio_embeddings_connector", 2048, 32, 64)?,
            hidden,
            embed_scale: (hidden as f64).sqrt() as f32,
            max_len: 1024,
            bos: tc.get("bos_token_id").and_then(|v| v.as_u64()).unwrap_or(2) as u32,
            pad: tc.get("pad_token_id").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
        })
    }

    /// Left-pad the prompt to the encoder's window, prepending BOS — the
    /// layout the reference tokenizer produces.
    pub fn pad_ids(&self, ids: &[u32]) -> (Vec<u32>, Vec<f32>) {
        let mut body = Vec::with_capacity(self.max_len);
        if ids.first() != Some(&self.bos) {
            body.push(self.bos);
        }
        body.extend_from_slice(ids);
        body.truncate(self.max_len);
        let padlen = self.max_len - body.len();
        let mut out = vec![self.pad; padlen];
        out.extend_from_slice(&body);
        let mut mask = vec![0f32; padlen];
        mask.extend(std::iter::repeat_n(1f32, body.len()));
        (out, mask)
    }

    /// Every one of the 49 hidden states, in HF's order: the scaled
    /// embedding first, then each layer's output.
    pub fn hidden_states(&self, ids: &[u32], mask: &[f32], pool: Option<&Pool>) -> Vec<Vec<f32>> {
        let t = ids.len();
        let d = self.hidden;
        let mut x = vec![0f32; t * d];
        for (i, &id) in ids.iter().enumerate() {
            self.embed.row_f32(id as usize, &mut x[i * d..(i + 1) * d]);
            for v in x[i * d..(i + 1) * d].iter_mut() {
                *v *= self.embed_scale;
            }
        }
        let mut out = vec![x.clone()];
        // one rotation table per (head_dim, theta) pair in use
        let rot_slide = Rot::build(t, self.layers[0].head_dim, 10000.0, 1.0);
        let full = self.layers.iter().find(|l| !l.sliding);
        let rot_full = full.map(|l| Rot::build(t, l.head_dim, 1_000_000.0, 0.25));

        for layer in &self.layers {
            let rot = if layer.sliding { &rot_slide } else { rot_full.as_ref().unwrap() };
            let mut h = vec![0f32; t * d];
            for i in 0..t {
                rms(&x[i * d..(i + 1) * d], &layer.in_norm, &mut h[i * d..(i + 1) * d]);
            }
            let attn = self.attention(layer, &h, t, mask, rot, pool);
            for i in 0..t {
                let mut n = vec![0f32; d];
                rms(&attn[i * d..(i + 1) * d], &layer.post_attn_norm, &mut n);
                for (v, &a) in x[i * d..(i + 1) * d].iter_mut().zip(&n) {
                    *v += a;
                }
            }
            let mut h2 = vec![0f32; t * d];
            for i in 0..t {
                rms(&x[i * d..(i + 1) * d], &layer.pre_ff_norm, &mut h2[i * d..(i + 1) * d]);
            }
            let mut g = layer.gate.apply(&h2, t, pool);
            let u = layer.up.apply(&h2, t, pool);
            for (a, &b) in g.iter_mut().zip(&u) {
                *a = gelu_tanh(*a) * b;
            }
            let ff = layer.down.apply(&g, t, pool);
            for i in 0..t {
                let mut n = vec![0f32; d];
                rms(&ff[i * d..(i + 1) * d], &layer.post_ff_norm, &mut n);
                for (v, &a) in x[i * d..(i + 1) * d].iter_mut().zip(&n) {
                    *v += a;
                }
            }
            for v in x.iter_mut() {
                *v *= layer.scalar;
            }
            out.push(x.clone());
        }
        // HF's hidden-state tuple is (embedding, layer 0 … layer 46,
        // norm(layer 47)): the last entry is the *normalized* final state,
        // not the raw one. The feature extractor reads all forty-nine, so
        // getting this wrong poisons a forty-ninth of every token's
        // features — and it is the entry with the largest magnitude.
        if let Some(last) = out.last_mut() {
            let mut n = vec![0f32; t * d];
            for i in 0..t {
                rms(&last[i * d..(i + 1) * d], &self.norm, &mut n[i * d..(i + 1) * d]);
            }
            *last = n;
        }
        out
    }

    fn attention(
        &self,
        l: &GemmaLayer,
        h: &[f32],
        t: usize,
        mask: &[f32],
        rot: &Rot,
        pool: Option<&Pool>,
    ) -> Vec<f32> {
        let hd = l.head_dim;
        let qi = l.q_heads * hd;
        let ki = l.kv_heads * hd;
        let mut q = l.q.apply(h, t, pool);
        let mut k = l.k.apply(h, t, pool);
        let mut v = match &l.v {
            Some(p) => p.apply(h, t, pool),
            None => k.clone(),
        };
        for i in 0..t {
            for hh in 0..l.q_heads {
                let r = &mut q[i * qi + hh * hd..i * qi + (hh + 1) * hd];
                let mut n = vec![0f32; hd];
                rms(r, &l.q_norm, &mut n);
                r.copy_from_slice(&n);
                rot.apply(i, r);
            }
            for hh in 0..l.kv_heads {
                let r = &mut k[i * ki + hh * hd..i * ki + (hh + 1) * hd];
                let mut n = vec![0f32; hd];
                rms(r, &l.k_norm, &mut n);
                r.copy_from_slice(&n);
                rot.apply(i, r);
                rms_nw(&mut v[i * ki + hh * hd..i * ki + (hh + 1) * hd]);
            }
        }
        // causal, with the sliding window and the prompt's left padding
        let window = if l.sliding { 1024usize } else { usize::MAX };
        let mut out = vec![0f32; t * qi];
        let dst = crate::ltxdit::Shared(out.as_mut_ptr());
        rows(pool, t, &|s, e| {
            let mut sc = vec![0f32; t];
            for i in s..e {
                let orow = unsafe { dst.at(i * qi, qi) };
                for hh in 0..l.q_heads {
                    let kv = hh * l.kv_heads / l.q_heads;
                    let qh = &q[i * qi + hh * hd..][..hd];
                    for (j, sj) in sc.iter_mut().enumerate() {
                        if j > i || (window != usize::MAX && i - j >= window) || mask[j] == 0.0 {
                            *sj = f32::NEG_INFINITY;
                            continue;
                        }
                        let kh = &k[j * ki + kv * hd..][..hd];
                        *sj = qh.iter().zip(kh).map(|(&a, &b)| a * b).sum::<f32>();
                    }
                    // a fully masked row (a left-pad position) attends nowhere
                    if sc.iter().all(|v| *v == f32::NEG_INFINITY) {
                        for d in orow[hh * hd..(hh + 1) * hd].iter_mut() {
                            *d = 0.0;
                        }
                        continue;
                    }
                    crate::ltxdit::softmax(&mut sc);
                    let oh = &mut orow[hh * hd..(hh + 1) * hd];
                    for d in oh.iter_mut() {
                        *d = 0.0;
                    }
                    for (j, &p) in sc.iter().enumerate() {
                        if p == 0.0 {
                            continue;
                        }
                        let vh = &v[j * ki + kv * hd..][..hd];
                        for d in 0..hd {
                            oh[d] += p * vh[d];
                        }
                    }
                }
            }
        });
        l.o.apply(&out, t, pool)
    }

    /// Hidden states → the two context tensors the DiT reads.
    pub fn encode_ids(
        &self,
        ids: &[u32],
        mask: &[f32],
        pool: Option<&Pool>,
    ) -> (Vec<f32>, Vec<f32>, usize) {
        let hs = self.hidden_states(ids, mask, pool);
        let (t, d, l) = (ids.len(), self.hidden, hs.len());
        // per-token, per-layer RMS over the hidden dimension, concatenated
        // layer-last: [T, d·L]
        let mut feats = vec![0f32; t * d * l];
        for (li, layer) in hs.iter().enumerate() {
            for i in 0..t {
                let row = &layer[i * d..(i + 1) * d];
                let var = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / d as f64;
                let inv = 1.0 / (var + 1e-6).sqrt();
                let keep = mask[i] != 0.0;
                for j in 0..d {
                    feats[i * d * l + j * l + li] = if keep { (row[j] as f64 * inv) as f32 } else { 0.0 };
                }
            }
        }
        // The aggregate projection is 4096x188160 — one weight buffer past
        // what a GPU binding may address, and it runs twice per prompt, not
        // per step. Keep it on the CPU rather than teach the device to page
        // a 2 GiB binding for a millisecond of work.
        let project = |agg: &Lin, out_dim: usize| -> Vec<f32> {
            let scale = ((out_dim as f64) / (d as f64)).sqrt() as f32;
            let scaled: Vec<f32> = feats.iter().map(|&v| v * scale).collect();
            crate::gpu::cpu_scope(|| agg.apply(&scaled, t, pool))
        };
        let vfeat = project(&self.video_agg, 4096);
        let afeat = project(&self.audio_agg, 2048);
        // the connectors want valid tokens first; the prompt arrives left-padded
        let order: Vec<usize> = (0..t)
            .filter(|&i| mask[i] != 0.0)
            .chain((0..t).filter(|&i| mask[i] == 0.0))
            .collect();
        let valid = mask.iter().filter(|&&m| m != 0.0).count();
        let reorder = |x: &[f32], dim: usize| -> Vec<f32> {
            let mut o = vec![0f32; t * dim];
            for (new, &old) in order.iter().enumerate() {
                o[new * dim..(new + 1) * dim].copy_from_slice(&x[old * dim..(old + 1) * dim]);
            }
            o
        };
        let v = self.v_conn.run(&reorder(&vfeat, 4096), t, valid, pool);
        let a = self.a_conn.run(&reorder(&afeat, 2048), t, valid, pool);
        (v, a, t)
    }
}

impl Connector {
    /// Eight gated-attention blocks over the whole window. Padded positions
    /// are first replaced by the learnable registers, tiled across the
    /// window, which is what makes the mask vanish: after the substitution
    /// every position is signal and attention is unmasked.
    fn run(&self, x: &[f32], t: usize, valid: usize, pool: Option<&Pool>) -> Vec<f32> {
        let d = self.dim;
        let regs = self.registers.len() / d;
        let mut h = x.to_vec();
        for i in valid..t {
            let r = i % regs;
            h[i * d..(i + 1) * d].copy_from_slice(&self.registers[r * d..(r + 1) * d]);
        }
        let pos: Vec<Vec<f64>> = (0..t).map(|i| vec![i as f64]).collect();
        let pe = Rope::build(&pos, &[self.max_pos], d, self.heads, 10000.0);
        for (attn, ff_in, ff_out) in &self.blocks {
            let mut n = vec![0f32; t * d];
            for i in 0..t {
                rms_plain(&h[i * d..(i + 1) * d], &mut n[i * d..(i + 1) * d]);
            }
            let a = attn.forward(&n, t, &n, t, Some(&pe), Some(&pe), None, pool);
            for (v, &y) in h.iter_mut().zip(&a) {
                *v += y;
            }
            let mut n2 = vec![0f32; t * d];
            for i in 0..t {
                rms_plain(&h[i * d..(i + 1) * d], &mut n2[i * d..(i + 1) * d]);
            }
            let mut g = ff_in.apply(&n2, t, pool);
            for v in g.iter_mut() {
                *v = gelu_tanh(*v);
            }
            let f = ff_out.apply(&g, t, pool);
            for (v, &y) in h.iter_mut().zip(&f) {
                *v += y;
            }
        }
        let mut out = vec![0f32; t * d];
        for i in 0..t {
            rms_plain(&h[i * d..(i + 1) * d], &mut out[i * d..(i + 1) * d]);
        }
        let _ = self.dh;
        out
    }
}
