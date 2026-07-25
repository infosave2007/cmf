//! Gemma-2 text ENCODER forward (the Lumina-Image 2.0 prompt encoder):
//! token ids → per-token hidden states of every layer.
//!
//! Second increment of the image-generation runtime
//! (docs/GENERATIVE.ru.md). A deliberately standalone, f32,
//! full-sequence forward — no KV cache, no sampling — loaded from a
//! diffusers/HF `text_encoder/` directory (config.json + sharded
//! safetensors). Gemma-2 specifics carried exactly: embedding scale
//! √hidden, RMSNorm with (1 + w), sandwich norms around attention AND
//! the GeGLU MLP, GQA with `query_pre_attn_scalar` scaling, attention
//! logit softcapping tanh(x/50)·50, RoPE θ=10000. Sliding-window
//! layers are exact for prompts shorter than the 4096 window (image
//! prompts are), enforced by an assert.
//!
//! Parity: `python/gemma_ref.py` + `tests/textenc_parity.rs` on the
//! real Lumina text-encoder weights.

use crate::dit::Proj;
use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::vae::{StTensor, read_safetensors};
use cortiq_core::CmfModel;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

struct Layer {
    input_norm: Vec<f32>,
    q: Proj, // [nh·hd, hidden]
    k: Proj, // [nkv·hd, hidden]
    v: Proj,
    o: Proj, // [hidden, nh·hd]
    post_attn_norm: Vec<f32>,
    pre_ffn_norm: Vec<f32>,
    gate: Proj, // [inter, hidden]
    up: Proj,
    down: Proj, // [hidden, inter]
    post_ffn_norm: Vec<f32>,
}

/// Gemma-2 encoder: exact f32 from a diffusers directory, or
/// CMF-quantized (mmap-resident, per-token embed dequant) from a
/// packaged file.
pub struct GemmaEncoder {
    embed: QTensor, // [vocab, hidden]
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    scale: f32, // 1/√query_pre_attn_scalar
    softcap: f32,
    theta: f32,
    eps: f64,
    window: usize,
}

fn rms_norm_gemma(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let n = x.len() as f64;
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| ((v as f64 * inv) as f32) * (1.0 + g))
        .collect()
}

fn gelu_tanh(v: f32) -> f32 {
    const C: f32 = 0.797_884_6; // √(2/π)
    0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh())
}

impl GemmaEncoder {
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("model.safetensors.index.json"))
                .map_err(|e| format!("index: {e}"))?,
        )
        .map_err(|e| format!("index: {e}"))?;
        let mut shards: Vec<String> = idx["weight_map"]
            .as_object()
            .ok_or("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shards.sort();
        shards.dedup();
        let mut t: HashMap<String, StTensor> = HashMap::new();
        for sh in &shards {
            t.extend(read_safetensors(&dir.join(sh))?);
        }
        // Some exports (the Lumina one) drop the "model." prefix.
        let take = |n: &str| -> Result<Vec<f32>, String> {
            t.get(n)
                .or_else(|| t.get(n.strip_prefix("model.").unwrap_or(n)))
                .map(|v| v.data.clone())
                .ok_or_else(|| format!("missing tensor {n}"))
        };
        let nl = cfg["num_hidden_layers"].as_u64().ok_or("layers")? as usize;
        let hidden = cfg["hidden_size"].as_u64().ok_or("hidden")? as usize;
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let p = format!("model.layers.{l}");
            let o = take(&format!("{p}.self_attn.o_proj.weight"))?;
            let o_cols = o.len() / hidden;
            let down = take(&format!("{p}.mlp.down_proj.weight"))?;
            let inter = down.len() / hidden;
            layers.push(Layer {
                input_norm: take(&format!("{p}.input_layernorm.weight"))?,
                q: Proj::f32(take(&format!("{p}.self_attn.q_proj.weight"))?, hidden),
                k: Proj::f32(take(&format!("{p}.self_attn.k_proj.weight"))?, hidden),
                v: Proj::f32(take(&format!("{p}.self_attn.v_proj.weight"))?, hidden),
                o: Proj::f32(o, o_cols),
                post_attn_norm: take(&format!("{p}.post_attention_layernorm.weight"))?,
                pre_ffn_norm: take(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                gate: Proj::f32(take(&format!("{p}.mlp.gate_proj.weight"))?, hidden),
                up: Proj::f32(take(&format!("{p}.mlp.up_proj.weight"))?, hidden),
                down: Proj::f32(down, inter),
                post_ffn_norm: take(&format!("{p}.post_feedforward_layernorm.weight"))?,
            });
        }
        let embed = take("model.embed_tokens.weight")?;
        let vocab = embed.len() / hidden;
        Ok(Self {
            embed: QTensor::from_f32(embed, vocab, hidden),
            layers,
            final_norm: take("model.norm.weight")?,
            pool: Pool::from_env(),
            hidden,
            nh: cfg["num_attention_heads"].as_u64().ok_or("nh")? as usize,
            nkv: cfg["num_key_value_heads"].as_u64().ok_or("nkv")? as usize,
            hd: cfg["head_dim"].as_u64().ok_or("hd")? as usize,
            scale: 1.0 / (cfg["query_pre_attn_scalar"].as_f64().unwrap_or(256.0) as f32).sqrt(),
            softcap: cfg["attn_logit_softcapping"].as_f64().unwrap_or(0.0) as f32,
            theta: cfg["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            window: cfg["sliding_window"].as_u64().unwrap_or(4096) as usize,
        })
    }

    /// Load from a packaged imagegen .cmf (`te.*` tensors +
    /// `te.config_json`). Quantized projections stay mmap-resident;
    /// embeddings dequantize per token.
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("te.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("te.config_json: {e}"))?;
        let f32v = |n: &str| -> Result<Vec<f32>, String> { crate::dit::cmf_f32(model, n) };
        let nl = cfg["num_hidden_layers"].as_u64().ok_or("layers")? as usize;
        let mut layers = Vec::with_capacity(nl);
        for l in 0..nl {
            let p = format!("te.layers.{l}");
            layers.push(Layer {
                input_norm: f32v(&format!("{p}.input_layernorm.weight"))?,
                q: Proj::from_model(model, &format!("{p}.self_attn.q_proj.weight"))?,
                k: Proj::from_model(model, &format!("{p}.self_attn.k_proj.weight"))?,
                v: Proj::from_model(model, &format!("{p}.self_attn.v_proj.weight"))?,
                o: Proj::from_model(model, &format!("{p}.self_attn.o_proj.weight"))?,
                post_attn_norm: f32v(&format!("{p}.post_attention_layernorm.weight"))?,
                pre_ffn_norm: f32v(&format!("{p}.pre_feedforward_layernorm.weight"))?,
                gate: Proj::from_model(model, &format!("{p}.mlp.gate_proj.weight"))?,
                up: Proj::from_model(model, &format!("{p}.mlp.up_proj.weight"))?,
                down: Proj::from_model(model, &format!("{p}.mlp.down_proj.weight"))?,
                post_ffn_norm: f32v(&format!("{p}.post_feedforward_layernorm.weight"))?,
            });
        }
        Ok(Self {
            embed: QTensor::from_model(model, "te.embed_tokens.weight")?,
            layers,
            final_norm: f32v("te.norm.weight")?,
            pool: Pool::from_env(),
            hidden: cfg["hidden_size"].as_u64().ok_or("hidden")? as usize,
            nh: cfg["num_attention_heads"].as_u64().ok_or("nh")? as usize,
            nkv: cfg["num_key_value_heads"].as_u64().ok_or("nkv")? as usize,
            hd: cfg["head_dim"].as_u64().ok_or("hd")? as usize,
            scale: 1.0 / (cfg["query_pre_attn_scalar"].as_f64().unwrap_or(256.0) as f32).sqrt(),
            softcap: cfg["attn_logit_softcapping"].as_f64().unwrap_or(0.0) as f32,
            theta: cfg["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
            eps: cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6),
            window: cfg["sliding_window"].as_u64().unwrap_or(4096) as usize,
        })
    }

    /// Full-sequence causal forward. Returns the FINAL-normed hidden
    /// states `[n, hidden]` and, if `keep_layer_inputs`, the residual
    /// stream entering every layer (what `hidden_states[i]` means in HF
    /// — index nl equals the pre-final-norm stream).
    pub fn encode(&self, ids: &[u32], keep_layer_inputs: bool) -> (Vec<f32>, Vec<Vec<f32>>) {
        let n = ids.len();
        assert!(
            n < self.window,
            "prompt of {n} tokens exceeds the sliding window {}",
            self.window
        );
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let emb_scale = (hs as f32).sqrt();
        let mut h = vec![0f32; n * hs];
        for (i, &id) in ids.iter().enumerate() {
            let row = &mut h[i * hs..(i + 1) * hs];
            self.embed.row_f32(id as usize, row);
            for v in row.iter_mut() {
                *v *= emb_scale;
            }
        }
        let mut streams = Vec::new();
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let hpk = nh / nkv;
        for layer in &self.layers {
            if keep_layer_inputs {
                streams.push(h.clone());
            }
            // ── attention (pre-norm, sandwich post-norm) ──
            let mut q_all = vec![0f32; n * nh * hd];
            let mut k_all = vec![0f32; n * nkv * hd];
            let mut v_all = vec![0f32; n * nkv * hd];
            let mut xn_all = vec![0f32; n * hs];
            for p in 0..n {
                xn_all[p * hs..(p + 1) * hs].copy_from_slice(&rms_norm_gemma(
                    &h[p * hs..(p + 1) * hs],
                    &layer.input_norm,
                    self.eps,
                ));
            }
            layer.q.matmat(&xn_all, n, &mut q_all, pool);
            layer.k.matmat(&xn_all, n, &mut k_all, pool);
            layer.v.matmat(&xn_all, n, &mut v_all, pool);
            // RoPE over the first hd dims of every head (full-dim rope).
            for (all, heads) in [(&mut q_all, nh), (&mut k_all, nkv)] {
                for p in 0..n {
                    for hh in 0..heads {
                        let v = &mut all[(p * heads + hh) * hd..(p * heads + hh + 1) * hd];
                        for i in 0..hd / 2 {
                            let freq = 1.0 / self.theta.powf(2.0 * i as f32 / hd as f32);
                            let (sin, cos) = (p as f32 * freq).sin_cos();
                            let (a, b) = (v[i], v[i + hd / 2]);
                            v[i] = a * cos - b * sin;
                            v[i + hd / 2] = a * sin + b * cos;
                        }
                    }
                }
            }
            let mut attn_out = vec![0f32; n * nh * hd];
            let mut row = vec![0f32; n];
            for hh in 0..nh {
                let kv = hh / hpk;
                for p in 0..n {
                    let qv = &q_all[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                    for (j, r) in row[..=p].iter_mut().enumerate() {
                        let kvv = &k_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                        let mut d = 0f32;
                        for (a, b) in qv.iter().zip(kvv) {
                            d += a * b;
                        }
                        let mut s = d * self.scale;
                        if self.softcap > 0.0 {
                            s = self.softcap * (s / self.softcap).tanh();
                        }
                        *r = s;
                    }
                    let mx = row[..=p].iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0f32;
                    for r in row[..=p].iter_mut() {
                        *r = (*r - mx).exp();
                        den += *r;
                    }
                    let inv = 1.0 / den;
                    let out = &mut attn_out[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                    for (j, &rw) in row[..=p].iter().enumerate() {
                        let vv = &v_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                        for (o, s) in out.iter_mut().zip(vv) {
                            *o += rw * inv * s;
                        }
                    }
                }
            }
            let mut proj_all = vec![0f32; n * hs];
            layer.o.matmat(&attn_out, n, &mut proj_all, pool);
            for p in 0..n {
                let post = rms_norm_gemma(
                    &proj_all[p * hs..(p + 1) * hs],
                    &layer.post_attn_norm,
                    self.eps,
                );
                for (dst, v) in h[p * hs..(p + 1) * hs].iter_mut().zip(&post) {
                    *dst += v;
                }
            }
            // ── GeGLU MLP (pre-norm, sandwich post-norm) ──
            let inter = layer.gate.rows();
            for p in 0..n {
                xn_all[p * hs..(p + 1) * hs].copy_from_slice(&rms_norm_gemma(
                    &h[p * hs..(p + 1) * hs],
                    &layer.pre_ffn_norm,
                    self.eps,
                ));
            }
            let mut g_all = vec![0f32; n * inter];
            let mut u_all = vec![0f32; n * inter];
            layer.gate.matmat(&xn_all, n, &mut g_all, pool);
            layer.up.matmat(&xn_all, n, &mut u_all, pool);
            for (g, u) in g_all.iter_mut().zip(&u_all) {
                *g = gelu_tanh(*g) * u;
            }
            let mut d_all = vec![0f32; n * hs];
            layer.down.matmat(&g_all, n, &mut d_all, pool);
            for p in 0..n {
                let post =
                    rms_norm_gemma(&d_all[p * hs..(p + 1) * hs], &layer.post_ffn_norm, self.eps);
                for (dst, v) in h[p * hs..(p + 1) * hs].iter_mut().zip(&post) {
                    *dst += v;
                }
            }
        }
        if keep_layer_inputs {
            streams.push(h.clone());
        }
        let mut out = Vec::with_capacity(n * hs);
        for p in 0..n {
            out.extend(rms_norm_gemma(
                &h[p * hs..(p + 1) * hs],
                &self.final_norm,
                self.eps,
            ));
        }
        (out, streams)
    }
}
