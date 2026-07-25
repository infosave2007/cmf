//! Lumina2 Next-DiT forward (the Lumina-Image 2.0 denoiser):
//! noisy latent + caption features + timestep → velocity prediction.
//!
//! Third increment of the image-generation runtime
//! (docs/GENERATIVE.ru.md). Standalone f32 forward loaded from a
//! diffusers `transformer/` directory, mirroring
//! Lumina2Transformer2DModel exactly: plain-w RMSNorm (not Gemma's
//! 1+w), per-head qk-norm, 3-axis complex-interleaved RoPE θ=10000
//! (caption tokens advance axis 0, image tokens sit at axis0=cap_len
//! with row/col on axes 1/2), AdaLN modulation with tanh gates from
//! the 1024-d timestep embedding, sandwich RMSNorms, SwiGLU FFN, and
//! a final LayerNorm(eps 1e-6, no affine) scaled by (1+scale) before
//! the patch projection. Attention is full/bidirectional.
//!
//! Parity: `python/nextdit_ref.py` + `tests/dit_parity.rs` on the
//! real Lumina transformer weights.

use crate::pool::Pool;
use crate::qtensor::QTensor;
use crate::vae::{StTensor, read_safetensors};
use cortiq_core::CmfModel;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A projection weight: exact f32 (diffusers load — Accelerate GEMM)
/// or a CMF-quantized tensor on the engine's batched dot kernels.
pub(crate) enum Proj {
    F32 {
        w: Vec<f32>,
        rows: usize,
        cols: usize,
    },
    Q(QTensor),
}

impl Proj {
    /// f32 weight `[?, cols]`; rows derived from the data length.
    pub(crate) fn f32(w: Vec<f32>, cols: usize) -> Self {
        let rows = w.len() / cols;
        debug_assert_eq!(w.len(), rows * cols);
        Proj::F32 { w, rows, cols }
    }

    /// Load from a CMF directory entry (mmap-resident when quantized;
    /// an F32 entry dequantizes into the exact-GEMM arm).
    pub(crate) fn from_model(model: &Arc<CmfModel>, name: &str) -> Result<Self, String> {
        Ok(match QTensor::from_model(model, name)? {
            QTensor::F32 { data, rows, cols } => Proj::F32 {
                w: data,
                rows,
                cols,
            },
            q => Proj::Q(q),
        })
    }

    pub(crate) fn rows(&self) -> usize {
        match self {
            Proj::F32 { rows, .. } => *rows,
            Proj::Q(q) => q.rows(),
        }
    }

    /// y[b, rows] = x[b, cols] · Wᵀ.
    pub(crate) fn matmat(&self, xs: &[f32], b: usize, out: &mut [f32], pool: Option<&Pool>) {
        match self {
            Proj::F32 { w, rows, cols } => {
                crate::fcd_ops::gemm_nt(xs, w, out, b, *cols, *rows, pool)
            }
            Proj::Q(q) => q.matmat(xs, b, out, pool),
        }
    }
}

struct Block {
    /// AdaLN: (linear [4·hidden, 1024], bias [4·hidden]); None = plain norm1.
    modulation: Option<(Proj, Vec<f32>)>,
    norm1: Vec<f32>,
    q: Proj, // [nh·hd, hidden]
    k: Proj, // [nkv·hd, hidden]
    v: Proj,
    o: Proj,          // [hidden, nh·hd]
    norm_q: Vec<f32>, // [hd]
    norm_k: Vec<f32>,
    norm2: Vec<f32>,
    ffn_norm1: Vec<f32>,
    w1: Proj, // gate [inter, hidden]
    w3: Proj, // up
    w2: Proj, // down [hidden, inter]
    ffn_norm2: Vec<f32>,
}

/// Next-DiT: exact f32 from a diffusers directory, or CMF-quantized
/// (mmap-resident) from a packaged file.
pub struct NextDit {
    x_emb: Proj, // [hidden, p·p·c]
    x_emb_b: Vec<f32>,
    t_lin1_w: Vec<f32>, // [temb, 256]
    t_lin1_b: Vec<f32>,
    t_lin2_w: Vec<f32>, // [temb, temb]
    t_lin2_b: Vec<f32>,
    cap_norm: Vec<f32>, // [cap_feat]
    cap_w: Proj,        // [hidden, cap_feat]
    cap_b: Vec<f32>,
    context_refiner: Vec<Block>,
    noise_refiner: Vec<Block>,
    layers: Vec<Block>,
    out_lin1_w: Vec<f32>, // [hidden, temb]
    out_lin1_b: Vec<f32>,
    out_lin2: Proj, // [p·p·c, hidden]
    out_lin2_b: Vec<f32>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    pub in_channels: usize,
    pub patch: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    axes_dim: Vec<usize>,
    eps: f64,
}

/// diffusers RMSNorm: x·rsqrt(mean x²+eps) · w (plain w).
fn rms_norm(x: &[f32], w: &[f32], eps: f64) -> Vec<f32> {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    x.iter()
        .zip(w)
        .map(|(&v, &g)| (v as f64 * inv) as f32 * g)
        .collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// y = x·Wᵀ + b for a single row.
fn linear(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let k = x.len();
    b.iter()
        .enumerate()
        .map(|(o, &bias)| {
            let row = &w[o * k..(o + 1) * k];
            bias + row.iter().zip(x).map(|(&a, &c)| a * c).sum::<f32>()
        })
        .collect()
}

/// Any CMF directory entry → f32 (norms, biases, f16 conv kernels).
pub(crate) fn cmf_f32(model: &CmfModel, name: &str) -> Result<Vec<f32>, String> {
    let entry = model
        .tensor(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    let bytes = model.entry_bytes(entry);
    let mut out = vec![0f32; entry.shape.iter().product()];
    cortiq_core::quant::dequant_tensor(entry, bytes, &mut out)?;
    Ok(out)
}

/// Per-token RoPE table: interleaved-pair rotation angles, f64.
/// `ids` is [n, 3]; each axis contributes dim/2 frequencies.
fn rope_table(ids: &[[u32; 3]], axes_dim: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let pairs: usize = axes_dim.iter().sum::<usize>() / 2;
    let mut cos = Vec::with_capacity(ids.len() * pairs);
    let mut sin = Vec::with_capacity(ids.len() * pairs);
    for id in ids {
        for (a, &d) in axes_dim.iter().enumerate() {
            for j in 0..d / 2 {
                let freq = 1.0 / 10000f64.powf(2.0 * j as f64 / d as f64);
                let ang = id[a] as f64 * freq;
                cos.push(ang.cos());
                sin.push(ang.sin());
            }
        }
    }
    (cos, sin)
}

impl NextDit {
    pub fn load_dir(dir: &Path) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("diffusion_pytorch_model.safetensors.index.json"))
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
        let mut take = |n: String| -> Result<Vec<f32>, String> {
            t.remove(&n)
                .map(|v| v.data)
                .ok_or_else(|| format!("missing tensor {n}"))
        };
        let hidden = cfg["hidden_size"].as_u64().ok_or("hidden")? as usize;
        let mut blocks = |pfx: &str, count: usize, modulated: bool| -> Result<Vec<Block>, String> {
            (0..count)
                .map(|l| {
                    let p = format!("{pfx}.{l}");
                    let w1 = take(format!("{p}.feed_forward.linear_1.weight"))?;
                    let inter = w1.len() / hidden;
                    Ok(Block {
                        modulation: if modulated {
                            let mw = take(format!("{p}.norm1.linear.weight"))?;
                            let cols = mw.len() / (4 * hidden);
                            Some((Proj::f32(mw, cols), take(format!("{p}.norm1.linear.bias"))?))
                        } else {
                            None
                        },
                        norm1: if modulated {
                            take(format!("{p}.norm1.norm.weight"))?
                        } else {
                            take(format!("{p}.norm1.weight"))?
                        },
                        q: Proj::f32(take(format!("{p}.attn.to_q.weight"))?, hidden),
                        k: Proj::f32(take(format!("{p}.attn.to_k.weight"))?, hidden),
                        v: Proj::f32(take(format!("{p}.attn.to_v.weight"))?, hidden),
                        o: {
                            let o = take(format!("{p}.attn.to_out.0.weight"))?;
                            let cols = o.len() / hidden;
                            Proj::f32(o, cols)
                        },
                        norm_q: take(format!("{p}.attn.norm_q.weight"))?,
                        norm_k: take(format!("{p}.attn.norm_k.weight"))?,
                        norm2: take(format!("{p}.norm2.weight"))?,
                        ffn_norm1: take(format!("{p}.ffn_norm1.weight"))?,
                        w1: Proj::f32(w1, hidden),
                        w3: Proj::f32(take(format!("{p}.feed_forward.linear_3.weight"))?, hidden),
                        w2: Proj::f32(take(format!("{p}.feed_forward.linear_2.weight"))?, inter),
                        ffn_norm2: take(format!("{p}.ffn_norm2.weight"))?,
                    })
                })
                .collect()
        };
        let nl = cfg["num_layers"].as_u64().ok_or("num_layers")? as usize;
        let nr = cfg["num_refiner_layers"].as_u64().unwrap_or(2) as usize;
        let context_refiner = blocks("context_refiner", nr, false)?;
        let noise_refiner = blocks("noise_refiner", nr, true)?;
        let layers = blocks("layers", nl, true)?;
        let nh = cfg["num_attention_heads"].as_u64().ok_or("nh")? as usize;
        let in_channels = cfg["in_channels"].as_u64().ok_or("in_channels")? as usize;
        let patch = cfg["patch_size"].as_u64().unwrap_or(2) as usize;
        let axes_dim: Vec<usize> = cfg["axes_dim_rope"]
            .as_array()
            .ok_or("axes_dim_rope")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let cap_norm = take("time_caption_embed.caption_embedder.0.weight".into())?;
        let cap_feat = cap_norm.len();
        Ok(Self {
            x_emb: Proj::f32(
                take("x_embedder.weight".into())?,
                patch * patch * in_channels,
            ),
            x_emb_b: take("x_embedder.bias".into())?,
            t_lin1_w: take("time_caption_embed.timestep_embedder.linear_1.weight".into())?,
            t_lin1_b: take("time_caption_embed.timestep_embedder.linear_1.bias".into())?,
            t_lin2_w: take("time_caption_embed.timestep_embedder.linear_2.weight".into())?,
            t_lin2_b: take("time_caption_embed.timestep_embedder.linear_2.bias".into())?,
            cap_norm,
            cap_w: Proj::f32(
                take("time_caption_embed.caption_embedder.1.weight".into())?,
                cap_feat,
            ),
            cap_b: take("time_caption_embed.caption_embedder.1.bias".into())?,
            context_refiner,
            noise_refiner,
            layers,
            out_lin1_w: take("norm_out.linear_1.weight".into())?,
            out_lin1_b: take("norm_out.linear_1.bias".into())?,
            out_lin2: Proj::f32(take("norm_out.linear_2.weight".into())?, hidden),
            out_lin2_b: take("norm_out.linear_2.bias".into())?,
            pool: Pool::from_env(),
            hidden,
            in_channels,
            patch,
            nh,
            nkv: cfg["num_kv_heads"].as_u64().unwrap_or(nh as u64) as usize,
            hd: hidden / nh,
            axes_dim,
            eps: cfg["norm_eps"].as_f64().unwrap_or(1e-5),
        })
    }

    /// Load from a packaged imagegen .cmf (`dit.*` tensors +
    /// `dit.config_json`). Quantized projections stay mmap-resident.
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("dit.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("dit.config_json: {e}"))?;
        let f32v = |n: &str| -> Result<Vec<f32>, String> { cmf_f32(model, n) };
        let hidden = cfg["hidden_size"].as_u64().ok_or("hidden")? as usize;
        let blocks = |pfx: &str, count: usize, modulated: bool| -> Result<Vec<Block>, String> {
            (0..count)
                .map(|l| {
                    let p = format!("dit.{pfx}.{l}");
                    Ok(Block {
                        modulation: if modulated {
                            Some((
                                Proj::from_model(model, &format!("{p}.norm1.linear.weight"))?,
                                f32v(&format!("{p}.norm1.linear.bias"))?,
                            ))
                        } else {
                            None
                        },
                        norm1: if modulated {
                            f32v(&format!("{p}.norm1.norm.weight"))?
                        } else {
                            f32v(&format!("{p}.norm1.weight"))?
                        },
                        q: Proj::from_model(model, &format!("{p}.attn.to_q.weight"))?,
                        k: Proj::from_model(model, &format!("{p}.attn.to_k.weight"))?,
                        v: Proj::from_model(model, &format!("{p}.attn.to_v.weight"))?,
                        o: Proj::from_model(model, &format!("{p}.attn.to_out.0.weight"))?,
                        norm_q: f32v(&format!("{p}.attn.norm_q.weight"))?,
                        norm_k: f32v(&format!("{p}.attn.norm_k.weight"))?,
                        norm2: f32v(&format!("{p}.norm2.weight"))?,
                        ffn_norm1: f32v(&format!("{p}.ffn_norm1.weight"))?,
                        w1: Proj::from_model(model, &format!("{p}.feed_forward.linear_1.weight"))?,
                        w3: Proj::from_model(model, &format!("{p}.feed_forward.linear_3.weight"))?,
                        w2: Proj::from_model(model, &format!("{p}.feed_forward.linear_2.weight"))?,
                        ffn_norm2: f32v(&format!("{p}.ffn_norm2.weight"))?,
                    })
                })
                .collect()
        };
        let nl = cfg["num_layers"].as_u64().ok_or("num_layers")? as usize;
        let nr = cfg["num_refiner_layers"].as_u64().unwrap_or(2) as usize;
        let nh = cfg["num_attention_heads"].as_u64().ok_or("nh")? as usize;
        let axes_dim: Vec<usize> = cfg["axes_dim_rope"]
            .as_array()
            .ok_or("axes_dim_rope")?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        Ok(Self {
            x_emb: Proj::from_model(model, "dit.x_embedder.weight")?,
            x_emb_b: f32v("dit.x_embedder.bias")?,
            t_lin1_w: f32v("dit.time_caption_embed.timestep_embedder.linear_1.weight")?,
            t_lin1_b: f32v("dit.time_caption_embed.timestep_embedder.linear_1.bias")?,
            t_lin2_w: f32v("dit.time_caption_embed.timestep_embedder.linear_2.weight")?,
            t_lin2_b: f32v("dit.time_caption_embed.timestep_embedder.linear_2.bias")?,
            cap_norm: f32v("dit.time_caption_embed.caption_embedder.0.weight")?,
            cap_w: Proj::from_model(model, "dit.time_caption_embed.caption_embedder.1.weight")?,
            cap_b: f32v("dit.time_caption_embed.caption_embedder.1.bias")?,
            context_refiner: blocks("context_refiner", nr, false)?,
            noise_refiner: blocks("noise_refiner", nr, true)?,
            layers: blocks("layers", nl, true)?,
            out_lin1_w: f32v("dit.norm_out.linear_1.weight")?,
            out_lin1_b: f32v("dit.norm_out.linear_1.bias")?,
            out_lin2: Proj::from_model(model, "dit.norm_out.linear_2.weight")?,
            out_lin2_b: f32v("dit.norm_out.linear_2.bias")?,
            pool: Pool::from_env(),
            hidden,
            in_channels: cfg["in_channels"].as_u64().ok_or("in_channels")? as usize,
            patch: cfg["patch_size"].as_u64().unwrap_or(2) as usize,
            nh,
            nkv: cfg["num_kv_heads"].as_u64().unwrap_or(nh as u64) as usize,
            hd: hidden / nh,
            axes_dim,
            eps: cfg["norm_eps"].as_f64().unwrap_or(1e-5),
        })
    }

    /// Sinusoidal(256, cos-first) → 2-layer MLP → temb [1024].
    fn time_embed(&self, t: f32) -> Vec<f32> {
        const HALF: usize = 128;
        let mut freq = [0f32; 2 * HALF];
        for i in 0..HALF {
            let ang = t as f64 * (-(10000f64.ln()) * i as f64 / HALF as f64).exp();
            freq[i] = ang.cos() as f32;
            freq[HALF + i] = ang.sin() as f32;
        }
        let mut h = linear(&freq, &self.t_lin1_w, &self.t_lin1_b);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        linear(&h, &self.t_lin2_w, &self.t_lin2_b)
    }

    fn block_forward(
        &self,
        blk: &Block,
        x: &mut [f32],
        rope: &(Vec<f64>, Vec<f64>),
        temb: Option<&[f32]>,
    ) {
        let (hs, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let pool = self.pool.as_deref();
        let n = x.len() / hs;
        let modv = blk.modulation.as_ref().zip(temb).map(|((w, b), t)| {
            let s: Vec<f32> = t.iter().map(|&v| silu(v)).collect();
            let mut m = vec![0f32; w.rows()];
            w.matmat(&s, 1, &mut m, pool);
            for (v, &bias) in m.iter_mut().zip(b) {
                *v += bias;
            }
            m
        });
        let (s_msa, g_msa, s_mlp, g_mlp) = match &modv {
            Some(m) => (
                Some(&m[..hs]),
                Some(&m[hs..2 * hs]),
                Some(&m[2 * hs..3 * hs]),
                Some(&m[3 * hs..]),
            ),
            None => (None, None, None, None),
        };
        // ── attention ──
        let mut xn = vec![0f32; n * hs];
        for p in 0..n {
            let mut row = rms_norm(&x[p * hs..(p + 1) * hs], &blk.norm1, self.eps);
            if let Some(s) = s_msa {
                for (r, &sc) in row.iter_mut().zip(s) {
                    *r *= 1.0 + sc;
                }
            }
            xn[p * hs..(p + 1) * hs].copy_from_slice(&row);
        }
        let mut q_all = vec![0f32; n * nh * hd];
        let mut k_all = vec![0f32; n * nkv * hd];
        let mut v_all = vec![0f32; n * nkv * hd];
        blk.q.matmat(&xn, n, &mut q_all, pool);
        blk.k.matmat(&xn, n, &mut k_all, pool);
        blk.v.matmat(&xn, n, &mut v_all, pool);
        // per-head qk-norm, then interleaved-pair RoPE
        let (cos, sin) = rope;
        let pairs = hd / 2;
        for (all, heads, w) in [
            (&mut q_all, nh, &blk.norm_q),
            (&mut k_all, nkv, &blk.norm_k),
        ] {
            for p in 0..n {
                for hh in 0..heads {
                    let v = &mut all[(p * heads + hh) * hd..(p * heads + hh + 1) * hd];
                    v.copy_from_slice(&rms_norm(v, w, 1e-5));
                    for j in 0..pairs {
                        let (c, s) = (cos[p * pairs + j], sin[p * pairs + j]);
                        let (a, b) = (v[2 * j] as f64, v[2 * j + 1] as f64);
                        v[2 * j] = (a * c - b * s) as f32;
                        v[2 * j + 1] = (a * s + b * c) as f32;
                    }
                }
            }
        }
        // full (bidirectional) softmax attention, GQA
        let scale = 1.0 / (hd as f32).sqrt();
        let hpk = nh / nkv;
        let mut attn = vec![0f32; n * nh * hd];
        let mut row = vec![0f32; n];
        for hh in 0..nh {
            let kv = hh / hpk;
            for p in 0..n {
                let qv = &q_all[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                for (j, r) in row.iter_mut().enumerate() {
                    let kvv = &k_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                    *r = qv.iter().zip(kvv).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                }
                let mx = row.iter().cloned().fold(f32::MIN, f32::max);
                let mut den = 0f32;
                for r in row.iter_mut() {
                    *r = (*r - mx).exp();
                    den += *r;
                }
                let inv = 1.0 / den;
                let out = &mut attn[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                for (j, &rw) in row.iter().enumerate() {
                    let vv = &v_all[(j * nkv + kv) * hd..(j * nkv + kv + 1) * hd];
                    for (o, s) in out.iter_mut().zip(vv) {
                        *o += rw * inv * s;
                    }
                }
            }
        }
        let mut proj = vec![0f32; n * hs];
        blk.o.matmat(&attn, n, &mut proj, pool);
        for p in 0..n {
            let post = rms_norm(&proj[p * hs..(p + 1) * hs], &blk.norm2, self.eps);
            let dst = &mut x[p * hs..(p + 1) * hs];
            match g_msa {
                Some(g) => {
                    for ((d, v), &gt) in dst.iter_mut().zip(&post).zip(g) {
                        *d += gt.tanh() * v;
                    }
                }
                None => {
                    for (d, v) in dst.iter_mut().zip(&post) {
                        *d += v;
                    }
                }
            }
        }
        // ── SwiGLU FFN ──
        for p in 0..n {
            let mut row = rms_norm(&x[p * hs..(p + 1) * hs], &blk.ffn_norm1, self.eps);
            if let Some(s) = s_mlp {
                for (r, &sc) in row.iter_mut().zip(s) {
                    *r *= 1.0 + sc;
                }
            }
            xn[p * hs..(p + 1) * hs].copy_from_slice(&row);
        }
        let inter = blk.w1.rows();
        let mut g_all = vec![0f32; n * inter];
        let mut u_all = vec![0f32; n * inter];
        blk.w1.matmat(&xn, n, &mut g_all, pool);
        blk.w3.matmat(&xn, n, &mut u_all, pool);
        for (g, u) in g_all.iter_mut().zip(&u_all) {
            *g = silu(*g) * u;
        }
        let mut d_all = vec![0f32; n * hs];
        blk.w2.matmat(&g_all, n, &mut d_all, pool);
        for p in 0..n {
            let post = rms_norm(&d_all[p * hs..(p + 1) * hs], &blk.ffn_norm2, self.eps);
            let dst = &mut x[p * hs..(p + 1) * hs];
            match g_mlp {
                Some(g) => {
                    for ((d, v), &gt) in dst.iter_mut().zip(&post).zip(g) {
                        *d += gt.tanh() * v;
                    }
                }
                None => {
                    for (d, v) in dst.iter_mut().zip(&post) {
                        *d += v;
                    }
                }
            }
        }
    }

    /// One denoising forward: latent `[c, h, w]` (NCHW), caption
    /// features `[cap_n, cap_feat]`, timestep `t` ∈ [0,1] (the
    /// pipeline's `1 − σ`). Returns the velocity prediction `[c, h, w]`.
    pub fn forward(
        &self,
        latent: &[f32],
        h: usize,
        w: usize,
        cap: &[f32],
        cap_n: usize,
        t: f32,
    ) -> Vec<f32> {
        let (c, p, hs) = (self.in_channels, self.patch, self.hidden);
        assert_eq!(latent.len(), c * h * w);
        let (hp, wp) = (h / p, w / p);
        let n_img = hp * wp;
        let temb = self.time_embed(t);

        // caption features → hidden
        let cap_feat = self.cap_norm.len();
        let mut cap_n_all = vec![0f32; cap_n * cap_feat];
        for i in 0..cap_n {
            cap_n_all[i * cap_feat..(i + 1) * cap_feat].copy_from_slice(&rms_norm(
                &cap[i * cap_feat..(i + 1) * cap_feat],
                &self.cap_norm,
                self.eps,
            ));
        }
        let mut cap_e = vec![0f32; cap_n * hs];
        self.cap_w
            .matmat(&cap_n_all, cap_n, &mut cap_e, self.pool.as_deref());
        for i in 0..cap_n {
            for (v, &b) in cap_e[i * hs..(i + 1) * hs].iter_mut().zip(&self.cap_b) {
                *v += b;
            }
        }

        // patchify (dy, dx, ch inner order) + x_embedder
        let pv = p * p * c;
        let mut tok = vec![0f32; n_img * pv];
        for ph in 0..hp {
            for pw in 0..wp {
                let dst = &mut tok[(ph * wp + pw) * pv..(ph * wp + pw + 1) * pv];
                for dy in 0..p {
                    for dx in 0..p {
                        for ch in 0..c {
                            dst[(dy * p + dx) * c + ch] =
                                latent[ch * h * w + (ph * p + dy) * w + pw * p + dx];
                        }
                    }
                }
            }
        }
        let mut img = vec![0f32; n_img * hs];
        self.x_emb
            .matmat(&tok, n_img, &mut img, self.pool.as_deref());
        for i in 0..n_img {
            for (v, &b) in img[i * hs..(i + 1) * hs].iter_mut().zip(&self.x_emb_b) {
                *v += b;
            }
        }

        // 3-axis position ids: caption (i,0,0), image (cap_n, row, col)
        let cap_ids: Vec<[u32; 3]> = (0..cap_n).map(|i| [i as u32, 0, 0]).collect();
        let img_ids: Vec<[u32; 3]> = (0..n_img)
            .map(|i| [cap_n as u32, (i / wp) as u32, (i % wp) as u32])
            .collect();
        let cap_rope = rope_table(&cap_ids, &self.axes_dim);
        let img_rope = rope_table(&img_ids, &self.axes_dim);

        for blk in &self.context_refiner {
            self.block_forward(blk, &mut cap_e, &cap_rope, None);
        }
        for blk in &self.noise_refiner {
            self.block_forward(blk, &mut img, &img_rope, Some(&temb));
        }

        // joint sequence: caption first
        let n = cap_n + n_img;
        let mut x = cap_e;
        x.extend_from_slice(&img);
        let joint_rope = (
            [cap_rope.0, img_rope.0].concat(),
            [cap_rope.1, img_rope.1].concat(),
        );
        for blk in &self.layers {
            self.block_forward(blk, &mut x, &joint_rope, Some(&temb));
        }

        // norm_out: LayerNorm(eps 1e-6, no affine) · (1+scale), project
        let silu_t: Vec<f32> = temb.iter().map(|&v| silu(v)).collect();
        let scale = linear(&silu_t, &self.out_lin1_w, &self.out_lin1_b);
        for row in x.chunks_exact_mut(hs) {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / hs as f64;
            let var = row
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / hs as f64;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for (v, &s) in row.iter_mut().zip(&scale) {
                *v = ((*v as f64 - mean) * inv) as f32 * (1.0 + s);
            }
        }
        let mut out = vec![0f32; n * pv];
        self.out_lin2.matmat(&x, n, &mut out, self.pool.as_deref());
        for i in 0..n {
            for (v, &b) in out[i * pv..(i + 1) * pv].iter_mut().zip(&self.out_lin2_b) {
                *v += b;
            }
        }

        // unpatchify image tokens → [c, h, w]
        let mut pred = vec![0f32; c * h * w];
        for ph in 0..hp {
            for pw in 0..wp {
                let src = &out[(cap_n + ph * wp + pw) * pv..(cap_n + ph * wp + pw + 1) * pv];
                for dy in 0..p {
                    for dx in 0..p {
                        for ch in 0..c {
                            pred[ch * h * w + (ph * p + dy) * w + pw * p + dx] =
                                src[(dy * p + dx) * c + ch];
                        }
                    }
                }
            }
        }
        pred
    }
}
