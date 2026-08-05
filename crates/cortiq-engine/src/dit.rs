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

/// `CMF_DIT_PROF=1`: wall-time totals per forward stage, accumulated
/// across every block of every step and dumped when the model drops.
/// Same spirit as `CMF_VAE_PROF` — the knife for "where do the
/// seconds go" before touching any kernel.
mod prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const MODNORM: usize = 0;
    pub const QKV: usize = 1;
    pub const ROPE: usize = 2;
    pub const APACK: usize = 3;
    pub const AQK: usize = 4;
    pub const SOFTMAX: usize = 5;
    pub const APV: usize = 6;
    pub const OPROJ: usize = 7;
    pub const FFN: usize = 8;
    pub const FFNEL: usize = 9;
    pub const HEADTAIL: usize = 10;
    pub const GPUBLK: usize = 11;
    const NAMES: [&str; 12] = [
        "mod+norms",
        "qkv-proj",
        "qknorm+rope",
        "attn-pack",
        "attn-qk",
        "softmax",
        "attn-pv",
        "o-proj",
        "ffn-mm",
        "ffn-silu",
        "head+tail",
        "gpu-block",
    ];
    static NS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("CMF_DIT_PROF").is_ok_and(|v| v != "0"))
    }

    /// RAII span: charges its category on drop. Free when prof is off.
    pub struct Span(Option<(std::time::Instant, usize)>);
    pub fn span(cat: usize) -> Span {
        Span(on().then(|| (std::time::Instant::now(), cat)))
    }
    impl Drop for Span {
        fn drop(&mut self) {
            if let Some((t0, c)) = self.0 {
                NS[c].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            }
        }
    }

    pub fn dump() {
        if !on() {
            return;
        }
        let total: u64 = NS.iter().map(|a| a.load(Ordering::Relaxed)).sum();
        if total == 0 {
            return;
        }
        eprintln!("dit prof ({:.1} s total in blocks):", total as f64 / 1e9);
        for (name, a) in NAMES.iter().zip(&NS) {
            let ns = a.load(Ordering::Relaxed);
            eprintln!(
                "  {name:<12} {:>7.2} s  {:>4.1}%",
                ns as f64 / 1e9,
                ns as f64 * 100.0 / total as f64
            );
        }
    }
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

/// `rms_norm` into a caller buffer — same math, no per-row alloc.
fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

/// `rms_norm` in place — same math, no per-row alloc.
fn rms_norm_inplace(v: &mut [f32], w: &[f32], eps: f64) {
    let ss = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for (x, &g) in v.iter_mut().zip(w) {
        *x = (*x as f64 * inv) as f32 * g;
    }
}

/// Rows of `n` items split across pool workers (serial without a pool).
fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
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

/// Row handout for pool workers over one flat buffer (Rust-2021
/// closures capture the raw pointer field, not the wrapper — hence
/// the accessor method).
struct SendRows(*mut f32);
unsafe impl Send for SendRows {}
unsafe impl Sync for SendRows {}
impl SendRows {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)] // the disjoint-rows contract IS the point
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }

    /// Single scattered element (transposed stores). SAFETY: as `row`.
    unsafe fn set(&self, off: usize, v: f32) {
        unsafe { *self.0.add(off) = v }
    }
}

/// Numerically stable in-place softmax of one full row (NEON exp on
/// aarch64, scalar elsewhere).
fn softmax_inplace(row: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        crate::attention::softmax_row(row);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let mut den = 0f32;
        for r in row.iter_mut() {
            *r = (*r - mx).exp();
            den += *r;
        }
        if den > 0.0 {
            let inv = 1.0 / den;
            for r in row.iter_mut() {
                *r *= inv;
            }
        }
    }
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

impl Drop for NextDit {
    fn drop(&mut self) {
        prof::dump();
    }
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

    /// Fused on-device SwiGLU FFN (Metal). Taken only once the wide
    /// GEMM probe has settled on the GPU arm: during probing the
    /// per-op path feeds the samples, after a CPU verdict (or a
    /// contention kill) the CPU path is the right one anyway. Carries
    /// the same work-proportional contention tripwire as the per-op
    /// route (cold ops exempt).
    fn gpu_ffn(&self, blk: &Block, xn: &[f32], n: usize, out: &mut [f32]) -> bool {
        use crate::gpu;
        if n < 128 || !gpu::enabled_here() || gpu::mm_killed() {
            return false;
        }
        // A fused block is not a wide matmat: see `fused_block_trusted`.
        if !gpu::fused_block_trusted()
            && (gpu::probe_deciding(gpu::OpClass::MatmatWide)
                || !matches!(gpu::probe_arm(gpu::OpClass::MatmatWide), gpu::ProbeArm::Gpu))
        {
            return false;
        }
        let (Proj::Q(q1), Proj::Q(q3), Proj::Q(q2)) = (&blk.w1, &blk.w3, &blk.w2) else {
            return false;
        };
        // q4t or q4tp — the fused chain exists for both, and picking by dtype
        // here is what keeps a q4tp image model off the unfused path (which
        // ships the [b, inter] intermediates across the CPU boundary twice per
        // layer: 28 s against 14 s on Lumina at 256px).
        let tp = q1.mapped_q4tp().is_some();
        let (Some((m, i1)), Some((_, i3)), Some((_, i2))) = (if tp {
            (q1.mapped_q4tp(), q3.mapped_q4tp(), q2.mapped_q4tp())
        } else {
            (q1.mapped_q4t(), q3.mapped_q4t(), q2.mapped_q4t())
        }) else {
            return false;
        };
        let inter = q1.rows();
        let t0 = std::time::Instant::now();
        let ok = if tp {
            gpu::q4tp_ffn(m, i1, i3, i2, xn, n, self.hidden, inter, out)
        } else {
            gpu::q4t_ffn(m, i1, i3, i2, xn, n, self.hidden, inter, out)
        };
        if !ok {
            return false;
        }
        let flops = 6.0 * n as f64 * self.hidden as f64 * inter as f64;
        let budget = std::time::Duration::from_secs_f64(flops / 1.5e12 * 8.0 + 0.020);
        let el = t0.elapsed();
        if el > budget && !gpu::probe_was_cold() {
            tracing::warn!(
                "gpu ffn took {el:?} (budget {budget:?}) — device contended, \
                 CPU for the rest of the process"
            );
            gpu::mm_kill();
        }
        true
    }

    /// All-heads attention on the device (same probe/kill gating as
    /// the fused FFN). Packs q/k/v head-major (pool-parallel), runs
    /// scores→softmax→P·V→unstack in one command buffer, and writes
    /// straight into the [n][nh·hd] attn layout the O-projection
    /// consumes. Returns false → caller runs the CPU per-head loop.
    fn gpu_attention(
        &self,
        q_all: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        n: usize,
        scale: f32,
        attn: &mut [f32],
    ) -> bool {
        use crate::gpu;
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        if n < 128 || !gpu::enabled_here() || gpu::mm_killed() {
            return false;
        }
        // A fused block is not a wide matmat: see `fused_block_trusted`.
        if !gpu::fused_block_trusted()
            && (gpu::probe_deciding(gpu::OpClass::MatmatWide)
                || !matches!(gpu::probe_arm(gpu::OpClass::MatmatWide), gpu::ProbeArm::Gpu))
        {
            return false;
        }
        let pool = self.pool.as_deref();
        let mut qh = vec![0f32; nh * n * hd];
        let mut kh = vec![0f32; nkv * n * hd];
        let mut vh = vec![0f32; nkv * n * hd];
        {
            let _s = prof::span(prof::APACK);
            let (sq, sk, sv) = (
                SendRows(qh.as_mut_ptr()),
                SendRows(kh.as_mut_ptr()),
                SendRows(vh.as_mut_ptr()),
            );
            pool_rows(pool, n, &|start, end| {
                for p in start..end {
                    for h in 0..nh {
                        // SAFETY: workers cover disjoint token ranges.
                        unsafe { sq.row((h * n + p) * hd, hd) }
                            .copy_from_slice(&q_all[(p * nh + h) * hd..(p * nh + h + 1) * hd]);
                    }
                    for h in 0..nkv {
                        unsafe { sk.row((h * n + p) * hd, hd) }
                            .copy_from_slice(&k_all[(p * nkv + h) * hd..(p * nkv + h + 1) * hd]);
                        unsafe { sv.row((h * n + p) * hd, hd) }
                            .copy_from_slice(&v_all[(p * nkv + h) * hd..(p * nkv + h + 1) * hd]);
                    }
                }
            });
        }
        let _s = prof::span(prof::AQK);
        let t0 = std::time::Instant::now();
        if !gpu::dit_attention(&qh, &kh, &vh, nh, nkv, n, hd, scale, attn) {
            return false;
        }
        let flops = 4.0 * nh as f64 * (n as f64) * (n as f64) * hd as f64;
        let budget = std::time::Duration::from_secs_f64(flops / 1.5e12 * 8.0 + 0.020);
        let el = t0.elapsed();
        if el > budget && !gpu::probe_was_cold() {
            tracing::warn!(
                "gpu attention took {el:?} (budget {budget:?}) — device contended, \
                 CPU for the rest of the process"
            );
            gpu::mm_kill();
        }
        true
    }

    /// One whole block on the device (norms → qkv → RoPE → attention
    /// → O → residual → FFN → residual, single command buffer). Same
    /// gating and contention tripwire as the per-stage GPU arms.
    fn gpu_block(
        &self,
        blk: &Block,
        x: &mut [f32],
        n: usize,
        rope32: &(Vec<f32>, Vec<f32>),
        m: &[f32],
    ) -> bool {
        use crate::gpu;
        let (hs, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        if n < 128 || !gpu::enabled_here() || gpu::mm_killed() {
            return false;
        }
        // A fused block is not a wide matmat: see `fused_block_trusted`.
        if !gpu::fused_block_trusted()
            && (gpu::probe_deciding(gpu::OpClass::MatmatWide)
                || !matches!(gpu::probe_arm(gpu::OpClass::MatmatWide), gpu::ProbeArm::Gpu))
        {
            return false;
        }
        // The pack kernel assumes the rope table covers the full head
        // dim (axes_dim sums to hd — true for Lumina; bail otherwise).
        if rope32.0.len() != n * hd / 2 {
            return false;
        }
        fn q(p: &Proj) -> Option<(&Arc<CmfModel>, usize)> {
            match p {
                Proj::Q(q) => q.mapped_q4t(),
                Proj::F32 { .. } => None,
            }
        }
        let (
            Some((model, wq)),
            Some((_, wk)),
            Some((_, wv)),
            Some((_, wo)),
            Some((_, w1)),
            Some((_, w3)),
            Some((_, w2)),
        ) = (
            q(&blk.q),
            q(&blk.k),
            q(&blk.v),
            q(&blk.o),
            q(&blk.w1),
            q(&blk.w3),
            q(&blk.w2),
        )
        else {
            return false;
        };
        let inter = blk.w1.rows();
        let gate_msa: Vec<f32> = m[hs..2 * hs].iter().map(|&v| v.tanh()).collect();
        let gate_mlp: Vec<f32> = m[3 * hs..].iter().map(|&v| v.tanh()).collect();
        let args = gpu::DitBlockArgs {
            n,
            hidden: hs,
            inter,
            nh,
            nkv,
            hd,
            eps: self.eps as f32,
            rope_cos: &rope32.0,
            rope_sin: &rope32.1,
            norm1: &blk.norm1,
            norm2: &blk.norm2,
            ffn_norm1: &blk.ffn_norm1,
            ffn_norm2: &blk.ffn_norm2,
            norm_q: &blk.norm_q,
            norm_k: &blk.norm_k,
            s_msa: &m[..hs],
            gate_msa: &gate_msa,
            s_mlp: &m[2 * hs..3 * hs],
            gate_mlp: &gate_mlp,
            wq,
            wk,
            wv,
            wo,
            w1,
            w3,
            w2,
        };
        let t0 = std::time::Instant::now();
        if !gpu::dit_block(model, &args, x) {
            return false;
        }
        let flops = 2.0 * n as f64 * hs as f64 * ((nh + 2 * nkv) * hd) as f64
            + 4.0 * nh as f64 * (n as f64) * (n as f64) * hd as f64
            + 2.0 * n as f64 * hs as f64 * (nh * hd) as f64
            + 6.0 * n as f64 * hs as f64 * inter as f64;
        let budget = std::time::Duration::from_secs_f64(flops / 1.5e12 * 8.0 + 0.030);
        let el = t0.elapsed();
        if el > budget && !gpu::probe_was_cold() {
            tracing::warn!(
                "gpu dit block took {el:?} (budget {budget:?}) — device contended, \
                 CPU for the rest of the process"
            );
            gpu::mm_kill();
        }
        true
    }


    /// Full bidirectional attention over ONE sequence: per head, pack
    /// q/k/v, scores as a GEMM, row softmax, P·V, scatter back. Lifted
    /// out of the block so a batched call can run it per segment
    /// without the segments ever meeting in a score matrix.
    fn attention_seq(
        &self,
        q_all: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        n: usize,
        scale: f32,
        attn: &mut [f32],
    ) {
        let (nh, nkv, hd) = (self.nh, self.nkv, self.hd);
        let hpk = nh / nkv;
        let pool = self.pool.as_deref();
            let mut qh = vec![0f32; n * hd];
            let mut kh = vec![0f32; n * hd];
            let mut vt = vec![0f32; hd * n]; // V transposed: gemm_nt's W layout
            let mut scores = vec![0f32; n * n];
            let mut oh = vec![0f32; n * hd];
            for hh in 0..nh {
                let kv = hh / hpk;
                {
                    let _s = prof::span(prof::APACK);
                    let (sq, sk, sv) = (
                        SendRows(qh.as_mut_ptr()),
                        SendRows(kh.as_mut_ptr()),
                        SendRows(vt.as_mut_ptr()),
                    );
                    pool_rows(pool, n, &|start, end| {
                        for p in start..end {
                            let qsrc = &q_all[(p * nh + hh) * hd..(p * nh + hh + 1) * hd];
                            // SAFETY: workers cover disjoint token ranges
                            // (`vt` columns are indexed by token too).
                            let qd = unsafe { sq.row(p * hd, hd) };
                            for (d, &v) in qsrc.iter().enumerate() {
                                qd[d] = v * scale;
                            }
                            unsafe { sk.row(p * hd, hd) }.copy_from_slice(
                                &k_all[(p * nkv + kv) * hd..(p * nkv + kv + 1) * hd],
                            );
                            let vv = &v_all[(p * nkv + kv) * hd..(p * nkv + kv + 1) * hd];
                            for (d, &val) in vv.iter().enumerate() {
                                unsafe { sv.set(d * n + p, val) };
                            }
                        }
                    });
                }
                {
                    let _s = prof::span(prof::AQK);
                    crate::fcd_ops::gemm_nt(&qh, &kh, &mut scores, n, hd, n, pool);
                }
                {
                    let _s = prof::span(prof::SOFTMAX);
                    let sp = SendRows(scores.as_mut_ptr());
                    let soft = |start: usize, end: usize| {
                        for r in start..end {
                            // SAFETY: workers cover disjoint row ranges.
                            softmax_inplace(unsafe { sp.row(r * n, n) });
                        }
                    };
                    match pool {
                        Some(p) => p.run_rows(n, &soft),
                        None => soft(0, n),
                    }
                }
                {
                    let _s = prof::span(prof::APV);
                    crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
                }
                let _s = prof::span(prof::APACK);
                let sa = SendRows(attn.as_mut_ptr());
                pool_rows(pool, n, &|start, end| {
                    for p in start..end {
                        // SAFETY: workers cover disjoint token ranges.
                        unsafe { sa.row((p * nh + hh) * hd, hd) }
                            .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
                    }
                });
            }
        
    }

    fn block_forward(
        &self,
        blk: &Block,
        x: &mut [f32],
        rope: &(Vec<f64>, Vec<f64>),
        rope32: Option<&(Vec<f32>, Vec<f32>)>,
        temb: Option<&[f32]>,
    ) {
        let n_all = x.len() / self.hidden;
        self.block_forward_seg(blk, x, rope, rope32, temb, &[n_all]);
    }

    /// `block_forward` over a CONCATENATION of independent sequences.
    /// Everything position-wise (norms, projections, FFN) sees one tall
    /// batch — the weights are read once for all of them, which is the
    /// whole point on a CPU or a phone — while attention runs per
    /// segment, so no sample ever attends to another's tokens. `segs`
    /// are the token counts in order; a single-element slice is the
    /// plain path, bit for bit.
    fn block_forward_seg(
        &self,
        blk: &Block,
        x: &mut [f32],
        rope: &(Vec<f64>, Vec<f64>),
        rope32: Option<&(Vec<f32>, Vec<f32>)>,
        temb: Option<&[f32]>,
        segs: &[usize],
    ) {
        let (hs, nh, nkv, hd) = (self.hidden, self.nh, self.nkv, self.hd);
        let pool = self.pool.as_deref();
        let n = x.len() / hs;
        let modv = {
            let _s = prof::span(prof::MODNORM);
            blk.modulation.as_ref().zip(temb).map(|((w, b), t)| {
                let s: Vec<f32> = t.iter().map(|&v| silu(v)).collect();
                let mut m = vec![0f32; w.rows()];
                w.matmat(&s, 1, &mut m, pool);
                for (v, &bias) in m.iter_mut().zip(b) {
                    *v += bias;
                }
                m
            })
        };
        if segs.len() <= 1 {
            if let (Some(m), Some(r32)) = (&modv, rope32) {
                let _s = prof::span(prof::GPUBLK);
                if self.gpu_block(blk, x, n, r32, m) {
                    return;
                }
            }
        }
        let modnorm = prof::span(prof::MODNORM);
        let (s_msa, g_msa, s_mlp, g_mlp) = match &modv {
            Some(m) => (
                Some(&m[..hs]),
                Some(&m[hs..2 * hs]),
                Some(&m[2 * hs..3 * hs]),
                Some(&m[3 * hs..]),
            ),
            None => (None, None, None, None),
        };
        // Gates: tanh once per block — every row shares the same gate
        // vector, and the naive per-element tanh in the residual loop
        // was billions of repeated evaluations per render.
        let gate_msa: Option<Vec<f32>> = g_msa.map(|g| g.iter().map(|&v| v.tanh()).collect());
        let gate_mlp: Option<Vec<f32>> = g_mlp.map(|g| g.iter().map(|&v| v.tanh()).collect());
        // Pool-parallel row helpers: dst = rms(src)·w · (1+s)  and
        // x += gate ⊙ rms(src)·w. Same math and per-row summation
        // order as the serial loops — rows are independent, so the
        // parallel split is bit-exact.
        let norm_scaled = |src: &[f32], w: &[f32], s: Option<&[f32]>, dst: &mut [f32]| {
            let sr = SendRows(dst.as_mut_ptr());
            pool_rows(pool, n, &|start, end| {
                for p in start..end {
                    // SAFETY: workers cover disjoint row ranges.
                    let row = unsafe { sr.row(p * hs, hs) };
                    rms_norm_into(&src[p * hs..(p + 1) * hs], w, self.eps, row);
                    if let Some(s) = s {
                        for (r, &sc) in row.iter_mut().zip(s) {
                            *r *= 1.0 + sc;
                        }
                    }
                }
            });
        };
        let residual = |src: &[f32], w: &[f32], gate: Option<&[f32]>, x: &mut [f32]| {
            let sr = SendRows(x.as_mut_ptr());
            pool_rows(pool, n, &|start, end| {
                let mut tmp = vec![0f32; hs];
                for p in start..end {
                    rms_norm_into(&src[p * hs..(p + 1) * hs], w, self.eps, &mut tmp);
                    // SAFETY: workers cover disjoint row ranges.
                    let dst = unsafe { sr.row(p * hs, hs) };
                    match gate {
                        Some(g) => {
                            for ((d, &v), &gt) in dst.iter_mut().zip(&tmp).zip(g) {
                                *d += gt * v;
                            }
                        }
                        None => {
                            for (d, &v) in dst.iter_mut().zip(&tmp) {
                                *d += v;
                            }
                        }
                    }
                }
            });
        };
        // ── attention ──
        let mut xn = vec![0f32; n * hs];
        norm_scaled(x, &blk.norm1, s_msa, &mut xn);
        drop(modnorm);
        let mut q_all = vec![0f32; n * nh * hd];
        let mut k_all = vec![0f32; n * nkv * hd];
        let mut v_all = vec![0f32; n * nkv * hd];
        {
            let _s = prof::span(prof::QKV);
            blk.q.matmat(&xn, n, &mut q_all, pool);
            blk.k.matmat(&xn, n, &mut k_all, pool);
            blk.v.matmat(&xn, n, &mut v_all, pool);
        }
        let rope_span = prof::span(prof::ROPE);
        // per-head qk-norm, then interleaved-pair RoPE
        let (cos, sin) = rope;
        let pairs = hd / 2;
        for (all, heads, w) in [
            (&mut q_all, nh, &blk.norm_q),
            (&mut k_all, nkv, &blk.norm_k),
        ] {
            let sr = SendRows(all.as_mut_ptr());
            pool_rows(pool, n, &|start, end| {
                for p in start..end {
                    for hh in 0..heads {
                        // SAFETY: workers cover disjoint token ranges.
                        let v = unsafe { sr.row((p * heads + hh) * hd, hd) };
                        rms_norm_inplace(v, w, 1e-5);
                        for j in 0..pairs {
                            let (c, s) = (cos[p * pairs + j], sin[p * pairs + j]);
                            let (a, b) = (v[2 * j] as f64, v[2 * j + 1] as f64);
                            v[2 * j] = (a * c - b * s) as f32;
                            v[2 * j + 1] = (a * s + b * c) as f32;
                        }
                    }
                }
            });
        }
        drop(rope_span);
        // full (bidirectional) softmax attention, GQA — per head:
        // scores = (Q·s)·Kᵀ and P·V as GEMMs (Accelerate/blocked),
        // pool-parallel row softmax between them. The naive
        // per-position loop was the depth wall: at 512px (1064
        // tokens) attention alone cost hundreds of serial GFLOP.
        let scale = 1.0 / (hd as f32).sqrt();
        let hpk = nh / nkv;
        let mut attn = vec![0f32; n * nh * hd];
        if segs.len() > 1 {
            // Each sequence attends within itself. The slices are row
            // ranges of the same buffers, so the per-head math below is
            // the one the single-sequence path runs.
            let mut off = 0usize;
            for &ns in segs {
                let (qs, ks, vs) = (
                    &q_all[off * nh * hd..(off + ns) * nh * hd],
                    &k_all[off * nkv * hd..(off + ns) * nkv * hd],
                    &v_all[off * nkv * hd..(off + ns) * nkv * hd],
                );
                let dst = &mut attn[off * nh * hd..(off + ns) * nh * hd];
                if !self.gpu_attention(qs, ks, vs, ns, scale, dst) {
                    self.attention_seq(qs, ks, vs, ns, scale, dst);
                }
                off += ns;
            }
        } else if !self.gpu_attention(&q_all, &k_all, &v_all, n, scale, &mut attn) {
            self.attention_seq(&q_all, &k_all, &v_all, n, scale, &mut attn);
        }
        let mut proj = vec![0f32; n * hs];
        {
            let _s = prof::span(prof::OPROJ);
            blk.o.matmat(&attn, n, &mut proj, pool);
        }
        let modnorm = prof::span(prof::MODNORM);
        residual(&proj, &blk.norm2, gate_msa.as_deref(), x);
        // ── SwiGLU FFN ──
        norm_scaled(x, &blk.ffn_norm1, s_mlp, &mut xn);
        drop(modnorm);
        let mut d_all = vec![0f32; n * hs];
        let fused = {
            let _s = prof::span(prof::FFN);
            self.gpu_ffn(blk, &xn, n, &mut d_all)
        };
        if !fused {
            let inter = blk.w1.rows();
            let mut g_all = vec![0f32; n * inter];
            let mut u_all = vec![0f32; n * inter];
            {
                let _s = prof::span(prof::FFN);
                blk.w1.matmat(&xn, n, &mut g_all, pool);
                blk.w3.matmat(&xn, n, &mut u_all, pool);
            }
            {
                let _s = prof::span(prof::FFNEL);
                let sg = SendRows(g_all.as_mut_ptr());
                pool_rows(pool, n, &|start, end| {
                    for p in start..end {
                        // SAFETY: workers cover disjoint token ranges.
                        let g = unsafe { sg.row(p * inter, inter) };
                        for (gv, &uv) in g.iter_mut().zip(&u_all[p * inter..(p + 1) * inter]) {
                            *gv = silu(*gv) * uv;
                        }
                    }
                });
            }
            {
                let _s = prof::span(prof::FFN);
                blk.w2.matmat(&g_all, n, &mut d_all, pool);
            }
        }
        let _modnorm = prof::span(prof::MODNORM);
        residual(&d_all, &blk.ffn_norm2, gate_mlp.as_deref(), x);
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
        self.forward_with_cap(latent, h, w, &self.refine_caption(cap, cap_n), cap_n, t)
    }

    /// Caption features → hidden, through the context refiner. Depends on
    /// NOTHING that moves during denoising — not the timestep, not the
    /// latents — so the whole thing is a constant of the prompt. The
    /// denoise loop hoists it out and hands the result to
    /// `forward_with_cap`; it used to be recomputed on every model call,
    /// which for 30 steps under CFG meant 60 evaluations of a value with
    /// two distinct instances.
    pub fn refine_caption(&self, cap: &[f32], cap_n: usize) -> Vec<f32> {
        let hs = self.hidden;
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
        let cap_ids: Vec<[u32; 3]> = (0..cap_n).map(|i| [i as u32, 0, 0]).collect();
        let cap_rope = rope_table(&cap_ids, &self.axes_dim);
        for blk in &self.context_refiner {
            self.block_forward(blk, &mut cap_e, &cap_rope, None, None);
        }
        cap_e
    }

    /// The rest of the forward, from an already-refined caption.
    /// Classifier-free guidance in ONE pass: the conditional and the
    /// unconditional sequence go through the joint stack as a single
    /// batch. Every weight is read once for both — which is the whole
    /// cost on a CPU or a phone — and the image branch (patchify,
    /// x_embedder, the noise refiners) is computed once instead of
    /// twice, because it does not depend on the caption at all.
    /// Attention stays per sequence, so the two never mix and each
    /// prediction equals what the single-sequence path returns.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cfg_pair(
        &self,
        latent: &[f32],
        h: usize,
        w: usize,
        cap_c: &[f32],
        cap_c_n: usize,
        cap_u: &[f32],
        cap_u_n: usize,
        t: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let (c, p, hs) = (self.in_channels, self.patch, self.hidden);
        let (hp, wp) = (h / p, w / p);
        let n_img = hp * wp;
        let head = prof::span(prof::HEADTAIL);
        let temb = self.time_embed(t);
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
        self.x_emb.matmat(&tok, n_img, &mut img, self.pool.as_deref());
        for i in 0..n_img {
            for (v, &b) in img[i * hs..(i + 1) * hs].iter_mut().zip(&self.x_emb_b) {
                *v += b;
            }
        }
        let img_ids: Vec<[u32; 3]> = (0..n_img)
            .map(|i| [0, (i / wp) as u32, (i % wp) as u32])
            .collect();
        let to32 = |r: &(Vec<f64>, Vec<f64>)| {
            (
                r.0.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
                r.1.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            )
        };
        // The refiners see the image alone — same for both branches.
        let img_rope_r = rope_table(
            &(0..n_img)
                .map(|i| [0u32, (i / wp) as u32, (i % wp) as u32])
                .collect::<Vec<_>>(),
            &self.axes_dim,
        );
        let img_rope32_r = to32(&img_rope_r);
        drop(head);
        for blk in &self.noise_refiner {
            self.block_forward(blk, &mut img, &img_rope_r, Some(&img_rope32_r), Some(&temb));
        }
        let _ = img_ids;
        // Joint sequences, concatenated: [cap_c | img] then [cap_u | img].
        let n_c = cap_c_n + n_img;
        let n_u = cap_u_n + n_img;
        let mut x = Vec::with_capacity((n_c + n_u) * hs);
        x.extend_from_slice(&cap_c[..cap_c_n * hs]);
        x.extend_from_slice(&img);
        x.extend_from_slice(&cap_u[..cap_u_n * hs]);
        x.extend_from_slice(&img);
        let rope_for = |cap_n: usize| -> (Vec<f64>, Vec<f64>) {
            let cap_ids: Vec<[u32; 3]> = (0..cap_n).map(|i| [i as u32, 0, 0]).collect();
            let im_ids: Vec<[u32; 3]> = (0..n_img)
                .map(|i| [cap_n as u32, (i / wp) as u32, (i % wp) as u32])
                .collect();
            let a = rope_table(&cap_ids, &self.axes_dim);
            let b = rope_table(&im_ids, &self.axes_dim);
            ([a.0, b.0].concat(), [a.1, b.1].concat())
        };
        let (rc, ru) = (rope_for(cap_c_n), rope_for(cap_u_n));
        let joint_rope = ([rc.0, ru.0].concat(), [rc.1, ru.1].concat());
        let joint_rope32 = to32(&joint_rope);
        let segs = [n_c, n_u];
        for blk in &self.layers {
            self.block_forward_seg(
                blk,
                &mut x,
                &joint_rope,
                Some(&joint_rope32),
                Some(&temb),
                &segs,
            );
        }
        let _tail = prof::span(prof::HEADTAIL);
        let n = n_c + n_u;
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
        let unpatch = |base: usize| -> Vec<f32> {
            let mut pred = vec![0f32; c * h * w];
            for ph in 0..hp {
                for pw in 0..wp {
                    let src = &out[(base + ph * wp + pw) * pv..(base + ph * wp + pw + 1) * pv];
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
        };
        (unpatch(cap_c_n), unpatch(n_c + cap_u_n))
    }

    pub fn forward_with_cap(
        &self,
        latent: &[f32],
        h: usize,
        w: usize,
        cap_e_in: &[f32],
        cap_n: usize,
        t: f32,
    ) -> Vec<f32> {
        let (c, p, hs) = (self.in_channels, self.patch, self.hidden);
        assert_eq!(latent.len(), c * h * w);
        let (hp, wp) = (h / p, w / p);
        let n_img = hp * wp;
        let head = prof::span(prof::HEADTAIL);
        let temb = self.time_embed(t);
        let mut cap_e = cap_e_in.to_vec();

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
        // f32 twins for the on-device block (values stay f64-derived).
        let to32 = |r: &(Vec<f64>, Vec<f64>)| {
            (
                r.0.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
                r.1.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
            )
        };
        let img_rope32 = to32(&img_rope);
        drop(head);

        // The context refiner already ran in `refine_caption` — it is a
        // constant of the prompt, not of this call.
        for blk in &self.noise_refiner {
            self.block_forward(blk, &mut img, &img_rope, Some(&img_rope32), Some(&temb));
        }

        // joint sequence: caption first
        let n = cap_n + n_img;
        let mut x = cap_e;
        x.extend_from_slice(&img);
        let joint_rope = (
            [cap_rope.0, img_rope.0].concat(),
            [cap_rope.1, img_rope.1].concat(),
        );
        let joint_rope32 = to32(&joint_rope);
        for blk in &self.layers {
            self.block_forward(blk, &mut x, &joint_rope, Some(&joint_rope32), Some(&temb));
        }

        // norm_out: LayerNorm(eps 1e-6, no affine) · (1+scale), project
        let _tail = prof::span(prof::HEADTAIL);
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
