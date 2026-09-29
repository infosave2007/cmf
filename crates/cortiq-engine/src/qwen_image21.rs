//! Qwen-Image-2.1 denoiser (`QwenImage21Transformer2DModel`, 32
//! single-stream blocks, 7 B) — the CPU-exact reference path and the host
//! glue around the device step.
//!
//! Semantics (diffusers `transformer_qwenimage21.py`):
//! - one joint sequence `[text/condition prefix, target image]`; text rows
//!   come from `txt_in` (zero-centred RMSNorm → Linear → GELU-tanh →
//!   Linear), image rows from `img_in` (64 latent channels, unpatched);
//! - ONE shared modulation for every block: `Linear(SiLU(temb))` →
//!   `[scale1, gate1, scale2, gate2]`; the block is
//!   `x += tanh(g1)·attn(LN(x)·(1+s1))`, `x += tanh(g2)·mlp(LN(x)·(1+s2))`
//!   with an affine-free LayerNorm and a bias-free SwiGLU (`out(silu(gate)·proj)`);
//! - `causal_condition`: text and condition-image rows are modulated from
//!   t = 0, target rows from the sampled t;
//! - block-causal attention: `(q ≥ kv) or same_image_block` — text is
//!   causal, every image block is bidirectional inside itself and sees
//!   everything before it; the target sees everything;
//! - per-head RMSNorm (weighted) on q and k, then the 3-axis complex RoPE
//!   (θ = 10000, axes [16, 56, 56]): text advances one position on all
//!   three axes, an image block freezes the frame axis at the running
//!   position and centres its (h, w) grid on zero, then the position
//!   advances by max(h, w).
//!
//! Because the prefix never attends to the target and never sees the
//! timestep, its per-layer keys and values are computed ONCE per prompt
//! ([`Qi21Dit::prefill`]) and every step recomputes only the target rows
//! against `[cached prefix, target]` — the pipeline's KV cache.
//!
//! Precision: f32 activations, f64 accumulation in every norm and in the
//! small linears of the time path; the big projections run the host GEMM
//! (f32 accumulation) over weights dequantized to f32.

use crate::dit::Proj;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// `header.arch.arch_name` of a Qwen-Image-2.1 container.
pub const ARCH_NAME: &str = "qwen_image21";

#[derive(Clone, Debug)]
pub struct Qi21Config {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub mlp_hidden: usize,
    pub context_in: usize,
    pub axes: [usize; 3],
    pub eps: f64,
    pub causal_condition: bool,
}

impl Qi21Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |k: &str, d: usize| v[k].as_u64().map(|x| x as usize).unwrap_or(d);
        let heads = u("num_attention_heads", 32);
        let head_dim = u("attention_head_dim", 128);
        let dim = heads * head_dim;
        let axes: Vec<usize> = v["axes_dims_rope"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
            .unwrap_or_else(|| vec![16, 56, 56]);
        if axes.len() != 3 || axes.iter().sum::<usize>() != head_dim || axes.iter().any(|a| a % 2 != 0) {
            return Err(format!("axes_dims_rope {axes:?} does not tile head_dim {head_dim}"));
        }
        if u("patch_size", 1) != 1 {
            return Err("Qwen-Image-2.1 consumes unpatched latents (patch_size 1)".into());
        }
        let in_channels = u("in_channels", 64);
        Ok(Self {
            dim,
            heads,
            head_dim,
            layers: u("num_layers", 32),
            in_channels,
            out_channels: u("out_channels", in_channels),
            mlp_hidden: dim * u("mlp_ratio", 3),
            context_in: u("context_in_dim", 4096),
            axes: [axes[0], axes[1], axes[2]],
            eps: v["eps"].as_f64().unwrap_or(1e-6),
            causal_condition: v["causal_condition"].as_bool().unwrap_or(true),
        })
    }
}

/// One run of the joint sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seg {
    Text(usize),
    /// An image block of `h × w` latent tokens (raster order).
    Image(usize, usize),
}

impl Seg {
    pub fn len(&self) -> usize {
        match *self {
            Seg::Text(n) => n,
            Seg::Image(h, w) => h * w,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The joint sequence: prefix runs, then the target image block (last).
#[derive(Clone, Debug)]
pub struct Qi21Layout {
    pub segs: Vec<Seg>,
}

impl Qi21Layout {
    /// Text-to-image: `[text(l), target(h, w)]`.
    pub fn t2i(text: usize, h: usize, w: usize) -> Self {
        Self {
            segs: vec![Seg::Text(text), Seg::Image(h, w)],
        }
    }

    pub fn total(&self) -> usize {
        self.segs.iter().map(|s| s.len()).sum()
    }

    /// Rows before the target block.
    pub fn prefix_len(&self) -> usize {
        self.total() - self.target().len()
    }

    pub fn target(&self) -> Seg {
        *self.segs.last().expect("empty layout")
    }

    pub fn validate(&self) -> Result<(), String> {
        match self.segs.last() {
            Some(Seg::Image(h, w)) if *h > 0 && *w > 0 => {}
            _ => return Err("the layout must end with a non-empty target image block".into()),
        }
        if self.prefix_len() == 0 {
            return Err("the prompt prefix is empty".into());
        }
        Ok(())
    }

    /// Per row: the image block id (−1 for text).
    pub fn block_ids(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.total());
        let mut id = 0i32;
        for s in &self.segs {
            match *s {
                Seg::Text(n) => out.extend(std::iter::repeat_n(-1, n)),
                Seg::Image(h, w) => {
                    out.extend(std::iter::repeat_n(id, h * w));
                    id += 1;
                }
            }
        }
        out
    }

    /// (frame, h, w) RoPE positions of every row (diffusers `QwenImage21Rope`).
    pub fn positions(&self) -> Vec<[i64; 3]> {
        let mut out = Vec::with_capacity(self.total());
        let mut pos = 0i64;
        for s in &self.segs {
            match *s {
                Seg::Text(n) => {
                    for _ in 0..n {
                        out.push([pos, pos, pos]);
                        pos += 1;
                    }
                }
                Seg::Image(h, w) => {
                    let (hi, wi) = (h as i64, w as i64);
                    for r in 0..hi {
                        for c in 0..wi {
                            out.push([pos, r - (hi - hi / 2), c - (wi - wi / 2)]);
                        }
                    }
                    pos += hi.max(wi);
                }
            }
        }
        out
    }
}

/// cos/sin of every row's rotation, `[rows, head_dim/2]` each, in the
/// complex-pair order the kernel applies them (frame, h, w).
pub fn rope_tables(pos: &[[i64; 3]], axes: [usize; 3], theta: f64) -> (Vec<f32>, Vec<f32>) {
    let half: usize = axes.iter().sum::<usize>() / 2;
    // torch: 1 / theta^(arange(0, d, 2)/d) in f32, angle = f32(index)·freq
    let freqs: Vec<Vec<f32>> = axes
        .iter()
        .map(|&d| {
            (0..d / 2)
                .map(|i| {
                    let e = (2 * i) as f32 / d as f32;
                    1.0f32 / (theta as f32).powf(e)
                })
                .collect()
        })
        .collect();
    let mut cos = vec![0f32; pos.len() * half];
    let mut sin = vec![0f32; pos.len() * half];
    for (r, p) in pos.iter().enumerate() {
        let mut j = 0;
        for (a, fr) in freqs.iter().enumerate() {
            for &f in fr {
                let ang = p[a] as f32 * f;
                cos[r * half + j] = ang.cos();
                sin[r * half + j] = ang.sin();
                j += 1;
            }
        }
    }
    (cos, sin)
}

struct Block {
    /// Tensor indices of the seven projections (device path).
    idx: Option<[usize; 7]>,
    q: Proj,
    k: Proj,
    v: Proj,
    o: Proj,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    gate: Proj,
    up: Proj,
    down: Proj,
}

/// Per-layer keys and values of the prefix (post qk-norm and RoPE),
/// `[prefix, dim]` each.
pub struct Qi21Prefix {
    pub layout: Qi21Layout,
    /// Host keys/values (empty when the prefix lives on the device).
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    /// The device program holding this prefix (`gpu::qi21_*`).
    pub device_key: Option<u64>,
    /// cos/sin of the target rows for THIS prefix's layout: the target's
    /// frame position follows its own prefix, so a positive and a negative
    /// prompt of different lengths rotate their targets differently.
    pub rope_t: (Vec<f32>, Vec<f32>),
}

impl Drop for Qi21Prefix {
    fn drop(&mut self) {
        if let Some(k) = self.device_key {
            crate::gpu::qi21_release_key(k);
        }
    }
}

impl Qi21Prefix {
    pub fn len(&self) -> usize {
        self.layout.prefix_len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Qi21Dit {
    pub cfg: Qi21Config,
    model: Option<Arc<CmfModel>>,
    img_in: Proj,
    txt_norm: Vec<f32>,
    txt_in1: Proj,
    txt_in2: Proj,
    t_lin1: Proj,
    t_lin2: Proj,
    modulation: Proj,
    norm_out: Proj,
    proj_out: Proj,
    blocks: Vec<Block>,
    pool: Option<Arc<Pool>>,
    /// f32 copies of `img_in` / `proj_out` for the device path.
    img_in_f32: Vec<f32>,
    proj_out_f32: Vec<f32>,
}

/// The device denoiser is allowed (`CMF_QI21_GPU=0` forces the host).
pub fn gpu_allowed() -> bool {
    std::env::var("CMF_QI21_GPU").as_deref() != Ok("0") && crate::gpu::enabled()
}

// ───────────────────────────── small helpers (copied, not shared) ─────

struct SendRows(*mut f32);
unsafe impl Send for SendRows {}
unsafe impl Sync for SendRows {}
impl SendRows {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn gelu_tanh(v: f32) -> f32 {
    let x = v as f64;
    (0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh()))
        as f32
}

/// Affine-free LayerNorm (biased variance), f64 accumulation.
fn layer_norm_into(x: &[f32], eps: f64, dst: &mut [f32]) {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    for (d, &v) in dst.iter_mut().zip(x) {
        *d = ((v as f64 - mean) * inv) as f32;
    }
}

fn rms_norm_inplace(v: &mut [f32], w: &[f32], eps: f64) {
    let ss = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for (x, &g) in v.iter_mut().zip(w) {
        *x = (*x as f64 * inv) as f32 * g;
    }
}

/// Row softmax over the first `valid` entries (the rest are zeroed).
fn softmax_prefix(row: &mut [f32], valid: usize) {
    let (live, dead) = row.split_at_mut(valid);
    let mx = live.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut den = 0f64;
    for r in live.iter_mut() {
        *r = (*r - mx).exp();
        den += *r as f64;
    }
    let inv = (1.0 / den) as f32;
    for r in live.iter_mut() {
        *r *= inv;
    }
    dead.fill(0.0);
}

/// y[n, rows] = x[n, cols] · Wᵀ on the host reference GEMM (a quantized
/// weight is dequantized to f32 in row chunks: weight-only error).
fn lin(p: &Proj, x: &[f32], n: usize, y: &mut [f32], pool: Option<&Pool>) {
    use crate::zimage::host_gemm;
    let (rows, cols) = (p.rows(), p.cols());
    match p {
        Proj::F32 { w, .. } => host_gemm::gemm_nt(x, w, y, n, cols, rows, pool),
        Proj::Q(q) => {
            const CH: usize = 768;
            let mut wbuf = vec![0f32; CH.min(rows) * cols];
            let mut r0 = 0;
            while r0 < rows {
                let rc = CH.min(rows - r0);
                {
                    let wp = SendRows(wbuf.as_mut_ptr());
                    pool_rows(pool, rc, &|lo, hi| {
                        for r in lo..hi {
                            // SAFETY: disjoint rows.
                            q.row_f32(r0 + r, unsafe { wp.row(r * cols, cols) });
                        }
                    });
                }
                host_gemm::gemm_nt_ld(x, &wbuf[..rc * cols], &mut y[r0..], rows, n, cols, rc, pool);
                r0 += rc;
            }
        }
    }
}

/// One row through a small projection, f64 accumulation.
fn lin_row(p: &Proj, x: &[f32], pool: Option<&Pool>) -> Vec<f32> {
    let (rows, cols) = (p.rows(), p.cols());
    let mut out = vec![0f32; rows];
    let op = SendRows(out.as_mut_ptr());
    pool_rows(pool, rows, &|lo, hi| {
        let mut wrow = vec![0f32; cols];
        for o in lo..hi {
            let w: &[f32] = match p {
                Proj::F32 { w, .. } => &w[o * cols..(o + 1) * cols],
                Proj::Q(q) => {
                    q.row_f32(o, &mut wrow);
                    &wrow
                }
            };
            let s: f64 = w.iter().zip(x).map(|(&a, &c)| a as f64 * c as f64).sum();
            // SAFETY: disjoint outputs.
            unsafe { op.row(o, 1)[0] = s as f32 };
        }
    });
    out
}

fn cfg_of(model: &CmfModel) -> Result<Qi21Config, String> {
    let raw = model
        .tensor_bytes("dit.config_json")
        .map_err(|e| format!("dit.config_json: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_slice(raw).map_err(|e| format!("dit.config_json: {e}"))?;
    Qi21Config::from_json(&v)
}

/// The modulation of one timestep: `(mods [4·dim], final_scale [dim])`.
pub struct Qi21Mods {
    pub mods: Vec<f32>,
    pub final_scale: Vec<f32>,
}

impl Qi21Dit {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        let cfg = cfg_of(model)?;
        let p = |n: &str| Proj::from_model(model, &format!("dit.{n}"));
        let f = |n: &str| crate::dit::cmf_f32(model, &format!("dit.{n}"));
        let mut blocks = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let b = format!("transformer_blocks.{l}");
            let names = [
                "attn.to_q.weight",
                "attn.to_k.weight",
                "attn.to_v.weight",
                "attn.to_out.0.weight",
                "img_mlp.gate_layer.weight",
                "img_mlp.proj.weight",
                "img_mlp.out.weight",
            ];
            let mut idx = [0usize; 7];
            let mut all = true;
            for (i, nm) in names.iter().enumerate() {
                match model.tensor_index(&format!("dit.{b}.{nm}")) {
                    Some(t) => idx[i] = t,
                    None => all = false,
                }
            }
            blocks.push(Block {
                idx: all.then_some(idx),
                q: p(&format!("{b}.attn.to_q.weight"))?,
                k: p(&format!("{b}.attn.to_k.weight"))?,
                v: p(&format!("{b}.attn.to_v.weight"))?,
                o: p(&format!("{b}.attn.to_out.0.weight"))?,
                norm_q: f(&format!("{b}.attn.norm_q.weight"))?,
                norm_k: f(&format!("{b}.attn.norm_k.weight"))?,
                gate: p(&format!("{b}.img_mlp.gate_layer.weight"))?,
                up: p(&format!("{b}.img_mlp.proj.weight"))?,
                down: p(&format!("{b}.img_mlp.out.weight"))?,
            });
        }
        let dit = Self {
            img_in: p("img_in.weight")?,
            txt_norm: f("txt_in.text_norm.weight")?,
            txt_in1: p("txt_in.in_layer.weight")?,
            txt_in2: p("txt_in.out_layer.weight")?,
            t_lin1: p("time_text_embed.timestep_embedder.linear_1.weight")?,
            t_lin2: p("time_text_embed.timestep_embedder.linear_2.weight")?,
            modulation: p("modulation.1.weight")?,
            norm_out: p("norm_out.linear.weight")?,
            proj_out: p("proj_out.weight")?,
            blocks,
            pool: Pool::from_env(),
            model: Some(model.clone()),
            img_in_f32: f("img_in.weight")?,
            proj_out_f32: f("proj_out.weight")?,
            cfg,
        };
        dit.check_shapes()?;
        Ok(dit)
    }

    fn check_shapes(&self) -> Result<(), String> {
        let c = &self.cfg;
        let want = |name: &str, p: &Proj, r: usize, k: usize| -> Result<(), String> {
            if p.rows() != r || p.cols() != k {
                return Err(format!(
                    "dit.{name}: [{}, {}], the config needs [{r}, {k}]",
                    p.rows(),
                    p.cols()
                ));
            }
            Ok(())
        };
        want("img_in", &self.img_in, c.dim, c.in_channels)?;
        want("txt_in.in_layer", &self.txt_in1, c.dim, c.context_in)?;
        want("modulation", &self.modulation, 4 * c.dim, c.dim)?;
        want("proj_out", &self.proj_out, c.out_channels, c.dim)?;
        let b = &self.blocks[0];
        want("to_q", &b.q, c.dim, c.dim)?;
        want("gate_layer", &b.gate, c.mlp_hidden, c.dim)?;
        want("out", &b.down, c.dim, c.mlp_hidden)?;
        Ok(())
    }

    pub fn model(&self) -> Option<&Arc<CmfModel>> {
        self.model.as_ref()
    }

    fn pool(&self) -> Option<&Pool> {
        self.pool.as_deref()
    }

    /// Sinusoidal timestep embedding → MLP: `temb [dim]` for the model
    /// input `t ∈ [0, 1]` (the pipeline passes `timestep / 1000`).
    pub fn temb(&self, t: f32) -> Vec<f32> {
        const HALF: usize = 128;
        let ts = 1000.0f32 * t;
        let mut e = vec![0f32; 2 * HALF];
        for i in 0..HALF {
            // torch: exp(-ln(10000) · arange(half) / half) in f32
            let f = (-(10000f32).ln() * i as f32 / HALF as f32).exp();
            let a = ts * f;
            e[i] = a.cos();
            e[HALF + i] = a.sin();
        }
        let pool = self.pool();
        let mut h = lin_row(&self.t_lin1, &e, pool);
        for v in h.iter_mut() {
            *v = silu(*v);
        }
        lin_row(&self.t_lin2, &h, pool)
    }

    /// Shared block modulation `[scale1, gate1, scale2, gate2]` and the
    /// final-norm scale of one timestep.
    pub fn mods(&self, t: f32) -> Qi21Mods {
        let temb = self.temb(t);
        let s: Vec<f32> = temb.iter().map(|&v| silu(v)).collect();
        let pool = self.pool();
        Qi21Mods {
            mods: lin_row(&self.modulation, &s, pool),
            final_scale: lin_row(&self.norm_out, &s, pool),
        }
    }

    /// `txt_in`: `[n, context_in]` → `[n, dim]`.
    pub fn embed_text(&self, feats: &[f32], n: usize) -> Vec<f32> {
        let c = &self.cfg;
        let pool = self.pool();
        let mut x = feats[..n * c.context_in].to_vec();
        for row in x.chunks_exact_mut(c.context_in) {
            let ss = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / c.context_in as f64;
            let inv = 1.0 / (ss + c.eps).sqrt();
            for (v, &w) in row.iter_mut().zip(&self.txt_norm) {
                *v = (*v as f64 * inv) as f32 * (w + 1.0);
            }
        }
        let mut h = vec![0f32; n * c.dim];
        lin(&self.txt_in1, &x, n, &mut h, pool);
        for v in h.iter_mut() {
            *v = gelu_tanh(*v);
        }
        let mut out = vec![0f32; n * c.dim];
        lin(&self.txt_in2, &h, n, &mut out, pool);
        out
    }

    /// `img_in`: latent tokens `[n, in_channels]` → `[n, dim]`.
    pub fn embed_image(&self, tok: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * self.cfg.dim];
        lin(&self.img_in, tok, n, &mut out, self.pool());
        out
    }

    /// dst = LN(src) · (1 + scale)
    fn norm_mod(&self, src: &[f32], scale: &[f32], dst: &mut [f32], n: usize) {
        let hs = self.cfg.dim;
        let eps = self.cfg.eps;
        let sr = SendRows(dst.as_mut_ptr());
        pool_rows(self.pool(), n, &|lo, hi| {
            for p in lo..hi {
                // SAFETY: disjoint rows.
                let row = unsafe { sr.row(p * hs, hs) };
                layer_norm_into(&src[p * hs..(p + 1) * hs], eps, row);
                for (r, &s) in row.iter_mut().zip(scale) {
                    *r *= 1.0 + s;
                }
            }
        });
    }

    /// x += tanh(gate) ⊙ y
    fn gated_add(&self, x: &mut [f32], y: &[f32], gate_tanh: &[f32], n: usize) {
        let hs = self.cfg.dim;
        let sr = SendRows(x.as_mut_ptr());
        pool_rows(self.pool(), n, &|lo, hi| {
            for p in lo..hi {
                // SAFETY: disjoint rows.
                let row = unsafe { sr.row(p * hs, hs) };
                for ((d, &v), &g) in row.iter_mut().zip(&y[p * hs..(p + 1) * hs]).zip(gate_tanh) {
                    *d += g * v;
                }
            }
        });
    }

    /// qk RMSNorm + RoPE in place on `[n, heads·hd]`.
    fn qk_norm_rope(&self, all: &mut [f32], w: &[f32], n: usize, cos: &[f32], sin: &[f32]) {
        let (nh, hd) = (self.cfg.heads, self.cfg.head_dim);
        let pairs = hd / 2;
        let eps = self.cfg.eps;
        let sr = SendRows(all.as_mut_ptr());
        pool_rows(self.pool(), n, &|lo, hi| {
            for p in lo..hi {
                for h in 0..nh {
                    // SAFETY: disjoint tokens.
                    let v = unsafe { sr.row((p * nh + h) * hd, hd) };
                    rms_norm_inplace(v, w, eps);
                    for j in 0..pairs {
                        let (c, s) = (cos[p * pairs + j], sin[p * pairs + j]);
                        let (a, b) = (v[2 * j], v[2 * j + 1]);
                        v[2 * j] = a * c - b * s;
                        v[2 * j + 1] = a * s + b * c;
                    }
                }
            }
        });
    }

    /// Attention of `n` query rows over `m` key rows. `keys_of(i)` = how
    /// many leading keys query `i` sees (a prefix of the key rows) plus an
    /// extra `[lo, hi)` range it also sees (its own image block); keys not
    /// covered are masked.
    fn attention(
        &self,
        q_all: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        n: usize,
        m: usize,
        visible: &(dyn Fn(usize) -> usize + Sync),
        out: &mut [f32],
    ) {
        use crate::zimage::host_gemm;
        let (nh, hd) = (self.cfg.heads, self.cfg.head_dim);
        let pool = self.pool();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; m * hd];
        let mut vt = vec![0f32; hd * m];
        let mut scores = vec![0f32; n * m];
        let mut oh = vec![0f32; n * hd];
        for h in 0..nh {
            for p in 0..n {
                for d in 0..hd {
                    qh[p * hd + d] = q_all[(p * nh + h) * hd + d] * scale;
                }
            }
            for p in 0..m {
                kh[p * hd..(p + 1) * hd].copy_from_slice(&k_all[(p * nh + h) * hd..(p * nh + h + 1) * hd]);
                for d in 0..hd {
                    vt[d * m + p] = v_all[(p * nh + h) * hd + d];
                }
            }
            host_gemm::gemm_nt(&qh, &kh, &mut scores, n, hd, m, pool);
            {
                let sp = SendRows(scores.as_mut_ptr());
                pool_rows(pool, n, &|lo, hi| {
                    for r in lo..hi {
                        // SAFETY: disjoint rows.
                        softmax_prefix(unsafe { sp.row(r * m, m) }, visible(r));
                    }
                });
            }
            host_gemm::gemm_nt(&scores, &vt, &mut oh, n, m, hd, pool);
            for p in 0..n {
                out[(p * nh + h) * hd..(p * nh + h + 1) * hd].copy_from_slice(&oh[p * hd..(p + 1) * hd]);
            }
        }
    }

    /// One block on the host, in place on `x` `[n, dim]` (the query rows).
    /// `kv_prefix` = keys/values of earlier rows (post norm+rope) that the
    /// queries also attend to; `visible(i)` counts the keys query `i` sees
    /// in `[kv_prefix ++ own]` order (a leading run). Returns this block's
    /// own keys and values.
    #[allow(clippy::too_many_arguments)]
    fn block_cpu(
        &self,
        l: usize,
        x: &mut [f32],
        n: usize,
        m: &[f32],
        rope: (&[f32], &[f32]),
        kv_prefix: Option<(&[f32], &[f32])>,
        visible: &(dyn Fn(usize) -> usize + Sync),
    ) -> (Vec<f32>, Vec<f32>) {
        let b = &self.blocks[l];
        let hs = self.cfg.dim;
        let pool = self.pool();
        let (s1, g1, s2, g2) = (&m[..hs], &m[hs..2 * hs], &m[2 * hs..3 * hs], &m[3 * hs..4 * hs]);
        let g1t: Vec<f32> = g1.iter().map(|v| v.tanh()).collect();
        let g2t: Vec<f32> = g2.iter().map(|v| v.tanh()).collect();
        let mut xn = vec![0f32; n * hs];
        self.norm_mod(x, s1, &mut xn, n);
        let mut q = vec![0f32; n * hs];
        let mut k = vec![0f32; n * hs];
        let mut v = vec![0f32; n * hs];
        lin(&b.q, &xn, n, &mut q, pool);
        lin(&b.k, &xn, n, &mut k, pool);
        lin(&b.v, &xn, n, &mut v, pool);
        self.qk_norm_rope(&mut q, &b.norm_q, n, rope.0, rope.1);
        self.qk_norm_rope(&mut k, &b.norm_k, n, rope.0, rope.1);
        let mut attn = vec![0f32; n * hs];
        match kv_prefix {
            Some((pk, pv)) => {
                let lp = pk.len() / hs;
                let mut ka = Vec::with_capacity((lp + n) * hs);
                ka.extend_from_slice(pk);
                ka.extend_from_slice(&k);
                let mut va = Vec::with_capacity((lp + n) * hs);
                va.extend_from_slice(pv);
                va.extend_from_slice(&v);
                self.attention(&q, &ka, &va, n, lp + n, visible, &mut attn);
            }
            None => self.attention(&q, &k, &v, n, n, visible, &mut attn),
        }
        let mut proj = vec![0f32; n * hs];
        lin(&b.o, &attn, n, &mut proj, pool);
        drop(attn);
        self.gated_add(x, &proj, &g1t, n);
        self.norm_mod(x, s2, &mut xn, n);
        let inter = self.cfg.mlp_hidden;
        let mut ga = vec![0f32; n * inter];
        let mut up = vec![0f32; n * inter];
        lin(&b.gate, &xn, n, &mut ga, pool);
        lin(&b.up, &xn, n, &mut up, pool);
        {
            let sg = SendRows(ga.as_mut_ptr());
            pool_rows(pool, n, &|lo, hi| {
                for p in lo..hi {
                    // SAFETY: disjoint rows.
                    let g = unsafe { sg.row(p * inter, inter) };
                    for (gv, &uv) in g.iter_mut().zip(&up[p * inter..(p + 1) * inter]) {
                        *gv = silu(*gv) * uv;
                    }
                }
            });
        }
        drop(up);
        lin(&b.down, &ga, n, &mut proj, pool);
        self.gated_add(x, &proj, &g2t, n);
        (k, v)
    }

    /// Run the prefix (text rows already `embed_text`-ed, condition-image
    /// rows `embed_image`-ed, in layout order) through every block with
    /// the t = 0 modulation and the block-causal mask; keep each layer's
    /// keys and values.
    pub fn prefill(&self, mut x: Vec<f32>, layout: &Qi21Layout) -> Result<Qi21Prefix, String> {
        layout.validate()?;
        let lp = layout.prefix_len();
        let hs = self.cfg.dim;
        if x.len() != lp * hs {
            return Err(format!("prefix rows: {} floats, the layout needs {}", x.len(), lp * hs));
        }
        let t0 = if self.cfg.causal_condition {
            self.mods(0.0)
        } else {
            return Err("causal_condition = false has no step-independent prefix".into());
        };
        let pos = layout.positions();
        let (cos, sin) = rope_tables(&pos[..lp], self.cfg.axes, 10000.0);
        // query i sees keys [0, end_i): its own image block's end, else i+1
        let ids = layout.block_ids();
        let mut end = vec![0usize; lp];
        for i in 0..lp {
            end[i] = if ids[i] < 0 {
                i + 1
            } else {
                let mut e = i + 1;
                while e < lp && ids[e] == ids[i] {
                    e += 1;
                }
                e
            };
        }
        if gpu_allowed() {
            if let Some(key) = self.prefill_device(&x, layout, &t0, (&cos, &sin), &end) {
                return Ok(Qi21Prefix {
                    layout: layout.clone(),
                    k: Vec::new(),
                    v: Vec::new(),
                    device_key: Some(key),
                    rope_t: self.target_rope(layout),
                });
            }
        }
        let visible = |i: usize| end[i];
        let mut ks = Vec::with_capacity(self.cfg.layers);
        let mut vs = Vec::with_capacity(self.cfg.layers);
        for l in 0..self.cfg.layers {
            let (k, v) = self.block_cpu(l, &mut x, lp, &t0.mods, (&cos, &sin), None, &visible);
            ks.push(k);
            vs.push(v);
        }
        Ok(Qi21Prefix {
            layout: layout.clone(),
            k: ks,
            v: vs,
            device_key: None,
            rope_t: self.target_rope(layout),
        })
    }

    fn geom(&self) -> crate::gpu::Qi21Geom {
        crate::gpu::Qi21Geom {
            hidden: self.cfg.dim,
            nh: self.cfg.heads,
            hd: self.cfg.head_dim,
            inter: self.cfg.mlp_hidden,
            in_ch: self.cfg.in_channels,
            eps: self.cfg.eps as f32,
        }
    }

    /// Build the device program and run the prefix there; `None` = the
    /// host path runs instead.
    fn prefill_device(
        &self,
        x: &[f32],
        layout: &Qi21Layout,
        t0: &Qi21Mods,
        rope_p: (&[f32], &[f32]),
        end: &[usize],
    ) -> Option<u64> {
        static KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let model = self.model.as_ref()?;
        let refs: Vec<crate::gpu::Qi21BlockRef> = self
            .blocks
            .iter()
            .map(|b| {
                b.idx.map(|w| crate::gpu::Qi21BlockRef {
                    w,
                    norm_q: &b.norm_q,
                    norm_k: &b.norm_k,
                })
            })
            .collect::<Option<_>>()?;
        let (ct, st) = self.target_rope(layout);
        let vis: Vec<u32> = end.iter().map(|&e| e as u32).collect();
        let key = KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let ok = crate::gpu::qi21_prefill(&crate::gpu::Qi21PrefillArgs {
            model,
            geom: self.geom(),
            blocks: &refs,
            img_in: &self.img_in_f32,
            proj_out: &self.proj_out_f32,
            key,
            x,
            lp: layout.prefix_len(),
            rope_p,
            rope_t: (&ct, &st),
            vis: &vis,
            mods0: &t0.mods,
            n: layout.target().len(),
        });
        ok.then_some(key)
    }

    /// One denoiser call: the device program when the prefix lives there,
    /// else the host path.
    pub fn step(&self, prefix: &Qi21Prefix, tok: &[f32], mods: &Qi21Mods) -> Result<Vec<f32>, String> {
        match prefix.device_key {
            Some(key) => {
                let n = prefix.layout.target().len();
                let fs: Vec<f32> = mods.final_scale.iter().map(|&v| 1.0 + v).collect();
                let mut out = vec![0f32; n * self.cfg.out_channels];
                if crate::gpu::qi21_step(key, tok, &mods.mods, &fs, &mut out) {
                    Ok(out)
                } else {
                    Err("the device denoiser step failed; rerun with CMF_QI21_GPU=0 for the host path".into())
                }
            }
            None => Ok(self.step_cpu(prefix, tok, mods)),
        }
    }

    /// Target-row RoPE tables for a prefix layout.
    pub fn target_rope(&self, layout: &Qi21Layout) -> (Vec<f32>, Vec<f32>) {
        let pos = layout.positions();
        rope_tables(&pos[layout.prefix_len()..], self.cfg.axes, 10000.0)
    }

    /// One denoiser call on the host: latent tokens `[n, in_channels]` →
    /// velocity `[n, out_channels]`, attending to the cached prefix (the
    /// target rows rotate by the prefix's own `rope_t`).
    pub fn step_cpu(&self, prefix: &Qi21Prefix, tok: &[f32], mods: &Qi21Mods) -> Vec<f32> {
        let rope = (&prefix.rope_t.0[..], &prefix.rope_t.1[..]);
        let n = prefix.layout.target().len();
        let lp = prefix.len();
        let hs = self.cfg.dim;
        let mut x = self.embed_image(tok, n);
        let all = |_: usize| lp + n;
        for l in 0..self.cfg.layers {
            self.block_cpu(
                l,
                &mut x,
                n,
                &mods.mods,
                rope,
                Some((&prefix.k[l], &prefix.v[l])),
                &all,
            );
        }
        let mut xn = vec![0f32; n * hs];
        self.norm_mod(&x, &mods.final_scale, &mut xn, n);
        let mut out = vec![0f32; n * self.cfg.out_channels];
        lin(&self.proj_out, &xn, n, &mut out, self.pool());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t2i_positions_follow_the_reference_rope_layout() {
        let l = Qi21Layout::t2i(3, 2, 3);
        let p = l.positions();
        assert_eq!(p[0], [0, 0, 0]);
        assert_eq!(p[2], [2, 2, 2]);
        // image frame = position after the text; h in [-(2-1), 1) = {-1, 0},
        // w in [-(3-1), 1) = {-2, -1, 0}
        assert_eq!(p[3], [3, -1, -2]);
        assert_eq!(p[5], [3, -1, 0]);
        assert_eq!(p[8], [3, 0, 0]);
        assert_eq!(l.prefix_len(), 3);
        assert_eq!(l.block_ids(), vec![-1, -1, -1, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn text_after_an_image_resumes_from_the_larger_side() {
        let l = Qi21Layout {
            segs: vec![Seg::Text(2), Seg::Image(2, 4), Seg::Text(1), Seg::Image(1, 1)],
        };
        let p = l.positions();
        // image block at frame 2, then the position advances by max(2, 4)
        assert_eq!(p[2][0], 2);
        assert_eq!(p[10], [6, 6, 6]);
        assert_eq!(p[11], [7, -1, -1]);
    }

    #[test]
    fn rope_tables_are_unit_rotations() {
        let l = Qi21Layout::t2i(4, 2, 2);
        let (c, s) = rope_tables(&l.positions(), [16, 56, 56], 10000.0);
        assert_eq!(c.len(), l.total() * 64);
        for (a, b) in c.iter().zip(&s) {
            assert!((a * a + b * b - 1.0).abs() < 1e-5);
        }
        // position 0 → identity
        assert!(c[..64].iter().all(|&v| (v - 1.0).abs() < 1e-7));
    }
}
