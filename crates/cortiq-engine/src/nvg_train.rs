//! NVG Ť on the device: layer-streamed distillation training of the metric
//! of a compressed model against its full-precision teacher, for the
//! Qwen3.5 family (GatedDeltaNet + gated full attention + SwiGLU).
//!
//! Every op runs on the card (`gpu_wgpu::{train_ops, train_mm, train_head,
//! gdn_train}`, each parity-tested against an f64 host reference):
//!
//! * teacher pass — the FP model (f16 planes streamed one layer at a time)
//!   over the training windows; per token its top-k next-token distribution
//!   is kept on the host;
//! * student step — forward over `nb` windows of `t` tokens with the layer
//!   inputs checkpointed, the loss `mean_t KL(p_T,topk ‖ p_S)` through the
//!   frozen head, then layer by layer in reverse: the layer's forward is
//!   recomputed from its checkpoint and its backward run, accumulating the
//!   gradients of the trainable metric (binary low-rank h, ℓ, g of every
//!   path and every RMSNorm gain);
//! * Adam on the metric; ℓ entries that only pad the rank to the GEMM tile
//!   stay zero (masked), so the bit budget never grows;
//! * optionally the signs themselves (`SignInit` at load, a `SignStepCfg`
//!   per step): one straight-through latent per sign, updated inside each
//!   path's backward right after its gradient is formed (`gpu_wgpu::train_sign`).

use crate::gpu_wgpu::gdn_train::{self as gdn, GdnTape, GdnTrainCfg};
use crate::gpu_wgpu::train_head::{self as th, Head, HeadScratch};
use crate::gpu_wgpu::train_mm::{self as tm, Blr, BlrPath, BlrScratch, Op};
use crate::gpu_wgpu::train_sign::{self as ts, SignCtx, SignInit, SignScratch, SignStepCfg};
use crate::gpu_wgpu::train_attn::{self as ta, AttScratch};
use crate::gpu_wgpu::train_ops::{self as to, AttnGeom, QkrGeom};
use cortiq_core::quant::{dequant_tensor, f16_to_f32, f32_to_f16};
use cortiq_core::{CmfModel, LayerType, NormStyle};
use std::collections::HashMap;

type Ctx = crate::gpu_wgpu::CtxRef;

/// The opt-in step profiler (`CMF_NVG_PROF`, see `gpu_wgpu::train_prof`).
pub use crate::gpu_wgpu::train_prof as prof;

// ───────────────────────────── geometry ─────────────────────────────

#[derive(Clone, Debug)]
pub struct Geom {
    pub hs: usize,
    pub nl: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub half: usize,
    pub inter: usize,
    pub vocab: usize,
    pub eps: f32,
    pub gemma: bool,
    pub gdn_layer: Vec<bool>,
    pub nv: usize,
    pub nk: usize,
    pub kk: usize,
    pub inv_freq: Vec<f64>,
}

impl Geom {
    pub fn from_model(m: &CmfModel) -> Result<Self, String> {
        let a = m.arch();
        let gdn_layer: Vec<bool> = (0..a.num_layers)
            .map(|l| matches!(a.layer_types.get(l), Some(LayerType::LinearAttention)))
            .collect();
        let (nv, nk, kk) = match &a.linear_core {
            Some(lc) => (
                lc.num_heads,
                a.linear_num_key_heads.ok_or("linear_num_key_heads")?,
                a.linear_conv_kernel_dim.ok_or("linear_conv_kernel_dim")?,
            ),
            None => (0, 0, 0),
        };
        if gdn_layer.iter().any(|&g| g) && (a.linear_key_head_dim != Some(128)) {
            return Err("the device GDN kernels take 128-wide heads".into());
        }
        let rotary = ((a.head_dim as f32 * a.partial_rotary_factor) as usize).max(2).min(a.head_dim);
        let inv_freq = (0..rotary / 2).map(|i| 1.0 / a.rope_theta.powf(2.0 * i as f64 / rotary as f64)).collect();
        Ok(Geom {
            hs: a.hidden_size,
            nl: a.num_layers,
            nh: a.num_attention_heads,
            nkv: a.num_kv_heads,
            hd: a.head_dim,
            half: rotary / 2,
            inter: a.intermediate_size,
            vocab: a.vocab_size,
            eps: a.rms_norm_eps as f32,
            gemma: matches!(a.norm_style, NormStyle::Gemma),
            gdn_layer,
            nv,
            nk,
            kk,
            inv_freq,
        })
    }
    fn vd(&self) -> usize {
        self.nv * 128
    }
    fn c_dim(&self) -> usize {
        2 * self.nk * 128 + self.vd()
    }
}

// ───────────────────────────── parameters ─────────────────────────────

/// A trainable f32 vector with its gradient and Adam moments.
pub struct Param {
    pub n: usize,
    pub w: wgpu::Buffer,
    pub g: wgpu::Buffer,
    m1: wgpu::Buffer,
    m2: wgpu::Buffer,
    mask: Option<wgpu::Buffer>,
    /// 0: plain Adam steps. > 0: steps relative to each value's magnitude
    /// (floored at this), for the metric scales whose magnitudes span orders
    /// (h, g ~ 1, ℓ carrying the weight's amplitude).
    rel: f32,
}

impl Param {
    fn new(c: Ctx, v: &[f32], mask: Option<&[f32]>) -> Self {
        Param {
            n: v.len(),
            w: to::upload(c, v),
            g: to::zeros(c, v.len()),
            m1: to::zeros(c, v.len()),
            m2: to::zeros(c, v.len()),
            mask: mask.map(|m| to::upload(c, m)),
            rel: 0.0,
        }
    }

    /// A metric scale: relative steps floored at 5 % of the live values' RMS.
    fn new_metric(c: Ctx, v: &[f32], mask: Option<&[f32]>) -> Self {
        let (mut ss, mut n) = (0f64, 0usize);
        for (i, &x) in v.iter().enumerate() {
            if mask.is_none_or(|m| m[i] != 0.0) {
                ss += (x as f64) * (x as f64);
                n += 1;
            }
        }
        let rms = (ss / n.max(1) as f64).sqrt() as f32;
        let mut p = Param::new(c, v, mask);
        p.rel = if rms > 0.0 { 0.05 * rms } else { 0.0 };
        p
    }
}

/// A linear layer: binary low-rank (trainable metric), a frozen f16 plane
/// (teacher), a small dense f32 matrix (frozen), or a dense matrix trained
/// in full (healing: f32 master weights with Adam state, the f16 plane and
/// its transpose refreshed after every step).
pub enum Lin {
    Blr { blr: Blr, role: &'static str, params: Vec<[Param; 3]> },
    Plane { w: wgpu::Buffer, wt: Option<wgpu::Buffer>, m: usize, k: usize },
    Dense { w: wgpu::Buffer, m: usize, k: usize },
    Train { p: Param, pl: wgpu::Buffer, plt: wgpu::Buffer, m: usize, k: usize },
    /// Not on the device: a layer outside the net's range (a healed tail
    /// fed from cached prefix activations, or a prefix net).
    Absent { m: usize },
}

impl Lin {
    /// Output rows.
    pub fn m(&self) -> usize {
        match self {
            Lin::Blr { blr, .. } => blr.m,
            Lin::Plane { m, .. } | Lin::Dense { m, .. } | Lin::Train { m, .. } | Lin::Absent { m } => *m,
        }
    }
}

/// Which weights a healing run trains (`Net::load_heal`): every layer from
/// `train_from` up — its dense linears in full (all of them, or only the
/// MLP's with `ffn_only`) and its norm gains; the layers below are frozen
/// and the backward stops at `train_from`.
#[derive(Clone, Copy, Debug)]
pub struct HealCfg {
    pub train_from: usize,
    pub ffn_only: bool,
    /// The frozen layers stay off the device: each step starts from their
    /// output, computed once by a [`Net::load_prefix`] net (`step_wh`).
    pub prefix_cached: bool,
}

pub enum Mix {
    Gdn { qkv: Lin, z: Lin, a: Lin, b: Lin, out: Lin, a_log: Vec<f32>, dt_bias: Vec<f32>, norm: Vec<f32>, conv: Vec<f32> },
    Attn { q: Lin, k: Lin, v: Lin, o: Lin, qn: Param, kn: Param },
}

pub struct Layer {
    pub ln1: Param,
    pub ln2: Param,
    pub mix: Mix,
    pub gate: Lin,
    pub up: Lin,
    pub down: Lin,
    /// DTG-MA: a gate on the MLP's intermediate ([`Net::enable_masks`]).
    pub mask: Option<MaskGate>,
}

/// A gate per MLP neuron between the SwiGLU and `down_proj`: `h·s` with
/// `s` set by the caller (σ(m) of the mask logits while training, 0/1 for
/// the hard mask), and `Σ_tokens dL/d(h·s)·h` — the gradient with respect
/// to `s` — accumulated on the card.
pub struct MaskGate {
    pub s: wgpu::Buffer,
    pub g: wgpu::Buffer,
    pub w: usize,
}

/// Every parameter laid end to end, for the optimizer: one dispatch for all
/// gradient norms and one for all Adam steps instead of thousands (values,
/// moments and gradients are copied in and back; the arithmetic per entry
/// is that of `train_mm::{sumsq, adam}`).
struct OptCat {
    off: Vec<usize>,
    total: usize,
    g: wgpu::Buffer,
    w: wgpu::Buffer,
    m1: wgpu::Buffer,
    m2: wgpu::Buffer,
    mask: wgpu::Buffer,
    relf: wgpu::Buffer,
    tab: wgpu::Buffer,
    part: wgpu::Buffer,
    /// `g` holds this step's gradients (set by `grad_norm`).
    g_fresh: std::sync::atomic::AtomicBool,
}

pub struct Net {
    pub geom: Geom,
    /// Lowest layer with trainable weights (0 unless healing a tail).
    pub train_from: usize,
    /// The layers on the device (`first..end`); the forward starts at
    /// `first` from given activations when it is not 0 and stops at `end`.
    pub first: usize,
    pub end: usize,
    /// Lowest layer the backward currently reaches (≥ `train_from`): a
    /// progressive unfreeze widens it epoch by epoch; layers below get no
    /// gradient, so Adam leaves them untouched.
    pub active_from: std::sync::atomic::AtomicUsize,
    pub layers: Vec<Layer>,
    pub final_norm: Param,
    pub head: Head,
    /// Embedding rows on the host (f16 bits), gathered per step.
    embed: Vec<u16>,
    /// Sign matrices with training state (ids 0..n_sign_mats).
    pub n_sign_mats: u32,
    opt: std::sync::OnceLock<OptCat>,
    /// Partial sums of squares of the dense trained weights' gradients.
    dense_opt: std::sync::OnceLock<wgpu::Buffer>,
}

fn deq(m: &CmfModel, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let e = m.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let n: usize = e.shape.iter().product();
    let mut v = vec![0f32; n];
    dequant_tensor(e, m.entry_bytes(e), &mut v).map_err(|x| format!("{name}: {x}"))?;
    Ok((v, e.shape.clone()))
}

fn f16_plane(c: Ctx, v: &[f32]) -> wgpu::Buffer {
    let mut h: Vec<u16> = v.iter().map(|&f| f32_to_f16(f)).collect();
    if h.len() % 2 == 1 {
        h.push(0);
    }
    let b = tm::f16_buf(c, h.len());
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(&h));
    b
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0f32; v.len()];
    for r in 0..rows {
        for k in 0..cols {
            t[k * rows + r] = v[r * cols + k];
        }
    }
    t
}

/// Parse a binary low-rank payload (dtype id 19, docs/BLR_NVG.md layout:
/// 16-byte header, then per path h f16[m], g f16[n], ℓ f16[r], U bits
/// [m][r/8], V bits [r][n/8]) into device paths, padding the rank to a
/// multiple of 128 with ℓ = 0 (masked).
fn load_blr(c: Ctx, bytes: &[u8], role: &'static str, sign: Option<&SignInit>, next_id: &mut u32) -> Result<Lin, String> {
    if bytes.len() < 16 || bytes[0] != 0x4C {
        return Err(format!("{role}: not a binary low-rank payload"));
    }
    let np = bytes[2] as usize;
    let r = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let m = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let rp = r.div_ceil(128) * 128;
    let wpr = rp / 32;
    let mut off = 16;
    let f16s = |b: &[u8], k: usize| -> Vec<f32> { (0..k).map(|i| f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))).collect() };
    let mut paths = Vec::new();
    let mut params = Vec::new();
    for _ in 0..np {
        let h = f16s(&bytes[off..], m);
        off += 2 * m;
        let g = f16s(&bytes[off..], n);
        off += 2 * n;
        let mut l = f16s(&bytes[off..], r);
        off += 2 * r;
        let ub = &bytes[off..off + m * r / 8];
        off += m * r / 8;
        let vb = &bytes[off..off + r * n / 8];
        off += r * n / 8;
        let mut bu = vec![0u32; m * wpr];
        for i in 0..m {
            for k in 0..r {
                if (ub[i * r / 8 + k / 8] >> (k % 8)) & 1 == 1 {
                    bu[i * wpr + k / 32] |= 1 << (k % 32);
                }
            }
        }
        let mut bv = vec![0u32; n * wpr];
        for k in 0..r {
            for j in 0..n {
                if (vb[k * n / 8 + j / 8] >> (j % 8)) & 1 == 1 {
                    bv[j * wpr + k / 32] |= 1 << (k % 32);
                }
            }
        }
        l.resize(rp, 0.0);
        let mask: Vec<f32> = (0..rp).map(|k| if k < r { 1.0 } else { 0.0 }).collect();
        let ub_buf = to::zeros(c, bu.len());
        c.queue.write_buffer(&ub_buf, 0, bytemuck::cast_slice(&bu));
        let vb_buf = to::zeros(c, bv.len());
        c.queue.write_buffer(&vb_buf, 0, bytemuck::cast_slice(&bv));
        let ph = Param::new_metric(c, &h, None);
        let pl = Param::new_metric(c, &l, Some(&mask));
        let pg = Param::new_metric(c, &g, None);
        let sp = match sign {
            Some(init) => {
                let u = ts::init_mat(c, &ub_buf, m, r, rp, init, *next_id).ok_or("sign init")?;
                let v = ts::init_mat(c, &vb_buf, n, r, rp, init, *next_id + 1).ok_or("sign init")?;
                *next_id += 2;
                Some(ts::SignPath { u, v })
            }
            None => None,
        };
        // The path reads the Param buffers directly (same wgpu::Buffer handles).
        paths.push(BlrPath {
            bits_u: ub_buf,
            bits_v: vb_buf,
            h: ph.w.clone(),
            l: pl.w.clone(),
            g: pg.w.clone(),
            dh: ph.g.clone(),
            dl: pl.g.clone(),
            dg: pg.g.clone(),
            sign: sp,
        });
        params.push([ph, pl, pg]);
    }
    Ok(Lin::Blr { blr: Blr { m, n, r: rp, paths }, role, params })
}

#[allow(clippy::too_many_arguments)]
fn load_lin(c: Ctx, m: &CmfModel, name: &str, role: &'static str, as_plane: bool, bwd: bool, sign: Option<&SignInit>, next_id: &mut u32) -> Result<Lin, String> {
    load_lin_t(c, m, name, role, as_plane, bwd, sign, next_id, false)
}

/// [`load_lin`]; `train`: a dense tensor becomes [`Lin::Train`] when its
/// shape suits the weight-gradient GEMM (rows and columns multiples of 128),
/// else it stays frozen.
#[allow(clippy::too_many_arguments)]
fn load_lin_t(c: Ctx, m: &CmfModel, name: &str, role: &'static str, as_plane: bool, bwd: bool, sign: Option<&SignInit>, next_id: &mut u32, train: bool) -> Result<Lin, String> {
    let e = m.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if e.dtype as u8 == 19 {
        return load_blr(c, m.entry_bytes(e), role, sign, next_id);
    }
    let (v, shape) = deq(m, name)?;
    let (rows, cols) = (shape[0], shape[1]);
    if train && rows % 128 == 0 && cols % 128 == 0 {
        let pl = f16_plane(c, &v);
        let plt = f16_plane(c, &transpose(&v, rows, cols));
        return Ok(Lin::Train { p: Param::new(c, &v, None), pl, plt, m: rows, k: cols });
    }
    if train {
        eprintln!("  note: {name} ({rows}×{cols}) stays frozen: the weight-gradient GEMM takes multiples of 128");
    }
    if as_plane && rows % 128 == 0 && cols % 32 == 0 {
        let wt = bwd.then(|| f16_plane(c, &transpose(&v, rows, cols)));
        Ok(Lin::Plane { w: f16_plane(c, &v), wt, m: rows, k: cols })
    } else {
        Ok(Lin::Dense { w: to::upload(c, &v), m: rows, k: cols })
    }
}

impl Net {
    /// Device copy of `model` (`plane` = frozen f16 planes for every
    /// non-binary linear: the teacher; the student's binary low-rank
    /// tensors always load as such).
    /// `sign`: give every binary low-rank path sign latents (sign training).
    pub fn load(c: Ctx, model: &CmfModel, plane: bool, bwd: bool, sign: Option<&SignInit>) -> Result<Self, String> {
        Self::load_with(c, model, plane, bwd, sign, None, None)
    }

    /// Layers `0..upto` only, forward planes: the frozen prefix of a heal,
    /// whose output ([`prefix_hidden`]) feeds a `prefix_cached` student.
    pub fn load_prefix(c: Ctx, model: &CmfModel, upto: usize) -> Result<Self, String> {
        Self::load_with(c, model, true, false, None, None, Some(0..upto))
    }

    /// A student for healing: dense linears of the layers `heal.train_from..`
    /// trained in full, everything below frozen without backward planes.
    pub fn load_heal(c: Ctx, model: &CmfModel, heal: HealCfg) -> Result<Self, String> {
        let nl = model.arch().num_layers;
        let range = heal.prefix_cached.then(|| heal.train_from.min(nl)..nl);
        Self::load_with(c, model, true, true, None, Some(heal), range)
    }

    fn load_with(c: Ctx, model: &CmfModel, plane: bool, bwd: bool, sign: Option<&SignInit>, heal: Option<HealCfg>, range: Option<std::ops::Range<usize>>) -> Result<Self, String> {
        let mut geom = Geom::from_model(model)?;
        let train_from = heal.map_or(0, |h| h.train_from.min(geom.nl));
        let range = range.unwrap_or(0..geom.nl);
        let absent = |name: &str| -> Result<Lin, String> {
            let e = model.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
            Ok(Lin::Absent { m: e.shape[0] })
        };
        let norm = |name: &str| -> Result<Param, String> { Ok(Param::new(c, &deq(model, name)?.0, None)) };
        let mut layers = Vec::with_capacity(geom.nl);
        let mut next_id = 0u32;
        for li in 0..geom.nl {
            let p = format!("model.layers.{li}.");
            // frozen layers below the trained tail run no backward
            let bwd = bwd && li >= train_from;
            let (t_mix, t_ffn) = match heal {
                Some(h) if li >= train_from => (!h.ffn_only, true),
                _ => (false, false),
            };
            let here = range.contains(&li);
            let mut lin = |n: &str, role: &'static str| {
                if !here {
                    return absent(&format!("{p}{n}"));
                }
                let t = if role.starts_with("mlp.") { t_ffn } else { t_mix };
                load_lin_t(c, model, &format!("{p}{n}"), role, plane, bwd, sign, &mut next_id, t)
            };
            let small = |n: &str| -> Result<Lin, String> {
                if !here {
                    return absent(&format!("{p}{n}"));
                }
                load_lin(c, model, &format!("{p}{n}"), "gdn.ab", false, bwd, None, &mut 0)
            };
            let mix = if geom.gdn_layer[li] {
                Mix::Gdn {
                    qkv: lin("linear_attn.in_proj_qkv.weight", "gdn.qkv")?,
                    z: lin("linear_attn.in_proj_z.weight", "gdn.z")?,
                    a: small("linear_attn.in_proj_a.weight")?,
                    b: small("linear_attn.in_proj_b.weight")?,
                    out: lin("linear_attn.out_proj.weight", "gdn.out")?,
                    a_log: deq(model, &format!("{p}linear_attn.A_log"))?.0,
                    dt_bias: deq(model, &format!("{p}linear_attn.dt_bias"))?.0,
                    norm: deq(model, &format!("{p}linear_attn.norm.weight"))?.0,
                    conv: deq(model, &format!("{p}linear_attn.conv1d.weight"))?.0,
                }
            } else {
                Mix::Attn {
                    q: lin("self_attn.q_proj.weight", "attn.q")?,
                    k: lin("self_attn.k_proj.weight", "attn.k")?,
                    v: lin("self_attn.v_proj.weight", "attn.v")?,
                    o: lin("self_attn.o_proj.weight", "attn.o")?,
                    qn: norm(&format!("{p}self_attn.q_norm.weight"))?,
                    kn: norm(&format!("{p}self_attn.k_norm.weight"))?,
                }
            };
            layers.push(Layer {
                ln1: norm(&format!("{p}input_layernorm.weight"))?,
                ln2: norm(&format!("{p}post_attention_layernorm.weight"))?,
                mix,
                gate: lin("mlp.gate_proj.weight", "mlp.gate")?,
                up: lin("mlp.up_proj.weight", "mlp.up")?,
                down: lin("mlp.down_proj.weight", "mlp.down")?,
                mask: None,
            });
            // Every upload is a queue write staged until the next submission;
            // a 27B teacher's ~49 GB of f16 planes would otherwise sit in
            // staging next to the planes themselves and exhaust the card.
            c.queue.submit(std::iter::empty());
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        }
        // per-layer MLP widths (a pruned model): the workspace takes the widest
        geom.inter = layers.iter().map(|l| l.gate.m()).max().unwrap_or(geom.inter);
        let final_norm = norm("model.norm.weight")?;
        let (emb, eshape) = deq(model, "model.embed_tokens.weight")?;
        let head_w = if model.tensor("lm_head.weight").is_some() { deq(model, "lm_head.weight")?.0 } else { emb.clone() };
        let (v, hs) = (eshape[0], eshape[1]);
        // The teacher only needs the head's forward planes (no transposes);
        // vocabulary slices keep each plane within one storage binding.
        let head = Head::new(c, &head_w, v, hs, bwd, None);
        let embed = emb.iter().map(|&f| f32_to_f16(f)).collect();
        Ok(Net {
            geom,
            train_from,
            first: range.start,
            end: range.end,
            active_from: std::sync::atomic::AtomicUsize::new(train_from),
            layers,
            final_norm,
            head,
            embed,
            n_sign_mats: next_id,
            opt: std::sync::OnceLock::new(),
            dense_opt: std::sync::OnceLock::new(),
        })
    }

    /// Every binary low-rank linear with its tensor name, in a fixed order
    /// (the order of [`Net::params`]).
    pub fn blr_lins(&self) -> Vec<(String, &Lin)> {
        let mut v: Vec<(String, &Lin)> = Vec::new();
        for (li, ly) in self.layers.iter().enumerate() {
            let mut list: Vec<(&str, &Lin)> = match &ly.mix {
                Mix::Gdn { qkv, z, out, .. } => vec![("linear_attn.in_proj_qkv.weight", qkv), ("linear_attn.in_proj_z.weight", z), ("linear_attn.out_proj.weight", out)],
                Mix::Attn { q, k, v: vv, o, .. } => vec![("self_attn.q_proj.weight", q), ("self_attn.k_proj.weight", k), ("self_attn.v_proj.weight", vv), ("self_attn.o_proj.weight", o)],
            };
            list.extend([("mlp.gate_proj.weight", &ly.gate), ("mlp.up_proj.weight", &ly.up), ("mlp.down_proj.weight", &ly.down)]);
            for (n, l) in list {
                if matches!(l, Lin::Blr { .. }) {
                    v.push((format!("model.layers.{li}.{n}"), l));
                }
            }
        }
        v
    }

    /// Live signs under training (Σ rows · true rank over every sign matrix).
    pub fn sign_live(&self) -> u64 {
        let mut s = 0;
        for (_, lin) in self.blr_lins() {
            if let Lin::Blr { blr, .. } = lin {
                for p in &blr.paths {
                    if let Some(sp) = &p.sign {
                        s += sp.u.live() + sp.v.live();
                    }
                }
            }
        }
        s
    }

    /// Every trainable parameter (metric of the binary low-rank paths, all
    /// RMSNorm gains).
    pub fn params(&self) -> Vec<&Param> {
        fn push_lin<'a>(l: &'a Lin, v: &mut Vec<&'a Param>) {
            if let Lin::Blr { params, .. } = l {
                for p in params {
                    v.extend(p.iter());
                }
            }
        }
        let mut v: Vec<&Param> = vec![&self.final_norm];
        for l in &self.layers[self.train_from..] {
            v.push(&l.ln1);
            v.push(&l.ln2);
            match &l.mix {
                Mix::Gdn { qkv, z, out, .. } => {
                    push_lin(qkv, &mut v);
                    push_lin(z, &mut v);
                    push_lin(out, &mut v);
                }
                Mix::Attn { q, k, v: vv, o, qn, kn } => {
                    push_lin(q, &mut v);
                    push_lin(k, &mut v);
                    push_lin(vv, &mut v);
                    push_lin(o, &mut v);
                    v.push(qn);
                    v.push(kn);
                }
            }
            push_lin(&l.gate, &mut v);
            push_lin(&l.up, &mut v);
            push_lin(&l.down, &mut v);
        }
        v
    }

    /// DTG-MA: put a gate (all open) on every on-device layer's MLP
    /// intermediate. The backward must reach the layers (a net loaded with
    /// `bwd`); the weights stay as they are.
    pub fn enable_masks(&mut self, c: Ctx) {
        for ly in &mut self.layers[self.first..self.end] {
            let w = ly.gate.m();
            ly.mask = Some(MaskGate { s: to::upload(c, &vec![1.0; w]), g: to::zeros(c, w), w });
        }
    }

    /// Set every gate (`s[l]` for layer `first + l`, its MLP width long).
    pub fn set_masks(&self, c: Ctx, s: &[Vec<f32>]) -> Option<()> {
        for (ly, v) in self.layers[self.first..self.end].iter().zip(s) {
            let mg = ly.mask.as_ref()?;
            if v.len() != mg.w {
                return None;
            }
            c.queue.write_buffer(&mg.s, 0, bytemuck::cast_slice(v));
        }
        Some(())
    }

    /// The gates' accumulated gradients `dL/ds`, read and cleared.
    pub fn mask_grads(&self, c: Ctx) -> Option<Vec<Vec<f32>>> {
        let mut out = Vec::new();
        for ly in &self.layers[self.first..self.end] {
            let mg = ly.mask.as_ref()?;
            out.push(to::download(c, &mg.g, mg.w)?);
            zero(c, &mg.g, mg.w);
        }
        Some(out)
    }

    /// Every linear trained in full ([`Lin::Train`]) with its tensor name.
    pub fn dense_lins(&self) -> Vec<(String, &Lin)> {
        let mut v: Vec<(String, &Lin)> = Vec::new();
        for (li, ly) in self.layers.iter().enumerate() {
            let mut list: Vec<(&str, &Lin)> = match &ly.mix {
                Mix::Gdn { qkv, z, out, .. } => vec![("linear_attn.in_proj_qkv.weight", qkv), ("linear_attn.in_proj_z.weight", z), ("linear_attn.out_proj.weight", out)],
                Mix::Attn { q, k, v: vv, o, .. } => vec![("self_attn.q_proj.weight", q), ("self_attn.k_proj.weight", k), ("self_attn.v_proj.weight", vv), ("self_attn.o_proj.weight", o)],
            };
            list.extend([("mlp.gate_proj.weight", &ly.gate), ("mlp.up_proj.weight", &ly.up), ("mlp.down_proj.weight", &ly.down)]);
            for (n, l) in list {
                if matches!(l, Lin::Train { .. }) {
                    v.push((format!("model.layers.{li}.{n}"), l));
                }
            }
        }
        v
    }

    /// The parameters of [`Net::dense_lins`], in that order.
    pub fn dense_params(&self) -> Vec<&Param> {
        self.dense_lins().into_iter().filter_map(|(_, l)| if let Lin::Train { p, .. } = l { Some(p) } else { None }).collect()
    }

    /// Entries trained: the parameters plus the dense weights.
    pub fn trained_entries(&self) -> usize {
        self.params().iter().map(|p| p.n).sum::<usize>() + self.dense_params().iter().map(|p| p.n).sum::<usize>()
    }
}

// ───────────────────────────── workspace ─────────────────────────────

/// Activation and gradient buffers for `nb` windows of `t` tokens, shared
/// by every layer (one layer is live at a time), plus the checkpoints.
pub struct Work {
    pub nb: usize,
    pub t: usize,
    ntok: usize,
    ckpt: Vec<wgpu::Buffer>,
    b: HashMap<&'static str, wgpu::Buffer>,
    blr: HashMap<&'static str, BlrScratch>,
    act16: wgpu::Buffer,
    tape: Option<GdnTape>,
    rope: wgpu::Buffer,
    head: HeadScratch,
    /// Sign-training workspace (when the net carries sign state).
    pub sg: Option<SignScratch>,
    /// Matrix-unit attention scratch (when the geometry allows it).
    att: Option<AttScratch>,
    /// The step's embedding rows as f16: a persistent mapped upload buffer
    /// (re-mapped right after its copy; `emb_mapped` set by the callback)
    /// and its device copy, converted to f32 into the first checkpoint.
    emb_up: wgpu::Buffer,
    emb16: wgpu::Buffer,
    emb_mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Input staging f16(x⊙g) of up to four blr paths reading the same x
    /// (`lin_fwd_shared`): two buffers for the widest input, two for
    /// `stage_small` columns.
    stage: Vec<wgpu::Buffer>,
    stage_small: usize,
    /// Transposed f16 staging of a trained linear's input and output
    /// gradient (`[rows][tokens]`), the operands of its weight gradient.
    xt16: Option<wgpu::Buffer>,
    dyt16: Option<wgpu::Buffer>,
}

impl Work {
    pub fn new(c: Ctx, net: &Net, nb: usize, t: usize) -> Self {
        let g = &net.geom;
        let ntok = nb * t;
        let pr = tm::pad_rows(ntok);
        let qd = g.nh * g.hd;
        let kd = g.nkv * g.hd;
        let mut b = HashMap::new();
        let mut z = |name: &'static str, width: usize| {
            b.insert(name, to::zeros(c, pr * width.max(1)));
        };
        for n in ["x", "n1", "x2", "n2", "mix", "dx", "dn1", "dx2", "dn2", "dmix", "tmp_h"] {
            z(n, g.hs);
        }
        for n in ["inv1", "inv2", "invf"] {
            z(n, 1);
        }
        for n in ["gg", "uu", "hh", "dgg", "duu", "dhh"] {
            z(n, g.inter);
        }
        if g.gdn_layer.iter().any(|&x| !x) {
            for (n, w) in [("qg", 2 * qd), ("q", qd), ("gate", qd), ("k", kd), ("v", kd), ("qr", qd), ("kr", kd), ("o", qd), ("og", qd),
                ("dqg", 2 * qd), ("dq", qd), ("dgate", qd), ("dk", kd), ("dv", kd), ("dqr", qd), ("dkr", kd), ("do", qd), ("dog", qd), ("dwq", qd), ("dwk", kd)] {
                z(n, w);
            }
            for n in ["qinv", "lse", "dsum"] {
                z(n, g.nh);
            }
            z("kinv", g.nkv);
        }
        if g.gdn_layer.iter().any(|&x| x) {
            z("gout", g.vd());
            z("dgout", g.vd());
            z("dnorm_scratch", g.vd());
        }
        z("loss", 1);
        // One scratch per role, sized for the largest instance: the encoder
        // picks one or two paths and their rank per tensor, so a role's
        // layers differ in both.
        let mut dims: HashMap<_, (usize, usize, usize, usize)> = HashMap::new();
        let mut act_w = g.hs.max(g.inter).max(2 * qd).max(g.c_dim().max(g.vd()));
        let (mut sg_rows, mut sg_rp) = (0usize, 0usize);
        for l in &net.layers {
            let mut add = |lin: &Lin| {
                if let Lin::Blr { blr: bl, role, .. } = lin {
                    act_w = act_w.max(bl.m).max(bl.n).max(bl.r);
                    let d = dims.entry(*role).or_insert((0, 0, 0, 0));
                    *d = (d.0.max(bl.m), d.1.max(bl.n), d.2.max(bl.r), d.3.max(bl.paths.len()));
                    if bl.paths.iter().any(|p| p.sign.is_some()) {
                        sg_rows = sg_rows.max(bl.m).max(bl.n);
                        sg_rp = sg_rp.max(bl.r);
                    }
                }
            };
            match &l.mix {
                Mix::Gdn { qkv, z, out, .. } => {
                    add(qkv);
                    add(z);
                    add(out);
                }
                Mix::Attn { q, k, v, o, .. } => {
                    add(q);
                    add(k);
                    add(v);
                    add(o);
                }
            }
            add(&l.gate);
            add(&l.up);
            add(&l.down);
        }
        // one set of transient blr buffers for every role (one layer runs at a time)
        let sh = {
            let mx = |f: &dyn Fn(&(usize, usize, usize, usize)) -> usize| dims.values().map(f).max().unwrap_or(1).max(1);
            tm::BlrShared::new(c, ntok, mx(&|d| d.0 * d.2), mx(&|d| d.1 * d.2), mx(&|d| d.0.max(d.1).max(d.2)), mx(&|d| d.0.max(d.1)), mx(&|d| d.2))
        };
        let blr: HashMap<_, _> = dims.into_iter().map(|(role, (m, _n, r, np))| (role, BlrScratch::with_shared(c, ntok, m, r, np, &sh))).collect();
        let tape = g.gdn_layer.iter().any(|&x| x).then(|| {
            let zeros = vec![0f32; g.nv];
            let conv = vec![0f32; g.c_dim() * g.kk];
            let norm = vec![1f32; 128];
            let cfg = GdnTrainCfg { nv: g.nv, nk: g.nk, kk: g.kk, rms_eps: g.eps as f64, a_log: &zeros, dt_bias: &zeros, norm: &norm, conv: &conv };
            gdn::gdn_tape(&cfg, nb, t).expect("gdn tape")
        });
        let mut tab = Vec::with_capacity(t * g.half * 2);
        for pos in 0..t {
            for &f in &g.inv_freq {
                let a = pos as f64 * f;
                tab.push(a.cos() as f32);
                tab.push(a.sin() as f32);
            }
        }
        Work {
            nb,
            t,
            ntok,
            ckpt: (0..=g.nl).map(|_| to::zeros(c, pr * g.hs)).collect(),
            b,
            blr,
            act16: tm::f16_buf(c, pr * act_w),
            tape,
            rope: to::upload(c, &tab),
            // 1024-row chunks: 4x fewer passes over the head planes, and the
            // dlogits·W GEMM gets 320 workgroups instead of 80 (the logits
            // buffer, 1 GB at a 248k vocabulary, stays under the binding limit)
            head: HeadScratch::new(c, 1024.min(ntok), g.vocab, g.hs),
            sg: (sg_rows > 0).then(|| SignScratch::new(c, ntok, sg_rows, sg_rp)),
            att: g
                .gdn_layer
                .iter()
                .any(|&x| !x)
                .then_some(AttnGeom { nb, t, nh: g.nh, nkv: g.nkv, hd: g.hd })
                .filter(|ag| ta::usable(c, ag))
                .map(|ag| AttScratch::new(c, &ag)),
            emb_up: c.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nvg_emb_up"),
                size: ((ntok * g.hs).div_ceil(2) * 4) as u64,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            }),
            emb16: tm::f16_buf(c, ntok * g.hs),
            emb_mapped: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            stage: {
                let wide = net
                    .layers
                    .iter()
                    .flat_map(|l| {
                        let mut v = vec![&l.gate, &l.up, &l.down];
                        match &l.mix {
                            Mix::Gdn { qkv, z, out, .. } => v.extend([qkv, z, out]),
                            Mix::Attn { q, k, v: vv, o, .. } => v.extend([q, k, vv, o]),
                        }
                        v
                    })
                    .filter_map(|l| if let Lin::Blr { blr, .. } = l { Some(blr.n) } else { None })
                    .max()
                    .unwrap_or(0);
                [wide, wide, g.hs, g.hs].iter().map(|&n| tm::f16_buf(c, pr * n.max(1))).collect()
            },
            stage_small: g.hs,
            xt16: {
                let kp = ntok.div_ceil(32) * 32;
                let mx = net.dense_lins().iter().map(|(_, l)| if let Lin::Train { k, .. } = l { tm::pad_rows(*k) } else { 0 }).max().unwrap_or(0);
                (mx > 0).then(|| tm::f16_buf(c, mx * kp))
            },
            dyt16: {
                let kp = ntok.div_ceil(32) * 32;
                let mx = net.dense_lins().iter().map(|(_, l)| if let Lin::Train { m, .. } = l { tm::pad_rows(*m) } else { 0 }).max().unwrap_or(0);
                (mx > 0).then(|| tm::f16_buf(c, mx * kp))
            },
        }
    }
    fn get(&self, n: &str) -> &wgpu::Buffer {
        &self.b[n]
    }
}

// ───────────────────────────── linear ops ─────────────────────────────

fn lin_fwd(c: Ctx, w: &Work, lin: &Lin, x: &wgpu::Buffer, y: &wgpu::Buffer, acc: bool) -> Option<()> {
    let ntok = w.ntok;
    match lin {
        Lin::Blr { blr, role, .. } => {
            if acc {
                return None;
            }
            tm::blr_forward(c, blr, &w.blr[role], x, y, ntok)
        }
        Lin::Plane { w: pl, m, k, .. } | Lin::Train { pl, m, k, .. } => {
            if acc {
                return None;
            }
            tm::plane_linear(c, pl, x, &w.act16, y, ntok, *m, *k, 1.0)
        }
        Lin::Dense { w: dw, m, k } => {
            tm::run(c, &[Op::D(tm::dense_fwd(c, x, dw, y, ntok, *k, *m, acc)?)]);
            Some(())
        }
        Lin::Absent { .. } => None,
    }
}

/// Forward of linears that all read `x` (each into its own output): the
/// input staging f16(x⊙g) of every binary low-rank path is done by one
/// `cvt_multi` per group of up to four paths, so `x` is read once a group
/// instead of once a path. The other linears run as [`lin_fwd`].
fn lin_fwd_shared(c: Ctx, w: &Work, lins: &[(&Lin, &wgpu::Buffer)], x: &wgpu::Buffer) -> Option<()> {
    let ntok = w.ntok;
    let paths = |l: &Lin| if let Lin::Blr { blr, .. } = l { blr.paths.len() } else { 0 };
    let mut i = 0;
    while i < lins.len() {
        let Lin::Blr { blr, .. } = lins[i].0 else {
            lin_fwd(c, w, lins[i].0, x, lins[i].1, false)?;
            i += 1;
            continue;
        };
        // a group of consecutive blr linears: ≤ 4 paths, buffers 2 and 3 only
        // for inputs that fit them
        let n = blr.n;
        let (mut j, mut np) = (i, 0usize);
        while j < lins.len() {
            let pj = paths(lins[j].0);
            let nj = if let Lin::Blr { blr: b, .. } = lins[j].0 { b.n } else { usize::MAX };
            if pj == 0 || nj != n || np + pj > 4 || (np + pj > 2 && n > w.stage_small) {
                break;
            }
            np += pj;
            j += 1;
        }
        if np < 2 {
            lin_fwd(c, w, lins[i].0, x, lins[i].1, false)?;
            i += 1;
            continue;
        }
        let mut cols: Vec<&wgpu::Buffer> = Vec::with_capacity(np);
        for (l, _) in &lins[i..j] {
            if let Lin::Blr { blr, .. } = l {
                cols.extend(blr.paths.iter().map(|p| &p.g));
            }
        }
        let dsts: Vec<&wgpu::Buffer> = w.stage[..np].iter().collect();
        tm::run(c, &[Op::D(tm::cvt_multi(c, x, &cols, &dsts, ntok, n, 1.0)?)]);
        let mut k = 0;
        for (l, y) in &lins[i..j] {
            if let Lin::Blr { blr, role, .. } = l {
                let pn = blr.paths.len();
                tm::blr_forward_staged(c, blr, &w.blr[role], x, y, ntok, Some(&w.stage[k..k + pn]))?;
                k += pn;
            }
        }
        i = j;
    }
    Some(())
}

/// Input gradient into `dx` (overwritten, or accumulated with `acc`), metric
/// gradients accumulated. Blr needs the scratch of its own last forward.
/// With `sign`, a blr linear's sign latents/bits are updated in the same
/// pass (after its input gradient is formed from the old bits).
#[allow(clippy::too_many_arguments)]
fn lin_bwd(c: Ctx, w: &Work, lin: &Lin, x: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, acc: bool, gscale: f32, sign: Option<&SignStepCfg>) -> Option<()> {
    let ntok = w.ntok;
    match lin {
        Lin::Blr { blr, role, .. } => {
            let ctx = match (sign, w.sg.as_ref()) {
                (Some(cfg), Some(sc)) => Some(SignCtx { sc, cfg, role: ts::role_id(role) }),
                _ => None,
            };
            if acc {
                let tmp = w.get("tmp_h");
                // tmp_h is hs wide: accumulation is only used for hs-wide inputs
                tm::blr_backward(c, blr, &w.blr[role], x, dy, tmp, ntok, gscale, ctx.as_ref())?;
                tm::run(c, &[Op::D(to::elt(c, "add", ntok * blr.n, dx, tmp, Some(dx), None, None, None)?)]);
                Some(())
            } else {
                tm::blr_backward(c, blr, &w.blr[role], x, dy, dx, ntok, gscale, ctx.as_ref())
            }
        }
        Lin::Plane { wt, m, k, .. } => {
            let wt = wt.as_ref()?;
            let target = if acc { w.get("tmp_h") } else { dx };
            tm::plane_linear(c, wt, dy, &w.act16, target, ntok, *k, *m, gscale)?;
            if acc {
                tm::run(c, &[Op::D(to::elt(c, "add", ntok * k, dx, target, Some(dx), None, None, None)?)]);
            }
            Some(())
        }
        Lin::Dense { w: dw, m, k } => {
            if !acc {
                zero(c, dx, ntok * k);
            }
            tm::run(c, &[Op::D(tm::dense_bwd(c, dy, dw, dx, ntok, *k, *m)?)]);
            Some(())
        }
        Lin::Train { p, plt, m, k, .. } => {
            let target = if acc { w.get("tmp_h") } else { dx };
            tm::plane_linear(c, plt, dy, &w.act16, target, ntok, *k, *m, gscale)?;
            if acc {
                tm::run(c, &[Op::D(to::elt(c, "add", ntok * k, dx, target, Some(dx), None, None, None)?)]);
            }
            // dW[m][k] = dyᵀ·x over the tokens: both staged transposed in
            // f16 (dy at the gradient scale), the token axis as the GEMM's K
            tm::weight_grad(c, x, dy, w.xt16.as_ref()?, w.dyt16.as_ref()?, &p.g, ntok, *m, *k, gscale)
        }
        Lin::Absent { .. } => None,
    }
}

fn zero(c: Ctx, b: &wgpu::Buffer, n: usize) {
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero") });
    let mut mk = prof::enc_begin(c, &mut enc);
    enc.clear_buffer(b, 0, Some((n * 4) as u64));
    prof::enc_mark(c, &mut enc, &mut mk, "clear");
    crate::gpu_wgpu::submit_pub(c, enc.finish());
}

fn copy(c: Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, n: usize) {
    copy_off(c, src, 0, dst, 0, n);
}

fn copy_off(c: Ctx, src: &wgpu::Buffer, so: usize, dst: &wgpu::Buffer, d_o: usize, n: usize) {
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
    let mut mk = prof::enc_begin(c, &mut enc);
    enc.copy_buffer_to_buffer(src, (so * 4) as u64, dst, (d_o * 4) as u64, (n * 4) as u64);
    prof::enc_mark(c, &mut enc, &mut mk, "copy");
    crate::gpu_wgpu::submit_pub(c, enc.finish());
}

// ───────────────────────────── layer forward / backward ─────────────────────────────

fn gdn_params(c: Ctx, w: &Work, g: &Geom, a_log: &[f32], dt_bias: &[f32], norm: &[f32], conv: &[f32]) {
    let tape = w.tape.as_ref().unwrap();
    let mut prm = Vec::with_capacity(2 * g.nv + 128 + conv.len());
    prm.extend_from_slice(a_log);
    prm.extend_from_slice(dt_bias);
    prm.extend_from_slice(norm);
    prm.extend_from_slice(conv);
    prof::count(prof::Cnt::Write, 1);
    prof::count(prof::Cnt::WriteBytes, (prm.len() * 4) as u64);
    c.queue.write_buffer(tape.prm(), 0, bytemuck::cast_slice(&prm));
}

/// Forward of layer `l` from `x` (`w.x` is the input) to `y`; intermediates
/// stay in the workspace for the backward (`rec`: the recompute right before
/// the layer's backward — only it records the GDN tape's checkpoints).
fn layer_fwd(c: Ctx, net: &Net, w: &Work, l: usize, x: &wgpu::Buffer, y: &wgpu::Buffer, rec: bool) -> Option<()> {
    let g = &net.geom;
    let ly = &net.layers[l];
    let n = w.ntok;
    to::run(c, &[to::rms_fwd(c, x, &ly.ln1.w, w.get("n1"), w.get("inv1"), n, g.hs, g.eps, g.gemma)?]);
    match &ly.mix {
        Mix::Gdn { qkv, z, a, b, out, a_log, dt_bias, norm, conv } => {
            let tape = w.tape.as_ref().unwrap();
            gdn_params(c, w, g, a_log, dt_bias, norm, conv);
            lin_fwd_shared(c, w, &[(qkv, tape.raw()), (z, tape.z())], w.get("n1"))?;
            // a and b into the two halves of the tape's a|b buffer
            lin_fwd(c, w, a, w.get("n1"), w.get("gout"), false)?;
            copy_off(c, w.get("gout"), 0, tape.ab(), 0, n * g.nv);
            lin_fwd(c, w, b, w.get("n1"), w.get("gout"), false)?;
            copy_off(c, w.get("gout"), 0, tape.ab(), n * g.nv, n * g.nv);
            gdn::gdn_forward_dev(tape, g.nk, g.nv, g.c_dim(), rec)?;
            copy(c, tape.out(), w.get("gout"), n * g.vd());
            lin_fwd_shared(c, w, &[(out, w.get("mix"))], w.get("gout"))?;
        }
        Mix::Attn { q, k, v, o, qn, kn } => {
            let (qd, kd) = (g.nh * g.hd, g.nkv * g.hd);
            lin_fwd_shared(c, w, &[(q, w.get("qg")), (k, w.get("k")), (v, w.get("v"))], w.get("n1"))?;
            let qg = QkrGeom { n, heads: g.nh, hd: g.hd, half: g.half, t: w.t, gemma: g.gemma, eps: g.eps };
            let kg = QkrGeom { heads: g.nkv, ..qg };
            let ag = AttnGeom { nb: w.nb, t: w.t, nh: g.nh, nkv: g.nkv, hd: g.hd };
            if let Some(sc) = w.att.as_ref() {
                let mut ops = vec![
                    Op::D(tm::split_qg(c, w.get("qg"), w.get("q"), w.get("gate"), n, g.nh, g.hd, false)?),
                    Op::D(to::qkr_fwd(c, qg, w.get("q"), &qn.w, w.get("qr"), w.get("qinv"), &w.rope)?),
                    Op::D(to::qkr_fwd(c, kg, w.get("k"), &kn.w, w.get("kr"), w.get("kinv"), &w.rope)?),
                ];
                ops.extend(ta::fwd_ops(c, ag, sc, w.get("qr"), w.get("kr"), w.get("v"), w.get("o"), w.get("lse"))?);
                ops.push(Op::D(to::elt(c, "gate_fwd", n * qd, w.get("o"), w.get("gate"), Some(w.get("og")), None, None, None)?));
                tm::run(c, &ops);
            } else {
                tm::run(c, &[Op::D(tm::split_qg(c, w.get("qg"), w.get("q"), w.get("gate"), n, g.nh, g.hd, false)?)]);
                to::run(c, &[
                    to::qkr_fwd(c, qg, w.get("q"), &qn.w, w.get("qr"), w.get("qinv"), &w.rope)?,
                    to::qkr_fwd(c, kg, w.get("k"), &kn.w, w.get("kr"), w.get("kinv"), &w.rope)?,
                    to::attn_fwd(c, ag, w.get("qr"), w.get("kr"), w.get("v"), w.get("o"), w.get("lse"))?,
                    to::elt(c, "gate_fwd", n * qd, w.get("o"), w.get("gate"), Some(w.get("og")), None, None, None)?,
                ]);
            }
            let _ = kd;
            lin_fwd_shared(c, w, &[(o, w.get("mix"))], w.get("og"))?;
        }
    }
    to::run(c, &[
        to::elt(c, "add", n * g.hs, x, w.get("mix"), Some(w.get("x2")), None, None, None)?,
        to::rms_fwd(c, w.get("x2"), &ly.ln2.w, w.get("n2"), w.get("inv2"), n, g.hs, g.eps, g.gemma)?,
    ]);
    lin_fwd_shared(c, w, &[(&ly.gate, w.get("gg")), (&ly.up, w.get("uu"))], w.get("n2"))?;
    let act = match &ly.mask {
        Some(mg) => to::swiglu_cols(c, n * mg.w, w.get("gg"), w.get("uu"), w.get("hh"), &mg.s, mg.w)?,
        None => to::elt(c, "swiglu_fwd", n * ly.gate.m(), w.get("gg"), w.get("uu"), Some(w.get("hh")), None, None, None)?,
    };
    to::run(c, &[act]);
    lin_fwd_shared(c, w, &[(&ly.down, w.get("mix"))], w.get("hh"))?;
    to::run(c, &[to::elt(c, "add", n * g.hs, w.get("x2"), w.get("mix"), Some(y), None, None, None)?]);
    Some(())
}

/// Backward of layer `l` (its forward just recomputed from `x`): `dy` is the
/// gradient at the layer output; `dx` gets the gradient at its input.
#[allow(clippy::too_many_arguments)]
fn layer_bwd(c: Ctx, net: &Net, w: &Work, l: usize, x: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, gs: f32, sg: Option<&SignStepCfg>) -> Option<()> {
    let g = &net.geom;
    let ly = &net.layers[l];
    let n = w.ntok;
    // MLP: y = x2 + down(swiglu(gate(n2), up(n2)))
    lin_bwd(c, w, &ly.down, w.get("hh"), dy, w.get("dhh"), false, gs, sg)?;
    match &ly.mask {
        Some(mg) => {
            // the gate's gradient Σ_tokens dhh·h, h the ungated SwiGLU
            // (recomputed into dgg, which the SwiGLU backward then overwrites)
            let [p1, p2] = tm::colsum(c, w.get("dhh"), tm::ColTerm::Prod(w.get("dgg")), &mg.g, n, mg.w, true)?;
            to::run(c, &[
                to::elt(c, "swiglu_fwd", n * mg.w, w.get("gg"), w.get("uu"), Some(w.get("dgg")), None, None, None)?,
                p1,
                p2,
                to::swiglu_cols_bwd(c, n * mg.w, w.get("gg"), w.get("uu"), &mg.s, w.get("dhh"), w.get("dgg"), w.get("duu"), mg.w)?,
            ]);
        }
        None => to::run(c, &[to::elt(c, "swiglu_bwd", n * ly.gate.m(), w.get("gg"), w.get("uu"), None, Some(w.get("dhh")), Some(w.get("dgg")), Some(w.get("duu")))?]),
    }
    lin_bwd(c, w, &ly.gate, w.get("n2"), w.get("dgg"), w.get("dn2"), false, gs, sg)?;
    lin_bwd(c, w, &ly.up, w.get("n2"), w.get("duu"), w.get("dn2"), true, gs, sg)?;
    let [dw1, dw2] = to::rms_dw(c, w.get("x2"), w.get("inv2"), w.get("dn2"), &ly.ln2.g, n, g.hs)?;
    to::run(c, &[
        to::rms_bwd(c, w.get("x2"), &ly.ln2.w, w.get("inv2"), w.get("dn2"), w.get("dx2"), n, g.hs, g.gemma)?,
        dw1,
        dw2,
        to::elt(c, "add", n * g.hs, w.get("dx2"), dy, Some(w.get("dx2")), None, None, None)?,
    ]);
    // mixer: x2 = x + mix(n1)
    match &ly.mix {
        Mix::Gdn { qkv, z, a, b, out, .. } => {
            let tape = w.tape.as_ref().unwrap();
            lin_bwd(c, w, out, w.get("gout"), w.get("dx2"), tape.dout(), false, gs, sg)?;
            gdn::gdn_backward_dev(tape, g.nk, g.nv, g.c_dim())?;
            lin_bwd(c, w, qkv, w.get("n1"), tape.draw(), w.get("dn1"), false, gs, sg)?;
            lin_bwd(c, w, z, w.get("n1"), tape.dz(), w.get("dn1"), true, gs, sg)?;
            copy_off(c, tape.dab(), 0, w.get("dgout"), 0, n * g.nv);
            lin_bwd(c, w, a, w.get("n1"), w.get("dgout"), w.get("dn1"), true, gs, sg)?;
            copy_off(c, tape.dab(), n * g.nv, w.get("dgout"), 0, n * g.nv);
            lin_bwd(c, w, b, w.get("n1"), w.get("dgout"), w.get("dn1"), true, gs, sg)?;
        }
        Mix::Attn { q, k, v, o, qn, kn } => {
            let qd = g.nh * g.hd;
            let qg = QkrGeom { n, heads: g.nh, hd: g.hd, half: g.half, t: w.t, gemma: g.gemma, eps: g.eps };
            let kg = QkrGeom { heads: g.nkv, ..qg };
            let ag = AttnGeom { nb: w.nb, t: w.t, nh: g.nh, nkv: g.nkv, hd: g.hd };
            lin_bwd(c, w, o, w.get("og"), w.get("dx2"), w.get("dog"), false, gs, sg)?;
            let gate_b = to::elt(c, "gate_bwd", n * qd, w.get("o"), w.get("gate"), None, Some(w.get("dog")), Some(w.get("do")), Some(w.get("dgate")))?;
            let ab = to::attn_bwd(c, ag, w.get("qr"), w.get("kr"), w.get("v"), w.get("o"), w.get("lse"), w.get("do"), w.get("dsum"), w.get("dqr"), w.get("dkr"), w.get("dv"))?;
            if let Some(sc) = w.att.as_ref() {
                // D = Σ dO⊙O (the f32 kernel's first pass), then the matrix-unit backward
                let dsum = ab.into_iter().next()?;
                let mut ops = vec![Op::D(gate_b), Op::D(dsum)];
                ops.extend(ta::bwd_ops(c, ag, sc, w.get("lse"), w.get("do"), w.get("dsum"), w.get("dqr"), w.get("dkr"), w.get("dv"), gs)?);
                tm::run(c, &ops);
            } else {
                to::run(c, &[gate_b]);
                to::run(c, &ab);
            }
            to::run(c, &[
                to::qkr_bwd(c, qg, w.get("q"), &qn.w, w.get("qinv"), &w.rope, w.get("dqr"), w.get("dq"), w.get("dwq"))?,
                to::qkr_bwd(c, kg, w.get("k"), &kn.w, w.get("kinv"), &w.rope, w.get("dkr"), w.get("dk"), w.get("dwk"))?,
            ]);
            // gain grads (summed over rows·heads) into the norms' gradients
            let [q1, q2] = to::qkr_dw(c, qg, w.get("dwq"), &qn.g, true)?;
            let [k1, k2] = to::qkr_dw(c, kg, w.get("dwk"), &kn.g, true)?;
            to::run(c, &[q1, q2, k1, k2]);
            tm::run(c, &[Op::D(tm::split_qg(c, w.get("dqg"), w.get("dq"), w.get("dgate"), n, g.nh, g.hd, true)?)]);
            lin_bwd(c, w, q, w.get("n1"), w.get("dqg"), w.get("dn1"), false, gs, sg)?;
            lin_bwd(c, w, k, w.get("n1"), w.get("dk"), w.get("dn1"), true, gs, sg)?;
            lin_bwd(c, w, v, w.get("n1"), w.get("dv"), w.get("dn1"), true, gs, sg)?;
        }
    }
    let [dw1, dw2] = to::rms_dw(c, x, w.get("inv1"), w.get("dn1"), &ly.ln1.g, n, g.hs)?;
    to::run(c, &[
        to::rms_bwd(c, x, &ly.ln1.w, w.get("inv1"), w.get("dn1"), dx, n, g.hs, g.gemma)?,
        dw1,
        dw2,
        to::elt(c, "add", n * g.hs, dx, w.get("dx2"), Some(dx), None, None, None)?,
    ]);
    Some(())
}

// ───────────────────────────── forward / step ─────────────────────────────

fn embed(c: Ctx, net: &Net, w: &Work, ids: &[u32]) {
    use std::sync::atomic::Ordering;
    let _p = prof::phase("embed");
    let hs = net.geom.hs;
    let n = ids.len() * hs;
    // the upload buffer was re-mapped after the last step's copy; wait only if
    // that has not completed yet (first use: mapped at creation)
    if !w.emb_mapped.load(Ordering::Acquire) {
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let bytes = n.div_ceil(2) * 4;
    {
        let mut view = w.emb_up.slice(..bytes as u64).get_mapped_range_mut().expect("embedding upload buffer mapped");
        // write-only mapped memory: each thread copies its rows' f16 bits to
        // a disjoint range (the conversion to f32 runs on the card)
        let mut dst = view.slice(..);
        let base = dst.as_raw_element_ptr().as_ptr() as usize;
        let per = ids.len().div_ceil(16).max(1);
        std::thread::scope(|sc| {
            for (ci, chunk) in ids.chunks(per).enumerate() {
                let emb = &net.embed;
                sc.spawn(move || {
                    for (t, &id) in chunk.iter().enumerate() {
                        let src = &emb[id as usize * hs..(id as usize + 1) * hs];
                        let at = base + ((ci * per + t) * hs) * 2;
                        // SAFETY: [at, at + 2·hs) lies inside the mapped write
                        // range (row ci·per + t < ids.len()) and no other thread
                        // writes it; `src` is host memory.
                        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, at as *mut u8, hs * 2) }
                    }
                });
            }
        });
    }
    w.emb_up.unmap();
    w.emb_mapped.store(false, Ordering::Release);
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("embed") });
    let mut mk = prof::enc_begin(c, &mut enc);
    enc.copy_buffer_to_buffer(&w.emb_up, 0, &w.emb16, 0, bytes as u64);
    prof::enc_mark(c, &mut enc, &mut mk, "copy");
    if let Some(d) = tm::f16_to_f32(c, &w.emb16, &w.ckpt[0], n) {
        tm::record(c, &mut enc, &[Op::D(d)]);
    }
    crate::gpu_wgpu::submit_pub(c, enc.finish());
    let flag = w.emb_mapped.clone();
    w.emb_up.slice(..).map_async(wgpu::MapMode::Write, move |r| {
        if r.is_ok() {
            flag.store(true, Ordering::Release);
        }
    });
}

/// Forward of all layers from the embedded `ids` (checkpoints filled);
/// returns the final-normed hidden in `w.b["n1"]`.
fn forward(c: Ctx, net: &Net, w: &Work, ids: &[u32]) -> Option<()> {
    forward_in(c, net, w, ids, None)
}

/// [`forward`] over the net's layers: from the embedding, or — for a net
/// starting past layer 0 — from `h0` (`[ntok][hs]`, the prefix's output);
/// the final norm only when the net runs to the last layer.
fn forward_in(c: Ctx, net: &Net, w: &Work, ids: &[u32], h0: Option<&[f32]>) -> Option<()> {
    if net.first == 0 {
        embed(c, net, w, ids);
    } else {
        let h = h0?;
        if h.len() != w.ntok * net.geom.hs {
            return None;
        }
        c.queue.write_buffer(&w.ckpt[net.first], 0, bytemuck::cast_slice(h));
    }
    for l in net.first..net.end {
        let _p = prof::phase(if net.geom.gdn_layer[l] { "fwd.gdn" } else { "fwd.attn" });
        layer_fwd(c, net, w, l, &w.ckpt[l], &w.ckpt[l + 1], false)?;
    }
    let g = &net.geom;
    if net.end < g.nl {
        return Some(());
    }
    let _p = prof::phase("fwd.final_norm");
    to::run(c, &[to::rms_fwd(c, &w.ckpt[g.nl], &net.final_norm.w, w.get("n1"), w.get("invf"), w.ntok, g.hs, g.eps, g.gemma)?]);
    Some(())
}

/// The output of a [`Net::load_prefix`] net for `ids` (`[ntok][hs]`).
pub fn prefix_hidden(c: Ctx, net: &Net, w: &Work, ids: &[u32]) -> Option<Vec<f32>> {
    forward_in(c, net, w, ids, None)?;
    to::download(c, &w.ckpt[net.end], w.ntok * net.geom.hs)
}

/// Forward of `ids` (`[nb·t]`, the first `ntok` real: padding rows after
/// them contribute nothing) up to the last layer of `grams`, adding each
/// listed layer's MLP down-projection input Gram `Σ_t h_t·h_tᵀ` into its
/// f32 `[w][w]` accumulator (`xt`: f16 staging of `[pad_rows(w)][ceil32(nb·t)]`,
/// `tmp`: f32 `[w][w]` for the widest layer).
pub fn capture_grams(c: Ctx, net: &Net, w: &Work, ids: &[u32], ntok: usize, grams: &[(usize, &wgpu::Buffer)], xt: &wgpu::Buffer, tmp: &wgpu::Buffer) -> Option<()> {
    let upto = grams.iter().map(|g| g.0).max()?;
    embed(c, net, w, ids);
    for l in 0..=upto {
        layer_fwd(c, net, w, l, &w.ckpt[l], &w.ckpt[l + 1], false)?;
        if let Some((_, acc)) = grams.iter().find(|g| g.0 == l) {
            tm::gram_acc(c, w.get("hh"), xt, acc, tmp, ntok, net.layers[l].gate.m())?;
        }
    }
    Some(())
}

/// [`capture_grams`] over layers `start..` only: from the embedded `ids`
/// when `start` is 0, else from `x0` (`[t][hs]` f32, the residual input of
/// layer `start` as a previous range returned it). With `keep`, returns the
/// residual input of the layer after the last captured one — where the next
/// range resumes, so a model captured in several passes runs each layer once.
#[allow(clippy::too_many_arguments)]
pub fn capture_grams_range(
    c: Ctx,
    net: &Net,
    w: &Work,
    ids: &[u32],
    ntok: usize,
    start: usize,
    x0: Option<&[f32]>,
    grams: &[(usize, &wgpu::Buffer)],
    xt: &wgpu::Buffer,
    tmp: &wgpu::Buffer,
    keep: bool,
) -> Option<Option<Vec<f32>>> {
    let upto = grams.iter().map(|g| g.0).max()?;
    if start == 0 {
        embed(c, net, w, ids);
    } else {
        let x = x0?;
        if x.len() != w.t * net.geom.hs {
            return None;
        }
        c.queue.write_buffer(&w.ckpt[start], 0, bytemuck::cast_slice(x));
    }
    for l in start..=upto {
        layer_fwd(c, net, w, l, &w.ckpt[l], &w.ckpt[l + 1], false)?;
        if let Some((_, acc)) = grams.iter().find(|g| g.0 == l) {
            tm::gram_acc(c, w.get("hh"), xt, acc, tmp, ntok, net.layers[l].gate.m())?;
        }
    }
    if !keep {
        return Some(None);
    }
    to::download(c, &w.ckpt[upto + 1], w.t * net.geom.hs).map(Some)
}

/// Teacher top-k for `ids` (`[nb·t]`): ids and renormalized probabilities.
pub fn teacher_topk(c: Ctx, net: &Net, w: &Work, ids: &[u32], k: usize) -> Option<(Vec<u32>, Vec<f32>)> {
    forward(c, net, w, ids)?;
    th::topk(c, &net.head, &w.head, w.get("n1"), w.ntok, k)
}

/// Statistics of one training step.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepStats {
    pub loss: f64,
    pub finite: bool,
    /// The backward ran (false: non-finite loss with sign training on).
    pub backward: bool,
    /// Sign flips this step, per role and side (`2·role + side`, see `train_sign::ROLES`).
    pub flips_role: [u64; 20],
    pub flips: u64,
    /// Non-finite sign-gradient entries (treated as 0).
    pub g_nonfinite: u64,
    /// Latent steps clipped at ±1, and steps clamped at ±c.
    pub saturated: u64,
    pub clamped: u64,
}

/// One forward + backward on `ids` against the teacher top-k; gradients
/// accumulate into the parameters' `g` buffers (zeroed by the caller).
/// With `sign`, every sign matrix is updated during the reverse sweep; a
/// non-finite loss then skips the backward (no sign update runs).
#[allow(clippy::too_many_arguments)]
pub fn step(c: Ctx, net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize, gscale: f32, sign: Option<&SignStepCfg>) -> Option<StepStats> {
    step_w(c, net, w, ids, tid, tp, k, gscale, sign, None)
}

/// [`step`] with per-token weights carried by the teacher rows: `total` =
/// Some(Σ weights) means a row's teacher probabilities sum to its weight
/// (0 = the row is not scored) and the loss is Σ_rows w·KL / total.
#[allow(clippy::too_many_arguments)]
pub fn step_w(c: Ctx, net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize, gscale: f32, sign: Option<&SignStepCfg>, total: Option<f64>) -> Option<StepStats> {
    step_wh(c, net, w, ids, None, tid, tp, k, gscale, sign, total)
}

/// [`step_w`] for a net that starts past layer 0: `h0` is its input.
#[allow(clippy::too_many_arguments)]
pub fn step_wh(c: Ctx, net: &Net, w: &Work, ids: &[u32], h0: Option<&[f32]>, tid: &[u32], tp: &[f32], k: usize, gscale: f32, sign: Option<&SignStepCfg>, total: Option<f64>) -> Option<StepStats> {
    let g = &net.geom;
    let n = w.ntok;
    prof::step_begin();
    let sign = if w.sg.is_some() { sign } else { None };
    if let (Some(_), Some(sc)) = (sign, w.sg.as_ref()) {
        let _p = prof::phase("counters");
        ts::reset_counters(c, sc);
    }
    forward_in(c, net, w, ids, h0)?;
    let ph = prof::phase("head");
    let btid = to::zeros(c, n * k);
    prof::count(prof::Cnt::Write, 1);
    prof::count(prof::Cnt::WriteBytes, (tid.len() * 4) as u64);
    c.queue.write_buffer(&btid, 0, bytemuck::cast_slice(tid));
    let btp = to::upload(c, tp);
    let norm = total.map_or(n as f64, |t| t.max(1e-9));
    th::head_kl_w(c, &net.head, &w.head, w.get("n1"), &btid, &btp, k, w.get("dn1"), w.get("loss"), n, norm as f32, gscale, total.is_some())?;
    // The backward is queued without reading the loss back (that drain left
    // the card idle): with sign training the card itself turns every
    // sign-state write off when a loss is not finite, which is what skipping
    // the backward did; the loss is read after the sweep.
    if let (Some(_), Some(sc)) = (sign, w.sg.as_ref()) {
        to::run(c, &[ts::loss_flag_op(c, sc, w.get("loss"), n)?]);
    }
    drop(ph);
    let mut st = StepStats::default();
    // final norm backward: dx at the last checkpoint
    let (da, db) = (w.get("dx"), w.get("dmix"));
    let ph = prof::phase("bwd.final_norm");
    let [dw1, dw2] = to::rms_dw(c, &w.ckpt[g.nl], w.get("invf"), w.get("dn1"), &net.final_norm.g, n, g.hs)?;
    to::run(c, &[to::rms_bwd(c, &w.ckpt[g.nl], &net.final_norm.w, w.get("invf"), w.get("dn1"), da, n, g.hs, g.gemma)?, dw1, dw2]);
    drop(ph);
    let mut cur = da;
    let mut nxt = db;
    let lowest = net.train_from.max(net.active_from.load(std::sync::atomic::Ordering::Relaxed));
    for l in (lowest..g.nl).rev() {
        let gdn = g.gdn_layer[l];
        let ph = prof::phase(if gdn { "recomp.gdn" } else { "recomp.attn" });
        layer_fwd(c, net, w, l, &w.ckpt[l], w.get("x"), true)?;
        drop(ph);
        let _p = prof::phase(if gdn { "bwd.gdn" } else { "bwd.attn" });
        layer_bwd(c, net, w, l, &w.ckpt[l], cur, nxt, gscale, sign)?;
        std::mem::swap(&mut cur, &mut nxt);
    }
    st.backward = true;
    if let (Some(_), Some(sc)) = (sign, w.sg.as_ref()) {
        let _p = prof::phase("counters");
        let cnt = ts::read_reset_counters(c, sc)?;
        for (i, f) in st.flips_role.iter_mut().enumerate() {
            *f = cnt[i] as u64;
        }
        st.flips = st.flips_role.iter().sum();
        st.g_nonfinite = cnt[ts::CNT_NONFINITE] as u64;
        st.saturated = cnt[ts::CNT_SATURATED] as u64;
        st.clamped = cnt[ts::CNT_CLAMPED] as u64;
    }
    st.loss = to::download(c, w.get("loss"), n)?.iter().map(|&x| x as f64).sum::<f64>() / norm;
    st.finite = st.loss.is_finite();
    if sign.is_some() && !st.finite {
        // the sign updates were skipped on the card: report the step as one
        // without a backward, as before
        st.backward = false;
    }
    Some(st)
}

/// Every sign matrix with its bitmap and counter slot, in [`Net::blr_lins`] order.
fn sign_mats(net: &Net) -> Vec<(&ts::SignMat, &wgpu::Buffer, u32)> {
    let mut v = Vec::new();
    for (_, lin) in net.blr_lins() {
        if let Lin::Blr { blr, role, .. } = lin {
            let rid = ts::role_id(role);
            for p in &blr.paths {
                if let Some(sp) = &p.sign {
                    v.push((&sp.u, &p.bits_u, 2 * rid));
                    v.push((&sp.v, &p.bits_v, 2 * rid + 1));
                }
            }
        }
    }
    v
}

/// Deferred mode (tests): apply every matrix's update from the gradient
/// its backward saved; returns the flips per counter slot.
pub fn apply_deferred(c: Ctx, net: &Net, w: &Work, cfg: &SignStepCfg) -> Option<[u32; ts::CNT_LEN]> {
    let sc = w.sg.as_ref()?;
    ts::reset_counters(c, sc);
    for (mat, bits, slot) in sign_mats(net) {
        ts::apply_deferred(c, cfg, sc, mat, bits, slot)?;
    }
    ts::read_reset_counters(c, sc)
}

/// Distinct flips against the starting bits per role and side:
/// (`role.U` / `role.V`, flipped, live). Needs `SignInit::keep_bits0`.
pub fn sign_stats(c: Ctx, net: &Net) -> Option<Vec<(String, u64, u64)>> {
    let mats = sign_mats(net);
    if mats.is_empty() {
        return Some(Vec::new());
    }
    let diff = to::zeros_u32(c, net.n_sign_mats as usize);
    // one submission per batch of matrices keeps the bind groups bounded
    for chunk in mats.chunks(64) {
        let ds: Vec<_> = chunk.iter().map(|(mat, bits, _)| ts::bits_diff_op(c, mat, bits, &diff, mat.id)).collect::<Option<Vec<_>>>()?;
        to::run(c, &ds);
    }
    let dv = to::download_u32(c, &diff, net.n_sign_mats as usize)?;
    let mut acc = [(0u64, 0u64); 20];
    for (mat, _, slot) in &mats {
        acc[*slot as usize].0 += dv[mat.id as usize] as u64;
        acc[*slot as usize].1 += mat.live();
    }
    Some(
        acc.iter()
            .enumerate()
            .filter(|(_, (_, live))| *live > 0)
            .map(|(i, &(f, live))| (format!("{}.{}", ts::ROLES[i / 2], ["U", "V"][i % 2]), f, live))
            .collect(),
    )
}
fn opt_cat(c: Ctx, net: &Net) -> &OptCat {
    net.opt.get_or_init(|| {
        let ps = net.params();
        let mut off = Vec::with_capacity(ps.len());
        let mut total = 0usize;
        let mut tab = Vec::with_capacity(2 * ps.len());
        let mut relf: Vec<u32> = Vec::new();
        for p in &ps {
            off.push(total);
            tab.push(total as u32);
            tab.push(p.n as u32);
            relf.extend(std::iter::repeat_n(p.rel.to_bits(), p.n));
            total += p.n;
        }
        let mask = to::upload(c, &vec![1f32; total]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("opt_cat") });
        for (p, &o) in ps.iter().zip(&off) {
            if let Some(m) = &p.mask {
                enc.copy_buffer_to_buffer(m, 0, &mask, (o * 4) as u64, (p.n * 4) as u64);
            }
        }
        crate::gpu_wgpu::submit_pub(c, enc.finish());
        let relf_b = to::zeros_u32(c, total);
        c.queue.write_buffer(&relf_b, 0, bytemuck::cast_slice(&relf));
        let tab_b = to::zeros_u32(c, tab.len());
        c.queue.write_buffer(&tab_b, 0, bytemuck::cast_slice(&tab));
        OptCat {
            g: to::zeros(c, total),
            w: to::zeros(c, total),
            m1: to::zeros(c, total),
            m2: to::zeros(c, total),
            mask,
            relf: relf_b,
            tab: tab_b,
            part: to::zeros(c, 64 * ps.len()),
            off,
            total,
            g_fresh: std::sync::atomic::AtomicBool::new(false),
        }
    })
}

/// Copy every parameter's gradient into the concatenation.
fn gather_grads(enc: &mut wgpu::CommandEncoder, ps: &[&Param], oc: &OptCat) {
    for (p, &o) in ps.iter().zip(&oc.off) {
        enc.copy_buffer_to_buffer(&p.g, 0, &oc.g, (o * 4) as u64, (p.n * 4) as u64);
    }
}

/// Global gradient norm over every parameter (NaN/inf when a gradient is):
/// each parameter's 64 partial sums of squares (the `sumsq` grid) summed on
/// the host in f64, parameters in [`Net::params`] order.
pub fn grad_norm(c: Ctx, net: &Net) -> Option<f64> {
    let _p = prof::phase("opt.grad_norm");
    let oc = opt_cat(c, net);
    let ps = net.params();
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("grad_norm") });
    let mut mk = prof::enc_begin(c, &mut enc);
    gather_grads(&mut enc, &ps, oc);
    prof::enc_mark(c, &mut enc, &mut mk, "copy");
    tm::record(c, &mut enc, &[Op::D(tm::sumsq_cat(c, &oc.g, &oc.tab, &oc.part, ps.len())?)]);
    crate::gpu_wgpu::submit_pub(c, enc.finish());
    oc.g_fresh.store(true, std::sync::atomic::Ordering::Relaxed);
    let part = to::download(c, &oc.part, 64 * ps.len())?;
    let mut s = 0f64;
    for i in 0..ps.len() {
        s += part[64 * i..64 * (i + 1)].iter().map(|&x| x as f64).sum::<f64>();
    }
    Some((s + dense_sumsq(c, net)?).sqrt())
}

/// Σ g² over the dense trained weights (0 without any): 64 partial sums per
/// weight, one read-back for all of them.
fn dense_sumsq(c: Ctx, net: &Net) -> Option<f64> {
    let dp = net.dense_params();
    if dp.is_empty() {
        return Some(0.0);
    }
    let part = net.dense_opt.get_or_init(|| to::zeros(c, 64 * dp.len()));
    let slot = to::zeros(c, 64);
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dense_norm") });
    for (i, p) in dp.iter().enumerate() {
        tm::record(c, &mut enc, &[Op::D(tm::sumsq_part(c, &p.g, p.n, &slot)?)]);
        enc.copy_buffer_to_buffer(&slot, 0, part, (i * 64 * 4) as u64, 64 * 4);
    }
    crate::gpu_wgpu::submit_pub(c, enc.finish());
    Some(to::download(c, part, 64 * dp.len())?.iter().map(|&x| x as f64).sum())
}

/// Refresh a trained linear's f16 plane and its transpose from the f32 weights.
fn refresh_planes(c: Ctx, lin: &Lin) -> Option<()> {
    if let Lin::Train { p, pl, plt, m, k } = lin {
        tm::run(c, &[Op::D(tm::cvt(c, &p.w, None, pl, *m, *k, 1.0)?), Op::D(tm::cvt_t(c, &p.w, None, plt, *m, *k, 1.0)?)]);
    }
    Some(())
}

/// Clear every gradient (a skipped step).
pub fn zero_grads(c: Ctx, net: &Net) {
    let _p = prof::phase("opt.zero");
    if let Some(oc) = net.opt.get() {
        oc.g_fresh.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    for p in net.params().into_iter().chain(net.dense_params()) {
        zero(c, &p.g, p.n);
    }
}

/// Adam over every parameter (masked entries stay put), the gradients
/// multiplied by `gmul` (clipping); clears the grads.
#[allow(clippy::too_many_arguments)]
pub fn adam(c: Ctx, net: &Net, lr: f32, step_no: u32, b1: f32, b2: f32, eps: f32, gmul: f32) -> Option<()> {
    adam_scaled(c, net, lr, step_no, b1, b2, eps, gmul, None)
}

/// [`adam`] with the dense trained weights' rate scaled per layer
/// (`layer_scale(l)`, e.g. a decay away from the lowest active layer).
#[allow(clippy::too_many_arguments)]
pub fn adam_scaled(c: Ctx, net: &Net, lr: f32, step_no: u32, b1: f32, b2: f32, eps: f32, gmul: f32, layer_scale: Option<&dyn Fn(usize) -> f32>) -> Option<()> {
    let _p = prof::phase("opt.adam");
    let oc = opt_cat(c, net);
    let ps = net.params();
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("adam") });
    let mut mk = prof::enc_begin(c, &mut enc);
    if !oc.g_fresh.swap(false, std::sync::atomic::Ordering::Relaxed) {
        gather_grads(&mut enc, &ps, oc);
    }
    for (p, &o) in ps.iter().zip(&oc.off) {
        let (bo, nb) = ((o * 4) as u64, (p.n * 4) as u64);
        enc.copy_buffer_to_buffer(&p.w, 0, &oc.w, bo, nb);
        enc.copy_buffer_to_buffer(&p.m1, 0, &oc.m1, bo, nb);
        enc.copy_buffer_to_buffer(&p.m2, 0, &oc.m2, bo, nb);
    }
    prof::enc_mark(c, &mut enc, &mut mk, "copy");
    tm::record(c, &mut enc, &[Op::D(tm::adam_cat(c, &oc.w, &oc.g, &oc.m1, &oc.m2, &oc.mask, &oc.relf, oc.total, lr, step_no, b1, b2, eps, gmul)?)]);
    let mut mk = prof::enc_begin(c, &mut enc);
    for (p, &o) in ps.iter().zip(&oc.off) {
        let (bo, nb) = ((o * 4) as u64, (p.n * 4) as u64);
        enc.copy_buffer_to_buffer(&oc.w, bo, &p.w, 0, nb);
        enc.copy_buffer_to_buffer(&oc.m1, bo, &p.m1, 0, nb);
        enc.copy_buffer_to_buffer(&oc.m2, bo, &p.m2, 0, nb);
        enc.clear_buffer(&p.g, 0, Some(nb));
    }
    prof::enc_mark(c, &mut enc, &mut mk, "copy");
    crate::gpu_wgpu::submit_pub(c, enc.finish());
    // dense trained weights: plain Adam each, then their f16 planes
    for (name, lin) in net.dense_lins() {
        let Lin::Train { p, .. } = lin else { continue };
        let scale = match layer_scale {
            Some(f) => name.strip_prefix("model.layers.").and_then(|t| t.split('.').next()).and_then(|v| v.parse().ok()).map_or(1.0, f),
            None => 1.0,
        };
        tm::run(c, &[Op::D(tm::adam(c, &p.w, &p.g, &p.m1, &p.m2, None, p.n, lr * scale, step_no, b1, b2, eps, gmul, 0.0)?)]);
        refresh_planes(c, lin)?;
        zero(c, &p.g, p.n);
    }
    Some(())
}

/// Host copy of every parameter's values and every sign bitmap
/// (best-checkpoint keeping).
pub struct Snap {
    pub params: Vec<Vec<f32>>,
    pub bits: Vec<Vec<u32>>,
}

/// Every path's bitmaps (U then V) with their word counts, in [`Net::blr_lins`] order.
fn bitmaps(net: &Net) -> Vec<(&wgpu::Buffer, usize)> {
    let mut v = Vec::new();
    for (_, lin) in net.blr_lins() {
        if let Lin::Blr { blr, .. } = lin {
            for p in &blr.paths {
                v.push((&p.bits_u, blr.m * blr.r / 32));
                v.push((&p.bits_v, blr.n * blr.r / 32));
            }
        }
    }
    v
}

/// `with_bits`: also copy the sign bitmaps (sign training).
pub fn snapshot(c: Ctx, net: &Net, with_bits: bool) -> Option<Snap> {
    let params = net.params().into_iter().chain(net.dense_params()).map(|p| to::download(c, &p.w, p.n)).collect::<Option<Vec<_>>>()?;
    let bits = if with_bits { bitmaps(net).iter().map(|(b, n)| to::download_u32(c, b, *n)).collect::<Option<Vec<_>>>()? } else { Vec::new() };
    Some(Snap { params, bits })
}

/// Restore a [`snapshot`].
pub fn restore(c: Ctx, net: &Net, snap: &Snap) {
    for (p, v) in net.params().into_iter().chain(net.dense_params()).zip(&snap.params) {
        c.queue.write_buffer(&p.w, 0, bytemuck::cast_slice(v));
    }
    for ((b, _), v) in bitmaps(net).iter().zip(&snap.bits) {
        c.queue.write_buffer(b, 0, bytemuck::cast_slice(v));
    }
    c.queue.submit(std::iter::empty());
    for (_, lin) in net.dense_lins() {
        refresh_planes(c, lin);
    }
}

/// Copy the trained metric and the sign bits of every binary low-rank tensor
/// and every norm gain back into payloads of `model` (same layout and
/// sizes): returns (tensor name, new bytes) for every changed entry.
pub fn export(c: Ctx, net: &Net, model: &CmfModel) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut out = Vec::new();
    let to_bytes_f16 = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|&f| f32_to_f16(f).to_le_bytes()).collect() };
    let mut norm = |name: String, p: &Param| -> Result<(), String> {
        let v = to::download(c, &p.w, p.n).ok_or("download")?;
        let e = model.tensor(&name).ok_or_else(|| format!("missing {name}"))?;
        let bytes = match e.dtype as u8 {
            1 => to_bytes_f16(&v),
            0 => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            other => return Err(format!("{name}: norm dtype {other} not writable")),
        };
        out.push((name, bytes));
        Ok(())
    };
    norm("model.norm.weight".into(), &net.final_norm)?;
    for (li, ly) in net.layers.iter().enumerate() {
        let p = format!("model.layers.{li}.");
        norm(format!("{p}input_layernorm.weight"), &ly.ln1)?;
        norm(format!("{p}post_attention_layernorm.weight"), &ly.ln2)?;
        if let Mix::Attn { qn, kn, .. } = &ly.mix {
            norm(format!("{p}self_attn.q_norm.weight"), qn)?;
            norm(format!("{p}self_attn.k_norm.weight"), kn)?;
        }
    }
    for (name, lin) in net.dense_lins() {
        let Lin::Train { p, .. } = lin else { continue };
        let v = to::download(c, &p.w, p.n).ok_or("download")?;
        let e = model.tensor(&name).ok_or_else(|| format!("missing {name}"))?;
        let bytes = match e.dtype as u8 {
            1 => to_bytes_f16(&v),
            0 => v.iter().flat_map(|f| f.to_le_bytes()).collect(),
            other => return Err(format!("{name}: dtype {other} not writable (heal an f16 or f32 student)")),
        };
        out.push((name, bytes));
    }
    for (name, lin) in net.blr_lins() {
        let Lin::Blr { blr, params, .. } = lin else { continue };
        let e = model.tensor(&name).ok_or_else(|| format!("missing {name}"))?;
        let mut bytes = model.entry_bytes(e).to_vec();
        let r = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let (m, n) = (blr.m, blr.n);
        if r % 8 != 0 || n % 8 != 0 {
            return Err(format!("{name}: rank {r} / width {n} not byte-aligned"));
        }
        let wpr = blr.r / 32;
        let mut off = 16;
        for (pr, path) in params.iter().zip(&blr.paths) {
            let h = to::download(c, &pr[0].w, m).ok_or("download")?;
            let l = to::download(c, &pr[1].w, pr[1].n).ok_or("download")?;
            let g = to::download(c, &pr[2].w, n).ok_or("download")?;
            bytes[off..off + 2 * m].copy_from_slice(&to_bytes_f16(&h));
            off += 2 * m;
            bytes[off..off + 2 * n].copy_from_slice(&to_bytes_f16(&g));
            off += 2 * n;
            bytes[off..off + 2 * r].copy_from_slice(&to_bytes_f16(&l[..r]));
            off += 2 * r;
            // signs: U [m][r/8] (rank-minor), V [r][n/8] (rank-major); pad ranks never written
            let bu = to::download_u32(c, &path.bits_u, m * wpr).ok_or("download")?;
            u_bits_to_file(&bu, m, r, wpr, &mut bytes[off..off + m * r / 8]);
            off += m * r / 8;
            let bv = to::download_u32(c, &path.bits_v, n * wpr).ok_or("download")?;
            v_bits_to_file(&bv, n, r, wpr, &mut bytes[off..off + r * n / 8]);
            off += r * n / 8;
        }
        out.push((name, bytes));
    }
    Ok(out)
}

/// Device U bitmap `[m][wpr]` → file `[m][r/8]` (byte b of row i = ranks 8b..8b+7).
fn u_bits_to_file(bu: &[u32], m: usize, r: usize, wpr: usize, out: &mut [u8]) {
    for i in 0..m {
        for b in 0..r / 8 {
            out[i * r / 8 + b] = ((bu[i * wpr + (8 * b) / 32] >> ((8 * b) % 32)) & 0xff) as u8;
        }
    }
}

/// Device V bitmap `[n][wpr]` (rank bits per input row) → file `[r][n/8]`
/// (input bits per rank): a bit transpose, parallel over ranks.
fn v_bits_to_file(bv: &[u32], n: usize, r: usize, wpr: usize, out: &mut [u8]) {
    let row = n / 8;
    let lanes = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4).min(16);
    let per = r.div_ceil(lanes).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(per * row).enumerate() {
            sc.spawn(move || {
                for (kk, dst) in chunk.chunks_mut(row).enumerate() {
                    let k = ci * per + kk;
                    dst.fill(0);
                    for j in 0..n {
                        if (bv[j * wpr + k / 32] >> (k % 32)) & 1 == 1 {
                            dst[j / 8] |= 1 << (j % 8);
                        }
                    }
                }
            });
        }
    });
}

// ───────────────────────────── resumable state ─────────────────────────────

/// Every device buffer a resumed run needs, in a fixed order: each
/// parameter with its Adam moments, every bitmap, every sign matrix's
/// latents, moments, factored statistics and starting bits.
fn state_bufs(net: &Net) -> Vec<(&wgpu::Buffer, usize)> {
    let mut v = Vec::new();
    for p in net.params().into_iter().chain(net.dense_params()) {
        v.push((&p.w, p.n));
        v.push((&p.m1, p.n));
        v.push((&p.m2, p.n));
    }
    v.extend(bitmaps(net));
    for (mat, _, _) in sign_mats(net) {
        v.push((&mat.lat, mat.lat_words()));
        if let Some(m) = &mat.mom {
            v.push((m, mat.mom_words()));
        }
        if let Some(x) = &mat.v {
            v.push((x, mat.rows * mat.rp));
        }
        v.push((&mat.vrow, mat.rows));
        v.push((&mat.vcol, mat.rp));
        if let Some(b0) = &mat.bits0 {
            v.push((b0, mat.rows * mat.wpr()));
        }
    }
    v
}

const STATE_MAGIC: &[u8; 8] = b"NVGSTAT1";

fn write_words(path: &std::path::Path, parts: &mut dyn Iterator<Item = Result<Vec<u32>, String>>, sizes: &[usize]) -> Result<u64, String> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut f = std::io::BufWriter::with_capacity(16 << 20, std::fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?);
    let io = |e: std::io::Error| e.to_string();
    f.write_all(STATE_MAGIC).map_err(io)?;
    f.write_all(&(sizes.len() as u64).to_le_bytes()).map_err(io)?;
    for &n in sizes {
        f.write_all(&(n as u64).to_le_bytes()).map_err(io)?;
    }
    let mut bytes = 0u64;
    for v in parts {
        let v = v?;
        f.write_all(bytemuck::cast_slice(&v)).map_err(io)?;
        bytes += v.len() as u64 * 4;
    }
    f.flush().map_err(io)?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(io)?;
    Ok(bytes)
}

fn read_words(path: &std::path::Path, sizes: &[usize], mut each: impl FnMut(usize, &[u32])) -> Result<(), String> {
    use std::io::Read;
    let mut f = std::io::BufReader::with_capacity(16 << 20, std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?);
    let io = |e: std::io::Error| e.to_string();
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr).map_err(io)?;
    if &hdr[..8] != STATE_MAGIC || u64::from_le_bytes(hdr[8..16].try_into().unwrap()) != sizes.len() as u64 {
        return Err(format!("{}: not a state file of this net", path.display()));
    }
    for &n in sizes {
        let mut w = [0u8; 8];
        f.read_exact(&mut w).map_err(io)?;
        if u64::from_le_bytes(w) != n as u64 {
            return Err(format!("{}: buffer sizes differ from this net", path.display()));
        }
    }
    for (i, &n) in sizes.iter().enumerate() {
        let mut v = vec![0u32; n];
        f.read_exact(bytemuck::cast_slice_mut(&mut v)).map_err(io)?;
        each(i, &v);
    }
    Ok(())
}

/// Write the training state (see [`state_bufs`]) to `path` (atomic rename).
/// Returns the bytes written.
pub fn save_state(c: Ctx, net: &Net, path: &std::path::Path) -> Result<u64, String> {
    let bufs = state_bufs(net);
    let sizes: Vec<usize> = bufs.iter().map(|b| b.1).collect();
    let mut it = bufs.iter().map(|(b, n)| to::download_u32(c, b, *n).ok_or_else(|| "download".to_string()));
    write_words(path, &mut it, &sizes)
}

/// Load a [`save_state`] file into a net of the same layout and options.
pub fn load_state(c: Ctx, net: &Net, path: &std::path::Path) -> Result<(), String> {
    let bufs = state_bufs(net);
    let sizes: Vec<usize> = bufs.iter().map(|b| b.1).collect();
    let mut pending = 0usize;
    read_words(path, &sizes, |i, v| {
        c.queue.write_buffer(bufs[i].0, 0, bytemuck::cast_slice(v));
        pending += v.len() * 4;
        // flush the staged writes every ~1 GB (a 27B state is tens of GB)
        if pending > 1 << 30 {
            c.queue.submit(std::iter::empty());
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            pending = 0;
        }
    })?;
    c.queue.submit(std::iter::empty());
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    for (_, lin) in net.dense_lins() {
        refresh_planes(c, lin).ok_or("plane refresh")?;
    }
    Ok(())
}

/// Write a [`Snap`] (best checkpoint) to `path`.
pub fn save_snap(snap: &Snap, path: &std::path::Path) -> Result<u64, String> {
    let sizes: Vec<usize> = snap.params.iter().map(|v| v.len()).chain(snap.bits.iter().map(|v| v.len())).collect();
    let mut it = snap.params.iter().map(|v| Ok(v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>())).chain(snap.bits.iter().map(|v| Ok(v.clone())));
    write_words(path, &mut it, &sizes)
}

/// Read a [`save_snap`] file shaped like `like` (a snapshot of the same net).
pub fn load_snap(like: &Snap, path: &std::path::Path) -> Result<Snap, String> {
    let np = like.params.len();
    let sizes: Vec<usize> = like.params.iter().map(|v| v.len()).chain(like.bits.iter().map(|v| v.len())).collect();
    let mut out = Snap { params: Vec::new(), bits: Vec::new() };
    read_words(path, &sizes, |i, v| {
        if i < np {
            out.params.push(v.iter().map(|&w| f32::from_bits(w)).collect());
        } else {
            out.bits.push(v.to_vec());
        }
    })?;
    Ok(out)
}

/// Mean KL of `ids` against the teacher top-k (forward only).
pub fn loss_only(c: Ctx, net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize) -> Option<f64> {
    Some(loss_sum(c, net, w, ids, tid, tp, k, false)? / w.ntok as f64)
}

/// Σ_rows of the (weighted, see [`step_w`]) KL of `ids` (forward only).
#[allow(clippy::too_many_arguments)]
pub fn loss_sum(c: Ctx, net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize, weighted: bool) -> Option<f64> {
    loss_sum_h(c, net, w, ids, None, tid, tp, k, weighted)
}

/// [`loss_sum`] for a net that starts past layer 0 (`h0` its input).
#[allow(clippy::too_many_arguments)]
pub fn loss_sum_h(c: Ctx, net: &Net, w: &Work, ids: &[u32], h0: Option<&[f32]>, tid: &[u32], tp: &[f32], k: usize, weighted: bool) -> Option<f64> {
    let n = w.ntok;
    forward_in(c, net, w, ids, h0)?;
    let btid = to::zeros(c, n * k);
    c.queue.write_buffer(&btid, 0, bytemuck::cast_slice(tid));
    let btp = to::upload(c, tp);
    th::head_kl_w(c, &net.head, &w.head, w.get("n1"), &btid, &btp, k, w.get("dn1"), w.get("loss"), n, n as f32, 1024.0, weighted)?;
    Some(to::download(c, w.get("loss"), n)?.iter().map(|&x| x as f64).sum::<f64>())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end gradient check through every layer's backward: the
    /// analytic gradient of an RMSNorm gain of layer 0 (and of the final
    /// norm) against a central finite difference of the loss, on a real
    /// model (`CMF_TRAIN_TEST_MODEL`, e.g. Qwen3.5-0.8B q8_2f), the student
    /// a perturbed copy of the teacher so the loss is not at its minimum.
    #[test]
    #[ignore]
    fn net_gradient_check() {
        let Ok(path) = std::env::var("CMF_TRAIN_TEST_MODEL") else { return };
        let Some(c) = to::dev() else { return };
        let model = CmfModel::open_sharded(&path).expect("model");
        let teacher = Net::load(c, &model, true, false, None).expect("teacher");
        let student = Net::load(c, &model, true, true, None).expect("student");
        let (nb, t, k) = (2usize, 64usize, 32usize);
        let w = Work::new(c, &teacher, nb, t);
        let vocab = teacher.geom.vocab as u64;
        let ids: Vec<u32> = (0..(nb * t) as u64).map(|i| ((i * 7919 + 13) % vocab.min(30000)) as u32).collect();
        let (tid, tp) = teacher_topk(c, &teacher, &w, &ids, k).expect("topk");
        // perturb every layer-0 input norm gain of the student by +3 %
        let hs = student.geom.hs;
        let ln = to::download(c, &student.layers[0].ln1.w, hs).unwrap();
        let pert: Vec<f32> = ln.iter().map(|v| v * 1.03).collect();
        c.queue.write_buffer(&student.layers[0].ln1.w, 0, bytemuck::cast_slice(&pert));
        let st = step(c, &student, &w, &ids, &tid, &tp, k, 1024.0, None).expect("step");
        eprintln!("loss {:.6}", st.loss);
        let g_ln = to::download(c, &student.layers[0].ln1.g, hs).unwrap();
        let g_fin = to::download(c, &student.final_norm.g, hs).unwrap();
        for (name, buf, grad, base) in [("layer0.ln1", &student.layers[0].ln1.w, &g_ln, pert.clone()), ("final", &student.final_norm.w, &g_fin, to::download(c, &student.final_norm.w, hs).unwrap())] {
            // the coordinates with the largest analytic gradient
            let mut order: Vec<usize> = (0..hs).collect();
            order.sort_by(|&a, &b| grad[b].abs().partial_cmp(&grad[a].abs()).unwrap());
            for &j in order.iter().take(4) {
                let mut fds = Vec::new();
                for frac in [0.02f32, 0.08] {
                    let eps = frac * base[j].abs().max(0.05);
                    let mut wp = base.clone();
                    wp[j] += eps;
                    c.queue.write_buffer(buf, 0, bytemuck::cast_slice(&wp));
                    let lp = loss_only(c, &student, &w, &ids, &tid, &tp, k).unwrap();
                    wp[j] -= 2.0 * eps;
                    c.queue.write_buffer(buf, 0, bytemuck::cast_slice(&wp));
                    let lm = loss_only(c, &student, &w, &ids, &tid, &tp, k).unwrap();
                    c.queue.write_buffer(buf, 0, bytemuck::cast_slice(&base));
                    fds.push((lp - lm) / (2.0 * eps as f64));
                }
                eprintln!("{name}[{j}]: analytic {:.5e}  fd(2%) {:.5e}  fd(8%) {:.5e}", grad[j], fds[0], fds[1]);
            }
        }
    }

    /// Healing a tail: the analytic gradient of the trained dense weights
    /// (every trained linear of the last two layers, rows half weighted out)
    /// against the central difference of the weighted loss along that
    /// gradient: d/dε L(W + ε·ĝ) = |g|. The student is the teacher with its
    /// last MLP down projection scaled by 1.05 (`CMF_TRAIN_TEST_MODEL`).
    #[test]
    #[ignore]
    fn heal_dense_gradient_check() {
        let Ok(path) = std::env::var("CMF_TRAIN_TEST_MODEL") else { return };
        let Some(c) = to::dev() else { return };
        let model = CmfModel::open_sharded(&path).expect("model");
        let teacher = Net::load(c, &model, true, false, None).expect("teacher");
        let nl = teacher.geom.nl;
        let student = Net::load_heal(c, &model, HealCfg { train_from: nl - 2, ffn_only: false, prefix_cached: false }).expect("student");
        assert!(student.dense_lins().len() >= 6, "trained linears: {}", student.dense_lins().len());
        let (nb, t, k) = (2usize, 64usize, 32usize);
        let w = Work::new(c, &student, nb, t);
        let tw = Work::new(c, &teacher, nb, t);
        let vocab = teacher.geom.vocab as u64;
        let ids: Vec<u32> = (0..(nb * t) as u64).map(|i| ((i * 7919 + 13) % vocab.min(30000)) as u32).collect();
        let (tid, mut tp) = teacher_topk(c, &teacher, &tw, &ids, k).expect("topk");
        drop(teacher);
        // every other row unscored
        let mut total = 0f64;
        for r in 0..nb * t {
            let wt: f32 = if r % 2 == 0 { 1.0 } else { 0.0 };
            total += wt as f64;
            for q in 0..k {
                tp[r * k + q] *= wt;
            }
        }
        let Lin::Train { p: down, .. } = &student.layers[nl - 1].down else { panic!("down not trained") };
        let wd = to::download(c, &down.w, down.n).unwrap();
        let pert: Vec<f32> = wd.iter().map(|v| v * 1.05).collect();
        c.queue.write_buffer(&down.w, 0, bytemuck::cast_slice(&pert));
        refresh_planes(c, &student.layers[nl - 1].down).unwrap();
        zero_grads(c, &student);
        let st = step_w(c, &student, &w, &ids, &tid, &tp, k, 1024.0, None, Some(total)).expect("step");
        let dp = student.dense_params();
        let g: Vec<Vec<f32>> = dp.iter().map(|p| to::download(c, &p.g, p.n).unwrap()).collect();
        let w0: Vec<Vec<f32>> = dp.iter().map(|p| to::download(c, &p.w, p.n).unwrap()).collect();
        let gfull = g.iter().flatten().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        // the direction: the 4096 largest gradient entries (a step spread
        // over every weight would sit below the f16 planes' resolution)
        let mut mags: Vec<f32> = g.iter().flatten().map(|x| x.abs()).collect();
        let cut = mags.len() - 4096;
        let thr = *mags.select_nth_unstable_by(cut, |a, b| a.partial_cmp(b).unwrap()).1;
        let g: Vec<Vec<f32>> = g.iter().map(|v| v.iter().map(|&x| if x.abs() >= thr { x } else { 0.0 }).collect()).collect();
        let gn = g.iter().flatten().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        eprintln!("loss {:.6}, |g| {gfull:.5e} (top-4096 part {gn:.5e}), {} trained tensors", st.loss, dp.len());
        let lw = |net: &Net| loss_sum(c, net, &w, &ids, &tid, &tp, k, true).unwrap() / total;
        let set = |eps: f64| {
            for ((p, w0), g) in dp.iter().zip(&w0).zip(&g) {
                let v: Vec<f32> = w0.iter().zip(g).map(|(&a, &b)| (a as f64 + eps * b as f64 / gn) as f32).collect();
                c.queue.write_buffer(&p.w, 0, bytemuck::cast_slice(&v));
            }
            c.queue.submit(std::iter::empty());
            for (_, lin) in student.dense_lins() {
                refresh_planes(c, lin).unwrap();
            }
        };
        // the same step with the frozen prefix cached: same loss, same gradients
        {
            let pnet = Net::load_prefix(c, &model, nl - 2).expect("prefix net");
            let pw = Work::new(c, &pnet, nb, t);
            let h0 = prefix_hidden(c, &pnet, &pw, &ids).expect("prefix");
            drop((pw, pnet));
            let cached = Net::load_heal(c, &model, HealCfg { train_from: nl - 2, ffn_only: false, prefix_cached: true }).expect("cached student");
            let Lin::Train { p: cd, .. } = &cached.layers[nl - 1].down else { panic!("down not trained") };
            c.queue.write_buffer(&cd.w, 0, bytemuck::cast_slice(&pert));
            refresh_planes(c, &cached.layers[nl - 1].down).unwrap();
            let cw = Work::new(c, &cached, nb, t);
            let sc = step_wh(c, &cached, &cw, &ids, Some(&h0), &tid, &tp, k, 1024.0, None, Some(total)).expect("cached step");
            let gc: Vec<f32> = cached.dense_params().iter().flat_map(|p| to::download(c, &p.g, p.n).unwrap()).collect();
            let g0: Vec<f32> = dp.iter().flat_map(|p| to::download(c, &p.g, p.n).unwrap()).collect();
            let dmax = g0.iter().zip(&gc).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let gm = g0.iter().fold(0f32, |a, x| a.max(x.abs()));
            eprintln!("prefix-cached step: loss {:.6} vs {:.6}, max |Δg| {dmax:.3e} of max |g| {gm:.3e}", sc.loss, st.loss);
            assert!((sc.loss - st.loss).abs() < 1e-4 * st.loss.abs().max(1e-3));
            assert!(dmax <= 1e-3 * gm);
        }
        let base = lw(&student);
        eprintln!("weighted loss: step {:.6} vs forward {:.6}", st.loss, base);
        assert!((st.loss - base).abs() < 1e-3 * base.abs().max(1e-3));
        let mut ok = false;
        for eps in [0.01f64, 0.03, 0.1] {
            set(eps);
            let lp = lw(&student);
            set(-eps);
            let lm = lw(&student);
            let fd = (lp - lm) / (2.0 * eps);
            eprintln!("eps {eps}: fd {fd:.5e} vs |g| {gn:.5e} (ratio {:.3})", fd / gn);
            ok |= (fd / gn - 1.0).abs() < 0.1;
        }
        set(0.0);
        assert!(ok, "directional derivative does not match the gradient norm");
    }

    /// DTG-MA gates: with every gate open the net is the plain one (same
    /// loss to the bit); the analytic gradient of a gate value against a
    /// central difference of the loss, for the largest-gradient gates of the
    /// first and the last layer, the gates set to a spread of values
    /// (`CMF_TRAIN_TEST_MODEL`).
    #[test]
    #[ignore]
    fn mask_gradient_check() {
        let Ok(path) = std::env::var("CMF_TRAIN_TEST_MODEL") else { return };
        let Some(c) = to::dev() else { return };
        let model = CmfModel::open_sharded(&path).expect("model");
        let teacher = Net::load(c, &model, true, false, None).expect("teacher");
        let (nb, t, k) = (2usize, 64usize, 32usize);
        let tw = Work::new(c, &teacher, nb, t);
        let vocab = teacher.geom.vocab as u64;
        let ids: Vec<u32> = (0..(nb * t) as u64).map(|i| ((i * 7919 + 13) % vocab.min(30000)) as u32).collect();
        let (tid, tp) = teacher_topk(c, &teacher, &tw, &ids, k).expect("topk");
        let plain = loss_only(c, &teacher, &tw, &ids, &tid, &tp, k).unwrap();
        drop((tw, teacher));
        let mut student = Net::load(c, &model, true, true, None).expect("student");
        student.enable_masks(c);
        let w = Work::new(c, &student, nb, t);
        let open = loss_only(c, &student, &w, &ids, &tid, &tp, k).unwrap();
        eprintln!("loss: plain {plain:.8}, every gate open {open:.8}");
        assert_eq!(plain, open, "open gates must leave the net unchanged");
        let nl = student.geom.nl;
        let mut s: Vec<Vec<f32>> = student.layers.iter().enumerate().map(|(l, ly)| (0..ly.gate.m()).map(|j| 0.4 + 0.6 * (((j * 37 + l * 11) % 101) as f32 / 100.0)).collect()).collect();
        student.set_masks(c, &s).unwrap();
        zero_grads(c, &student);
        let st = step(c, &student, &w, &ids, &tid, &tp, k, 1024.0, None).expect("step");
        let g = student.mask_grads(c).unwrap();
        eprintln!("loss with gates {:.6}", st.loss);
        let mut worst = 0f64;
        for l in [0, nl - 1] {
            let mut order: Vec<usize> = (0..g[l].len()).collect();
            order.sort_by(|&a, &b| g[l][b].abs().partial_cmp(&g[l][a].abs()).unwrap());
            for &j in order.iter().take(4) {
                // two step sizes: through all the layers above, the f16
                // forward's rounding swamps a small one (the layer-0 norm
                // gains of `net_gradient_check` behave alike)
                let base = s[l][j];
                let mut fds = Vec::new();
                for eps in [0.02f32, 0.08] {
                    s[l][j] = base + eps;
                    student.set_masks(c, &s).unwrap();
                    let lp = loss_only(c, &student, &w, &ids, &tid, &tp, k).unwrap();
                    s[l][j] = base - eps;
                    student.set_masks(c, &s).unwrap();
                    let lm = loss_only(c, &student, &w, &ids, &tid, &tp, k).unwrap();
                    fds.push((lp - lm) / (2.0 * eps as f64));
                }
                s[l][j] = base;
                let r = fds.iter().map(|fd| (fd / g[l][j] as f64 - 1.0).abs()).fold(f64::MAX, f64::min);
                // the last layer's difference is clean (≤ 2 %); through all
                // the layers above the first one it scatters around the
                // gradient by the f16 forward's rounding: the two step sizes
                // must bracket it (15 % slack)
                let (lo, hi) = (fds[0].min(fds[1]), fds[0].max(fds[1]));
                let a = g[l][j] as f64;
                let ok = if l + 1 == nl { r < 0.02 } else { r < 0.15 || (a >= lo - 0.15 * a.abs() && a <= hi + 0.15 * a.abs()) };
                worst = worst.max(if ok { 0.0 } else { r });
                eprintln!("L{l} gate {j}: analytic {:.5e}  fd(0.02) {:.5e}  fd(0.08) {:.5e}  (rel {r:.3}){}", g[l][j], fds[0], fds[1], if ok { "" } else { "  ✗" });
            }
        }
        student.set_masks(c, &s).unwrap();
        assert!(worst == 0.0, "gate gradient off by {worst:.3}");
    }

    /// Where a healing step's time goes on this device: the teacher pass
    /// and six steps of a full-depth MLP heal of `CMF_TRAIN_TEST_MODEL` at
    /// the compress shape (2 windows × 2048 tokens), host-timed, then the
    /// profiler's table (run with `CMF_NVG_PROF=2` or `1`).
    #[test]
    #[ignore]
    fn heal_step_profile() {
        let Ok(path) = std::env::var("CMF_TRAIN_TEST_MODEL") else { return };
        let Some(c) = to::dev() else { return };
        let model = CmfModel::open_sharded(&path).expect("model");
        let (nb, t, k) = (2usize, 2048usize, 65usize);
        let vocab = model.arch().vocab_size as u64;
        let ids: Vec<u32> = (0..(nb * t) as u64).map(|i| ((i * 7919 + 13) % vocab.min(30000)) as u32).collect();
        let (tid, tp) = {
            let teacher = Net::load(c, &model, true, false, None).expect("teacher");
            let tw = Work::new(c, &teacher, nb, t);
            let _ = teacher_topk(c, &teacher, &tw, &ids, k).expect("warm");
            let t0 = std::time::Instant::now();
            let r = teacher_topk(c, &teacher, &tw, &ids, k).expect("topk");
            eprintln!("teacher pass: {:.2} s for {} tokens ({:.0} tok/s)", t0.elapsed().as_secs_f64(), nb * t, (nb * t) as f64 / t0.elapsed().as_secs_f64());
            r
        };
        let student = Net::load_heal(c, &model, HealCfg { train_from: 0, ffn_only: true, prefix_cached: false }).expect("student");
        let w = Work::new(c, &student, nb, t);
        for i in 0..6 {
            let t0 = std::time::Instant::now();
            let st = step_w(c, &student, &w, &ids, &tid, &tp, k, 1024.0, None, None).expect("step");
            prof::step_end();
            let ts = t0.elapsed().as_secs_f64();
            adam(c, &student, 1e-6, i + 1, 0.9, 0.999, 1e-8, 1.0).expect("adam");
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            eprintln!("step {i}: loss {:.4}, fwd+bwd {:.2} s, with adam {:.2} s", st.loss, ts, t0.elapsed().as_secs_f64());
        }
        prof::report();
    }

    /// Teacher top-k on one small batch of the 0.8B check models
    /// (`CMF_TRAIN_TEST_BLR` = blr student, `CMF_TRAIN_TEST_TEACHER` = its teacher).
    #[allow(clippy::type_complexity)]
    fn blr_fixture(sign: Option<&SignInit>) -> Option<(Ctx, CmfModel, Net, Work, Vec<u32>, Vec<u32>, Vec<f32>, usize)> {
        blr_fixture_sized(sign, 2, 64, 32)
    }

    #[allow(clippy::type_complexity)]
    fn blr_fixture_sized(sign: Option<&SignInit>, nb: usize, t: usize, k: usize) -> Option<(Ctx, CmfModel, Net, Work, Vec<u32>, Vec<u32>, Vec<f32>, usize)> {
        let sp = std::env::var("CMF_TRAIN_TEST_BLR").ok()?;
        let tp_path = std::env::var("CMF_TRAIN_TEST_TEACHER").ok()?;
        let c = to::dev()?;
        let ids: Vec<u32>;
        let (tid, tp);
        {
            let tmodel = CmfModel::open_sharded(&tp_path).expect("teacher model");
            let teacher = Net::load(c, &tmodel, true, false, None).expect("teacher");
            let tw = Work::new(c, &teacher, nb, t);
            let vocab = teacher.geom.vocab as u64;
            ids = (0..(nb * t) as u64).map(|i| ((i * 7919 + 13) % vocab.min(30000)) as u32).collect();
            (tid, tp) = teacher_topk(c, &teacher, &tw, &ids, k).expect("topk");
        }
        let smodel = CmfModel::open_sharded(&sp).expect("student model");
        let student = Net::load(c, &smodel, true, true, sign).expect("student");
        let w = Work::new(c, &student, nb, t);
        Some((c, smodel, student, w, ids, tid, tp, k))
    }

    fn sign_cfg(eta: f32, defer: bool) -> SignStepCfg {
        SignStepCfg { opt: ts::SignOpt::Fact, eta, step: 1, b1: 0.9, b2: 0.999, eps: 1e-3, clamp: 4.0, hyst: 0.0, k_ss: 8.0, seed: 42, defer }
    }

    fn grads(c: Ctx, net: &Net) -> Vec<Vec<u32>> {
        net.params().iter().map(|p| to::download_u32(c, &p.g, p.n).unwrap()).collect()
    }

    /// With η = 0 the sign ops change nothing the metric path reads: bits
    /// unchanged, loss and every metric gradient bit-identical to a step
    /// without sign training (guards buffer aliasing and hazards).
    #[test]
    #[ignore]
    fn step_sign_lr0_equals_metric_only() {
        let init = SignInit { opt: ts::SignOpt::Fact, tau: 0.25, seed: 42, keep_bits0: true, defer: false };
        let Some((c, _m, net, w, ids, tid, tp, k)) = blr_fixture(Some(&init)) else { return };
        let b0 = snapshot(c, &net, true).unwrap().bits;
        let a = step(c, &net, &w, &ids, &tid, &tp, k, 1024.0, None).unwrap();
        let ga = grads(c, &net);
        zero_grads(c, &net);
        let b = step(c, &net, &w, &ids, &tid, &tp, k, 1024.0, Some(&sign_cfg(0.0, false))).unwrap();
        let gb = grads(c, &net);
        let b1 = snapshot(c, &net, true).unwrap().bits;
        let gdiff: usize = ga.iter().zip(&gb).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count()).sum();
        let bdiff: usize = b0.iter().zip(&b1).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count()).sum();
        eprintln!("loss {:.9} vs {:.9}; metric-gradient words differing {gdiff}; bitmap words differing {bdiff}; flips {}", a.loss, b.loss, b.flips);
        assert_eq!(a.loss.to_bits(), b.loss.to_bits());
        assert_eq!(gdiff, 0);
        assert_eq!(bdiff, 0);
        assert_eq!(b.flips, 0);
        let st = sign_stats(c, &net).unwrap();
        assert!(st.iter().all(|(_, f, _)| *f == 0));
    }

    /// Latent words and bitmaps of every sign matrix (fused vs deferred comparison).
    fn sign_state(c: Ctx, net: &Net) -> Vec<Vec<u32>> {
        let mut v = Vec::new();
        for (mat, bits, _) in sign_mats(net) {
            v.push(to::download_u32(c, &mat.lat, mat.lat_words()).unwrap());
            v.push(to::download_u32(c, bits, mat.rows * mat.wpr()).unwrap());
            v.push(to::download_u32(c, &mat.vrow, mat.rows).unwrap());
            v.push(to::download_u32(c, &mat.vcol, mat.rp).unwrap());
        }
        v
    }

    fn set_sign_state(c: Ctx, net: &Net, st: &[Vec<u32>]) {
        for ((mat, bits, _), ch) in sign_mats(net).iter().zip(st.chunks(4)) {
            c.queue.write_buffer(&mat.lat, 0, bytemuck::cast_slice(&ch[0]));
            c.queue.write_buffer(bits, 0, bytemuck::cast_slice(&ch[1]));
            c.queue.write_buffer(&mat.vrow, 0, bytemuck::cast_slice(&ch[2]));
            c.queue.write_buffer(&mat.vcol, 0, bytemuck::cast_slice(&ch[3]));
        }
        c.queue.submit(std::iter::empty());
    }

    /// The in-place update inside the reverse sweep equals updating every
    /// matrix after the sweep from saved gradients: identical bits and latents.
    #[test]
    #[ignore]
    fn fused_equals_deferred() {
        let init = SignInit { opt: ts::SignOpt::Fact, tau: 0.25, seed: 42, keep_bits0: true, defer: true };
        let Some((c, _m, net, w, ids, tid, tp, k)) = blr_fixture(Some(&init)) else { return };
        let s0 = sign_state(c, &net);
        let a = step(c, &net, &w, &ids, &tid, &tp, k, 1024.0, Some(&sign_cfg(1e-2, false))).unwrap();
        let sa = sign_state(c, &net);
        set_sign_state(c, &net, &s0);
        zero_grads(c, &net);
        let b = step(c, &net, &w, &ids, &tid, &tp, k, 1024.0, Some(&sign_cfg(1e-2, true))).unwrap();
        let cnt = apply_deferred(c, &net, &w, &sign_cfg(1e-2, true)).unwrap();
        let sb = sign_state(c, &net);
        let fl_b: u64 = cnt[..20].iter().map(|&x| x as u64).sum();
        let diff: usize = sa.iter().zip(&sb).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p != q).count()).sum();
        let live = net.sign_live();
        eprintln!("loss {:.6} / {:.6}; flips fused {} deferred {fl_b} of {live} signs; state words differing {diff}", a.loss, b.loss, a.flips);
        assert!(a.flips > 0);
        assert_eq!(a.flips, fl_b);
        assert_eq!(diff, 0);
    }

    /// Export writes the bits: no training → payloads byte-identical; known
    /// flips survive export and a reload of the payload.
    #[test]
    #[ignore]
    fn export_bits_roundtrip() {
        let Some((c, model, net, _w, ..)) = blr_fixture(None) else { return };
        let out = export(c, &net, &model).unwrap();
        let mut nblr = 0;
        for (name, bytes) in &out {
            let e = model.tensor(name).unwrap();
            assert_eq!(bytes.as_slice(), model.entry_bytes(e), "{name} changed without training");
            nblr += (e.dtype as u8 == 19) as usize;
        }
        // flip known bits of the first blr tensor's first path
        let (name, lin) = net.blr_lins().into_iter().next().unwrap();
        let Lin::Blr { blr, role, .. } = lin else { unreachable!() };
        let p0 = &blr.paths[0];
        let wpr = blr.r / 32;
        let mut bu = to::download_u32(c, &p0.bits_u, blr.m * wpr).unwrap();
        let mut bv = to::download_u32(c, &p0.bits_v, blr.n * wpr).unwrap();
        for &(i, k) in &[(0usize, 0usize), (3, 5), (blr.m - 1, 9), (10, 31)] {
            bu[i * wpr + k / 32] ^= 1 << (k % 32);
        }
        for &(j, k) in &[(0usize, 0usize), (7, 9), (blr.n - 1, 33), (100, 1)] {
            bv[j * wpr + k / 32] ^= 1 << (k % 32);
        }
        c.queue.write_buffer(&p0.bits_u, 0, bytemuck::cast_slice(&bu));
        c.queue.write_buffer(&p0.bits_v, 0, bytemuck::cast_slice(&bv));
        let out = export(c, &net, &model).unwrap();
        let bytes = &out.iter().find(|(n, _)| *n == name).unwrap().1;
        let re = load_blr(c, bytes, role, None, &mut 0).unwrap();
        let Lin::Blr { blr: rb, .. } = &re else { unreachable!() };
        let ru = to::download_u32(c, &rb.paths[0].bits_u, blr.m * wpr).unwrap();
        let rv = to::download_u32(c, &rb.paths[0].bits_v, blr.n * wpr).unwrap();
        eprintln!("{nblr} blr payloads round-tripped; {name}: reloaded U equal {}, V equal {}", ru == bu, rv == bv);
        assert_eq!(ru, bu);
        assert_eq!(rv, bv);
    }

    /// On the real network: flipping one of the 32 largest-|G| signs of the
    /// layer-0 gate changes the loss in the direction −2s·G predicts for at
    /// least 90 % of them.
    #[test]
    #[ignore]
    fn net_sign_flip_check() {
        let init = SignInit { opt: ts::SignOpt::Fact, tau: 0.25, seed: 42, keep_bits0: true, defer: true };
        let Some((c, _m, net, w, ids, tid, tp, k)) = blr_fixture(Some(&init)) else { return };
        let st = step(c, &net, &w, &ids, &tid, &tp, k, 1024.0, Some(&sign_cfg(0.0, true))).unwrap();
        let Lin::Blr { blr, .. } = &net.layers[0].gate else { panic!("layer-0 gate is not blr") };
        let p0 = &blr.paths[0];
        let sp = p0.sign.as_ref().unwrap();
        let (rows, rp, r) = (sp.u.rows, sp.u.rp, sp.u.r_true);
        let g = to::download(c, sp.u.gsave.as_ref().unwrap(), rows * rp).unwrap();
        let bits = to::download_u32(c, &p0.bits_u, rows * rp / 32).unwrap();
        let mut idx: Vec<usize> = (0..rows * rp).filter(|e| e % rp < r).collect();
        idx.sort_by(|&a, &b| g[b].abs().partial_cmp(&g[a].abs()).unwrap());
        let base = loss_only(c, &net, &w, &ids, &tid, &tp, k).unwrap();
        eprintln!("loss {:.6} (step {:.6})", base, st.loss);
        let mut agree = 0;
        for &e in idx.iter().take(32) {
            let (i, kk) = (e / rp, e % rp);
            let word = i * (rp / 32) + kk / 32;
            let s = if (bits[word] >> (kk % 32)) & 1 == 1 { 1.0 } else { -1.0 };
            let mut fl = bits.clone();
            fl[word] ^= 1 << (kk % 32);
            c.queue.write_buffer(&p0.bits_u, 0, bytemuck::cast_slice(&fl));
            let l1 = loss_only(c, &net, &w, &ids, &tid, &tp, k).unwrap();
            c.queue.write_buffer(&p0.bits_u, 0, bytemuck::cast_slice(&bits));
            let pred = -2.0 * s * g[e] as f64;
            let meas = l1 - base;
            agree += (meas.signum() == pred.signum()) as usize;
            eprintln!("  U[{i},{kk}] G {:+.3e}  ΔL {meas:+.3e}  first-order {pred:+.3e}", g[e]);
        }
        eprintln!("sign agreement {agree}/32");
        assert!(agree >= 29, "only {agree}/32 flips moved the loss as predicted");
    }

    /// Cost of sign training per step: the same net and batch, steps with and
    /// without the sign ops alternating (in-process timers, so a shared card
    /// slows both arms alike). `CMF_TRAIN_TEST_NB` × `CMF_TRAIN_TEST_T`
    /// tokens (default 8 × 512, the A/B's step), `CMF_TRAIN_TEST_REPS` pairs.
    #[test]
    #[ignore]
    fn sign_step_cost() {
        let env = |n: &str, d: usize| std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (nb, t, reps) = (env("CMF_TRAIN_TEST_NB", 8), env("CMF_TRAIN_TEST_T", 512), env("CMF_TRAIN_TEST_REPS", 5));
        let opt = ts::SignOpt::parse(&std::env::var("CMF_TRAIN_TEST_OPT").unwrap_or_else(|_| "fact-mom".into())).expect("CMF_TRAIN_TEST_OPT");
        let init = SignInit { opt, tau: 0.25, seed: 42, keep_bits0: true, defer: false };
        let Some((c, _m, net, w, ids, tid, tp, k)) = blr_fixture_sized(Some(&init), nb, t, 64) else { return };
        let live = net.sign_live();
        eprintln!(
            "{} signs, {} matrices; {} state {:.3} GB, sign scratch {:.3} GB",
            live,
            net.n_sign_mats,
            opt.name(),
            live as f64 * opt.bytes_per_sign() as f64 / 1e9,
            w.sg.as_ref().map(|s| s.bytes() as f64 / 1e9).unwrap_or(0.0)
        );
        let time = |sign: Option<&SignStepCfg>| -> f64 {
            let t0 = std::time::Instant::now();
            step(c, &net, &w, &ids, &tid, &tp, k, 1024.0, sign).unwrap();
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
            let dt = t0.elapsed().as_secs_f64();
            zero_grads(c, &net);
            dt
        };
        let _ = time(None);
        let mut cfg0 = sign_cfg(1e-3, false);
        cfg0.opt = opt;
        let _ = time(Some(&cfg0));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for r in 0..reps {
            let mut cfg = cfg0;
            cfg.step = 2 + r as u32;
            if r % 2 == 0 {
                a.push(time(None));
                b.push(time(Some(&cfg)));
            } else {
                b.push(time(Some(&cfg)));
                a.push(time(None));
            }
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        let (ma, mb) = (med(&mut a), med(&mut b));
        eprintln!("{} tokens per step: metric only {ma:.3} s, with signs {mb:.3} s (median of {reps}; +{:.3} s, ×{:.3})", nb * t, mb - ma, mb / ma);
    }
}
