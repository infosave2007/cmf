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
use crate::gpu_wgpu::train_ops::{self as to, AttnGeom, QkrGeom};
use cortiq_core::quant::{dequant_tensor, f16_to_f32, f32_to_f16};
use cortiq_core::{CmfModel, LayerType, NormStyle};
use std::collections::HashMap;

type Ctx = crate::gpu_wgpu::CtxRef;

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
/// (teacher), or a small dense f32 matrix (frozen).
pub enum Lin {
    Blr { blr: Blr, role: &'static str, params: Vec<[Param; 3]> },
    Plane { w: wgpu::Buffer, wt: Option<wgpu::Buffer>, m: usize, k: usize },
    Dense { w: wgpu::Buffer, m: usize, k: usize },
}

impl Lin {
    fn m(&self) -> usize {
        match self {
            Lin::Blr { blr, .. } => blr.m,
            Lin::Plane { m, .. } | Lin::Dense { m, .. } => *m,
        }
    }
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
}

pub struct Net {
    pub geom: Geom,
    pub layers: Vec<Layer>,
    pub final_norm: Param,
    pub head: Head,
    /// Embedding rows on the host (f16 bits), gathered per step.
    embed: Vec<u16>,
    /// Sign matrices with training state (ids 0..n_sign_mats).
    pub n_sign_mats: u32,
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
    let e = m.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if e.dtype as u8 == 19 {
        return load_blr(c, m.entry_bytes(e), role, sign, next_id);
    }
    let (v, shape) = deq(m, name)?;
    let (rows, cols) = (shape[0], shape[1]);
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
        let geom = Geom::from_model(model)?;
        let norm = |name: &str| -> Result<Param, String> { Ok(Param::new(c, &deq(model, name)?.0, None)) };
        let mut layers = Vec::with_capacity(geom.nl);
        let mut next_id = 0u32;
        for li in 0..geom.nl {
            let p = format!("model.layers.{li}.");
            let mut lin = |n: &str, role: &'static str| load_lin(c, model, &format!("{p}{n}"), role, plane, bwd, sign, &mut next_id);
            let mix = if geom.gdn_layer[li] {
                Mix::Gdn {
                    qkv: lin("linear_attn.in_proj_qkv.weight", "gdn.qkv")?,
                    z: lin("linear_attn.in_proj_z.weight", "gdn.z")?,
                    a: load_lin(c, model, &format!("{p}linear_attn.in_proj_a.weight"), "gdn.a", false, bwd, None, &mut 0)?,
                    b: load_lin(c, model, &format!("{p}linear_attn.in_proj_b.weight"), "gdn.b", false, bwd, None, &mut 0)?,
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
            });
            // Every upload is a queue write staged until the next submission;
            // a 27B teacher's ~49 GB of f16 planes would otherwise sit in
            // staging next to the planes themselves and exhaust the card.
            c.queue.submit(std::iter::empty());
            let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let final_norm = norm("model.norm.weight")?;
        let (emb, eshape) = deq(model, "model.embed_tokens.weight")?;
        let head_w = if model.tensor("lm_head.weight").is_some() { deq(model, "lm_head.weight")?.0 } else { emb.clone() };
        let (v, hs) = (eshape[0], eshape[1]);
        // The teacher only needs the head's forward planes (no transposes);
        // vocabulary slices keep each plane within one storage binding.
        let head = Head::new(c, &head_w, v, hs, bwd, None);
        let embed = emb.iter().map(|&f| f32_to_f16(f)).collect();
        Ok(Net { geom, layers, final_norm, head, embed, n_sign_mats: next_id })
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
        for l in &self.layers {
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
        let blr: HashMap<_, _> = dims.into_iter().map(|(role, (m, n, r, np))| (role, BlrScratch::new(c, ntok, m, n, r, np))).collect();
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
            head: HeadScratch::new(c, 256.min(ntok), g.vocab, g.hs),
            sg: (sg_rows > 0).then(|| SignScratch::new(c, ntok, sg_rows, sg_rp)),
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
        Lin::Plane { w: pl, m, k, .. } => {
            if acc {
                return None;
            }
            tm::plane_linear(c, pl, x, &w.act16, y, ntok, *m, *k, 1.0)
        }
        Lin::Dense { w: dw, m, k } => {
            tm::run(c, &[Op::D(tm::dense_fwd(c, x, dw, y, ntok, *k, *m, acc)?)]);
            Some(())
        }
    }
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
    }
}

fn zero(c: Ctx, b: &wgpu::Buffer, n: usize) {
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero") });
    enc.clear_buffer(b, 0, Some((n * 4) as u64));
    crate::gpu_wgpu::submit_pub(c, enc.finish());
}

fn copy(c: Ctx, src: &wgpu::Buffer, dst: &wgpu::Buffer, n: usize) {
    copy_off(c, src, 0, dst, 0, n);
}

fn copy_off(c: Ctx, src: &wgpu::Buffer, so: usize, dst: &wgpu::Buffer, d_o: usize, n: usize) {
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("copy") });
    enc.copy_buffer_to_buffer(src, (so * 4) as u64, dst, (d_o * 4) as u64, (n * 4) as u64);
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
    c.queue.write_buffer(tape.prm(), 0, bytemuck::cast_slice(&prm));
}

/// Forward of layer `l` from `x` (`w.x` is the input) to `y`; intermediates
/// stay in the workspace for the backward.
fn layer_fwd(c: Ctx, net: &Net, w: &Work, l: usize, x: &wgpu::Buffer, y: &wgpu::Buffer) -> Option<()> {
    let g = &net.geom;
    let ly = &net.layers[l];
    let n = w.ntok;
    to::run(c, &[to::rms_fwd(c, x, &ly.ln1.w, w.get("n1"), w.get("inv1"), n, g.hs, g.eps, g.gemma)?]);
    match &ly.mix {
        Mix::Gdn { qkv, z, a, b, out, a_log, dt_bias, norm, conv } => {
            let tape = w.tape.as_ref().unwrap();
            gdn_params(c, w, g, a_log, dt_bias, norm, conv);
            lin_fwd(c, w, qkv, w.get("n1"), tape.raw(), false)?;
            lin_fwd(c, w, z, w.get("n1"), tape.z(), false)?;
            // a and b into the two halves of the tape's a|b buffer
            lin_fwd(c, w, a, w.get("n1"), w.get("gout"), false)?;
            copy_off(c, w.get("gout"), 0, tape.ab(), 0, n * g.nv);
            lin_fwd(c, w, b, w.get("n1"), w.get("gout"), false)?;
            copy_off(c, w.get("gout"), 0, tape.ab(), n * g.nv, n * g.nv);
            gdn::gdn_forward_dev(tape, g.nk, g.nv, g.c_dim())?;
            copy(c, tape.out(), w.get("gout"), n * g.vd());
            lin_fwd(c, w, out, w.get("gout"), w.get("mix"), false)?;
        }
        Mix::Attn { q, k, v, o, qn, kn } => {
            let (qd, kd) = (g.nh * g.hd, g.nkv * g.hd);
            lin_fwd(c, w, q, w.get("n1"), w.get("qg"), false)?;
            lin_fwd(c, w, k, w.get("n1"), w.get("k"), false)?;
            lin_fwd(c, w, v, w.get("n1"), w.get("v"), false)?;
            let qg = QkrGeom { n, heads: g.nh, hd: g.hd, half: g.half, t: w.t, gemma: g.gemma, eps: g.eps };
            let kg = QkrGeom { heads: g.nkv, ..qg };
            let ag = AttnGeom { nb: w.nb, t: w.t, nh: g.nh, nkv: g.nkv, hd: g.hd };
            tm::run(c, &[Op::D(tm::split_qg(c, w.get("qg"), w.get("q"), w.get("gate"), n, g.nh, g.hd, false)?)]);
            to::run(c, &[
                to::qkr_fwd(c, qg, w.get("q"), &qn.w, w.get("qr"), w.get("qinv"), &w.rope)?,
                to::qkr_fwd(c, kg, w.get("k"), &kn.w, w.get("kr"), w.get("kinv"), &w.rope)?,
                to::attn_fwd(c, ag, w.get("qr"), w.get("kr"), w.get("v"), w.get("o"), w.get("lse"))?,
                to::elt(c, "gate_fwd", n * qd, w.get("o"), w.get("gate"), Some(w.get("og")), None, None, None)?,
            ]);
            let _ = kd;
            lin_fwd(c, w, o, w.get("og"), w.get("mix"), false)?;
        }
    }
    to::run(c, &[
        to::elt(c, "add", n * g.hs, x, w.get("mix"), Some(w.get("x2")), None, None, None)?,
        to::rms_fwd(c, w.get("x2"), &ly.ln2.w, w.get("n2"), w.get("inv2"), n, g.hs, g.eps, g.gemma)?,
    ]);
    lin_fwd(c, w, &ly.gate, w.get("n2"), w.get("gg"), false)?;
    lin_fwd(c, w, &ly.up, w.get("n2"), w.get("uu"), false)?;
    to::run(c, &[to::elt(c, "swiglu_fwd", n * g.inter, w.get("gg"), w.get("uu"), Some(w.get("hh")), None, None, None)?]);
    lin_fwd(c, w, &ly.down, w.get("hh"), w.get("mix"), false)?;
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
    to::run(c, &[to::elt(c, "swiglu_bwd", n * g.inter, w.get("gg"), w.get("uu"), None, Some(w.get("dhh")), Some(w.get("dgg")), Some(w.get("duu")))?]);
    lin_bwd(c, w, &ly.gate, w.get("n2"), w.get("dgg"), w.get("dn2"), false, gs, sg)?;
    lin_bwd(c, w, &ly.up, w.get("n2"), w.get("duu"), w.get("dn2"), true, gs, sg)?;
    to::run(c, &[
        to::rms_bwd(c, w.get("x2"), &ly.ln2.w, w.get("inv2"), w.get("dn2"), w.get("dx2"), n, g.hs, g.gemma)?,
        to::rms_dw(c, w.get("x2"), w.get("inv2"), w.get("dn2"), &ly.ln2.g, n, g.hs)?,
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
            to::run(c, &[to::elt(c, "gate_bwd", n * qd, w.get("o"), w.get("gate"), None, Some(w.get("dog")), Some(w.get("do")), Some(w.get("dgate")))?]);
            to::run(c, &to::attn_bwd(c, ag, w.get("qr"), w.get("kr"), w.get("v"), w.get("o"), w.get("lse"), w.get("do"), w.get("dsum"), w.get("dqr"), w.get("dkr"), w.get("dv"))?);
            to::run(c, &[
                to::qkr_bwd(c, qg, w.get("q"), &qn.w, w.get("qinv"), &w.rope, w.get("dqr"), w.get("dq"), w.get("dwq"))?,
                to::qkr_bwd(c, kg, w.get("k"), &kn.w, w.get("kinv"), &w.rope, w.get("dkr"), w.get("dk"), w.get("dwk"))?,
            ]);
            // gain grads (summed over rows·heads) into the norms' gradients
            let gq = to::zeros(c, g.hd);
            let gk = to::zeros(c, g.hd);
            to::run(c, &[to::qkr_dw(c, qg, w.get("dwq"), &gq)?, to::qkr_dw(c, kg, w.get("dwk"), &gk)?]);
            to::run(c, &[
                to::elt(c, "add", g.hd, &qn.g, &gq, Some(&qn.g), None, None, None)?,
                to::elt(c, "add", g.hd, &kn.g, &gk, Some(&kn.g), None, None, None)?,
            ]);
            tm::run(c, &[Op::D(tm::split_qg(c, w.get("dqg"), w.get("dq"), w.get("dgate"), n, g.nh, g.hd, true)?)]);
            lin_bwd(c, w, q, w.get("n1"), w.get("dqg"), w.get("dn1"), false, gs, sg)?;
            lin_bwd(c, w, k, w.get("n1"), w.get("dk"), w.get("dn1"), true, gs, sg)?;
            lin_bwd(c, w, v, w.get("n1"), w.get("dv"), w.get("dn1"), true, gs, sg)?;
        }
    }
    to::run(c, &[
        to::rms_bwd(c, x, &ly.ln1.w, w.get("inv1"), w.get("dn1"), dx, n, g.hs, g.gemma)?,
        to::rms_dw(c, x, w.get("inv1"), w.get("dn1"), &ly.ln1.g, n, g.hs)?,
        to::elt(c, "add", n * g.hs, dx, w.get("dx2"), Some(dx), None, None, None)?,
    ]);
    Some(())
}

// ───────────────────────────── forward / step ─────────────────────────────

fn embed(c: Ctx, net: &Net, w: &Work, ids: &[u32]) {
    let hs = net.geom.hs;
    let mut rows = vec![0f32; ids.len() * hs];
    for (t, &id) in ids.iter().enumerate() {
        let src = &net.embed[id as usize * hs..(id as usize + 1) * hs];
        for (d, &s) in rows[t * hs..(t + 1) * hs].iter_mut().zip(src) {
            *d = f16_to_f32(s);
        }
    }
    c.queue.write_buffer(&w.ckpt[0], 0, bytemuck::cast_slice(&rows));
}

/// Forward of all layers from the embedded `ids` (checkpoints filled);
/// returns the final-normed hidden in `w.b["n1"]`.
fn forward(c: Ctx, net: &Net, w: &Work, ids: &[u32]) -> Option<()> {
    embed(c, net, w, ids);
    for l in 0..net.geom.nl {
        layer_fwd(c, net, w, l, &w.ckpt[l], &w.ckpt[l + 1])?;
    }
    let g = &net.geom;
    to::run(c, &[to::rms_fwd(c, &w.ckpt[g.nl], &net.final_norm.w, w.get("n1"), w.get("invf"), w.ntok, g.hs, g.eps, g.gemma)?]);
    Some(())
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
    let g = &net.geom;
    let n = w.ntok;
    let sign = if w.sg.is_some() { sign } else { None };
    if let (Some(_), Some(sc)) = (sign, w.sg.as_ref()) {
        ts::reset_counters(c, sc);
    }
    forward(c, net, w, ids)?;
    let btid = to::zeros(c, n * k);
    c.queue.write_buffer(&btid, 0, bytemuck::cast_slice(tid));
    let btp = to::upload(c, tp);
    th::head_kl(c, &net.head, &w.head, w.get("n1"), &btid, &btp, k, w.get("dn1"), w.get("loss"), n, n, gscale)?;
    let loss: f64 = to::download(c, w.get("loss"), n)?.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mut st = StepStats { loss, finite: loss.is_finite(), ..Default::default() };
    if sign.is_some() && !st.finite {
        return Some(st);
    }
    // final norm backward: dx at the last checkpoint
    let (da, db) = (w.get("dx"), w.get("dmix"));
    to::run(c, &[
        to::rms_bwd(c, &w.ckpt[g.nl], &net.final_norm.w, w.get("invf"), w.get("dn1"), da, n, g.hs, g.gemma)?,
        to::rms_dw(c, &w.ckpt[g.nl], w.get("invf"), w.get("dn1"), &net.final_norm.g, n, g.hs)?,
    ]);
    let mut cur = da;
    let mut nxt = db;
    for l in (0..g.nl).rev() {
        layer_fwd(c, net, w, l, &w.ckpt[l], w.get("x"))?;
        layer_bwd(c, net, w, l, &w.ckpt[l], cur, nxt, gscale, sign)?;
        std::mem::swap(&mut cur, &mut nxt);
    }
    st.backward = true;
    if let (Some(_), Some(sc)) = (sign, w.sg.as_ref()) {
        let cnt = ts::read_reset_counters(c, sc)?;
        for (i, f) in st.flips_role.iter_mut().enumerate() {
            *f = cnt[i] as u64;
        }
        st.flips = st.flips_role.iter().sum();
        st.g_nonfinite = cnt[ts::CNT_NONFINITE] as u64;
        st.saturated = cnt[ts::CNT_SATURATED] as u64;
        st.clamped = cnt[ts::CNT_CLAMPED] as u64;
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
/// Global gradient norm over every parameter (NaN/inf when a gradient is).
pub fn grad_norm(c: Ctx, net: &Net) -> Option<f64> {
    let mut s = 0f64;
    for p in net.params() {
        s += tm::sumsq(c, &p.g, p.n)?;
    }
    Some(s.sqrt())
}

/// Clear every gradient (a skipped step).
pub fn zero_grads(c: Ctx, net: &Net) {
    for p in net.params() {
        zero(c, &p.g, p.n);
    }
}

/// Adam over every parameter (masked entries stay put), the gradients
/// multiplied by `gmul` (clipping); clears the grads.
#[allow(clippy::too_many_arguments)]
pub fn adam(c: Ctx, net: &Net, lr: f32, step_no: u32, b1: f32, b2: f32, eps: f32, gmul: f32) -> Option<()> {
    for p in net.params() {
        tm::run(c, &[Op::D(tm::adam(c, &p.w, &p.g, &p.m1, &p.m2, p.mask.as_ref(), p.n, lr, step_no, b1, b2, eps, gmul, p.rel)?)]);
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
    let params = net.params().iter().map(|p| to::download(c, &p.w, p.n)).collect::<Option<Vec<_>>>()?;
    let bits = if with_bits { bitmaps(net).iter().map(|(b, n)| to::download_u32(c, b, *n)).collect::<Option<Vec<_>>>()? } else { Vec::new() };
    Some(Snap { params, bits })
}

/// Restore a [`snapshot`].
pub fn restore(c: Ctx, net: &Net, snap: &Snap) {
    for (p, v) in net.params().iter().zip(&snap.params) {
        c.queue.write_buffer(&p.w, 0, bytemuck::cast_slice(v));
    }
    for ((b, _), v) in bitmaps(net).iter().zip(&snap.bits) {
        c.queue.write_buffer(b, 0, bytemuck::cast_slice(v));
    }
    c.queue.submit(std::iter::empty());
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
    for p in net.params() {
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
    let n = w.ntok;
    forward(c, net, w, ids)?;
    let btid = to::zeros(c, n * k);
    c.queue.write_buffer(&btid, 0, bytemuck::cast_slice(tid));
    let btp = to::upload(c, tp);
    th::head_kl(c, &net.head, &w.head, w.get("n1"), &btid, &btp, k, w.get("dn1"), w.get("loss"), n, n, 1024.0)?;
    Some(to::download(c, w.get("loss"), n)?.iter().map(|&x| x as f64).sum::<f64>() / n as f64)
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
        let init = SignInit { opt: ts::SignOpt::Fact, tau: 0.25, seed: 42, keep_bits0: true, defer: false };
        let Some((c, _m, net, w, ids, tid, tp, k)) = blr_fixture_sized(Some(&init), nb, t, 64) else { return };
        let live = net.sign_live();
        eprintln!(
            "{} signs, {} matrices; latents {:.3} GB, sign scratch {:.3} GB",
            live,
            net.n_sign_mats,
            live as f64 * 2.0 / 1e9,
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
        let _ = time(Some(&sign_cfg(1e-3, false)));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for r in 0..reps {
            let mut cfg = sign_cfg(1e-3, false);
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
