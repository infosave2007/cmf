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
//!   stay zero (masked), so the bit budget never grows.

use crate::gpu_wgpu::gdn_train::{self as gdn, GdnTape, GdnTrainCfg};
use crate::gpu_wgpu::train_head::{self as th, Head, HeadScratch};
use crate::gpu_wgpu::train_mm::{self as tm, Blr, BlrPath, BlrScratch, Op};
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
fn load_blr(c: Ctx, bytes: &[u8], role: &'static str) -> Result<Lin, String> {
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
        });
        params.push([ph, pl, pg]);
    }
    Ok(Lin::Blr { blr: Blr { m, n, r: rp, paths }, role, params })
}

fn load_lin(c: Ctx, m: &CmfModel, name: &str, role: &'static str, as_plane: bool, bwd: bool) -> Result<Lin, String> {
    let e = m.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if e.dtype as u8 == 19 {
        return load_blr(c, m.entry_bytes(e), role);
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
    pub fn load(c: Ctx, model: &CmfModel, plane: bool, bwd: bool) -> Result<Self, String> {
        let geom = Geom::from_model(model)?;
        let norm = |name: &str| -> Result<Param, String> { Ok(Param::new(c, &deq(model, name)?.0, None)) };
        let mut layers = Vec::with_capacity(geom.nl);
        for li in 0..geom.nl {
            let p = format!("model.layers.{li}.");
            let lin = |n: &str, role: &'static str| load_lin(c, model, &format!("{p}{n}"), role, plane, bwd);
            let mix = if geom.gdn_layer[li] {
                Mix::Gdn {
                    qkv: lin("linear_attn.in_proj_qkv.weight", "gdn.qkv")?,
                    z: lin("linear_attn.in_proj_z.weight", "gdn.z")?,
                    a: load_lin(c, model, &format!("{p}linear_attn.in_proj_a.weight"), "gdn.a", false, bwd)?,
                    b: load_lin(c, model, &format!("{p}linear_attn.in_proj_b.weight"), "gdn.b", false, bwd)?,
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
        Ok(Net { geom, layers, final_norm, head, embed })
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
        for l in &net.layers {
            let mut add = |lin: &Lin| {
                if let Lin::Blr { blr: bl, role, .. } = lin {
                    act_w = act_w.max(bl.m).max(bl.n).max(bl.r);
                    let d = dims.entry(*role).or_insert((0, 0, 0, 0));
                    *d = (d.0.max(bl.m), d.1.max(bl.n), d.2.max(bl.r), d.3.max(bl.paths.len()));
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
#[allow(clippy::too_many_arguments)]
fn lin_bwd(c: Ctx, w: &Work, lin: &Lin, x: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, acc: bool, gscale: f32) -> Option<()> {
    let ntok = w.ntok;
    match lin {
        Lin::Blr { blr, role, .. } => {
            if acc {
                let tmp = w.get("tmp_h");
                // tmp_h is hs wide: accumulation is only used for hs-wide inputs
                tm::blr_backward(c, blr, &w.blr[role], x, dy, tmp, ntok, gscale)?;
                tm::run(c, &[Op::D(to::elt(c, "add", ntok * blr.n, dx, tmp, Some(dx), None, None, None)?)]);
                Some(())
            } else {
                tm::blr_backward(c, blr, &w.blr[role], x, dy, dx, ntok, gscale)
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
fn layer_bwd(c: Ctx, net: &Net, w: &Work, l: usize, x: &wgpu::Buffer, dy: &wgpu::Buffer, dx: &wgpu::Buffer, gs: f32) -> Option<()> {
    let g = &net.geom;
    let ly = &net.layers[l];
    let n = w.ntok;
    // MLP: y = x2 + down(swiglu(gate(n2), up(n2)))
    lin_bwd(c, w, &ly.down, w.get("hh"), dy, w.get("dhh"), false, gs)?;
    to::run(c, &[to::elt(c, "swiglu_bwd", n * g.inter, w.get("gg"), w.get("uu"), None, Some(w.get("dhh")), Some(w.get("dgg")), Some(w.get("duu")))?]);
    lin_bwd(c, w, &ly.gate, w.get("n2"), w.get("dgg"), w.get("dn2"), false, gs)?;
    lin_bwd(c, w, &ly.up, w.get("n2"), w.get("duu"), w.get("dn2"), true, gs)?;
    to::run(c, &[
        to::rms_bwd(c, w.get("x2"), &ly.ln2.w, w.get("inv2"), w.get("dn2"), w.get("dx2"), n, g.hs, g.gemma)?,
        to::rms_dw(c, w.get("x2"), w.get("inv2"), w.get("dn2"), &ly.ln2.g, n, g.hs)?,
        to::elt(c, "add", n * g.hs, w.get("dx2"), dy, Some(w.get("dx2")), None, None, None)?,
    ]);
    // mixer: x2 = x + mix(n1)
    match &ly.mix {
        Mix::Gdn { qkv, z, a, b, out, .. } => {
            let tape = w.tape.as_ref().unwrap();
            lin_bwd(c, w, out, w.get("gout"), w.get("dx2"), tape.dout(), false, gs)?;
            gdn::gdn_backward_dev(tape, g.nk, g.nv, g.c_dim())?;
            lin_bwd(c, w, qkv, w.get("n1"), tape.draw(), w.get("dn1"), false, gs)?;
            lin_bwd(c, w, z, w.get("n1"), tape.dz(), w.get("dn1"), true, gs)?;
            copy_off(c, tape.dab(), 0, w.get("dgout"), 0, n * g.nv);
            lin_bwd(c, w, a, w.get("n1"), w.get("dgout"), w.get("dn1"), true, gs)?;
            copy_off(c, tape.dab(), n * g.nv, w.get("dgout"), 0, n * g.nv);
            lin_bwd(c, w, b, w.get("n1"), w.get("dgout"), w.get("dn1"), true, gs)?;
        }
        Mix::Attn { q, k, v, o, qn, kn } => {
            let qd = g.nh * g.hd;
            let qg = QkrGeom { n, heads: g.nh, hd: g.hd, half: g.half, t: w.t, gemma: g.gemma, eps: g.eps };
            let kg = QkrGeom { heads: g.nkv, ..qg };
            let ag = AttnGeom { nb: w.nb, t: w.t, nh: g.nh, nkv: g.nkv, hd: g.hd };
            lin_bwd(c, w, o, w.get("og"), w.get("dx2"), w.get("dog"), false, gs)?;
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
            lin_bwd(c, w, q, w.get("n1"), w.get("dqg"), w.get("dn1"), false, gs)?;
            lin_bwd(c, w, k, w.get("n1"), w.get("dk"), w.get("dn1"), true, gs)?;
            lin_bwd(c, w, v, w.get("n1"), w.get("dv"), w.get("dn1"), true, gs)?;
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
}

/// One forward + backward on `ids` against the teacher top-k; gradients
/// accumulate into the parameters' `g` buffers (zeroed by the caller).
pub fn step(c: Ctx, net: &Net, w: &Work, ids: &[u32], tid: &[u32], tp: &[f32], k: usize, gscale: f32) -> Option<StepStats> {
    let g = &net.geom;
    let n = w.ntok;
    forward(c, net, w, ids)?;
    let btid = to::zeros(c, n * k);
    c.queue.write_buffer(&btid, 0, bytemuck::cast_slice(tid));
    let btp = to::upload(c, tp);
    th::head_kl(c, &net.head, &w.head, w.get("n1"), &btid, &btp, k, w.get("dn1"), w.get("loss"), n, n, gscale)?;
    let loss: f64 = to::download(c, w.get("loss"), n)?.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
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
        layer_bwd(c, net, w, l, &w.ckpt[l], cur, nxt, gscale)?;
        std::mem::swap(&mut cur, &mut nxt);
    }
    Some(StepStats { loss, finite: loss.is_finite() })
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

/// Host copy of every parameter's values (best-checkpoint keeping).
pub fn snapshot(c: Ctx, net: &Net) -> Option<Vec<Vec<f32>>> {
    net.params().iter().map(|p| to::download(c, &p.w, p.n)).collect()
}

/// Restore a [`snapshot`].
pub fn restore(c: Ctx, net: &Net, snap: &[Vec<f32>]) {
    for (p, v) in net.params().iter().zip(snap) {
        c.queue.write_buffer(&p.w, 0, bytemuck::cast_slice(v));
    }
}

/// Copy the trained metric of every binary low-rank tensor and every norm
/// gain back into payloads of `model` (same layout and sizes): returns
/// (tensor name, new bytes) for every changed entry.
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
    let mut blr = |name: String, lin: &Lin| -> Result<(), String> {
        let Lin::Blr { blr, params, .. } = lin else { return Ok(()) };
        let e = model.tensor(&name).ok_or_else(|| format!("missing {name}"))?;
        let mut bytes = model.entry_bytes(e).to_vec();
        let r = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let (m, n) = (blr.m, blr.n);
        let mut off = 16;
        for pr in params {
            let h = to::download(c, &pr[0].w, m).ok_or("download")?;
            let l = to::download(c, &pr[1].w, pr[1].n).ok_or("download")?;
            let g = to::download(c, &pr[2].w, n).ok_or("download")?;
            bytes[off..off + 2 * m].copy_from_slice(&to_bytes_f16(&h));
            off += 2 * m;
            bytes[off..off + 2 * n].copy_from_slice(&to_bytes_f16(&g));
            off += 2 * n;
            bytes[off..off + 2 * r].copy_from_slice(&to_bytes_f16(&l[..r]));
            off += 2 * r + m * r / 8 + r * n / 8;
        }
        out.push((name, bytes));
        Ok(())
    };
    for (li, ly) in net.layers.iter().enumerate() {
        let p = format!("model.layers.{li}.");
        match &ly.mix {
            Mix::Gdn { qkv, z, out: o, .. } => {
                blr(format!("{p}linear_attn.in_proj_qkv.weight"), qkv)?;
                blr(format!("{p}linear_attn.in_proj_z.weight"), z)?;
                blr(format!("{p}linear_attn.out_proj.weight"), o)?;
            }
            Mix::Attn { q, k, v, o, .. } => {
                blr(format!("{p}self_attn.q_proj.weight"), q)?;
                blr(format!("{p}self_attn.k_proj.weight"), k)?;
                blr(format!("{p}self_attn.v_proj.weight"), v)?;
                blr(format!("{p}self_attn.o_proj.weight"), o)?;
            }
        }
        blr(format!("{p}mlp.gate_proj.weight"), &ly.gate)?;
        blr(format!("{p}mlp.up_proj.weight"), &ly.up)?;
        blr(format!("{p}mlp.down_proj.weight"), &ly.down)?;
    }
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
        let teacher = Net::load(c, &model, true, false).expect("teacher");
        let student = Net::load(c, &model, true, true).expect("student");
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
        let st = step(c, &student, &w, &ids, &tid, &tp, k, 1024.0).expect("step");
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
}
