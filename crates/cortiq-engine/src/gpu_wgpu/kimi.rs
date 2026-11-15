//! Kimi-Linear's two token mixers on the wgpu graphs (`GraphAttn::Kda` and
//! `GraphAttn::Mla`), the kernels the token graph (one row) and the batch
//! graph (a prompt chunk) share. The CUDA hybrid arm (`gpu_cuda/hyb.cu`) has
//! the same steps; `linear_core::kda_step` and `pipeline::mla_attention` the
//! semantics.
//!
//! KDA, after the graph projected q / k / v (its own kernels), the low-rank
//! decay `f = f_b · (f_a · x)`, `b · x` and the output gate:
//!
//! * `kda_front`: the depthwise causal conv of every q | k | v channel over
//!   the ring and the rows before it, SiLU, the L2 norm of each q and k head
//!   (q also over sqrt(dk)), a workgroup per (head-wide unit, row);
//! * `kda_gates`: the decay of every key channel, `exp(log-decay(f + bias))`,
//!   and beta = sigmoid(b);
//! * `kda_scan`: the delta rule with a decay per key channel, the state in
//!   registers (a lane a state column, four 32-row quarters a workgroup), 16
//!   rows at a time through workgroup tiles — `hy_kda_scan4`'s layout;
//! * `kda_ring`: the chunk's last kk - 1 raw rows into the ring (after the
//!   front read the old one);
//! * `kda_norm`: the per-head RMS norm, times the norm weight and
//!   sigmoid(gate), in place over the scan's output.
//!
//! MLA runs as multi-head attention with heads `rope + nope` wide (the
//! graphs' per-layer-geometry attention, `GraphAttnGeom`): `mla_lat` is the
//! latent's RMS norm (kv_b's input), `mla_kv` the heads' keys
//! `[k_rope | k_nope]` and `v`-wide values from the latent's rope key and
//! kv_b's rows.

use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

include!("kimi_src.rs");

struct Pipe {
    p: wgpu::ComputePipeline,
    l: wgpu::BindGroupLayout,
}

struct Pipes {
    front: Pipe,
    lr: Pipe,
    scan: Pipe,
    norm: Pipe,
    lat: Pipe,
    kv: Pipe,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            if c.device.limits().max_compute_workgroup_storage_size < 26 * 1024 {
                tracing::warn!("Kimi-Linear kernels: workgroup storage under 26 KiB");
                return None;
            }
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("kimi"),
                source: wgpu::ShaderSource::Wgsl(src().into()),
            });
            let mk = |entry: &str| {
                let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                });
                let l = p.get_bind_group_layout(0);
                Pipe { p, l }
            };
            let out = Pipes {
                front: mk("kda_front"),
                lr: mk("kda_lr_out"),
                scan: mk("kda_scan"),
                norm: mk("kda_norm"),
                lat: mk("mla_lat"),
                kv: mk("mla_kv"),
            };
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("Kimi-Linear kernels rejected: {e}");
                return None;
            }
            Some(out)
        })
        .as_ref()
}

/// Did the kernels build on this device (the graphs decline Kimi otherwise).
pub(crate) fn available(c: &Ctx) -> bool {
    pipes(c).is_some()
}

/// Scratch buffers by name, grown when a call needs more (a token graph's
/// one row, a prompt chunk's rows).
static POOL: Mutex<Option<HashMap<&'static str, (wgpu::Buffer, usize)>>> = Mutex::new(None);

/// A scratch buffer of at least `words` f32 under `name`.
pub(crate) fn scratch(c: &Ctx, name: &'static str, words: usize) -> wgpu::Buffer {
    let mut g = POOL.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    let words = words.max(64);
    if let Some((b, n)) = m.get(name) {
        if *n >= words {
            return b.clone();
        }
    }
    let n = words.next_power_of_two();
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    m.insert(name, (b.clone(), n));
    b
}

/// Uniforms by content: a buffer per distinct parameter set, never
/// rewritten, so two uses in one submit cannot see each other's values.
static UNIFORMS: Mutex<Option<HashMap<[u32; 16], wgpu::Buffer>>> = Mutex::new(None);

fn uniform16(c: &Ctx, v: [u32; 16]) -> wgpu::Buffer {
    let mut g = UNIFORMS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(b) = m.get(&v) {
        return b.clone();
    }
    let b = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("kimi-u"),
        contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    m.insert(v, b.clone());
    b
}

/// A KDA layer's token-invariant constants on the device.
pub(crate) struct KdaConsts {
    /// `[q | k | v channels][kk]`.
    taps: wgpu::Buffer,
    alog: wgpu::Buffer,
    a_len: usize,
    dtb: wgpu::Buffer,
    onorm: wgpu::Buffer,
}

/// Constants by the address and a sample of the layer's conv taps (stable
/// for a loaded model; the sample catches another model mapped at the same
/// address).
static CONSTS: Mutex<Option<HashMap<(usize, usize), (u64, std::sync::Arc<KdaConsts>)>>> = Mutex::new(None);

fn sample_fp(parts: &[&[f32]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        h = (h ^ p.len() as u64).wrapping_mul(0x100_0000_01b3);
        let n = p.len();
        for i in [0, n / 3, n / 2, (2 * n) / 3, n.saturating_sub(1)] {
            if let Some(v) = p.get(i) {
                h = (h ^ v.to_bits() as u64).wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    h
}

fn stor_f32(c: &Ctx, label: &str, data: &[f32]) -> wgpu::Buffer {
    let b = c.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (data.len().max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
    b
}

/// The host-side constants of one KDA layer.
pub(crate) struct KdaHost<'a> {
    pub conv_q: &'a [f32],
    pub conv_k: &'a [f32],
    pub conv_v: &'a [f32],
    pub dt_bias: &'a [f32],
    pub a_log: &'a [f32],
    pub o_norm: &'a [f32],
}

pub(crate) fn kda_consts(c: &Ctx, h: &KdaHost<'_>) -> std::sync::Arc<KdaConsts> {
    let key = (h.conv_q.as_ptr() as usize, h.conv_q.len());
    let fp = sample_fp(&[h.conv_q, h.conv_k, h.conv_v, h.dt_bias, h.a_log, h.o_norm]);
    let mut g = CONSTS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some((f, k)) = m.get(&key) {
        if *f == fp {
            return k.clone();
        }
    }
    let taps: Vec<f32> = h.conv_q.iter().chain(h.conv_k).chain(h.conv_v).copied().collect();
    let k = std::sync::Arc::new(KdaConsts {
        taps: stor_f32(c, "kda-taps", &taps),
        alog: stor_f32(c, "kda-alog", h.a_log),
        a_len: h.a_log.len(),
        dtb: stor_f32(c, "kda-dtb", h.dt_bias),
        onorm: stor_f32(c, "kda-onorm", h.o_norm),
    });
    m.insert(key, (fp, k.clone()));
    k
}

/// One KDA layer's geometry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct KdaDims {
    pub nh: usize,
    pub dk: usize,
    pub dv: usize,
    pub kk: usize,
}

impl KdaDims {
    pub(crate) fn kd(&self) -> usize {
        self.nh * self.dk
    }
    pub(crate) fn vd(&self) -> usize {
        self.nh * self.dv
    }
    pub(crate) fn cdim(&self) -> usize {
        2 * self.kd() + self.vd()
    }
    /// Device state: the conv ring `[kk - 1][q | k | v]` and S `[nh][dk][dv]`.
    pub(crate) fn ring_words(&self) -> usize {
        self.kk.saturating_sub(1) * self.cdim()
    }
    pub(crate) fn s_words(&self) -> usize {
        self.nh * self.dk * self.dv
    }
    /// The kernels' contract: 128-wide heads, a conv of at most 9 taps.
    pub(crate) fn admitted(&self) -> bool {
        self.dk == KDA_DK && self.dv == KDA_DK && self.nh > 0 && self.kk >= 1 && self.kk <= 9
    }
}

/// The host's KDA state `[q ring | k ring | v ring | S]` (each ring
/// `[kk - 1][its channels]`) in the device layout `[ring [kk - 1][q | k |
/// v] | S]`.
pub(crate) fn device_seed<'h>(d: &KdaDims, host: &'h [f32]) -> Option<(Vec<f32>, &'h [f32])> {
    let (kd, vd, rows) = (d.kd(), d.vd(), d.kk.saturating_sub(1));
    let ring_n = d.ring_words();
    if host.len() != ring_n + d.s_words() {
        return None;
    }
    let cdim = d.cdim();
    let mut r = vec![0f32; ring_n];
    for j in 0..rows {
        let dst = &mut r[j * cdim..(j + 1) * cdim];
        dst[..kd].copy_from_slice(&host[j * kd..(j + 1) * kd]);
        dst[kd..2 * kd].copy_from_slice(&host[rows * kd + j * kd..rows * kd + (j + 1) * kd]);
        dst[2 * kd..].copy_from_slice(&host[2 * rows * kd + j * vd..2 * rows * kd + (j + 1) * vd]);
    }
    Some((r, &host[ring_n..]))
}

/// The buffers one KDA step reads and writes.
pub(crate) struct KdaIo<'b> {
    /// Raw projections `[rows][kd]`, `[rows][kd]`, `[rows][vd]`.
    pub xq: &'b wgpu::Buffer,
    pub xk: &'b wgpu::Buffer,
    pub xv: &'b wgpu::Buffer,
    /// The low-rank stages' first halves: `f_a · x` `[rows][rank_f]`,
    /// `g_a · x` `[rows][rank_g]` (unread when the gate is full rank).
    pub fl: &'b wgpu::Buffer,
    pub gl: &'b wgpu::Buffer,
    /// `b · x`, `[rows][nh]`.
    pub braw: &'b wgpu::Buffer,
    /// The output gate's pre-activation, `[rows][vd]`: written here from
    /// `gl` (low rank) or by the caller's full-rank `g_a`.
    pub gate: &'b wgpu::Buffer,
    pub ring: &'b wgpu::Buffer,
    pub s: &'b wgpu::Buffer,
    /// The gated per-head output `[rows][vd]`, the output projection's input.
    pub o: &'b wgpu::Buffer,
}

/// The low-rank second stages: f32 `f_b` `[kd][rank_f]`, `g_b` `[vd][rank_g]`.
pub(crate) struct KdaLowRank<'b> {
    pub fb: &'b wgpu::Buffer,
    pub gb: Option<&'b wgpu::Buffer>,
    pub rank_f: usize,
    pub rank_g: usize,
}

/// Bind groups of a layer while its buffers stay the same.
struct KdaBgs {
    bufs: Vec<wgpu::Buffer>,
    front: wgpu::BindGroup,
    lr: wgpu::BindGroup,
    scan: wgpu::BindGroup,
    norm: wgpu::BindGroup,
}

static KDA_BGS: Mutex<Option<HashMap<(u64, usize, usize), KdaBgs>>> = Mutex::new(None);

/// Encode one KDA layer's step over `rows` rows after its first-stage
/// projections: conv and norms, the low-rank second stages with the gates,
/// the recurrence, the gated norm into `io.o` and the ring. `key` = (kv id,
/// absolute layer, a call-site tag) for the bind groups.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_kda(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    key: (u64, usize, usize),
    d: &KdaDims,
    consts: &KdaConsts,
    lr: &KdaLowRank<'_>,
    lower: Option<f32>,
    eps: f32,
    rows: usize,
    io: &KdaIo<'_>,
) -> bool {
    let Some(p) = pipes(c) else { return false };
    if rows == 0 || !d.admitted() || rows > 65535 || lr.rank_f == 0 {
        return false;
    }
    let (kd, vd, cdim, nh) = (d.kd(), d.vd(), d.cdim(), d.nh);
    let (lb_on, lb) = match lower {
        Some(v) => (1u32, v),
        None => (0u32, 0.0f32),
    };
    let u = uniform16(
        c,
        [
            rows as u32,
            nh as u32,
            d.dk as u32,
            d.dv as u32,
            d.kk as u32,
            cdim as u32,
            kd as u32,
            vd as u32,
            consts.a_len as u32,
            lb_on,
            lb.to_bits(),
            eps.to_bits(),
            lr.rank_f as u32,
            lr.rank_g as u32,
            u32::from(lr.gb.is_some()),
            0,
        ],
    );
    let cq = scratch(c, "kda-cq", rows * cdim);
    let gd = scratch(c, "kda-gd", rows * kd);
    let beta = scratch(c, "kda-beta", rows * nh);
    // full rank: the lr kernel's gate rows are never dispatched, its g_b
    // slots take stand-ins
    let (gbw, gl) = match lr.gb {
        Some(g) => (g, io.gl),
        None => (lr.fb, io.fl),
    };
    let bufs = vec![
        u.clone(),
        io.xq.clone(),
        io.xk.clone(),
        io.xv.clone(),
        io.fl.clone(),
        gl.clone(),
        io.braw.clone(),
        io.gate.clone(),
        io.ring.clone(),
        io.s.clone(),
        io.o.clone(),
        cq.clone(),
        gd.clone(),
        beta.clone(),
        lr.fb.clone(),
        gbw.clone(),
        consts.taps.clone(),
        consts.alog.clone(),
        consts.dtb.clone(),
        consts.onorm.clone(),
    ];
    let mut g = KDA_BGS.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    let fresh = m.get(&key).is_none_or(|b| b.bufs != bufs);
    if fresh {
        let bp = |l: &wgpu::BindGroupLayout, pairs: &[(u32, &wgpu::Buffer)]| bind_pairs(c, l, pairs);
        let front = bp(
            &p.front.l,
            &[(0, &u), (1, io.xq), (2, io.xk), (3, io.xv), (4, &consts.taps), (5, io.ring), (6, &cq)],
        );
        let lrb = bp(
            &p.lr.l,
            &[
                (0, &u),
                (8, io.braw),
                (9, &consts.alog),
                (10, &consts.dtb),
                (11, &gd),
                (12, &beta),
                (27, lr.fb),
                (28, gbw),
                (29, io.fl),
                (30, gl),
                (31, io.gate),
            ],
        );
        let scan = bp(&p.scan.l, &[(0, &u), (6, &cq), (11, &gd), (12, &beta), (13, io.s), (14, io.o)]);
        let norm = bp(
            &p.norm.l,
            &[(0, &u), (1, io.xq), (2, io.xk), (3, io.xv), (5, io.ring), (14, io.o), (15, io.gate), (16, &consts.onorm)],
        );
        m.insert(key, KdaBgs { bufs, front, lr: lrb, scan, norm });
    }
    let b = m.get(&key).unwrap();
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.front.p);
    pass.set_bind_group(0, &b.front, &[]);
    pass.dispatch_workgroups((cdim / KDA_DK) as u32, rows as u32, 1);
    pass.set_pipeline(&p.lr.p);
    pass.set_bind_group(0, &b.lr, &[]);
    pass.dispatch_workgroups((kd + lr.gb.map_or(0, |_| vd)) as u32, rows as u32, 1);
    pass.set_pipeline(&p.scan.p);
    pass.set_bind_group(0, &b.scan, &[]);
    pass.dispatch_workgroups((nh * d.dv / 32) as u32, 1, 1);
    // the gated norm, and the ring after the front read it
    pass.set_pipeline(&p.norm.p);
    pass.set_bind_group(0, &b.norm, &[]);
    pass.dispatch_workgroups(nh as u32, rows as u32, 1);
    true
}

/// One MLA layer's widths.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct MlaDims {
    pub nh: usize,
    pub rope: usize,
    pub nope: usize,
    pub vd: usize,
    pub lora: usize,
}

impl MlaDims {
    pub(crate) fn hd(&self) -> usize {
        self.rope + self.nope
    }
    /// kv_a's rows: the latent, then the rope key.
    pub(crate) fn kva_rows(&self) -> usize {
        self.lora + self.rope
    }
    /// kv_b's rows: per head `[k_nope | v]`.
    pub(crate) fn kvb_rows(&self) -> usize {
        self.nh * (self.nope + self.vd)
    }
}

/// The latent's norm of `rows` kv_a rows into `lat` `[rows][lora]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mla_lat(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    key: Option<(u64, usize, u32)>,
    d: &MlaDims,
    eps: f32,
    rows: usize,
    kva: &wgpu::Buffer,
    norm: &wgpu::Buffer,
    lat: &wgpu::Buffer,
) -> bool {
    let Some(p) = pipes(c) else { return false };
    if rows == 0 || rows > 65535 {
        return false;
    }
    let u = mla_uniform(c, d, eps, rows);
    let bgl = bind_cached(c, key, &p.lat.l, &[(20, &u), (21, kva), (22, norm), (23, lat)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.lat.p);
    pass.set_bind_group(0, &bgl, &[]);
    pass.dispatch_workgroups(rows as u32, 1, 1);
    true
}

/// The heads' keys `[rows][nh][hd]` and values `[rows][nh][vd]` from kv_a's
/// rows (their rope key) and kv_b's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mla_kv(
    c: &Ctx,
    enc: &mut wgpu::CommandEncoder,
    key: Option<(u64, usize, u32)>,
    d: &MlaDims,
    rows: usize,
    kva: &wgpu::Buffer,
    kvb: &wgpu::Buffer,
    k: &wgpu::Buffer,
    v: &wgpu::Buffer,
) -> bool {
    let Some(p) = pipes(c) else { return false };
    if rows == 0 || rows > 65535 {
        return false;
    }
    let u = mla_uniform(c, d, 0.0, rows);
    let bgk = bind_cached(c, key, &p.kv.l, &[(20, &u), (21, kva), (24, kvb), (25, k), (26, v)]);
    let mut pass = begin_pass(enc);
    pass.set_pipeline(&p.kv.p);
    pass.set_bind_group(0, &bgk, &[]);
    pass.dispatch_workgroups((d.nh * d.hd()).div_ceil(256) as u32, rows as u32, 1);
    true
}

fn mla_uniform(c: &Ctx, d: &MlaDims, eps: f32, rows: usize) -> wgpu::Buffer {
    uniform16(
        c,
        [
            rows as u32,
            d.nh as u32,
            d.rope as u32,
            d.nope as u32,
            d.vd as u32,
            d.lora as u32,
            d.kva_rows() as u32,
            eps.to_bits(),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0x4d4c_4100,
        ],
    )
}

/// Bind groups by (kv id, absolute layer, site), valid while the buffers
/// they were made over are the same: the token graph builds a layer's
/// matvec and kernel bind groups once, not every token (a bind group costs
/// microseconds of host time, and Kimi-Linear's 27 layers wanted ~250).
pub(crate) struct CachedBind {
    bufs: Vec<wgpu::Buffer>,
    pub(crate) p: Option<wgpu::ComputePipeline>,
    pub(crate) bg: wgpu::BindGroup,
    pub(crate) wg: u32,
}

static BINDS: Mutex<Option<HashMap<(u64, usize, u32), CachedBind>>> = Mutex::new(None);

/// The cached (pipeline, bind group, workgroups) of `key` over `bufs`.
pub(crate) fn bind_get(key: (u64, usize, u32), bufs: &[&wgpu::Buffer]) -> Option<(Option<wgpu::ComputePipeline>, wgpu::BindGroup, u32)> {
    let g = BINDS.lock().unwrap();
    let e = g.as_ref()?.get(&key)?;
    (e.bufs.len() == bufs.len() && e.bufs.iter().zip(bufs).all(|(a, b)| a == *b)).then(|| (e.p.clone(), e.bg.clone(), e.wg))
}

pub(crate) fn bind_put(key: (u64, usize, u32), bufs: &[&wgpu::Buffer], p: Option<wgpu::ComputePipeline>, bg: wgpu::BindGroup, wg: u32) {
    let mut g = BINDS.lock().unwrap();
    g.get_or_insert_with(HashMap::new).insert(key, CachedBind { bufs: bufs.iter().map(|b| (*b).clone()).collect(), p, bg, wg });
}

/// `bind_pairs` through the cache: `key` names the site, `pairs` the buffers.
pub(crate) fn bind_cached(c: &Ctx, key: Option<(u64, usize, u32)>, l: &wgpu::BindGroupLayout, pairs: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let Some(key) = key else { return bind_pairs(c, l, pairs) };
    let bufs: Vec<&wgpu::Buffer> = pairs.iter().map(|(_, b)| *b).collect();
    if let Some((_, bg, _)) = bind_get(key, &bufs) {
        return bg;
    }
    let bg = bind_pairs(c, l, pairs);
    bind_put(key, &bufs, None, bg.clone(), 0);
    bg
}

/// Drop the cached bind groups of a sequence (its mirrors went away).
pub(crate) fn binds_forget(kv_id: u64) {
    if let Some(m) = BINDS.lock().unwrap().as_mut() {
        m.retain(|k, _| k.0 != kv_id);
    }
    if let Some(m) = KDA_BGS.lock().unwrap().as_mut() {
        m.retain(|k, _| k.0 != kv_id);
    }
}

/// `CMF_GRAPH_HOSTPROF=1`: host encode time of the token graph's mixers.
pub(crate) fn hostprof_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_GRAPH_HOSTPROF").as_deref() == Ok("1"))
}

/// One layer part's encode time: kind 0 an attention mixer (MLA
/// included), 1 a KDA mixer, 2 the rest of the layer (norms, FFN), 3 the
/// token graph's setup before its layer loop, 4 a mid-stack piece submit.
pub(crate) fn hostprof_add(kind: usize, dt: std::time::Duration) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    static N: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    let k = kind.min(4);
    NS[k].fetch_add(dt.as_nanos() as u64, Ordering::Relaxed);
    let n = N[k].fetch_add(1, Ordering::Relaxed) + 1;
    if n % [700, 2000, 2700, 100, 600][k] == 0 {
        eprintln!(
            "hostprof: {} encode {:.1} us a layer over {n} layers",
            ["attention mixer", "KDA mixer", "norms + FFN", "setup before the layers (a token)", "piece submit"][k],
            NS[k].load(Ordering::Relaxed) as f64 / n as f64 / 1e3
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((*seed >> 33) as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn upload(c: &Ctx, data: &[f32]) -> wgpu::Buffer {
        let b = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kimi-test"),
            size: (data.len().max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        c.queue.write_buffer(&b, 0, bytemuck::cast_slice(data));
        b
    }

    fn download(c: &Ctx, b: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let st = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kimi-test-rb"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(b, 0, &st, 0, (n * 4) as u64);
        c.queue.submit([enc.finish()]);
        let sl = st.slice(..);
        sl.map_async(wgpu::MapMode::Read, |_| {});
        c.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let v = bytemuck::cast_slice::<u8, f32>(&sl.get_mapped_range().unwrap()).to_vec();
        st.unmap();
        v
    }

    fn sig(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// The KDA kernels over a few rows against an f64 transcription of
    /// `linear_core::kda_step` (conv per projection ring, the delta rule
    /// with a decay per key channel, the sigmoid-gated norm), ring and S too.
    #[test]
    fn kda_kernels_match_host() {
        let Some(c) = ctx().or_else(trainer_ctx) else {
            eprintln!("kda_kernels_match_host: no wgpu device, skipped");
            return;
        };
        if pipes(c).is_none() {
            eprintln!("kda_kernels_match_host: kernels unavailable, skipped");
            return;
        }
        let d = KdaDims { nh: 2, dk: 128, dv: 128, kk: 4 };
        let (kd, vd, cdim, nh, kk) = (d.kd(), d.vd(), d.cdim(), d.nh, d.kk);
        let rows = 19; // two scan tiles, one partial
        let mut sd = 7u64;
        let xq = rnd(&mut sd, rows * kd, 1.0);
        let xk = rnd(&mut sd, rows * kd, 1.0);
        let xv = rnd(&mut sd, rows * vd, 1.0);
        // the low-rank stages: f = f_b · fl, gate = g_b · gl
        let rank = 16;
        let fl = rnd(&mut sd, rows * rank, 1.0);
        let fbw = rnd(&mut sd, kd * rank, 0.5);
        let gl = rnd(&mut sd, rows * rank, 1.0);
        let gbw = rnd(&mut sd, vd * rank, 0.5);
        let lowrank = |w: &[f32], x: &[f32], out_rows: usize| -> Vec<f32> {
            let mut y = vec![0f32; rows * out_rows];
            for t in 0..rows {
                for r in 0..out_rows {
                    y[t * out_rows + r] = (0..rank).map(|i| w[r * rank + i] as f64 * x[t * rank + i] as f64).sum::<f64>() as f32;
                }
            }
            y
        };
        let f = lowrank(&fbw, &fl, kd);
        let gate = lowrank(&gbw, &gl, vd);
        let braw = rnd(&mut sd, rows * nh, 2.0);
        let conv_q = rnd(&mut sd, kd * kk, 0.5);
        let conv_k = rnd(&mut sd, kd * kk, 0.5);
        let conv_v = rnd(&mut sd, vd * kk, 0.5);
        let dt_bias = rnd(&mut sd, kd, 1.0);
        let a_log = rnd(&mut sd, nh, 1.0);
        let o_norm: Vec<f32> = rnd(&mut sd, d.dv, 0.5).iter().map(|v| 1.0 + v).collect();
        // host state [q ring | k ring | v ring | S], nonzero
        let host0 = rnd(&mut sd, d.ring_words() + d.s_words(), 0.3);
        let eps = 1e-6f32;

        // reference
        let mut ring_q: Vec<f64> = host0[..(kk - 1) * kd].iter().map(|&v| v as f64).collect();
        let mut ring_k: Vec<f64> = host0[(kk - 1) * kd..2 * (kk - 1) * kd].iter().map(|&v| v as f64).collect();
        let mut ring_v: Vec<f64> = host0[2 * (kk - 1) * kd..d.ring_words()].iter().map(|&v| v as f64).collect();
        let mut sref: Vec<f64> = host0[d.ring_words()..].iter().map(|&v| v as f64).collect();
        let conv = |raw: &[f32], taps: &[f32], ring: &mut Vec<f64>, cn: usize| -> Vec<f64> {
            let mut out = vec![0f64; cn];
            for c in 0..cn {
                let t = &taps[c * kk..(c + 1) * kk];
                let mut acc = raw[c] as f64 * t[kk - 1] as f64;
                for j in 0..kk - 1 {
                    acc += ring[j * cn + c] * t[j] as f64;
                }
                out[c] = acc * sig(acc);
            }
            ring.copy_within(cn.., 0);
            let tail = (kk - 2) * cn;
            for c in 0..cn {
                ring[tail + c] = raw[c] as f64;
            }
            out
        };
        let mut oref = vec![0f64; rows * vd];
        for t in 0..rows {
            let cq = conv(&xq[t * kd..(t + 1) * kd], &conv_q, &mut ring_q, kd);
            let ck = conv(&xk[t * kd..(t + 1) * kd], &conv_k, &mut ring_k, kd);
            let cv = conv(&xv[t * vd..(t + 1) * vd], &conv_v, &mut ring_v, vd);
            for h in 0..nh {
                let qs = h * 128;
                let nq: f64 = (0..128).map(|i| cq[qs + i] * cq[qs + i]).sum();
                let nk: f64 = (0..128).map(|i| ck[qs + i] * ck[qs + i]).sum();
                let invq = 1.0 / ((nq + 1e-6).sqrt() * (128f64).sqrt());
                let invk = 1.0 / (nk + 1e-6).sqrt();
                let gd: Vec<f64> = (0..128)
                    .map(|i| {
                        let x = f[t * kd + qs + i] as f64 + dt_bias[qs + i] as f64;
                        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
                        (-(a_log[h] as f64).exp() * sp).exp()
                    })
                    .collect();
                let beta = sig(braw[t * nh + h] as f64);
                let s = &mut sref[h * 128 * 128..(h + 1) * 128 * 128];
                let mut kv = vec![0f64; 128];
                for i in 0..128 {
                    let kg = ck[qs + i] * invk * gd[i];
                    for j in 0..128 {
                        kv[j] += s[i * 128 + j] * kg;
                    }
                }
                let mut o = vec![0f64; 128];
                for i in 0..128 {
                    for j in 0..128 {
                        let dl = (cv[h * 128 + j] - kv[j]) * beta;
                        let cell = gd[i] * s[i * 128 + j] + ck[qs + i] * invk * dl;
                        s[i * 128 + j] = cell;
                        o[j] += cq[qs + i] * invq * cell;
                    }
                }
                let ss: f64 = o.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss / 128.0 + eps as f64).sqrt();
                for j in 0..128 {
                    oref[t * vd + h * 128 + j] = o[j] * inv * o_norm[j] as f64 * sig(gate[t * vd + h * 128 + j] as f64);
                }
            }
        }

        // device
        let host = KdaHost { conv_q: &conv_q, conv_k: &conv_k, conv_v: &conv_v, dt_bias: &dt_bias, a_log: &a_log, o_norm: &o_norm };
        let consts = kda_consts(c, &host);
        let (r0, s0) = device_seed(&d, &host0).unwrap();
        let (bq, bk, bv, bb) = (upload(c, &xq), upload(c, &xk), upload(c, &xv), upload(c, &braw));
        let (bfl, bgl, bfw, bgw) = (upload(c, &fl), upload(c, &gl), upload(c, &fbw), upload(c, &gbw));
        let bg = upload(c, &vec![0f32; rows * vd]);
        let (ring, s) = (upload(c, &r0), upload(c, s0));
        let o = upload(c, &vec![0f32; rows * vd]);
        let io = KdaIo { xq: &bq, xk: &bk, xv: &bv, fl: &bfl, gl: &bgl, braw: &bb, gate: &bg, ring: &ring, s: &s, o: &o };
        let lr = KdaLowRank { fb: &bfw, gb: Some(&bgw), rank_f: rank, rank_g: rank };
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(encode_kda(c, &mut enc, (u64::MAX, 0, 9), &d, &consts, &lr, None, eps, rows, &io));
        flush_pass(&enc);
        c.queue.submit([enc.finish()]);
        let og = download(c, &o, rows * vd);
        let sg = download(c, &s, d.s_words());
        let rg = download(c, &ring, d.ring_words());
        let err = |a: &[f32], b: &[f64]| -> f64 {
            let mut m = 0f64;
            for (x, y) in a.iter().zip(b) {
                m = m.max((*x as f64 - y).abs() / (1.0 + y.abs()));
            }
            m
        };
        let eo = err(&og, &oref);
        let es = err(&sg, &sref);
        // the ring: [kk-1][q | k | v] = the last kk - 1 raw rows
        let mut rref = vec![0f64; d.ring_words()];
        for j in 0..kk - 1 {
            let t = rows - (kk - 1) + j;
            for c2 in 0..cdim {
                rref[j * cdim + c2] = if c2 < kd {
                    xq[t * kd + c2]
                } else if c2 < 2 * kd {
                    xk[t * kd + c2 - kd]
                } else {
                    xv[t * vd + c2 - 2 * kd]
                } as f64;
            }
        }
        let er = err(&rg, &rref);
        eprintln!("kda kernels vs host: o {eo:.2e}, S {es:.2e}, ring {er:.2e}");
        assert!(eo < 1e-4 && es < 1e-4 && er == 0.0, "o {eo} S {es} ring {er}");
    }

    /// The MLA latent norm and the K / V assembly against the host.
    #[test]
    fn mla_kernels_match_host() {
        let Some(c) = ctx().or_else(trainer_ctx) else {
            eprintln!("mla_kernels_match_host: no wgpu device, skipped");
            return;
        };
        if pipes(c).is_none() {
            return;
        }
        let d = MlaDims { nh: 3, rope: 64, nope: 128, vd: 128, lora: 512 };
        let rows = 5;
        let mut sd = 11u64;
        let kva = rnd(&mut sd, rows * d.kva_rows(), 1.0);
        let w = rnd(&mut sd, d.lora, 1.0);
        let kvb = rnd(&mut sd, rows * d.kvb_rows(), 1.0);
        let eps = 1e-6f32;
        let (bkva, bw, bkvb) = (upload(c, &kva), upload(c, &w), upload(c, &kvb));
        let lat = upload(c, &vec![0f32; rows * d.lora]);
        let k = upload(c, &vec![0f32; rows * d.nh * d.hd()]);
        let v = upload(c, &vec![0f32; rows * d.nh * d.vd]);
        let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        assert!(encode_mla_lat(c, &mut enc, None, &d, eps, rows, &bkva, &bw, &lat));
        assert!(encode_mla_kv(c, &mut enc, None, &d, rows, &bkva, &bkvb, &k, &v));
        flush_pass(&enc);
        c.queue.submit([enc.finish()]);
        let (lg, kg, vg) = (download(c, &lat, rows * d.lora), download(c, &k, rows * d.nh * d.hd()), download(c, &v, rows * d.nh * d.vd));
        let mut el = 0f64;
        for t in 0..rows {
            let x = &kva[t * d.kva_rows()..t * d.kva_rows() + d.lora];
            let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv = 1.0 / (ss / d.lora as f64 + eps as f64).sqrt();
            for i in 0..d.lora {
                el = el.max((lg[t * d.lora + i] as f64 - x[i] as f64 * inv * w[i] as f64).abs());
            }
            for h in 0..d.nh {
                for e in 0..d.hd() {
                    let want = if e < d.rope { kva[t * d.kva_rows() + d.lora + e] } else { kvb[(t * d.nh + h) * (d.nope + d.vd) + e - d.rope] };
                    assert_eq!(kg[(t * d.nh + h) * d.hd() + e], want);
                }
                for e in 0..d.vd {
                    assert_eq!(vg[(t * d.nh + h) * d.vd + e], kvb[(t * d.nh + h) * (d.nope + d.vd) + d.nope + e]);
                }
            }
        }
        eprintln!("mla latent norm vs host: {el:.2e}");
        assert!(el < 1e-5);
    }

    /// The module parses and validates without a device (naga's uniformity
    /// analysis and reserved words reject a shader silently on the pod).
    #[test]
    fn kimi_shader_validates() {
        let src = super::src();
        let module = wgpu::naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("kimi WGSL must parse: {}", e.emit_to_string(&src)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("kimi WGSL must validate: {}", e.emit_to_string(&src)));
    }
}
