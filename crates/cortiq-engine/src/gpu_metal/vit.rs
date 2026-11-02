//! A vision tower's encoder resident on Metal (`gpu::vit_encode`,
//! `gpu::sam_encode`), the OCR models' twin of `gpu_wgpu::vit`: the embedded
//! rows go up once, every block's activations stay on the device, one readback.
//!
//! The GEMMs and the softmax are the EmbeddingGemma encoders' kernels
//! (`egemma2`: 64×64×32 tiles, the host's sum order). Attention is their
//! GEMM form, `S = Q·Kᵀ`, the softmax in place, `O = P·V`; SAM's global blocks
//! add the decomposed relative positions to `S` between the first two.
//!
//! An NT GEMM's k is a multiple of 32, so a narrower operand carries zeros in a
//! padded tail: SigLIP's 72-wide heads go through `vt_pad_q`, its 4304-wide FFN
//! through `vt_bias_act`.
//!
//! Only the identity rotation is taken (`VitArgs::sin` all zero, the OCR
//! towers): rotated towers keep their per-op path.

use super::egemma2::{self, EgAtt, EgMm, Arena, Prog, Rec, Rounds, Scratch, att_plan, proj, set_p, wrap, write};
use super::Ctx;
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::sync::{Mutex, OnceLock};

const VMSL: &str = include_str!("vit_msl.metal");

struct Pipes {
    ln: ComputePipelineState,
    act: ComputePipelineState,
    add: ComputePipelineState,
    pad_q: ComputePipelineState,
    part: ComputePipelineState,
    unpart: ComputePipelineState,
    win: ComputePipelineState,
    bias: ComputePipelineState,
    conv: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    let r = PIPES.get_or_init(|| {
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c
            ._device
            .new_library_with_source(VMSL, &opts)
            .map_err(|e| format!("vit MSL compile: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| format!("pipeline {name}: {e}"))
        };
        Ok(Pipes {
            ln: pso("vt_ln")?,
            act: pso("vt_bias_act")?,
            add: pso("vt_add")?,
            pad_q: pso("vt_pad_q")?,
            part: pso("sam_part")?,
            unpart: pso("sam_unpart")?,
            win: pso("sam_win")?,
            bias: pso("sam_bias")?,
            conv: pso("sam_conv")?,
        })
    });
    match r {
        Ok(p) => Some(p),
        Err(e) => {
            static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!("vision tower kernels unavailable ({e})");
            }
            None
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VtRow {
    rows: u32,
    cols: u32,
    ldx: u32,
    ldy: u32,
    eps: f32,
    act: u32,
    hd: u32,
    hs: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SamP {
    rows: u32,
    d: u32,
    g: u32,
    w: u32,
    nh: u32,
    hd: u32,
    ld: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SamBias {
    s_off: u32,
    nq: u32,
    lds: u32,
    q0: u32,
    h: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ConvP {
    g: u32,
    ci: u32,
    co: u32,
    k: u32,
    stride: u32,
    pad: u32,
    g2: u32,
    _a: u32,
}

struct LayerB {
    ln1_w: Buffer,
    ln1_b: Buffer,
    qkv: Buffer,
    qkv_b: Buffer,
    proj: Buffer,
    proj_b: Buffer,
    ln2_w: Buffer,
    ln2_b: Buffer,
    fc1: Buffer,
    fc1_b: Buffer,
    fc2: Buffer,
    fc2_b: Buffer,
    rel: Option<Buffer>,
}

/// A weight whose rows a padded k reads `extra` floats past: wrapped when the
/// wrap's page rounding covers them, copied with a zero tail otherwise.
fn wrap_past(c: &Ctx, v: &[f32], extra: usize) -> Buffer {
    let page = super::page_size();
    let bytes = v.len() * 4;
    if extra == 0 || bytes.div_ceil(page) * page >= bytes + extra * 4 {
        return wrap(c, v);
    }
    let b = c._device.new_buffer(((v.len() + extra) * 4) as u64, MTLResourceOptions::StorageModeShared);
    write(&b, v);
    unsafe {
        std::ptr::write_bytes((b.contents() as *mut f32).add(v.len()), 0, extra);
    }
    b
}

/// The tower the buffers wrap, by key and by the first weight's address (the
/// wraps alias the host allocation).
struct Tower {
    key: u64,
    at: usize,
    layers: Vec<LayerB>,
    convs: Vec<Buffer>,
    norms: Vec<Buffer>,
    rel: Option<(usize, Vec<Buffer>)>,
}

static VIT: Mutex<Option<Tower>> = Mutex::new(None);
static SAM: Mutex<Option<Tower>> = Mutex::new(None);
static SCRATCH: Mutex<Scratch> = Mutex::new(Scratch::new());

const S_X: usize = 0;
const S_XN: usize = 1;
const S_WIDE: usize = 2;
const S_TMP: usize = 3;
const S_ATT: usize = 4;
const S_S: usize = 5;
const S_Q: usize = 6;
const S_WN: usize = 7;
const S_ITEMS: usize = 8;
const S_OUT0: usize = 9;

fn layer_norm(g: &mut Prog, vp: &Pipes, src: &Buffer, w: &Buffer, b: &Buffer, dst: &Buffer, rows: usize, cols: usize, eps: f32) {
    let enc = g.rec.enc("rows");
    enc.set_compute_pipeline_state(&vp.ln);
    enc.set_buffer(0, Some(src), 0);
    enc.set_buffer(1, Some(w), 0);
    enc.set_buffer(2, Some(b), 0);
    enc.set_buffer(3, Some(dst), 0);
    set_p(enc, 4, &VtRow { rows: rows as u32, cols: cols as u32, ldx: cols as u32, ldy: cols as u32, eps, ..Default::default() });
    enc.dispatch_thread_groups(MTLSize::new((rows as u64).div_ceil(4), 1, 1), MTLSize::new(128, 1, 1));
}

fn bias_act(g: &mut Prog, vp: &Pipes, y: &Buffer, b: &Buffer, rows: usize, cols: usize, ld: usize, act: u32) {
    let enc = g.rec.enc("rows");
    enc.set_compute_pipeline_state(&vp.act);
    enc.set_buffer(0, Some(y), 0);
    enc.set_buffer(1, Some(b), 0);
    set_p(enc, 2, &VtRow { rows: rows as u32, cols: cols as u32, ldy: ld as u32, act, ..Default::default() });
    enc.dispatch_thread_groups(MTLSize::new((rows as u64).div_ceil(4), 1, 1), MTLSize::new(128, 1, 1));
}

fn add(g: &mut Prog, vp: &Pipes, x: &Buffer, y: &Buffer, b: &Buffer, rows: usize, cols: usize) {
    let enc = g.rec.enc("rows");
    enc.set_compute_pipeline_state(&vp.add);
    enc.set_buffer(0, Some(x), 0);
    enc.set_buffer(1, Some(y), 0);
    enc.set_buffer(2, Some(b), 0);
    set_p(enc, 3, &VtRow { rows: rows as u32, cols: cols as u32, ldx: cols as u32, ldy: cols as u32, ..Default::default() });
    enc.dispatch_thread_groups(MTLSize::new((rows as u64).div_ceil(4), 1, 1), MTLSize::new(128, 1, 1));
}

/// `[n][nh·hd]` attention over `qkv = [n][q|k|v]`. `q` is the padded q plane of
/// a head width off the k grid, else the same rows. `rel` adds SAM's relative
/// bias to the scores.
#[allow(clippy::too_many_arguments)]
fn attention(
    g: &mut Prog,
    vp: &Pipes,
    rounds: &Rounds,
    q: (&Buffer, usize, usize),
    qkv: &Buffer,
    s: &Buffer,
    att: &Buffer,
    d: usize,
    hd: usize,
    scale: f32,
    rel: Option<(&Buffer, usize)>,
) -> Result<(), String> {
    let ld = 3 * d;
    let (qb, ldq, hs) = q;
    for round in rounds {
        if round.is_empty() {
            continue;
        }
        let s_items: Vec<EgMm> = round
            .iter()
            .map(|(a, it)| EgMm {
                n: it.nq as u32,
                m: it.nk as u32,
                k: hs as u32,
                ldx: ldq as u32,
                ldw: ld as u32,
                ldy: a.lds,
                x_off: (it.q0 * ldq + it.h * hs) as u32,
                w_off: (it.k0 * ld + d + it.kvh * hd) as u32,
                y_off: a.s_off,
                alpha: scale,
                wrows: it.nk as u32,
                pad1: 0,
            })
            .collect();
        g.mm(false, qb, [qkv, qkv, qkv], s, &s_items, "attn")?;
        if let Some((rb, grid)) = rel {
            let items: Vec<SamBias> = round
                .iter()
                .map(|(a, it)| SamBias {
                    s_off: a.s_off,
                    nq: it.nq as u32,
                    lds: a.lds,
                    q0: it.q0 as u32,
                    h: it.h as u32,
                    pad: [0; 3],
                })
                .collect();
            let at = g.arena.push(&items)?;
            let maxq = round.iter().map(|(_, it)| it.nq).max().unwrap() as u64;
            let buf = g.arena.buf.clone();
            let enc = g.rec.enc("attn");
            enc.set_compute_pipeline_state(&vp.bias);
            enc.set_buffer(0, Some(qkv), 0);
            enc.set_buffer(1, Some(rb), 0);
            enc.set_buffer(2, Some(s), 0);
            enc.set_buffer(3, Some(&buf), at);
            set_p(enc, 4, &SamP { g: grid as u32, hd: hd as u32, ld: ld as u32, ..Default::default() });
            enc.dispatch_threads(MTLSize::new(maxq, round.len() as u64, 1), MTLSize::new(32, 1, 1));
        }
        let atts: Vec<EgAtt> = round.iter().map(|(a, _)| *a).collect();
        let at = g.arena.push(&atts)?;
        let maxq = round.iter().map(|(_, it)| it.nq).max().unwrap() as u64;
        let buf = g.arena.buf.clone();
        let p = g.p;
        let enc = g.rec.enc("attn");
        enc.set_compute_pipeline_state(&p.softmax);
        enc.set_buffer(0, Some(s), 0);
        enc.set_buffer(1, Some(&buf), at);
        enc.dispatch_thread_groups(MTLSize::new(maxq.div_ceil(4), round.len() as u64, 1), MTLSize::new(128, 1, 1));
        let o_items: Vec<EgMm> = round
            .iter()
            .map(|(a, it)| EgMm {
                n: it.nq as u32,
                m: hd as u32,
                k: a.lds,
                ldx: a.lds,
                ldw: ld as u32,
                ldy: d as u32,
                x_off: a.s_off,
                w_off: (it.k0 * ld + 2 * d + it.kvh * hd) as u32,
                y_off: (it.q0 * d + it.h * hd) as u32,
                alpha: 1.0,
                wrows: it.nk as u32,
                pad1: 0,
            })
            .collect();
        g.mm(true, s, [qkv, qkv, qkv], att, &o_items, "attn")?;
    }
    Ok(())
}

fn arena_cap(rounds: &[&Rounds], layers: usize) -> usize {
    let per_item = std::mem::size_of::<EgAtt>() + 2 * std::mem::size_of::<EgMm>() + std::mem::size_of::<SamBias>();
    let items: usize = rounds.iter().map(|r| r.iter().map(|x| x.len() * per_item + 4 * 256).sum::<usize>()).sum();
    items + layers * 8 * (3 * std::mem::size_of::<EgMm>() + 256) + 4096
}

fn prof_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_OCR_PROF").is_some())
}

/// The tower's encoder layers over `x` (`[n, d]`, embedded rows), in place.
/// `false` = refused; the caller runs its per-op path.
pub fn encode(a: &crate::gpu::VitArgs, x: &mut [f32]) -> bool {
    let Some(c) = super::ctx() else { return false };
    let Some(vp) = pipes(c) else { return false };
    let Ok(p) = egemma2::pipes(c) else { return false };
    let (n, d, nh, hd, inter) = (a.n, a.d, a.heads, a.head_dim, a.inter);
    if n == 0
        || nh * hd != d
        || !d.is_multiple_of(32)
        || !hd.is_multiple_of(4)
        || !inter.is_multiple_of(4)
        || x.len() != n * d
        || a.sin.iter().any(|&s| s != 0.0)
        || a.layers.is_empty()
    {
        return false;
    }
    let t0 = std::time::Instant::now();
    let hs = hd.div_ceil(32) * 32;
    let inter_p = inter.div_ceil(32) * 32;
    let mut guard = VIT.lock().unwrap_or_else(|e| e.into_inner());
    let at = a.layers[0].qkv.as_ptr() as usize;
    if guard.as_ref().is_none_or(|t| t.key != a.key || t.at != at || t.layers.len() != a.layers.len()) {
        let layers = a
            .layers
            .iter()
            .map(|l| LayerB {
                ln1_w: wrap(c, l.ln1_w),
                ln1_b: wrap(c, l.ln1_b),
                qkv: wrap_past(c, l.qkv, 0),
                qkv_b: wrap(c, l.qkv_b),
                proj: wrap(c, l.proj),
                proj_b: wrap(c, l.proj_b),
                ln2_w: wrap(c, l.ln2_w),
                ln2_b: wrap(c, l.ln2_b),
                fc1: wrap(c, l.fc1),
                fc1_b: wrap(c, l.fc1_b),
                fc2: wrap_past(c, l.fc2, inter_p - inter),
                fc2_b: wrap(c, l.fc2_b),
                rel: None,
            })
            .collect();
        *guard = Some(Tower { key: a.key, at, layers, convs: Vec::new(), norms: Vec::new(), rel: None });
    }
    let t = guard.as_ref().expect("set above");
    let (rounds, s_need) = att_plan(&[(0, n)], nh, nh, 0);
    let ld = 3 * d;
    let mut sc = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    let bx = sc.get(c, S_X, n * d * 4);
    let bn = sc.get(c, S_XN, n * d * 4);
    // Rows of padding: a tile reads past the last valid column.
    let bwide = sc.get(c, S_WIDE, (n + 32) * ld.max(inter_p) * 4);
    let bt = sc.get(c, S_TMP, n * d * 4);
    let batt = sc.get(c, S_ATT, n * d * 4);
    let bs = sc.get(c, S_S, s_need.max(1) * 4);
    let bq = if hs != hd { sc.get(c, S_Q, n * nh * hs * 4) } else { bwide.clone() };
    let per_layer: Vec<&Rounds> = t.layers.iter().map(|_| &rounds).collect();
    let cap = arena_cap(&per_layer, t.layers.len());
    let bi = sc.get(c, S_ITEMS, cap);
    write(&bx, x);
    let eps = a.eps as f32;
    let act = if a.quick_gelu {
        3
    } else if a.gelu_tanh {
        2
    } else {
        1
    };
    let scale = (hd as f32).powf(-0.5);
    let mut g = Prog { p, rec: Rec::new(c), arena: Arena { buf: bi, cap, off: 0 }, eps };
    for l in &t.layers {
        layer_norm(&mut g, vp, &bx, &l.ln1_w, &l.ln1_b, &bn, n, d, eps);
        if g.mm(false, &bn, [&l.qkv, &l.qkv, &l.qkv], &bwide, &[proj(n, ld, d, d, ld, 0)], "gemm").is_err() {
            return false;
        }
        bias_act(&mut g, vp, &bwide, &l.qkv_b, n, ld, ld, 0);
        let qp = if hs != hd {
            let enc = g.rec.enc("rows");
            enc.set_compute_pipeline_state(&vp.pad_q);
            enc.set_buffer(0, Some(&bwide), 0);
            enc.set_buffer(1, Some(&bq), 0);
            set_p(enc, 2, &VtRow { cols: nh as u32, ldx: ld as u32, ldy: (nh * hs) as u32, hd: hd as u32, hs: hs as u32, ..Default::default() });
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(32, 1, 1));
            (&bq, nh * hs, hs)
        } else {
            (&bwide, ld, hd)
        };
        if attention(&mut g, vp, &rounds, qp, &bwide, &bs, &batt, d, hd, scale, None).is_err() {
            return false;
        }
        if g.mm(false, &batt, [&l.proj, &l.proj, &l.proj], &bt, &[proj(n, d, d, d, d, 0)], "gemm").is_err() {
            return false;
        }
        add(&mut g, vp, &bx, &bt, &l.proj_b, n, d);
        layer_norm(&mut g, vp, &bx, &l.ln2_w, &l.ln2_b, &bn, n, d, eps);
        if g.mm(false, &bn, [&l.fc1, &l.fc1, &l.fc1], &bwide, &[proj(n, inter, d, d, inter_p, 0)], "gemm").is_err() {
            return false;
        }
        bias_act(&mut g, vp, &bwide, &l.fc1_b, n, inter, inter_p, act);
        let mut down = proj(n, d, inter_p, inter_p, d, 0);
        down.ldw = inter as u32;
        if g.mm(false, &bwide, [&l.fc2, &l.fc2, &l.fc2], &bt, &[down], "gemm").is_err() {
            return false;
        }
        add(&mut g, vp, &bx, &bt, &l.fc2_b, n, d);
        g.rec.close();
    }
    let Prog { rec, .. } = g;
    let host = t0.elapsed().as_secs_f64();
    let times = match rec.finish() {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("vision tower failed on the device: {e}");
            return false;
        }
    };
    if prof_on() {
        eprintln!("vit metal: {n} rows {} layers, encode {:.1} ms, total {:.1} ms, {times:?}", a.layers.len(), host * 1e3, t0.elapsed().as_secs_f64() * 1e3);
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bx.contents() as *const f32, x.as_mut_ptr(), n * d);
    }
    true
}

/// SAM's blocks and neck over the grid `x` (`[g², d]`); `out` the last conv's
/// channels. `false` = refused.
pub fn sam_encode(a: &crate::gpu::SamArgs, x: &[f32], out: &mut Vec<f32>) -> bool {
    let Some(c) = super::ctx() else { return false };
    let Some(vp) = pipes(c) else { return false };
    let Ok(p) = egemma2::pipes(c) else { return false };
    let (grid, d, nh, hd, inter) = (a.g, a.d, a.heads, a.head_dim, a.inter);
    let n = grid * grid;
    if x.len() != n * d
        || nh * hd != d
        || !hd.is_multiple_of(32)
        || !d.is_multiple_of(32)
        || !inter.is_multiple_of(32)
        || grid > 128
        || a.layers.is_empty()
    {
        return false;
    }
    let t0 = std::time::Instant::now();
    let mut guard = SAM.lock().unwrap_or_else(|e| e.into_inner());
    let at = a.layers[0].qkv.as_ptr() as usize;
    if guard.as_ref().is_none_or(|t| t.key != a.key || t.at != at) {
        let layers = a
            .layers
            .iter()
            .map(|l| LayerB {
                ln1_w: wrap(c, l.ln1_w),
                ln1_b: wrap(c, l.ln1_b),
                qkv: wrap(c, l.qkv),
                qkv_b: wrap(c, l.qkv_b),
                proj: wrap(c, l.proj),
                proj_b: wrap(c, l.proj_b),
                ln2_w: wrap(c, l.ln2_w),
                ln2_b: wrap(c, l.ln2_b),
                fc1: wrap(c, l.fc1),
                fc1_b: wrap(c, l.fc1_b),
                fc2: wrap(c, l.fc2),
                fc2_b: wrap(c, l.fc2_b),
                rel: Some(wrap(c, l.rel)),
            })
            .collect();
        let convs = a.convs.iter().map(|v| wrap(c, v.w)).collect();
        let norms = a.norms.iter().map(|v| wrap(c, v)).collect();
        *guard = Some(Tower { key: a.key, at, layers, convs, norms, rel: None });
    }
    let t = guard.as_mut().expect("set above");
    if t.rel.as_ref().is_none_or(|r| r.0 != grid) {
        let copies = a
            .rel_global
            .iter()
            .map(|v| {
                c._device.new_buffer_with_data(
                    v.as_ptr() as *const std::ffi::c_void,
                    (v.len() * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect();
        t.rel = Some((grid, copies));
    }
    let t = &*t;
    let grel = &t.rel.as_ref().expect("set above").1;
    let w0 = a.layers.iter().map(|l| l.window).max().unwrap_or(1).max(1);
    let wrows = grid.div_ceil(w0) * grid.div_ceil(w0) * w0 * w0;
    let rows_max = wrows.max(n);
    let ld = 3 * d;
    let (rounds, s_need) = att_plan(&[(0, n)], nh, nh, 0);
    let per: Vec<&Rounds> = a.layers.iter().filter(|l| l.window == 0).map(|_| &rounds).collect();
    let cap = arena_cap(&per, a.layers.len());
    let mut sc = SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    let bx = sc.get(c, S_X, n * d * 4);
    let bn = sc.get(c, S_XN, n * d * 4);
    let bw = sc.get(c, S_WN, rows_max * d * 4);
    let bwide = sc.get(c, S_WIDE, (rows_max + 32) * ld.max(inter) * 4);
    let bt = sc.get(c, S_TMP, rows_max * d * 4);
    let batt = sc.get(c, S_ATT, rows_max * d * 4);
    let bs = sc.get(c, S_S, s_need.max(1) * 4);
    let bi = sc.get(c, S_ITEMS, cap);
    write(&bx, x);
    let eps = a.eps as f32;
    let scale = (hd as f32).powf(-0.5);
    let mut g = Prog { p, rec: Rec::new(c), arena: Arena { buf: bi, cap, off: 0 }, eps };
    let mut gi = 0usize;
    for (l, spec) in t.layers.iter().zip(&a.layers) {
        layer_norm(&mut g, vp, &bx, &l.ln1_w, &l.ln1_b, &bn, n, d, eps);
        if spec.window > 0 {
            let w = spec.window;
            let rows = grid.div_ceil(w) * grid.div_ceil(w) * w * w;
            let prm = SamP {
                rows: rows as u32,
                d: d as u32,
                g: grid as u32,
                w: w as u32,
                nh: nh as u32,
                hd: hd as u32,
                ld: ld as u32,
                scale,
            };
            {
                let enc = g.rec.enc("rows");
                enc.set_compute_pipeline_state(&vp.part);
                enc.set_buffer(0, Some(&bn), 0);
                enc.set_buffer(1, Some(&bw), 0);
                set_p(enc, 2, &prm);
                enc.dispatch_threads(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
            }
            if g.mm(false, &bw, [&l.qkv, &l.qkv, &l.qkv], &bwide, &[proj(rows, ld, d, d, ld, 0)], "gemm").is_err() {
                return false;
            }
            bias_act(&mut g, vp, &bwide, &l.qkv_b, rows, ld, ld, 0);
            {
                let enc = g.rec.enc("attn");
                enc.set_compute_pipeline_state(&vp.win);
                enc.set_buffer(0, Some(&bwide), 0);
                enc.set_buffer(1, Some(l.rel.as_ref().expect("sam layer")), 0);
                enc.set_buffer(2, Some(&batt), 0);
                set_p(enc, 3, &prm);
                enc.dispatch_threads(MTLSize::new(nh as u64, rows as u64, 1), MTLSize::new(32, 1, 1));
            }
            if g.mm(false, &batt, [&l.proj, &l.proj, &l.proj], &bt, &[proj(rows, d, d, d, d, 0)], "gemm").is_err() {
                return false;
            }
            let enc = g.rec.enc("rows");
            enc.set_compute_pipeline_state(&vp.unpart);
            enc.set_buffer(0, Some(&bx), 0);
            enc.set_buffer(1, Some(&bt), 0);
            enc.set_buffer(2, Some(&l.proj_b), 0);
            set_p(enc, 3, &prm);
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(32, 1, 1));
        } else {
            if g.mm(false, &bn, [&l.qkv, &l.qkv, &l.qkv], &bwide, &[proj(n, ld, d, d, ld, 0)], "gemm").is_err() {
                return false;
            }
            bias_act(&mut g, vp, &bwide, &l.qkv_b, n, ld, ld, 0);
            let rel = &grel[gi];
            gi += 1;
            if attention(&mut g, vp, &rounds, (&bwide, ld, hd), &bwide, &bs, &batt, d, hd, scale, Some((rel, grid))).is_err() {
                return false;
            }
            if g.mm(false, &batt, [&l.proj, &l.proj, &l.proj], &bt, &[proj(n, d, d, d, d, 0)], "gemm").is_err() {
                return false;
            }
            add(&mut g, vp, &bx, &bt, &l.proj_b, n, d);
        }
        layer_norm(&mut g, vp, &bx, &l.ln2_w, &l.ln2_b, &bn, n, d, eps);
        if g.mm(false, &bn, [&l.fc1, &l.fc1, &l.fc1], &bwide, &[proj(n, inter, d, d, inter, 0)], "gemm").is_err() {
            return false;
        }
        bias_act(&mut g, vp, &bwide, &l.fc1_b, n, inter, inter, 1);
        if g.mm(false, &bwide, [&l.fc2, &l.fc2, &l.fc2], &bt, &[proj(n, d, inter, inter, d, 0)], "gemm").is_err() {
            return false;
        }
        add(&mut g, vp, &bx, &bt, &l.fc2_b, n, d);
        g.rec.close();
    }
    // The neck and the strided convs.
    let mut cur = bx.clone();
    let mut cg = grid;
    let mut norm_at = 0usize;
    let mut slot = S_OUT0;
    let mut last = 0usize;
    for (cv, wb) in a.convs.iter().zip(&t.convs) {
        let g2 = (cg + 2 * cv.pad - cv.k) / cv.stride + 1;
        let cells = g2 * g2 * cv.co;
        let y = sc.get(c, slot, cells * 4);
        slot += 1;
        {
            let enc = g.rec.enc("rows");
            enc.set_compute_pipeline_state(&vp.conv);
            enc.set_buffer(0, Some(wb), 0);
            enc.set_buffer(1, Some(&cur), 0);
            enc.set_buffer(2, Some(&y), 0);
            set_p(
                enc,
                3,
                &ConvP {
                    g: cg as u32,
                    ci: cv.ci as u32,
                    co: cv.co as u32,
                    k: cv.k as u32,
                    stride: cv.stride as u32,
                    pad: cv.pad as u32,
                    g2: g2 as u32,
                    _a: 0,
                },
            );
            enc.dispatch_threads(MTLSize::new(cells as u64, 1, 1), MTLSize::new(32, 1, 1));
        }
        if cv.norm {
            let (w, b) = (&t.norms[norm_at], &t.norms[norm_at + 1]);
            norm_at += 2;
            let yn = sc.get(c, slot, cells * 4);
            slot += 1;
            layer_norm(&mut g, vp, &y, w, b, &yn, g2 * g2, cv.co, eps);
            cur = yn;
        } else {
            cur = y;
        }
        cg = g2;
        last = cells;
    }
    let Prog { rec, .. } = g;
    let host = t0.elapsed().as_secs_f64();
    let times = match rec.finish() {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("SAM failed on the device: {e}");
            return false;
        }
    };
    if prof_on() {
        eprintln!("sam metal: grid {grid}, encode {:.1} ms, total {:.1} ms, {times:?}", host * 1e3, t0.elapsed().as_secs_f64() * 1e3);
    }
    out.clear();
    out.resize(last, 0.0);
    unsafe {
        std::ptr::copy_nonoverlapping(cur.contents() as *const f32, out.as_mut_ptr(), last);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(st: &mut u64, n: usize, a: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *st ^= *st << 13;
                *st ^= *st >> 7;
                *st ^= *st << 17;
                ((*st >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * a
            })
            .collect()
    }

    fn lnorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
        let d = w.len();
        let mut out = vec![0f32; x.len()];
        for (r, o) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
            let mean = r.iter().sum::<f32>() / d as f32;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = (var + eps).sqrt().recip();
            for i in 0..d {
                o[i] = (r[i] - mean) * inv * w[i] + b[i];
            }
        }
        out
    }

    fn mm(x: &[f32], w: &[f32], b: &[f32], k: usize, m: usize) -> Vec<f32> {
        let n = x.len() / k;
        let mut y = vec![0f32; n * m];
        for i in 0..n {
            for j in 0..m {
                let s: f32 = (0..k).map(|c| x[i * k + c] * w[j * k + c]).sum();
                y[i * m + j] = s + b[j];
            }
        }
        y
    }

    /// One block on the host; `rel` the decomposed bias of a global SAM block.
    #[allow(clippy::too_many_arguments)]
    fn block(x: &mut [f32], l: &crate::gpu::VitLayer, n: usize, d: usize, nh: usize, hd: usize, inter: usize, eps: f32, act: u32, rel: Option<(&[f32], usize)>) {
        let xn = lnorm(x, l.ln1_w, l.ln1_b, eps);
        let qkv = mm(&xn, l.qkv, l.qkv_b, d, 3 * d);
        let scale = (hd as f32).powf(-0.5);
        let mut ao = vec![0f32; n * d];
        for h in 0..nh {
            for q in 0..n {
                let qr = &qkv[q * 3 * d + h * hd..q * 3 * d + (h + 1) * hd];
                let mut sc: Vec<f32> = (0..n)
                    .map(|k| {
                        let kr = &qkv[k * 3 * d + d + h * hd..k * 3 * d + d + (h + 1) * hd];
                        qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                if let Some((rel, g)) = rel {
                    let (qy, qx) = (q / g, q % g);
                    let rl = 2 * g - 1;
                    for k in 0..n {
                        let (ky, kx) = (k / g, k % g);
                        let ry = &rel[(qy + g - 1 - ky) * hd..][..hd];
                        let rx = &rel[(rl + qx + g - 1 - kx) * hd..][..hd];
                        sc[k] += qr.iter().zip(ry).map(|(a, b)| a * b).sum::<f32>();
                        sc[k] += qr.iter().zip(rx).map(|(a, b)| a * b).sum::<f32>();
                    }
                }
                let mx = sc.iter().fold(f32::MIN, |a, &b| a.max(b));
                let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
                let den: f32 = e.iter().sum();
                for j in 0..hd {
                    let s: f32 = (0..n).map(|k| e[k] * qkv[k * 3 * d + 2 * d + h * hd + j]).sum();
                    ao[q * d + h * hd + j] = s / den;
                }
            }
        }
        let t = mm(&ao, l.proj, l.proj_b, d, d);
        for (a, b) in x.iter_mut().zip(&t) {
            *a += b;
        }
        let xn = lnorm(x, l.ln2_w, l.ln2_b, eps);
        let mut f = mm(&xn, l.fc1, l.fc1_b, d, inter);
        for v in &mut f {
            let u = *v;
            *v = match act {
                1 => 0.5 * u * (1.0 + crate::inference::erf_f32(u * std::f32::consts::FRAC_1_SQRT_2)),
                2 => 0.5 * u * (1.0 + (0.7978845608 * (u + 0.044715 * u * u * u)).clamp(-15.0, 15.0).tanh()),
                _ => u / (1.0 + (-1.702 * u).exp()),
            };
        }
        let t = mm(&f, l.fc2, l.fc2_b, inter, d);
        for (a, b) in x.iter_mut().zip(&t) {
            *a += b;
        }
    }

    fn check(n: usize, d: usize, nh: usize, hd: usize, inter: usize, act: u32) {
        if super::super::ctx().is_none() {
            return;
        }
        let mut st = 0x9e37_79b9_7f4a_7c15u64 ^ (d as u64) << 7;
        let eps = 1e-6f32;
        let ws: Vec<Vec<f32>> = [d, d, 3 * d * d, 3 * d, d * d, d, d, d, inter * d, inter, d * inter, d]
            .iter()
            .map(|&s| rnd(&mut st, s, 0.3))
            .collect();
        let layer = || crate::gpu::VitLayer {
            ln1_w: &ws[0],
            ln1_b: &ws[1],
            qkv: &ws[2],
            qkv_b: &ws[3],
            proj: &ws[4],
            proj_b: &ws[5],
            ln2_w: &ws[6],
            ln2_b: &ws[7],
            fc1: &ws[8],
            fc1_b: &ws[9],
            fc2: &ws[10],
            fc2_b: &ws[11],
        };
        let x0 = rnd(&mut st, n * d, 2.0);
        let (cos, sin) = (vec![1f32; n * hd], vec![0f32; n * hd]);
        let args = crate::gpu::VitArgs {
            key: d as u64 + ((hd as u64) << 16) + ((n as u64) << 32),
            layers: vec![layer(), layer(), layer()],
            n,
            d,
            heads: nh,
            head_dim: hd,
            inter,
            eps: eps as f64,
            gelu_tanh: act == 2,
            quick_gelu: act == 3,
            rope_halves: true,
            cos: &cos,
            sin: &sin,
            prof: false,
        };
        let mut got = x0.clone();
        // A runner without the matrix kernels refuses: the host path runs there.
        if !encode(&args, &mut got) {
            return;
        }
        let mut want = x0;
        for _ in 0..3 {
            block(&mut want, &layer(), n, d, nh, hd, inter, eps, act, None);
        }
        let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let err = got.iter().zip(&want).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        assert!(err / scale < 1e-3, "n {n} d {d} hd {hd} inter {inter}: {err} of {scale}");
    }

    #[test]
    fn vit_matches_host_on_the_k_grid() {
        check(70, 128, 2, 64, 256, 3);
    }

    #[test]
    fn vit_matches_host_with_padded_heads_and_ffn() {
        check(70, 160, 4, 40, 164, 2);
    }

    #[test]
    fn vit_matches_host_siglip_width() {
        check(160, 1152, 16, 72, 4304, 2);
    }
}
