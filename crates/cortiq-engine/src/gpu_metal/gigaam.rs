//! The GigaAM encoder resident on Metal (`gpu::gigaam_encode`): the three
//! Zapformer stacks of one segment with the activations on the device from
//! the first projection to the output norm — one upload, one readback.
//!
//! The weights go up once a model (`key`) as f32 planes, filled on a few
//! threads straight into the device buffers, in the order the forward takes
//! them: the first segment's layers are encoded and run behind the fill, a
//! product waiting only for its own plane. The products and the attention
//! are the EmbeddingGemma encoders' kernels (`egemma2`: 64×64×32 tiles in the
//! host's sum order; `S = Q·Kᵀ`, the softmax in place, `O = P·V`), everything
//! else the kernels of `gigaam_msl.metal`. q, k and v stay in the rows their
//! products wrote (`ga_prep` rotates and gates them in place), and a stack
//! with a fold of 2 reads the frames as rows of pairs where they lie, the
//! frame after an odd last one kept at zero.
//!
//! A command buffer a layer, committed as it is encoded and waited for once
//! at the end. A product of a short segment is a few thread groups, each
//! walking its whole k alone with the device mostly idle: under
//! `CMF_GIGAAM_TG_MIN` groups (48; 0 = never) its k runs in parts side by
//! side, summed after (`ga_sum`).
//!
//! `CMF_GIGAAM_PROF=1` prints the segment's time; `CMF_EGEMMA2_GPU_PROF=1`
//! beside it splits the GPU time by kernel class (a command buffer a class:
//! slower, for the split only).

use super::egemma2::{self, Arena, EgAtt, EgMm, Prog, Rec, Rounds, Scratch, att_plan, proj, set_p, write};
use super::Ctx;
use crate::gpu::{GaLin, GigaamArgs};
use metal::{Buffer, ComputePipelineState, MTLResourceOptions, MTLSize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

const GMSL: &str = include_str!("gigaam_msl.metal");

struct Pipes {
    sum: ComputePipelineState,
    bias_act: ComputePipelineState,
    add: ComputePipelineState,
    scale2: ComputePipelineState,
    rms: ComputePipelineState,
    prep: ComputePipelineState,
    gate: ComputePipelineState,
    mean: ComputePipelineState,
    conv: ComputePipelineState,
}

static PIPES: OnceLock<Result<Pipes, String>> = OnceLock::new();

fn pipes(c: &Ctx) -> Result<&'static Pipes, String> {
    PIPES
        .get_or_init(|| {
            let opts = metal::CompileOptions::new();
            opts.set_language_version(metal::MTLLanguageVersion::V3_0);
            let lib = c._device.new_library_with_source(GMSL, &opts).map_err(|e| format!("gigaam MSL compile: {e}"))?;
            let pso = |name: &str| -> Result<ComputePipelineState, String> {
                let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
                c._device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("pipeline {name}: {e}"))
            };
            Ok(Pipes {
                sum: pso("ga_sum")?,
                bias_act: pso("ga_bias_act")?,
                add: pso("ga_add")?,
                scale2: pso("ga_scale2")?,
                rms: pso("ga_rms")?,
                prep: pso("ga_prep")?,
                gate: pso("ga_gate")?,
                mean: pso("ga_mean")?,
                conv: pso("ga_conv")?,
            })
        })
        .as_ref()
        .map_err(|e| e.clone())
}

/// The kernels' parameters (`GaP` of `gigaam_msl.metal`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct GaP {
    rows: u32,
    cols: u32,
    flags: u32,
    aux: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
}

/// A product on the device: the weight plane and its bias (None: no bias).
struct LinB {
    w: Buffer,
    b: Option<Buffer>,
    rows: usize,
    cols: usize,
    /// the plane's place in the fill order
    id: usize,
}

struct ConvB {
    inp: LinB,
    taps: Buffer,
    bias: Buffer,
    mean_w: Buffer,
    kernel: usize,
    out: LinB,
}

struct LayerB {
    ff1: (LinB, LinB),
    in_norm: Buffer,
    qk: LinB,
    vg: LinB,
    attn_out: LinB,
    conv: Option<ConvB>,
    ff2: Option<(LinB, LinB)>,
    norm: Buffer,
}

struct StackB {
    proj: LinB,
    proj_t: LinB,
    layers: Vec<LayerB>,
}

struct Model {
    key: u64,
    stacks: Vec<StackB>,
    cos: Buffer,
    sin: Buffer,
}

struct State {
    model: Option<Model>,
    /// working buffers by slot; they only grow
    scratch: Scratch,
}

static STATE: Mutex<State> = Mutex::new(State { model: None, scratch: Scratch::new() });

// working buffers
const S_X: usize = 0;
const S_H: usize = 1;
const S_S: usize = 2;
const S_TMP: usize = 3;
const S_RES: usize = 4;
const S_BYPASS: usize = 5;
const S_WIDE: usize = 6;
const S_QK: usize = 7;
const S_SCORES: usize = 8;
const S_MEAN: usize = 9;
const S_ITEMS: usize = 10;
const S_PARTS: usize = 11;

/// A product of fewer thread groups than this (64 rows by 64 features each)
/// runs its k in parts side by side. On the M4 a 64-row product of 3072 into
/// 768 takes 0.47 ms whole, where four times the rows take 0.67 ms.
fn tg_min() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_GIGAAM_TG_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(48))
}
/// The most parts of one product.
const PARTS_MAX: usize = 8;

fn floats(c: &Ctx, v: &[f32]) -> Buffer {
    c._device.new_buffer_with_data(v.as_ptr() as *const std::ffi::c_void, (v.len() * 4).max(16) as u64, MTLResourceOptions::StorageModeShared)
}

/// The planes of a model being filled: `jobs` in the forward's order, each a
/// product and the address of its plane.
struct Fill<'x, 'b> {
    jobs: Vec<(&'x GaLin<'b>, usize)>,
    next: AtomicUsize,
    /// which planes are whole, and whether a filler gave up
    done: Mutex<(Vec<bool>, bool)>,
    turn: Condvar,
    since: std::time::Instant,
    /// when the last plane was whole, ms after `since`
    took: Mutex<f64>,
}

impl Fill<'_, '_> {
    /// One filler: the next plane until none is left.
    fn work(&self) {
        struct Gave<'a, 'x, 'b>(&'a Fill<'x, 'b>);
        impl Drop for Gave<'_, '_, '_> {
            fn drop(&mut self) {
                if std::thread::panicking() {
                    self.0.done.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
                    self.0.turn.notify_all();
                }
            }
        }
        let _gave = Gave(self);
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(&(l, at)) = self.jobs.get(i) else { break };
            // SAFETY: the plane's own shared buffer, `rows·cols` floats, this job's alone;
            // the device reads it only after `wait` has seen it whole.
            l.weight.fill(unsafe { std::slice::from_raw_parts_mut(at as *mut f32, l.rows * l.cols) });
            let mut d = self.done.lock().unwrap_or_else(|e| e.into_inner());
            d.0[i] = true;
            if d.0.iter().all(|v| *v) {
                *self.took.lock().unwrap_or_else(|e| e.into_inner()) = self.since.elapsed().as_secs_f64() * 1e3;
            }
            drop(d);
            self.turn.notify_all();
        }
    }

    fn whole(&self, id: usize) -> bool {
        self.done.lock().unwrap_or_else(|e| e.into_inner()).0[id]
    }

    /// Until plane `id` is whole.
    fn wait(&self, id: usize) -> Result<(), String> {
        let mut d = self.done.lock().unwrap_or_else(|e| e.into_inner());
        while !d.0[id] {
            if d.1 {
                return Err("a weight could not be read".into());
            }
            d = self.turn.wait(d).unwrap_or_else(|e| e.into_inner());
        }
        Ok(())
    }
}

/// The planes of `a` on the device, allocated and not yet filled, and their
/// fill (a q8 file's planes are widened as they are asked).
fn planes<'x, 'b>(c: &Ctx, a: &'x GigaamArgs<'b>) -> Result<(Model, Fill<'x, 'b>), String> {
    let mut jobs: Vec<(&GaLin, usize)> = Vec::new();
    let mut failed = false;
    let mut lin = |l: &'x GaLin<'b>| -> LinB {
        let w = c._device.new_buffer((l.rows * l.cols * 4).max(16) as u64, MTLResourceOptions::StorageModeShared);
        let at = w.contents() as usize;
        failed |= at == 0;
        jobs.push((l, at));
        LinB { w, b: if l.bias.is_empty() { None } else { Some(floats(c, l.bias)) }, rows: l.rows, cols: l.cols, id: jobs.len() - 1 }
    };
    let mut stacks = Vec::with_capacity(a.stacks.len());
    for st in &a.stacks {
        // in the order the forward asks
        let proj = lin(&st.proj);
        let mut layers = Vec::with_capacity(st.layers.len());
        for l in &st.layers {
            layers.push(LayerB {
                ff1: (lin(&l.ff1_in), lin(&l.ff1_out)),
                in_norm: floats(c, l.in_norm),
                qk: lin(&l.qk),
                vg: lin(&l.vg),
                attn_out: lin(&l.attn_out),
                conv: l.conv.as_ref().map(|cv| ConvB {
                    inp: lin(&cv.inp),
                    taps: floats(c, cv.taps),
                    bias: floats(c, cv.bias),
                    mean_w: floats(c, cv.mean_w),
                    kernel: cv.kernel,
                    out: lin(&cv.out),
                }),
                ff2: l.ff2.as_ref().map(|(i, o)| (lin(i), lin(o))),
                norm: floats(c, l.norm),
            });
        }
        stacks.push(StackB { proj, proj_t: lin(&st.proj_t), layers });
    }
    if failed {
        return Err("no device memory for the weights".into());
    }
    let done = Mutex::new((vec![false; jobs.len()], false));
    let fill = Fill { jobs, next: AtomicUsize::new(0), done, turn: Condvar::new(), since: std::time::Instant::now(), took: Mutex::new(0.0) };
    Ok((Model { key: a.key, stacks, cos: floats(c, a.cos), sin: floats(c, a.sin) }, fill))
}

/// What the kernels take: f32 rows on the GEMM's grid of 32 (every k) and in
/// float4 lanes (every width), 32-wide-grid heads, the shapes of the interface.
fn fits(a: &GigaamArgs, x: &[f32], t: usize) -> bool {
    let (w, hd) = (a.width, a.head_dim);
    let lin = |l: &GaLin, rows: usize, cols: usize| l.rows == rows && l.cols == cols && (l.bias.is_empty() || l.bias.len() == rows);
    let grid = |v: usize| v > 0 && v.is_multiple_of(32);
    if t == 0 || t > 1 << 20 || x.len() != t * w || !grid(w) || !grid(hd) || a.cos.len() < t * hd / 2 || a.sin.len() != a.cos.len() || a.stacks.is_empty() {
        return false;
    }
    a.stacks.iter().all(|st| {
        let (d, hw) = (st.dim, st.heads * hd);
        (st.fold == 1 || st.fold == 2)
            && grid(d)
            && st.heads > 0
            && st.residual_scales.len() == st.layers.len() + 1
            && lin(&st.proj, d, st.fold * w)
            && lin(&st.proj_t, st.fold * w, d)
            && st.layers.iter().all(|l| {
                let ff = |i: &GaLin, o: &GaLin| grid(i.rows) && lin(i, i.rows, d) && lin(o, d, i.rows);
                ff(&l.ff1_in, &l.ff1_out)
                    && l.in_norm.len() == d
                    && l.norm.len() == d
                    && lin(&l.qk, 2 * hw, d)
                    && lin(&l.vg, 3 * hw, d)
                    && lin(&l.attn_out, d, hw)
                    && l.conv.as_ref().is_none_or(|cv| {
                        cv.kernel > 0
                            && cv.kernel < 64
                            && cv.taps.len() == cv.kernel * d
                            && cv.bias.len() == d
                            && cv.mean_w.len() == d
                            && lin(&cv.inp, 2 * d, d)
                            && lin(&cv.out, d, d)
                    })
                    && l.ff2.as_ref().is_none_or(|(i, o)| ff(i, o))
            })
    })
}

/// A dispatch of one thread a float4 over `[rows][cols]`.
fn cells(enc: &metal::ComputeCommandEncoderRef, rows: usize, cols: usize) {
    let w = (cols / 4) as u64;
    enc.dispatch_threads(MTLSize::new(w, rows as u64, 1), MTLSize::new(w.min(256), 1, 1));
}

struct Ops<'a, 'c> {
    g: Prog<'c>,
    gp: &'a Pipes,
    /// the model's planes are still being filled
    fill: Option<&'a Fill<'a, 'a>>,
    /// the partial products of a product run in parts
    parts: Buffer,
}

impl Ops<'_, '_> {
    /// y `[n][rows]` = x `[n][cols]`·wᵀ; the bias is the caller's.
    fn mm(&mut self, l: &LinB, x: &Buffer, y: &Buffer, n: usize) -> Result<(), String> {
        if let Some(f) = self.fill
            && !f.whole(l.id)
        {
            // what is encoded so far runs while this plane fills
            self.g.rec.close();
            f.wait(l.id)?;
        }
        let groups = n.div_ceil(64) * l.rows.div_ceil(64);
        let steps = l.cols / 32;
        let parts = tg_min().div_ceil(groups).min(PARTS_MAX).min(steps);
        if parts <= 1 {
            return self.g.mm(false, x, [&l.w, &l.w, &l.w], y, &[proj(n, l.rows, l.cols, l.cols, l.rows, 0)], "gemm");
        }
        let items: Vec<EgMm> = (0..parts)
            .map(|i| {
                let (k0, k1) = (i * steps / parts * 32, (i + 1) * steps / parts * 32);
                EgMm { k: (k1 - k0) as u32, x_off: k0 as u32, w_off: k0 as u32, ..proj(n, l.rows, l.cols, l.cols, l.rows, i * n * l.rows) }
            })
            .collect();
        self.g.mm(false, x, [&l.w, &l.w, &l.w], &self.parts, &items, "gemm")?;
        let enc = self.g.rec.enc("rows");
        enc.set_compute_pipeline_state(&self.gp.sum);
        enc.set_buffer(0, Some(&self.parts), 0);
        enc.set_buffer(1, Some(y), 0);
        set_p(enc, 2, &GaP { rows: n as u32, cols: l.rows as u32, aux: parts as u32, ..Default::default() });
        cells(enc, n, l.rows);
        Ok(())
    }

    /// y += b a row, then SiLU when asked.
    fn bias_act(&mut self, y: &Buffer, b: Option<&Buffer>, rows: usize, cols: usize, silu: bool) {
        if b.is_none() && !silu {
            return;
        }
        let enc = self.g.rec.enc("rows");
        enc.set_compute_pipeline_state(&self.gp.bias_act);
        enc.set_buffer(0, Some(y), 0);
        enc.set_buffer(1, Some(b.unwrap_or(y)), 0);
        set_p(enc, 2, &GaP { rows: rows as u32, cols: cols as u32, flags: b.is_some() as u32 | (silu as u32) << 1, ..Default::default() });
        cells(enc, rows, cols);
    }

    /// The product with its bias and activation.
    fn lin(&mut self, l: &LinB, x: &Buffer, y: &Buffer, n: usize, silu: bool) -> Result<(), String> {
        self.mm(l, x, y, n)?;
        self.bias_act(y, l.b.as_ref(), n, l.rows, silu);
        Ok(())
    }

    /// dst = a + y (+ b a row); dst may be a.
    fn add(&mut self, a: &Buffer, y: &Buffer, b: Option<&Buffer>, dst: &Buffer, rows: usize, cols: usize) {
        let enc = self.g.rec.enc("rows");
        enc.set_compute_pipeline_state(&self.gp.add);
        enc.set_buffer(0, Some(a), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(b.unwrap_or(y)), 0);
        enc.set_buffer(3, Some(dst), 0);
        set_p(enc, 4, &GaP { rows: rows as u32, cols: cols as u32, flags: b.is_some() as u32, ..Default::default() });
        cells(enc, rows, cols);
    }

    /// dst = (res + mix·(s − res))·rsqrt(mean(.²) + eps)·w a row; `scalar`:
    /// that scale for every lane instead of w; `bypass`: += its scale · dst.
    #[allow(clippy::too_many_arguments)]
    fn rms(&mut self, res: &Buffer, s: &Buffer, w: Option<&Buffer>, dst: &Buffer, bypass: Option<(&Buffer, f32)>, rows: usize, cols: usize, eps: f32, mix: f32, scalar: f32) {
        let enc = self.g.rec.enc("rows");
        enc.set_compute_pipeline_state(&self.gp.rms);
        enc.set_buffer(0, Some(res), 0);
        enc.set_buffer(1, Some(s), 0);
        enc.set_buffer(2, Some(w.unwrap_or(res)), 0);
        enc.set_buffer(3, Some(dst), 0);
        enc.set_buffer(4, Some(bypass.map_or(dst, |b| b.0)), 0);
        let flags = w.is_none() as u32 | (bypass.is_some() as u32) << 1;
        set_p(enc, 5, &GaP { rows: rows as u32, cols: cols as u32, flags, aux: 0, f0: eps, f1: mix, f2: scalar, f3: bypass.map_or(0.0, |b| b.1) });
        enc.dispatch_thread_groups(MTLSize::new((rows as u64).div_ceil(4), 1, 1), MTLSize::new(128, 1, 1));
    }

    /// `ctx[:, h·hd..] = softmax(scale·Q_h·K_hᵀ)·V_h` over every item of
    /// `rounds`: q and k in the `[n][2·hw]` rows of `qk`, v the first third of
    /// the `[n][3·hw]` rows of `vg`.
    #[allow(clippy::too_many_arguments)]
    fn attention(&mut self, rounds: &Rounds, qk: &Buffer, vg: &Buffer, s: &Buffer, ctx: &Buffer, hw: usize, hd: usize, scale: f32) -> Result<(), String> {
        for round in rounds {
            if round.is_empty() {
                continue;
            }
            let s_items: Vec<EgMm> = round
                .iter()
                .map(|(a, it)| EgMm {
                    n: it.nq as u32,
                    m: it.nk as u32,
                    k: hd as u32,
                    ldx: (2 * hw) as u32,
                    ldw: (2 * hw) as u32,
                    ldy: a.lds,
                    x_off: (it.q0 * 2 * hw + it.h * hd) as u32,
                    w_off: (it.k0 * 2 * hw + hw + it.kvh * hd) as u32,
                    y_off: a.s_off,
                    alpha: scale,
                    wrows: it.nk as u32,
                    pad1: 0,
                })
                .collect();
            self.g.mm(false, qk, [qk, qk, qk], s, &s_items, "attn")?;
            let atts: Vec<EgAtt> = round.iter().map(|(a, _)| *a).collect();
            let at = self.g.arena.push(&atts)?;
            let maxq = round.iter().map(|(_, it)| it.nq).max().unwrap_or(1) as u64;
            let buf = self.g.arena.buf.clone();
            let p = self.g.p;
            let enc = self.g.rec.enc("attn");
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
                    ldw: (3 * hw) as u32,
                    ldy: hw as u32,
                    x_off: a.s_off,
                    w_off: (it.k0 * 3 * hw + it.kvh * hd) as u32,
                    y_off: (it.q0 * hw + it.h * hd) as u32,
                    alpha: 1.0,
                    wrows: it.nk as u32,
                    pad1: 0,
                })
                .collect();
            self.g.mm(true, s, [vg, vg, vg], ctx, &o_items, "attn")?;
        }
        Ok(())
    }
}

/// Item-table bytes of one forward: the products of every layer and stack,
/// and three tables an attention round.
fn arena_cap(plans: &[(Rounds, usize)]) -> usize {
    let (mm, att) = (std::mem::size_of::<EgMm>(), std::mem::size_of::<EgAtt>());
    plans.iter().map(|(rounds, layers)| (2 + layers * 9) * (PARTS_MAX * mm + 256) + layers * rounds.iter().map(|r| r.len() * (2 * mm + att) + 3 * 256).sum::<usize>()).sum::<usize>() + 4096
}

fn said_once(what: &str) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("gigaam: the Metal encoder declined ({what}); the per-product path runs");
    }
}

/// The encoder stacks over one segment `x` (`[t, width]`) in place;
/// `false` = declined, the caller runs its own path.
pub fn encode(a: &GigaamArgs, x: &mut [f32], t: usize) -> bool {
    let Some(c) = super::ctx() else { return false };
    if !fits(a, x, t) {
        return false;
    }
    let (gp, p) = match (pipes(c), egemma2::pipes(c)) {
        (Ok(gp), Ok(p)) => (gp, p),
        (Err(e), _) | (_, Err(e)) => {
            said_once(&e);
            return false;
        }
    };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    match run(c, gp, p, &mut state, a, x, t) {
        Ok(()) => true,
        Err(e) => {
            // the caller does not come back after a decline: the planes go
            state.model = None;
            said_once(&e);
            false
        }
    }
}

fn run(c: &Ctx, gp: &Pipes, p: &'static egemma2::Pipes, state: &mut State, a: &GigaamArgs, x: &mut [f32], t: usize) -> Result<(), String> {
    if state.model.as_ref().is_some_and(|m| m.key == a.key && m.stacks.len() == a.stacks.len()) {
        let State { model, scratch } = state;
        return forward(c, gp, p, model.as_ref().expect("checked above"), scratch, a, x, t, None);
    }
    // another model: its planes fill while this segment's first layers run
    state.model = None;
    let (model, fill) = planes(c, a)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, 8).min(fill.jobs.len().max(1));
    let done = std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| fill.work());
        }
        forward(c, gp, p, &model, &mut state.scratch, a, x, t, Some(&fill))
    });
    // every plane is whole here, whatever the forward said
    state.model = Some(model);
    done
}

#[allow(clippy::too_many_arguments)]
fn forward(c: &Ctx, gp: &Pipes, p: &'static egemma2::Pipes, model: &Model, sc: &mut Scratch, a: &GigaamArgs, x: &mut [f32], t: usize, fill: Option<&Fill>) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let (w, hd) = (a.width, a.head_dim);
    // every stack's rows and attention plan
    let mut s_need = 1;
    let plans: Vec<(Rounds, usize)> = a
        .stacks
        .iter()
        .map(|st| {
            let (rounds, need) = att_plan(&[(0, t.div_ceil(st.fold))], st.heads, st.heads, 0);
            s_need = s_need.max(need);
            (rounds, st.layers.len())
        })
        .collect();
    let rows_d = a.stacks.iter().map(|st| t.div_ceil(st.fold) * st.dim).max().unwrap_or(0);
    let rows_wide = a
        .stacks
        .iter()
        .map(|st| {
            let widest = st.layers.iter().map(|l| l.ff1_in.rows.max(l.vg.rows).max(l.conv.as_ref().map_or(0, |cv| cv.inp.rows)).max(l.ff2.as_ref().map_or(0, |f| f.0.rows))).max().unwrap_or(0);
            t.div_ceil(st.fold) * widest.max(st.fold * w)
        })
        .max()
        .unwrap_or(0);
    let rows_qk = a.stacks.iter().map(|st| t.div_ceil(st.fold) * 2 * st.heads * hd).max().unwrap_or(0);
    let max_dim = a.stacks.iter().map(|st| st.dim).max().unwrap_or(w);
    let cap = arena_cap(&plans);
    let bx = sc.get(c, S_X, (t + 1) * w * 4);
    let (bh, bs, btmp, bres, bbyp) = (sc.get(c, S_H, rows_d * 4), sc.get(c, S_S, rows_d * 4), sc.get(c, S_TMP, rows_d * 4), sc.get(c, S_RES, rows_d * 4), sc.get(c, S_BYPASS, rows_d * 4));
    let bwide = sc.get(c, S_WIDE, rows_wide * 4);
    let bqk = sc.get(c, S_QK, rows_qk * 4);
    let bsc = sc.get(c, S_SCORES, s_need * 4);
    let bmean = sc.get(c, S_MEAN, max_dim * 4);
    let bi = sc.get(c, S_ITEMS, cap);
    // a product in parts has under `tg_min` groups of 64×64, and its parts under twice that
    let bparts = sc.get(c, S_PARTS, 2 * tg_min() * 4096 * 4);
    if [&bx, &bh, &bs, &btmp, &bres, &bbyp, &bwide, &bqk, &bsc, &bmean, &bi, &bparts].iter().any(|b| b.contents().is_null()) {
        return Err("no device memory for the working buffers".into());
    }
    write(&bx, x);
    // the frame after the last stays zero: an odd segment's last pair
    unsafe {
        std::ptr::write_bytes((bx.contents() as *mut f32).add(t * w), 0, w);
    }
    let scale = (hd as f32).powf(-0.5);
    let mut o = Ops { g: Prog { p, rec: Rec::new(c), arena: Arena { buf: bi, cap, off: 0 }, eps: 1e-6 }, gp, fill, parts: bparts };
    let mut body = || -> Result<(), String> {
        for ((st, dev), (rounds, _)) in a.stacks.iter().zip(&model.stacks).zip(&plans) {
            let (n, d, hw) = (t.div_ceil(st.fold), st.dim, st.heads * hd);
            // rows of this stack: the frames, or pairs of them where they lie
            o.lin(&dev.proj, &bx, &bh, n, false)?;
            {
                let enc = o.g.rec.enc("rows");
                enc.set_compute_pipeline_state(&gp.scale2);
                enc.set_buffer(0, Some(&bh), 0);
                enc.set_buffer(1, Some(&bbyp), 0);
                set_p(enc, 2, &GaP { rows: n as u32, cols: d as u32, f0: st.input_scale, f1: st.residual_scales[0], ..Default::default() });
                cells(enc, n, d);
            }
            for (li, l) in dev.layers.iter().enumerate() {
                // the layer input stays in h; the stream is s
                o.lin(&l.ff1.0, &bh, &bwide, n, true)?;
                o.mm(&l.ff1.1, &bwide, &btmp, n)?;
                o.add(&bh, &btmp, l.ff1.1.b.as_ref(), &bs, n, d);
                // attention
                o.rms(&bh, &bh, Some(&l.in_norm), &bres, None, n, d, 1e-6, 1.0, 0.0);
                o.mm(&l.qk, &bres, &bqk, n)?;
                o.mm(&l.vg, &bs, &bwide, n)?;
                {
                    let enc = o.g.rec.enc("rows");
                    enc.set_compute_pipeline_state(&gp.prep);
                    enc.set_buffer(0, Some(&bqk), 0);
                    enc.set_buffer(1, Some(&bwide), 0);
                    enc.set_buffer(2, Some(l.qk.b.as_ref().unwrap_or(&bqk)), 0);
                    enc.set_buffer(3, Some(l.vg.b.as_ref().unwrap_or(&bwide)), 0);
                    enc.set_buffer(4, Some(&model.cos), 0);
                    enc.set_buffer(5, Some(&model.sin), 0);
                    let flags = l.qk.b.is_some() as u32 | (l.vg.b.is_some() as u32) << 1;
                    set_p(enc, 6, &GaP { rows: n as u32, cols: hw as u32, flags, aux: hd as u32, ..Default::default() });
                    cells(enc, n, hw);
                }
                o.attention(rounds, &bqk, &bwide, &bsc, &bres, hw, hd, scale)?;
                {
                    let enc = o.g.rec.enc("rows");
                    enc.set_compute_pipeline_state(&gp.gate);
                    enc.set_buffer(0, Some(&bres), 0);
                    enc.set_buffer(1, Some(&bwide), 0);
                    set_p(enc, 2, &GaP { rows: n as u32, cols: hw as u32, ..Default::default() });
                    cells(enc, n, hw);
                }
                o.mm(&l.attn_out, &bres, &btmp, n)?;
                o.add(&bs, &btmp, l.attn_out.b.as_ref(), &bs, n, d);
                if let Some(cv) = &l.conv {
                    o.mm(&cv.inp, &bs, &bwide, n)?;
                    let flags = cv.inp.b.is_some() as u32;
                    let prm = GaP { rows: n as u32, cols: d as u32, flags, aux: cv.kernel as u32, ..Default::default() };
                    let enc = o.g.rec.enc("rows");
                    enc.set_compute_pipeline_state(&gp.mean);
                    enc.set_buffer(0, Some(&bwide), 0);
                    enc.set_buffer(1, Some(cv.inp.b.as_ref().unwrap_or(&bwide)), 0);
                    enc.set_buffer(2, Some(&cv.mean_w), 0);
                    enc.set_buffer(3, Some(&bmean), 0);
                    set_p(enc, 4, &prm);
                    enc.dispatch_threads(MTLSize::new((d / 4) as u64, 1, 1), MTLSize::new(32, 1, 1));
                    enc.set_compute_pipeline_state(&gp.conv);
                    enc.set_buffer(0, Some(&bwide), 0);
                    enc.set_buffer(1, Some(cv.inp.b.as_ref().unwrap_or(&bwide)), 0);
                    enc.set_buffer(2, Some(&cv.taps), 0);
                    enc.set_buffer(3, Some(&cv.bias), 0);
                    enc.set_buffer(4, Some(&bmean), 0);
                    enc.set_buffer(5, Some(&bres), 0);
                    set_p(enc, 6, &prm);
                    cells(enc, n, d);
                    o.mm(&cv.out, &bres, &btmp, n)?;
                    o.add(&bs, &btmp, cv.out.b.as_ref(), &bs, n, d);
                }
                if let Some(ff2) = &l.ff2 {
                    o.lin(&ff2.0, &bs, &bwide, n, true)?;
                    o.mm(&ff2.1, &bwide, &btmp, n)?;
                    o.add(&bs, &btmp, ff2.1.b.as_ref(), &bs, n, d);
                }
                // h = norm(h + ¼ (stream − h)), and the bypass takes its share
                o.rms(&bh, &bs, Some(&l.norm), &bh, Some((&bbyp, st.residual_scales[li + 1])), n, d, 1e-6, 0.25, 0.0);
                o.g.rec.close();
            }
            // x += proj_t(bypass), the frames of an odd last pair's zero half left alone
            o.lin(&dev.proj_t, &bbyp, &bwide, n, false)?;
            o.add(&bx, &bwide, None, &bx, t, w);
        }
        // the output norm: one scale for every lane
        o.rms(&bx, &bx, None, &bx, None, t, w, a.out_eps * a.out_eps, 1.0, a.out_scale);
        Ok(())
    };
    let built = body();
    let Ops { g: Prog { rec, .. }, .. } = o;
    let host_ms = t0.elapsed().as_secs_f64() * 1e3;
    // what was encoded is waited for either way
    let times = rec.finish();
    built?;
    let times = times?;
    let out = unsafe { std::slice::from_raw_parts(bx.contents() as *const f32, t * w) };
    if !out.iter().all(|v| v.is_finite()) {
        return Err("the encoder came back with a value that is not finite".into());
    }
    x.copy_from_slice(out);
    if a.prof {
        let gpu: Vec<String> = times.iter().map(|(l, ms)| format!("{l} {ms:.1}")).collect();
        let first = fill.map_or(String::new(), |f| format!(", the weights filled behind it in {:.1} ms", *f.took.lock().unwrap_or_else(|e| e.into_inner())));
        eprintln!("gigaam (metal, resident): {t} frames, {:.1} ms (encode {host_ms:.1} ms, GPU {}{first})", t0.elapsed().as_secs_f64() * 1e3, gpu.join(", "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GaConv, GaLayer, GaStack, GaWeight};

    struct Plane(Vec<f32>);

    impl GaWeight for Plane {
        fn fill(&self, dst: &mut [f32]) {
            dst.copy_from_slice(&self.0);
        }
    }

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

    fn sigm(v: f32) -> f32 {
        1.0 / (1.0 + (-v).exp())
    }

    fn lin(l: &GaLin, x: &[f32], n: usize) -> Vec<f32> {
        let mut w = vec![0f32; l.rows * l.cols];
        l.weight.fill(&mut w);
        let mut y = vec![0f32; n * l.rows];
        for i in 0..n {
            for j in 0..l.rows {
                let s: f32 = (0..l.cols).map(|c| x[i * l.cols + c] * w[j * l.cols + c]).sum();
                y[i * l.rows + j] = s + l.bias.get(j).copied().unwrap_or(0.0);
            }
        }
        y
    }

    fn rms(x: &[f32], d: usize, w: &[f32], eps: f32) -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        for (r, o) in x.chunks_exact(d).zip(y.chunks_exact_mut(d)) {
            let inv = (r.iter().map(|v| v * v).sum::<f32>() / d as f32 + eps).sqrt().recip();
            for i in 0..d {
                o[i] = r[i] * inv * w[i];
            }
        }
        y
    }

    fn add(s: &mut [f32], y: &[f32]) {
        s.iter_mut().zip(y).for_each(|(a, b)| *a += b);
    }

    /// One layer on the host, as `gpu::GaLayer` words it.
    fn layer(l: &GaLayer, h: &[f32], n: usize, d: usize, nh: usize, a: &GigaamArgs) -> Vec<f32> {
        let (hd, hw) = (a.head_dim, nh * a.head_dim);
        let ff = |i: &GaLin, o: &GaLin, x: &[f32]| {
            let mut f = lin(i, x, n);
            f.iter_mut().for_each(|v| *v *= sigm(*v));
            lin(o, &f, n)
        };
        let mut s = h.to_vec();
        add(&mut s, &ff(&l.ff1_in, &l.ff1_out, h));
        let mut qk = lin(&l.qk, &rms(h, d, l.in_norm, 1e-6), n);
        let vg = lin(&l.vg, &s, n);
        for (r, row) in qk.chunks_exact_mut(hw).enumerate() {
            let pos = r / 2;
            for (i, pair) in row.chunks_exact_mut(2).enumerate() {
                let (c, sn) = (a.cos[pos * hd / 2 + i % (hd / 2)], a.sin[pos * hd / 2 + i % (hd / 2)]);
                (pair[0], pair[1]) = (pair[0] * c - pair[1] * sn, pair[0] * sn + pair[1] * c);
            }
        }
        let scale = (hd as f32).powf(-0.5);
        let mut ctx = vec![0f32; n * hw];
        for hh in 0..nh {
            for q in 0..n {
                let qr = &qk[q * 2 * hw + hh * hd..][..hd];
                let sc: Vec<f32> = (0..n).map(|k| qr.iter().zip(&qk[k * 2 * hw + hw + hh * hd..][..hd]).map(|(x, y)| x * y).sum::<f32>() * scale).collect();
                let mx = sc.iter().fold(f32::MIN, |m, &v| m.max(v));
                let e: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
                let den: f32 = e.iter().sum();
                for j in 0..hd {
                    let col = hh * hd + j;
                    let o: f32 = (0..n).map(|k| e[k] * vg[k * 3 * hw + col] * sigm(vg[k * 3 * hw + hw + col])).sum();
                    ctx[q * hw + col] = o / den * sigm(vg[q * 3 * hw + 2 * hw + col]);
                }
            }
        }
        add(&mut s, &lin(&l.attn_out, &ctx, n));
        if let Some(cv) = &l.conv {
            let cg = lin(&cv.inp, &s, n);
            let side = (cv.kernel - 1) / 2;
            let mut z = vec![0f32; n * d];
            for c in 0..d {
                let mean = (0..n).map(|r| cg[r * 2 * d + c]).sum::<f32>() / n as f32 * cv.mean_w[c];
                for r in 0..n {
                    let conv: f32 = (0..cv.kernel).filter(|j| r + j >= side && r + j - side < n).map(|j| cv.taps[j * d + c] * cg[(r + j - side) * 2 * d + c]).sum();
                    let v = (conv + cv.bias[c] + mean) * sigm(cg[r * 2 * d + d + c]);
                    z[r * d + c] = v * sigm(v);
                }
            }
            add(&mut s, &lin(&cv.out, &z, n));
        }
        if let Some((i, o)) = &l.ff2 {
            let f = ff(i, o, &s);
            add(&mut s, &f);
        }
        let mixed: Vec<f32> = h.iter().zip(&s).map(|(r, v)| r + 0.25 * (v - r)).collect();
        rms(&mixed, d, l.norm, 1e-6)
    }

    fn reference(a: &GigaamArgs, x: &[f32], t: usize) -> Vec<f32> {
        let w = a.width;
        let mut x = x.to_vec();
        for st in &a.stacks {
            let n = t.div_ceil(st.fold);
            let mut rows = x.clone();
            rows.resize(n * st.fold * w, 0.0);
            let mut h = lin(&st.proj, &rows, n);
            let mut bypass: Vec<f32> = h.iter().map(|v| v * st.residual_scales[0]).collect();
            h.iter_mut().for_each(|v| *v *= st.input_scale);
            for (i, l) in st.layers.iter().enumerate() {
                h = layer(l, &h, n, st.dim, st.heads, a);
                bypass.iter_mut().zip(&h).for_each(|(b, v)| *b += st.residual_scales[i + 1] * v);
            }
            let back = lin(&st.proj_t, &bypass, n);
            x.iter_mut().zip(&back).for_each(|(v, b)| *v += b);
        }
        for row in x.chunks_exact_mut(w) {
            let k = a.out_scale / (row.iter().map(|v| v * v).sum::<f32>() / w as f32 + a.out_eps * a.out_eps).sqrt();
            row.iter_mut().for_each(|v| *v *= k);
        }
        x
    }

    /// A small encoder of every part — a stack with the convolution module and
    /// the second feed-forward, a stack of frame pairs without, biases on and
    /// off — against the host, on an odd and a tile-crossing segment.
    #[test]
    fn gigaam_matches_host() {
        if super::super::ctx().is_none() {
            return;
        }
        let (w, hd) = (32usize, 32usize);
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        // (fold, dim, heads, layers, conv kernel)
        let shapes = [(1usize, 64usize, 2usize, 1usize, 5usize), (2, 96, 1, 2, 0), (1, 32, 1, 1, 3)];
        let mut planes: Vec<Plane> = Vec::new();
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        // per product: (rows, cols, plane, bias vec or none)
        let mut lins: Vec<(usize, usize, usize, Option<usize>)> = Vec::new();
        let mut mk = |rows: usize, cols: usize, bias: bool, planes: &mut Vec<Plane>, vecs: &mut Vec<Vec<f32>>| {
            planes.push(Plane(rnd(&mut st, rows * cols, 2.0 / (cols as f32).sqrt())));
            let b = bias.then(|| {
                vecs.push(rnd(&mut st, rows, 0.2));
                vecs.len() - 1
            });
            lins.push((rows, cols, planes.len() - 1, b));
        };
        for &(fold, d, nh, layers, kernel) in &shapes {
            let hw = nh * hd;
            mk(d, fold * w, false, &mut planes, &mut vecs);
            mk(fold * w, d, false, &mut planes, &mut vecs);
            for li in 0..layers {
                mk(4 * d, d, true, &mut planes, &mut vecs);
                mk(d, 4 * d, li == 0, &mut planes, &mut vecs);
                mk(2 * hw, d, true, &mut planes, &mut vecs);
                mk(3 * hw, d, li == 0, &mut planes, &mut vecs);
                mk(d, hw, true, &mut planes, &mut vecs);
                if kernel > 0 {
                    mk(2 * d, d, true, &mut planes, &mut vecs);
                    mk(d, d, true, &mut planes, &mut vecs);
                    mk(4 * d, d, true, &mut planes, &mut vecs);
                    mk(d, 4 * d, true, &mut planes, &mut vecs);
                }
            }
        }
        let mut st2 = 0x1234_5678_9abc_def1u64;
        // per layer: in_norm, norm, then taps, conv bias, mean_w where there is a convolution
        let mut extra: Vec<Vec<f32>> = Vec::new();
        let mut scales: Vec<Vec<f32>> = Vec::new();
        for &(_, d, _, layers, kernel) in &shapes {
            scales.push(rnd(&mut st2, layers + 1, 1.0).iter().map(|v| 0.5 + v).collect());
            for _ in 0..layers {
                extra.push(rnd(&mut st2, d, 1.0).iter().map(|v| 1.0 + v).collect());
                extra.push(rnd(&mut st2, d, 1.0).iter().map(|v| 1.0 + v).collect());
                if kernel > 0 {
                    extra.push(rnd(&mut st2, kernel * d, 1.0));
                    extra.push(rnd(&mut st2, d, 0.2));
                    extra.push(rnd(&mut st2, d, 1.0));
                }
            }
        }
        let positions = 128usize;
        let (mut cos, mut sin) = (Vec::new(), Vec::new());
        for p in 0..positions {
            for i in 0..hd / 2 {
                let ang = p as f32 / 10000f32.powf(2.0 * i as f32 / hd as f32);
                cos.push(ang.cos());
                sin.push(ang.sin());
            }
        }
        let (mut at, mut ex) = (0usize, 0usize);
        let mut next = || {
            let (rows, cols, plane, bias) = lins[at];
            at += 1;
            GaLin { rows, cols, weight: &planes[plane], bias: bias.map_or(&[][..], |b| &vecs[b]) }
        };
        let mut stacks = Vec::new();
        for (si, &(fold, d, nh, layers, kernel)) in shapes.iter().enumerate() {
            let (proj, proj_t) = (next(), next());
            let mut ls = Vec::new();
            for _ in 0..layers {
                let (ff1_in, ff1_out, qk, vg, attn_out) = (next(), next(), next(), next(), next());
                let (in_norm, norm) = (&extra[ex][..], &extra[ex + 1][..]);
                ex += 2;
                let (conv, ff2) = if kernel > 0 {
                    let (inp, out) = (next(), next());
                    let cv = GaConv { inp, taps: &extra[ex], bias: &extra[ex + 1], kernel, mean_w: &extra[ex + 2], out };
                    ex += 3;
                    (Some(cv), Some((next(), next())))
                } else {
                    (None, None)
                };
                ls.push(GaLayer { ff1_in, ff1_out, in_norm, qk, vg, attn_out, conv, ff2, norm });
            }
            stacks.push(GaStack { fold, dim: d, heads: nh, proj, proj_t, residual_scales: &scales[si], input_scale: 0.7, layers: ls });
        }
        let args = GigaamArgs { key: 0x6a61_7465_7374, width: w, head_dim: hd, stacks, cos: &cos, sin: &sin, out_scale: 1.3, out_eps: 0.05, prof: false, front: None };
        let mut st3 = 0x0f1e_2d3c_4b5a_6978u64;
        // the first call fills the planes behind the forward, the others find them
        for t in [37usize, 70, 1, 64] {
            let x0 = rnd(&mut st3, t * w, 2.0);
            let mut got = x0.clone();
            // a runner without the matrix kernels declines: the host path runs there
            if !encode(&args, &mut got, t) {
                return;
            }
            let want = reference(&args, &x0, t);
            let scale = want.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let err = got.iter().zip(&want).fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            assert!(err / scale < 2e-5, "t {t}: {err} of {scale}");
        }
    }
}
