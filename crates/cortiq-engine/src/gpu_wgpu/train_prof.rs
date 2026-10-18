//! Opt-in profiler of the layer-streamed trainer (`CMF_NVG_PROF`).
//!
//! * `1`: GPU timestamps around every dispatch, GEMM, copy and clear the
//!   trainer records during a step (written inside passes and encoders, so
//!   the submission structure is the one an unprofiled step has), each
//!   attributed to the phase the trainer is in ([`phase`]) and to its
//!   kernel; host time per phase, host waits, and the submits, buffers,
//!   bind groups and queue writes one step creates. Busy time is the union
//!   of the timestamped intervals, idle time the gaps between them.
//! * `2`: the same counts, with phase times from the host clock and a device
//!   drain at every phase change (no timestamps; host and device serialize,
//!   so the phases add up to the step). Also the fallback of `1` on a device
//!   without timestamps inside passes and encoders.
//!
//! Off (unset or `0`), every hook is one relaxed load of [`on`], false.
//! `CMF_NVG_PROF_SKIP` (default 2): leading steps left out of the average.

use super::Ctx;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Profiling mode from `CMF_NVG_PROF` (0 off, 1 timestamps, 2 drains).
pub fn mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| std::env::var("CMF_NVG_PROF").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Inside a profiled step (always false with profiling off).
#[inline]
pub fn on() -> bool {
    ACTIVE.load(Relaxed)
}

/// Per-step counters of the objects the step creates.
#[derive(Clone, Copy)]
pub enum Cnt {
    Buf = 0,
    BindGroup,
    Write,
    WriteBytes,
    Wait,
    WaitNs,
    Disp,
    Mm,
    Pass,
    Copy,
    SubmitNs,
}
const NCNT: usize = 11;
const CNT_NAMES: [&str; NCNT] = ["buffers", "bind_groups", "queue_writes", "queue_write_bytes", "host_waits", "host_wait_ns", "dispatches", "gemms", "passes", "copies_clears", "host_submit_ns"];
static CNT: [AtomicU64; NCNT] = [const { AtomicU64::new(0) }; NCNT];

#[inline]
pub fn count(k: Cnt, n: u64) {
    if on() {
        CNT[k as usize].fetch_add(n, Relaxed);
    }
}

// ───────────────────────────── kernel names ─────────────────────────────

static NAMES: OnceLock<Mutex<HashMap<usize, &'static str>>> = OnceLock::new();

/// Remember a trainer pipeline's entry point (pipelines are leaked statics,
/// so the address is a stable key).
pub fn name_pipe(p: &wgpu::ComputePipeline, name: &'static str) {
    if mode() == 0 {
        return;
    }
    NAMES.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap().insert(p as *const _ as usize, name);
}

pub fn pipe_name(p: &wgpu::ComputePipeline) -> &'static str {
    NAMES.get().and_then(|m| m.lock().unwrap().get(&(p as *const _ as usize)).copied()).unwrap_or("?")
}

/// `tag.name` as a static string (interned; profiling only).
pub fn tagged(tag: &'static str, name: &'static str) -> &'static str {
    if tag.is_empty() {
        return name;
    }
    static T: OnceLock<Mutex<HashMap<(&'static str, &'static str), &'static str>>> = OnceLock::new();
    let mut m = T.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    m.entry((tag, name)).or_insert_with(|| Box::leak(format!("{tag}.{name}").into_boxed_str()))
}

// ───────────────────────────── state ─────────────────────────────

const PER_SET: u32 = 4096; // wgpu's QUERY_SET_MAX_QUERIES
const SETS: u32 = 64;

struct Ts {
    sets: Vec<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    stage: wgpu::Buffer,
    period_ns: f64,
}

struct Rec {
    phase: &'static str,
    kern: &'static str,
    q0: u32,
    q1: u32,
    mnk: [u32; 3],
}

#[derive(Default)]
struct Acc {
    steps: u64,
    wall: f64,
    busy: f64,
    span: f64,
    submits: f64,
    cnt: [f64; NCNT],
    /// (phase, kernel) -> (GPU s, calls, flops)
    pk: HashMap<(&'static str, &'static str), (f64, f64, f64)>,
    /// phase -> (host s, idle-before s, inside submit s)
    ph: HashMap<&'static str, (f64, f64, f64)>,
    /// GEMM shape -> (GPU s, calls)
    shapes: HashMap<[u32; 3], (f64, f64)>,
    dropped: u64,
}

struct State {
    stack: Vec<&'static str>,
    last: Instant,
    t0: Instant,
    submits0: u64,
    host: HashMap<&'static str, f64>,
    submit: HashMap<&'static str, f64>,
    recs: Vec<Rec>,
    nq: u32,
    dropped: u64,
    seen: u64,
    acc: Acc,
}

fn state() -> &'static Mutex<State> {
    static S: OnceLock<Mutex<State>> = OnceLock::new();
    S.get_or_init(|| {
        Mutex::new(State {
            stack: Vec::new(),
            last: Instant::now(),
            t0: Instant::now(),
            submits0: 0,
            host: HashMap::new(),
            submit: HashMap::new(),
            recs: Vec::new(),
            nq: 0,
            dropped: 0,
            seen: 0,
            acc: Acc::default(),
        })
    })
}

fn ts(c: &Ctx) -> Option<&'static Ts> {
    static T: OnceLock<Option<Ts>> = OnceLock::new();
    T.get_or_init(|| {
        let need = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if mode() != 1 || !c.device.features().contains(need) {
            if mode() == 1 {
                eprintln!("[nvgprof] no timestamps inside passes/encoders on this device: falling back to drains (mode 2)");
            }
            return None;
        }
        let sets = (0..SETS)
            .map(|_| c.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("nvgprof"), ty: wgpu::QueryType::Timestamp, count: PER_SET }))
            .collect();
        let size = (SETS * PER_SET) as u64 * 8;
        let resolve = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nvgprof_resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nvgprof_stage"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Ts { sets, resolve, stage, period_ns: c.queue.get_timestamp_period() as f64 })
    })
    .as_ref()
}

fn drains() -> bool {
    mode() == 2 || super::ctx().is_some_and(|c| ts(c).is_none())
}

/// Attribute host time since the last switch to the phase on top.
fn switch(s: &mut State) {
    if drains()
        && let Some(c) = super::ctx()
    {
        let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    }
    let now = Instant::now();
    let top = *s.stack.last().unwrap_or(&"step");
    *s.host.entry(top).or_insert(0.0) += (now - s.last).as_secs_f64();
    s.last = now;
}

/// Phase guard: everything recorded until it drops belongs to `name`
/// (nested phases override; host time is exclusive).
pub struct Phase(bool);

pub fn phase(name: &'static str) -> Phase {
    if !on() {
        return Phase(false);
    }
    let mut s = state().lock().unwrap();
    switch(&mut s);
    s.stack.push(name);
    Phase(true)
}

impl Drop for Phase {
    fn drop(&mut self) {
        if self.0 && on() {
            let mut s = state().lock().unwrap();
            switch(&mut s);
            s.stack.pop();
        }
    }
}

/// Start a profiled step (no-op with profiling off).
pub fn step_begin() {
    if mode() == 0 {
        return;
    }
    let mut s = state().lock().unwrap();
    s.stack.clear();
    s.host.clear();
    s.submit.clear();
    s.recs.clear();
    s.nq = 0;
    s.dropped = 0;
    for x in &CNT {
        x.store(0, Relaxed);
    }
    s.submits0 = super::SUBMITS.load(Relaxed);
    s.t0 = Instant::now();
    s.last = s.t0;
    ACTIVE.store(true, Relaxed);
}

/// Host time inside one `queue.submit`, attributed to the current phase.
pub fn submit_ns(ns: u64) {
    if !on() {
        return;
    }
    CNT[Cnt::SubmitNs as usize].fetch_add(ns, Relaxed);
    let mut s = state().lock().unwrap();
    let top = *s.stack.last().unwrap_or(&"step");
    *s.submit.entry(top).or_insert(0.0) += ns as f64 * 1e-9;
}

// ───────────────────────────── timestamps ─────────────────────────────

fn alloc(c: &Ctx) -> Option<(&'static wgpu::QuerySet, u32, u32)> {
    let t = ts(c)?;
    let mut s = state().lock().unwrap();
    if s.nq >= SETS * PER_SET {
        s.dropped += 1;
        return None;
    }
    let q = s.nq;
    s.nq += 1;
    Some((&t.sets[(q / PER_SET) as usize], q % PER_SET, q))
}

fn push(q0: u32, q1: u32, kern: &'static str, mnk: [u32; 3]) {
    let mut s = state().lock().unwrap();
    let phase = *s.stack.last().unwrap_or(&"step");
    s.recs.push(Rec { phase, kern, q0, q1, mnk });
}

/// Timestamp marks inside one compute pass or encoder.
pub struct Marks(Option<u32>);

/// Open marks at the start of a pass (writes the first timestamp).
pub fn pass_begin(c: &Ctx, ps: &mut wgpu::ComputePass) -> Marks {
    if !on() {
        return Marks(None);
    }
    CNT[Cnt::Pass as usize].fetch_add(1, Relaxed);
    match alloc(c) {
        Some((qs, i, q)) => {
            ps.write_timestamp(qs, i);
            Marks(Some(q))
        }
        None => Marks(None),
    }
}

/// After one dispatch: its interval runs from the previous mark to here.
pub fn pass_mark(c: &Ctx, ps: &mut wgpu::ComputePass, m: &mut Marks, kern: &'static str, mnk: [u32; 3]) {
    if !on() {
        return;
    }
    CNT[if mnk[0] > 0 { Cnt::Mm } else { Cnt::Disp } as usize].fetch_add(1, Relaxed);
    let Some(q0) = m.0 else { return };
    match alloc(c) {
        Some((qs, i, q)) => {
            ps.write_timestamp(qs, i);
            push(q0, q, kern, mnk);
            m.0 = Some(q);
        }
        None => m.0 = None,
    }
}

pub fn enc_begin(c: &Ctx, enc: &mut wgpu::CommandEncoder) -> Marks {
    if !on() {
        return Marks(None);
    }
    match alloc(c) {
        Some((qs, i, q)) => {
            enc.write_timestamp(qs, i);
            Marks(Some(q))
        }
        None => Marks(None),
    }
}

/// After one copy or clear recorded on the encoder.
pub fn enc_mark(c: &Ctx, enc: &mut wgpu::CommandEncoder, m: &mut Marks, kern: &'static str) {
    if !on() {
        return;
    }
    CNT[Cnt::Copy as usize].fetch_add(1, Relaxed);
    let Some(q0) = m.0 else { return };
    match alloc(c) {
        Some((qs, i, q)) => {
            enc.write_timestamp(qs, i);
            push(q0, q, kern, [0; 3]);
            m.0 = Some(q);
        }
        None => m.0 = None,
    }
}

// ───────────────────────────── step end / report ─────────────────────────────

fn read_ts(c: &Ctx, t: &Ts, nq: u32) -> Option<Vec<u64>> {
    if nq == 0 {
        return Some(Vec::new());
    }
    let mut enc = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nvgprof_resolve") });
    let mut q = 0;
    while q < nq {
        let set = q / PER_SET;
        let n = (nq - q).min(PER_SET);
        enc.resolve_query_set(&t.sets[set as usize], 0..n, &t.resolve, (set * PER_SET) as u64 * 8);
        q += n;
    }
    enc.copy_buffer_to_buffer(&t.resolve, 0, &t.stage, 0, nq as u64 * 8);
    c.queue.submit(Some(enc.finish()));
    let slice = t.stage.slice(..nq as u64 * 8);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    c.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    let v = bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range().ok()?).to_vec();
    t.stage.unmap();
    Some(v)
}

/// Close the step: resolve its timestamps, print a one-line summary and
/// fold it into the running average (no-op with profiling off).
pub fn step_end() {
    if !on() {
        return;
    }
    let Some(c) = super::ctx() else { return };
    {
        let mut s = state().lock().unwrap();
        switch(&mut s);
    }
    ACTIVE.store(false, Relaxed);
    let mut s = state().lock().unwrap();
    let wall = s.t0.elapsed().as_secs_f64();
    let submits = (super::SUBMITS.load(Relaxed) - s.submits0) as f64;
    let cnt: Vec<f64> = CNT.iter().map(|x| x.load(Relaxed) as f64).collect();
    let mut pk: HashMap<(&'static str, &'static str), (f64, f64, f64)> = HashMap::new();
    let mut idle: HashMap<&'static str, f64> = HashMap::new();
    let mut shapes: HashMap<[u32; 3], (f64, f64)> = HashMap::new();
    let (mut busy, mut span) = (0f64, 0f64);
    if let Some(t) = ts(c) {
        let tsv = read_ts(c, t, s.nq).unwrap_or_default();
        let ns = |q: u32| tsv.get(q as usize).copied().unwrap_or(0) as f64 * t.period_ns * 1e-9;
        let mut iv: Vec<(f64, f64, usize)> = s.recs.iter().enumerate().map(|(i, r)| (ns(r.q0), ns(r.q1).max(ns(r.q0)), i)).collect();
        iv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if let (Some(f), Some(l)) = (iv.first(), iv.iter().map(|x| x.1).reduce(f64::max)) {
            span = l - f.0;
            let mut end = f.0;
            for &(a, b, i) in &iv {
                if a > end {
                    *idle.entry(s.recs[i].phase).or_insert(0.0) += a - end;
                }
                busy += (b - a.max(end)).max(0.0);
                end = end.max(b);
            }
        }
        for r in &s.recs {
            let d = (ns(r.q1) - ns(r.q0)).max(0.0);
            let fl = 2.0 * r.mnk[0] as f64 * r.mnk[1] as f64 * r.mnk[2] as f64;
            let e = pk.entry((r.phase, r.kern)).or_insert((0.0, 0.0, 0.0));
            e.0 += d;
            e.1 += 1.0;
            e.2 += fl;
            if r.mnk[0] > 0 {
                let e = shapes.entry(r.mnk).or_insert((0.0, 0.0));
                e.0 += d;
                e.1 += 1.0;
            }
        }
    }
    s.seen += 1;
    let gpu_sum: f64 = pk.values().map(|v| v.0).sum();
    eprintln!(
        "\n[nvgprof] step {} wall {:.3} s  gpu busy {:.3} s ({:.1}%)  span {:.3} s  kernels {:.3} s  submits {}  dispatches {}  gemms {}  passes {}  copies {}  buffers {}  bind groups {}  writes {} ({:.1} MB)  waits {} ({:.3} s)  in submit {:.3} s  queries {}{}",
        s.seen,
        wall,
        busy,
        100.0 * busy / wall.max(1e-9),
        span,
        gpu_sum,
        submits,
        cnt[Cnt::Disp as usize],
        cnt[Cnt::Mm as usize],
        cnt[Cnt::Pass as usize],
        cnt[Cnt::Copy as usize],
        cnt[Cnt::Buf as usize],
        cnt[Cnt::BindGroup as usize],
        cnt[Cnt::Write as usize],
        cnt[Cnt::WriteBytes as usize] / 1e6,
        cnt[Cnt::Wait as usize],
        cnt[Cnt::WaitNs as usize] * 1e-9,
        cnt[Cnt::SubmitNs as usize] * 1e-9,
        s.nq,
        if s.dropped > 0 { format!("  DROPPED {}", s.dropped) } else { String::new() }
    );
    let skip: u64 = std::env::var("CMF_NVG_PROF_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    if s.seen <= skip {
        return;
    }
    let host = std::mem::take(&mut s.host);
    let subm = std::mem::take(&mut s.submit);
    let dropped = s.dropped;
    let a = &mut s.acc;
    a.steps += 1;
    a.wall += wall;
    a.busy += busy;
    a.span += span;
    a.submits += submits;
    a.dropped += dropped;
    for (i, v) in cnt.iter().enumerate() {
        a.cnt[i] += v;
    }
    for (k, v) in pk {
        let e = a.pk.entry(k).or_insert((0.0, 0.0, 0.0));
        e.0 += v.0;
        e.1 += v.1;
        e.2 += v.2;
    }
    for (k, v) in host {
        a.ph.entry(k).or_insert((0.0, 0.0, 0.0)).0 += v;
    }
    for (k, v) in idle {
        a.ph.entry(k).or_insert((0.0, 0.0, 0.0)).1 += v;
    }
    for (k, v) in subm {
        a.ph.entry(k).or_insert((0.0, 0.0, 0.0)).2 += v;
    }
    for (k, v) in shapes {
        let e = a.shapes.entry(k).or_insert((0.0, 0.0));
        e.0 += v.0;
        e.1 += v.1;
    }
}

/// Family of a kernel label for the roll-up table.
fn family(kern: &str) -> &'static str {
    let (sign, k) = match kern.strip_prefix("sign.") {
        Some(k) => (true, k),
        None => (false, kern),
    };
    if sign {
        return match k {
            "mm" => "gemm.sign_grad",
            "cvt_t" => "sign.stage_t",
            _ => "sign.update",
        };
    }
    match k {
        "mm" => "gemm",
        "expand" | "expand_n" | "expand_t" => "blr.expand",
        "cvt" | "cvt_col" | "cvt_cols" => "stage.cvt",
        "scale_acc" => "blr.scale_acc",
        "colsum_prod" | "colsum_part" | "colsum_fin" => "colsum(all)",
        "colsum_fused" | "colsum_fused_t" => "colsum+eltwise",
        "rms_fwd" | "rms_bwd" => "norm",
        "rms_dw" => "norm.dw",
        "add" | "acc_add" | "swiglu_fwd" | "swiglu_bwd" | "gate_fwd" | "gate_bwd" | "split_qg" => "eltwise",
        "qkr_fwd" | "qkr_bwd" => "attn.qk_norm_rope",
        "qkr_dw" => "attn.qkr_dw",
        "attn_fwd" | "att_softmax" => "attn.core_fwd",
        "att_mm_s" | "att_mm_o" | "att_mm_d" => "attn.mm",
        "att_ds" => "attn.core_bwd",
        "attn_dsum" | "attn_dkv" | "attn_dq" => "attn.core_bwd",
        "gdn.conv_fwd" | "gdn.qk_prep" | "gdn.gate_prep" => "gdn.prep_fwd",
        "gdn.rec_fwd" => "gdn.rec_fwd",
        "gdn.rec_bwd" => "gdn.rec_bwd",
        "gdn.qk_bwd" | "gdn.conv_bwd" => "gdn.prep_bwd",
        "dense_fwd" | "dense_bwd" | "dense_fwd_t" | "dense_bwd_t" => "dense(a,b)",
        "copy" | "clear" | "readback" => "copy/clear",
        "kl_rows" => "head.kl_rows",
        "adam" | "sumsq" => "optimizer",
        _ => "other",
    }
}

/// Print the average over the profiled steps (after the skipped ones).
pub fn report() {
    if mode() == 0 {
        return;
    }
    let s = state().lock().unwrap();
    let a = &s.acc;
    if a.steps == 0 {
        eprintln!("[nvgprof] no steps past the skipped ones");
        return;
    }
    let n = a.steps as f64;
    let wall = a.wall / n;
    let busy = a.busy / n;
    let gpu_sum: f64 = a.pk.values().map(|v| v.0).sum::<f64>() / n;
    eprintln!("\n[nvgprof] ===== average over {} steps (mode {}) =====", a.steps, mode());
    eprintln!(
        "[nvgprof] wall {:.3} s/step | GPU busy {:.3} s ({:.1}% of wall) | idle {:.3} s | span {:.3} s | sum of kernel intervals {:.3} s",
        wall,
        busy,
        100.0 * busy / wall,
        wall - busy,
        a.span / n,
        gpu_sum
    );
    let c: Vec<String> = (0..NCNT).map(|i| format!("{} {:.0}", CNT_NAMES[i], a.cnt[i] / n)).collect();
    eprintln!("[nvgprof] per step: submits {:.0} | {} | dropped queries {}", a.submits / n, c.join(" | "), a.dropped);
    let host: f64 = a.ph.values().map(|v| v.0).sum::<f64>() / n;
    let (wait, sub) = (a.cnt[Cnt::WaitNs as usize] * 1e-9 / n, a.cnt[Cnt::SubmitNs as usize] * 1e-9 / n);
    eprintln!(
        "[nvgprof] host: {:.3} s in phases = {:.3} s in readback waits + {:.3} s inside queue.submit + {:.3} s other (encoding, buffer/bind group creation, host loops)",
        host,
        wait,
        sub,
        host - wait - sub
    );
    // phases
    let mut ph: HashMap<&'static str, (f64, f64, f64, f64)> = HashMap::new();
    for (&(p, _), v) in &a.pk {
        ph.entry(p).or_insert((0.0, 0.0, 0.0, 0.0)).0 += v.0 / n;
    }
    for (&p, v) in &a.ph {
        let e = ph.entry(p).or_insert((0.0, 0.0, 0.0, 0.0));
        e.1 = v.0 / n;
        e.2 = v.1 / n;
        e.3 = v.2 / n;
    }
    let mut rows: Vec<_> = ph.into_iter().collect();
    rows.sort_by(|x, y| (y.1.0 + y.1.2).max(y.1.1).partial_cmp(&(x.1.0 + x.1.2).max(x.1.1)).unwrap());
    eprintln!("[nvgprof] {:<20} {:>9} {:>7} {:>9} {:>9} {:>10}", "phase", "gpu s", "% wall", "idle s", "host s", "submit s");
    for (p, (g, h, i, sb)) in &rows {
        eprintln!("[nvgprof] {:<20} {:>9.3} {:>6.1}% {:>9.3} {:>9.3} {:>10.3}", p, g, 100.0 * g / wall, i, h, sb);
    }
    // families
    let mut fam: HashMap<&'static str, (f64, f64, f64)> = HashMap::new();
    for (&(_, k), v) in &a.pk {
        let e = fam.entry(family(k)).or_insert((0.0, 0.0, 0.0));
        e.0 += v.0 / n;
        e.1 += v.1 / n;
        e.2 += v.2 / n;
    }
    let mut rows: Vec<_> = fam.into_iter().collect();
    rows.sort_by(|x, y| y.1.0.partial_cmp(&x.1.0).unwrap());
    eprintln!("[nvgprof] {:<20} {:>9} {:>7} {:>9} {:>10} {:>9}", "family", "gpu s", "% wall", "calls", "ms/call", "TFLOP/s");
    for (f, (g, k, fl)) in &rows {
        let tf = if *fl > 0.0 { format!("{:.1}", fl / g / 1e12) } else { String::new() };
        eprintln!("[nvgprof] {:<20} {:>9.3} {:>6.1}% {:>9.0} {:>10.3} {:>9}", f, g, 100.0 * g / wall, k, 1e3 * g / k.max(1.0), tf);
    }
    // phase x kernel
    let mut rows: Vec<_> = a.pk.iter().map(|(&(p, k), v)| (p, k, v.0 / n, v.1 / n, v.2 / n)).collect();
    rows.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
    eprintln!("[nvgprof] {:<14} {:<18} {:>9} {:>7} {:>8} {:>10} {:>9}", "phase", "kernel", "gpu s", "% wall", "calls", "ms/call", "TFLOP/s");
    for (p, k, g, cl, fl) in rows.iter().take(80) {
        let tf = if *fl > 0.0 { format!("{:.1}", fl / g / 1e12) } else { String::new() };
        eprintln!("[nvgprof] {:<14} {:<18} {:>9.4} {:>6.2}% {:>8.0} {:>10.4} {:>9}", p, k, g, 100.0 * g / wall, cl, 1e3 * g / cl.max(1.0), tf);
    }
    // GEMM shapes
    let mut rows: Vec<_> = a.shapes.iter().map(|(k, v)| (*k, v.0 / n, v.1 / n)).collect();
    rows.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    eprintln!("[nvgprof] {:>6} {:>6} {:>6} {:>8} {:>9} {:>10} {:>9}", "M", "N", "K", "calls", "gpu s", "ms/call", "TFLOP/s");
    for (k, g, cl) in rows.iter().take(30) {
        let fl = 2.0 * k[0] as f64 * k[1] as f64 * k[2] as f64 * cl;
        eprintln!("[nvgprof] {:>6} {:>6} {:>6} {:>8.0} {:>9.4} {:>10.4} {:>9.1}", k[0], k[1], k[2], cl, g, 1e3 * g / cl.max(1.0), fl / g / 1e12);
    }
}
