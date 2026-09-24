//! CPU share of the DiT GEMMs (WP3, M8): the M4's GPU issues its matrix
//! instructions at ~3.55 TF/s and the chain's GEMMs already run at 91–94 %
//! of that, while the CPU's matrix unit (Accelerate `cblas_sgemm`) adds
//! 1.1–1.2 TF/s beside a busy GPU (30 s co-run: GPU 3.28 → 3.10 TF/s, CPU
//! 1.14–1.23). So every large GEMM hands its last output features to the
//! CPU: the GPU computes rows [0, rg), the CPU rows [rg, rows), both write
//! straight into the same shared output buffer.
//!
//! Ordering is one `MTLSharedEvent`: the GPU signals `ready` after the
//! producer of the GEMM's input, runs its part, then waits for `done`
//! before the consumer; the host thread (after committing every command
//! buffer) walks the jobs in order — wait `ready`, convert the half input
//! to f32, dequantize its int8 rows with the row scale, one sgemm, signal
//! `done`. Any failure releases the GPU (the event jumps past every
//! value) and the step declines.

use crate::pool::Pool;
use metal::{CommandBuffer, SharedEventRef};
use std::sync::OnceLock;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

/// `CMF_ZI_CPU_FRAC`: a number = that fixed share (0 = off), `auto` = the
/// adaptive controller, unset = the measured per-chip default.
enum Mode {
    Fixed(f64),
    Auto,
    Default,
}

fn mode() -> &'static Mode {
    static M: OnceLock<Mode> = OnceLock::new();
    M.get_or_init(|| match std::env::var("CMF_ZI_CPU_FRAC") {
        Ok(v) if v == "auto" => Mode::Auto,
        Ok(v) => Mode::Fixed(v.parse::<f64>().unwrap_or(0.0).clamp(0.0, 0.6)),
        Err(_) => Mode::Default,
    })
}

/// The default share: fixed (so a seed gives the same image every run),
/// measured on the Mac mini M4 (10-core GPU) only — Turbo 512² −13 % per
/// image at 0.25, 1024² −6 % at 0.20 (whole-CLI A/B, alternating); other
/// chips have other GPU/CPU ratios (a larger GPU would wait for the CPU)
/// and run without a CPU share unless `CMF_ZI_CPU_FRAC` says otherwise.
fn default_frac(device: &str, rows: usize) -> f64 {
    if device == "Apple M4" {
        if rows <= 1600 { 0.25 } else { 0.20 }
    } else {
        0.0
    }
}

/// Starting share of the adaptive controller, its step per forward and
/// its bounds. The controller keeps the CPU and GPU parts finishing
/// together: after every forward, if most CPU parts finished after the
/// GPU's part of the same GEMM (the GPU then waits), the share drops by
/// one step, if most finished first it grows. It follows the machine's
/// state (a hot M4 moves power from the CPU to the GPU) and other chips
/// (a larger GPU pushes the share toward the floor).
pub(super) const START_FRAC: f64 = 0.2;
const STEP_FRAC: f64 = 0.03;
const MIN_FRAC: f64 = 0.04;
const MAX_FRAC: f64 = 0.45;

fn state() -> &'static std::sync::Mutex<std::collections::HashMap<usize, f64>> {
    static S: OnceLock<std::sync::Mutex<std::collections::HashMap<usize, f64>>> = OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// The share for a program of `rows` rows on `device` (0 = off).
pub(super) fn frac(device: &str, rows: usize) -> f64 {
    match mode() {
        Mode::Fixed(f) => *f,
        Mode::Default => default_frac(device, rows),
        Mode::Auto => *state().lock().unwrap().entry(rows).or_insert(START_FRAC),
    }
}

/// Feed one forward's votes back (`auto` only).
fn adapt(rows: usize, cpu_late: usize, cpu_early: usize) {
    if !matches!(mode(), Mode::Auto) || cpu_late + cpu_early == 0 {
        return;
    }
    let mut st = state().lock().unwrap();
    let f = st.entry(rows).or_insert(START_FRAC);
    let tot = (cpu_late + cpu_early) as f64;
    if cpu_late as f64 > 0.6 * tot {
        *f = (*f - STEP_FRAC).max(MIN_FRAC);
    } else if cpu_early as f64 > 0.6 * tot {
        *f = (*f + STEP_FRAC).min(MAX_FRAC);
    }
}

/// One GEMM's CPU part. Raw pointers into the file mapping and into shared
/// Metal buffers; valid for the step that built it.
pub(super) struct CpuJob {
    pub ready: u64,
    pub done: u64,
    /// the value the GPU signals on the second event after its part
    pub gdone: u64,
    /// half activations: element offsets per z, row stride
    pub x: *const u16,
    pub x_off: [usize; 3],
    pub ldx: usize,
    /// per z: int8 rows [rows][k] (file) and f32 row scales [rows]
    pub w: [*const i8; 3],
    pub rs: [*const f32; 3],
    pub nz: usize,
    pub k: usize,
    /// the CPU's feature range
    pub r0: usize,
    pub r1: usize,
    pub n: usize,
    pub mul: f32,
    /// output base (f32 or half elements), element offsets per z (row0·ldy
    /// + the z column offset included), row stride
    pub y: *mut u8,
    pub y_half: bool,
    pub y_off: [usize; 3],
    pub ldy: usize,
}

unsafe impl Send for CpuJob {}

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

fn pool() -> &'static Pool {
    static P: OnceLock<Pool> = OnceLock::new();
    P.get_or_init(|| Pool::new(env_threads()))
}

fn env_threads() -> usize {
    std::env::var("CMF_ZI_CPU_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).max(1)
}

fn f16_lut() -> &'static [f32] {
    static L: OnceLock<Vec<f32>> = OnceLock::new();
    L.get_or_init(|| (0..=u16::MAX).map(cortiq_core::quant::f16_to_f32).collect())
}

#[derive(Default)]
pub(super) struct Scratch {
    x: Vec<f32>,
    w: Vec<f32>,
    c: Vec<f32>,
    /// seconds: wait, x convert, w dequant, sgemm, y convert
    t: [f64; 5],
}

/// Wait until the event reaches `v`; false on timeout or a failed buffer.
fn wait(ev: &SharedEventRef, v: u64, cmds: &[CommandBuffer]) -> bool {
    let t0 = std::time::Instant::now();
    let mut spins = 0u32;
    while ev.signaled_value() < v {
        spins += 1;
        if spins < 2000 {
            std::hint::spin_loop();
            continue;
        }
        std::thread::sleep(std::time::Duration::from_micros(50));
        if spins % 2000 == 0 {
            if cmds.iter().any(|c| c.status() == metal::MTLCommandBufferStatus::Error) {
                return false;
            }
            if t0.elapsed().as_secs() > 600 {
                return false;
            }
        }
    }
    true
}

fn run(j: &CpuJob, s: &mut Scratch) {
    let (n, k) = (j.n, j.k);
    let rc = j.r1 - j.r0;
    let lut = f16_lut();
    let p = pool();
    s.x.resize(n * k, 0.0);
    s.w.resize(rc * k, 0.0);
    if j.y_half {
        s.c.resize(n * rc, 0.0);
    }
    for z in 0..j.nz {
        let tt = std::time::Instant::now();
        // X half → f32
        {
            let xs = SendPtr(s.x.as_mut_ptr());
            let (xb, xo, ldx) = (SendPtr(j.x as *mut u16), j.x_off[z], j.ldx);
            p.run_rows(n, &|lo, hi| {
                let (xs, xb) = (&xs, &xb);
                for t in lo..hi {
                    // SAFETY: rows lo..hi are this worker's; the source is a
                    // live shared buffer the GPU finished writing (event).
                    unsafe {
                        let src = xb.0.add(xo + t * ldx);
                        let dst = xs.0.add(t * k);
                        for i in 0..k {
                            *dst.add(i) = lut[*src.add(i) as usize];
                        }
                    }
                }
            });
        }
        s.t[1] += tt.elapsed().as_secs_f64();
        let tt = std::time::Instant::now();
        // int8 rows r0..r1 → f32 · row scale
        {
            let ws = SendPtr(s.w.as_mut_ptr());
            let (wb, rsb, r0) = (SendPtr(j.w[z] as *mut i8), SendPtr(j.rs[z] as *mut f32), j.r0);
            p.run_rows(rc, &|lo, hi| {
                let (ws, wb, rsb) = (&ws, &wb, &rsb);
                for r in lo..hi {
                    // SAFETY: disjoint rows; the weights are the mapped file.
                    unsafe {
                        let sc = *rsb.0.add(r0 + r);
                        let src = wb.0.add((r0 + r) * k);
                        let dst = ws.0.add(r * k);
                        for i in 0..k {
                            *dst.add(i) = *src.add(i) as f32 * sc;
                        }
                    }
                }
            });
        }
        s.t[2] += tt.elapsed().as_secs_f64();
        let tt = std::time::Instant::now();
        // C[n][rc] = mul · X · Wᵀ
        let (cp, ldc) = if j.y_half {
            (s.c.as_mut_ptr(), rc)
        } else {
            // SAFETY: f32 output, this job's columns of the shared buffer
            (unsafe { (j.y as *mut f32).add(j.y_off[z] + j.r0) }, j.ldy)
        };
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                n as i32,
                rc as i32,
                k as i32,
                j.mul,
                s.x.as_ptr(),
                k as i32,
                s.w.as_ptr(),
                k as i32,
                0.0,
                cp,
                ldc as i32,
            );
        }
        s.t[3] += tt.elapsed().as_secs_f64();
        let tt = std::time::Instant::now();
        if j.y_half {
            let cs = SendPtr(s.c.as_mut_ptr());
            let (yb, yo, ldy, r0) = (SendPtr(j.y as *mut u16), j.y_off[z], j.ldy, j.r0);
            p.run_rows(n, &|lo, hi| {
                let (cs, yb) = (&cs, &yb);
                for t in lo..hi {
                    // SAFETY: disjoint rows of this job's columns.
                    unsafe {
                        let src = cs.0.add(t * rc);
                        let dst = yb.0.add(yo + t * ldy + r0);
                        for i in 0..rc {
                            *dst.add(i) = cortiq_core::quant::f32_to_f16(*src.add(i));
                        }
                    }
                }
            });
        }
        s.t[4] += tt.elapsed().as_secs_f64();
    }
}

/// Run every job in order against the GPU's progress. On failure the GPU
/// is released (the event jumps past every job) and false returned.
pub(super) fn execute(
    jobs: &[CpuJob],
    ev: &SharedEventRef,
    ev2: &SharedEventRef,
    rows: usize,
    cmds: &[CommandBuffer],
) -> bool {
    let mut s = Scratch::default();
    let (mut late, mut early) = (0usize, 0usize);
    for j in jobs {
        let tw = std::time::Instant::now();
        if !wait(ev, j.ready, cmds) {
            // Release the GPU (its waits on our `done` values are satisfied
            // by a jump past every job) and move the value counter past the
            // jump: the next step's first `ready` must be a value the GPU
            // has not signaled yet, or the CPU would read that step's input
            // before the GPU wrote it.
            let past = jobs.last().map_or(0, |l| l.done) + 1;
            ev.set_signaled_value(past);
            super::ZEVENT_NEXT.fetch_max(past + 1, std::sync::atomic::Ordering::SeqCst);
            return false;
        }
        s.t[0] += tw.elapsed().as_secs_f64();
        run(j, &mut s);
        // did the GPU finish its part of this GEMM before we did?
        if ev2.signaled_value() >= j.gdone {
            late += 1;
        } else {
            early += 1;
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        ev.set_signaled_value(j.done);
    }
    adapt(rows, late, early);
    if std::env::var("CMF_ZI_CPU_PROF").as_deref() == Ok("1") {
        eprintln!(
            "zimage cpu share {:.2}: {} jobs, cpu late {late} / early {early} · wait {:.3}s · x->f32 {:.3}s · w dequant {:.3}s · sgemm {:.3}s · y->f16 {:.3}s",
            jobs.first().map_or(0.0, |j| (j.r1 - j.r0) as f64 / j.r1 as f64),
            jobs.len(),
            s.t[0],
            s.t[1],
            s.t[2],
            s.t[3],
            s.t[4]
        );
    }
    true
}
