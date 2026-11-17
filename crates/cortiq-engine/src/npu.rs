//! NPU lane (Android): the FFN's `up` matvec on the Hexagon DSP while the
//! CPU computes `gate`.
//!
//! The DSP is reached through `libcortiq_npu.so` + `libcortiq_hvx_skel.so`
//! (`npu/hvx`: our own FastRPC module, integer HVX, unsigned PD — a few tens
//! of kilobytes, no vendor runtime). It is loaded with `dlopen` only when the
//! lane is asked for (`set_enabled` / `CMF_NPU=1`).
//!
//! Nothing is requantized: a q4tp matrix is copied into the module's tiled
//! layout code for code (32 rows in a vector's lanes), its per-group scales
//! become Q15 ratios of the row's largest, and the activation is the same
//! int8 split with float outliers the CPU's a8w8 kernel uses. The DSP returns
//! exact integer sums; the result differs from the CPU's only by the Q15
//! rounding of the scales.
//!
//! Two uses, each with its own verdict (`Mode`): decode — rows `row0..` of a
//! gate/up pair as one call while the CPU's fused kernel computes the rest —
//! and the prompt, where a chunk's `matmat` is split the same way and the
//! DSP multiplies eight activation vectors per pass over the codes. Whether
//! a use stays on is measured per process (`arm`/`trial_sample`): calls
//! alternate between the NPU split and the CPU alone, and the NPU stays
//! only when it wins by a margin.
//!
//! Snapdragon 778G (Hexagon V68), LFM3-1.7B q4tp: a 256-token prompt at
//! 30.9 tokens/s with the lane against 22.7 on the CPU (35.6 when 640 MB of
//! weights may be copied to the DSP), decode after it unchanged; the decode
//! split does not pay there — a token is bound by DDR, which the CPU and the
//! DSP share — and is off unless `CMF_NPU_DECODE=1`. `npu/hvx/README.md` has
//! the kernel, the build and the rest of the numbers.

use crate::qtensor::QTensor;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

/// Bytes of one (tile, group) record: four code vectors and one of scales.
pub const REC: usize = 640;

struct Api {
    alloc: unsafe extern "C" fn(usize) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
    add: unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int) -> c_int,
    run: unsafe extern "C" fn(c_int, c_int, *const i8, *mut i32, c_int) -> c_int,
    gemm: unsafe extern "C" fn(c_int, c_int, *const i8, *mut i32, c_int) -> c_int,
    err: unsafe extern "C" fn() -> *const c_char,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static LIB_DIR: Mutex<Option<String>> = Mutex::new(None);
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

/// Turn the lane on or off at run time (the app's switch). `CMF_NPU=1` turns
/// it on too; `CMF_NPU=0` keeps it off whatever the caller asks.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

/// Where `libcortiq_npu.so` and `libcortiq_hvx_skel.so` live (an app's native
/// library directory). Must be called before the first use; without it the
/// loader's default search path is used.
pub fn set_lib_dir(dir: &str) {
    *LIB_DIR.lock().unwrap() = Some(dir.to_string());
}

fn wanted() -> bool {
    match std::env::var("CMF_NPU").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => ENABLED.load(Ordering::Relaxed),
    }
}

fn fail(why: String) -> Option<Api> {
    eprintln!("[cortiq] npu: {why}");
    *LAST_ERROR.lock().unwrap() = why;
    None
}

fn api() -> Option<&'static Api> {
    if !wanted() {
        return None;
    }
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| {
        let dir = LIB_DIR.lock().unwrap().clone();
        let lib = std::env::var("CMF_NPU_LIB").ok().unwrap_or_else(|| match &dir {
            Some(d) => format!("{d}/libcortiq_npu.so"),
            None => "libcortiq_npu.so".into(),
        });
        if let Some(d) = &dir {
            // The DSP loads its skeleton from this list; the vendor
            // directories stay on it for the device's own libraries.
            if std::env::var_os("ADSP_LIBRARY_PATH").is_none() {
                let v = CString::new(format!("{d};/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp;/dsp")).ok()?;
                unsafe { libc::setenv(c"ADSP_LIBRARY_PATH".as_ptr(), v.as_ptr(), 1) };
            }
        }
        let lib_c = CString::new(lib.clone()).ok()?;
        unsafe {
            let h = libc::dlopen(lib_c.as_ptr(), libc::RTLD_NOW);
            if h.is_null() {
                let e = libc::dlerror();
                let why = if e.is_null() { String::new() } else { CStr::from_ptr(e).to_string_lossy().into_owned() };
                return fail(format!("cannot load {lib}: {why}"));
            }
            let sym = |n: &str| {
                let c = CString::new(n).unwrap();
                let p = libc::dlsym(h, c.as_ptr());
                if p.is_null() { None } else { Some(p) }
            };
            let (Some(init), Some(alloc), Some(free), Some(add), Some(run), Some(gemm), Some(err)) = (
                sym("cq_npu_init"),
                sym("cq_npu_alloc"),
                sym("cq_npu_free"),
                sym("cq_npu_add"),
                sym("cq_npu_run"),
                sym("cq_npu_gemm"),
                sym("cq_npu_error"),
            ) else {
                return fail(format!("{lib} is not a cortiq-npu module"));
            };
            let init: unsafe extern "C" fn(*const c_char) -> c_int = std::mem::transmute(init);
            let api = Api {
                alloc: std::mem::transmute(alloc),
                free: std::mem::transmute(free),
                add: std::mem::transmute(add),
                run: std::mem::transmute(run),
                gemm: std::mem::transmute(gemm),
                err: std::mem::transmute(err),
            };
            let dir_c = dir.as_ref().and_then(|d| CString::new(d.as_str()).ok());
            if init(dir_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) != 0 {
                return fail(format!("the Hexagon module did not come up: {}", CStr::from_ptr((api.err)()).to_string_lossy()));
            }
            eprintln!("[cortiq] npu: Hexagon HVX module up ({lib}; {})", CStr::from_ptr((api.err)()).to_string_lossy());
            Some(api)
        }
    })
    .as_ref()
}

/// A matrix's rows from `row0` on, in the module's tiled layout.
pub struct Tiled {
    pub tiles: usize,
    pub gpr: usize,
    pub shift: u32,
    /// Each row's largest group scale; the DSP's scales are Q15 ratios of it.
    pub row_max: Vec<f32>,
}

/// One matrix's NPU share: its rows `row0..`.
pub struct Lane {
    handle: c_int,
    /// First row the DSP computes; rows before it are the CPU's.
    pub row0: usize,
    pub gpr: usize,
    pub tiles: usize,
    pub shift: u32,
    pub row_max: Vec<f32>,
    /// The tiled buffer (shared with the DSP), for the host's outlier terms.
    w: *const u8,
}
unsafe impl Send for Lane {}
unsafe impl Sync for Lane {}

impl Lane {
    /// Level (code - 8) and scale of weight (i, j), `i` counted from `row0`,
    /// read from the tiled copy.
    #[inline]
    pub fn weight(&self, i: usize, j: usize) -> (f32, f32) {
        let (g, k) = (j / 32, j % 32);
        let (t, lane) = (i / 32, i % 32);
        let rec = unsafe { self.w.add((t * self.gpr + g) * REC) };
        let byte = unsafe { *rec.add((k / 8) * 128 + lane * 4 + (k % 8) / 2) };
        let n = if k & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let q = unsafe { (rec.add(512 + lane * 4) as *const i32).read_unaligned() } >> 16;
        ((n as i32 - 8) as f32, self.row_max[i] * q as f32 * (1.0 / 32767.0))
    }

    /// Scale from the DSP's integer to the product, before the activation's own scale.
    #[inline]
    pub fn unit(&self) -> f32 {
        (32768.0 / 32767.0) / (1u32 << self.shift) as f32
    }
}

fn lanes() -> &'static Mutex<HashMap<usize, Option<Arc<Lane>>>> {
    static M: OnceLock<Mutex<HashMap<usize, Option<Arc<Lane>>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(HashMap::new()))
}

pub static CALLS: AtomicU64 = AtomicU64::new(0);
pub static FALLBACKS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// What the lane is being used for; each use is judged on its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// One token: a gate/up pair's rows on the DSP.
    Decode = 0,
    /// A prompt chunk: a matrix times many activation vectors.
    Batch = 1,
}

struct Trial {
    /// 0 unknown, 1 use the NPU, 2 the CPU is faster here.
    verdict: AtomicU32,
    turn: AtomicU32,
    /// The matrices the trial measures with.
    keys: Mutex<Vec<usize>>,
    samples: Mutex<(Vec<u64>, Vec<u64>)>,
    medians: Mutex<(u64, u64)>,
}

static TRIALS: [Trial; 2] = [Trial::new(), Trial::new()];

impl Trial {
    const fn new() -> Self {
        Trial {
            verdict: AtomicU32::new(0),
            turn: AtomicU32::new(0),
            keys: Mutex::new(Vec::new()),
            samples: Mutex::new((Vec::new(), Vec::new())),
            medians: Mutex::new((0, 0)),
        }
    }
}

/// The lane of the matrix whose bytes start at `key`, built at first use by
/// `fill` (which writes rows `row0..` into the buffer it is given). `None`:
/// the NPU is off or this use lost its trial, the matrix is too small to be
/// worth a call, the budget is spent, or the DSP refused it.
pub fn lane_for(
    mode: Mode,
    key: usize,
    rows: usize,
    cols: usize,
    fill: &dyn Fn(&mut [u8], usize) -> Option<Tiled>,
) -> Option<Arc<Lane>> {
    if !wanted() {
        return None;
    }
    load_verdicts();
    let trial = &TRIALS[mode as usize];
    let verdict = trial.verdict.load(Ordering::Relaxed);
    if verdict == 2 {
        return None;
    }
    let api = api()?;
    if verdict == 0 {
        // Until the trial has spoken it measures with a few matrices only:
        // every lane is a copy of weights on the DSP. A prompt chunk calls
        // each matrix once, so one matrix would take many chunks to judge.
        let mut keys = trial.keys.lock().unwrap();
        if !keys.contains(&key) {
            if keys.len() >= TRIAL_KEYS {
                return None;
            }
            keys.push(key);
        }
    }
    let mut map = lanes().lock().unwrap();
    if let Some(slot) = map.get(&key) {
        return slot.clone();
    }
    // The DSP's share of the rows (`CMF_NPU_SHARE`).
    let share = env_f32("CMF_NPU_SHARE", 0.65).clamp(0.05, 1.0);
    let row0 = (((rows as f32) * (1.0 - share)) as usize / 32) * 32;
    let tiles = (rows - row0).div_ceil(32);
    let gpr = cols / 32;
    let min = (env_f32("CMF_NPU_MIN_M", 2.0) * 1e6) as u64;
    let budget = budget_bytes();
    let bytes = (tiles * gpr * REC) as u64;
    if cols % 32 != 0 || ((rows * cols) as u64) < min || tiles == 0 || BYTES.load(Ordering::Relaxed) + bytes > budget {
        map.insert(key, None);
        return None;
    }
    let t0 = std::time::Instant::now();
    let buf = unsafe { (api.alloc)(bytes as usize) } as *mut u8;
    let slot = (|| {
        if buf.is_null() {
            eprintln!("[cortiq] npu: no shared memory for {rows}x{cols}");
            return None;
        }
        let Some(t) = fill(unsafe { std::slice::from_raw_parts_mut(buf, bytes as usize) }, row0) else {
            unsafe { (api.free)(buf as *mut c_void) };
            return None;
        };
        let h = unsafe { (api.add)(buf as *mut c_void, t.tiles as c_int, t.gpr as c_int, t.shift as c_int) };
        if h < 0 {
            eprintln!("[cortiq] npu: {rows}x{cols} refused: {}", unsafe { CStr::from_ptr((api.err)()).to_string_lossy() });
            unsafe { (api.free)(buf as *mut c_void) };
            return None;
        }
        BYTES.fetch_add(bytes, Ordering::Relaxed);
        if std::env::var("CMF_NPU_LOG").is_ok() {
            eprintln!(
                "[cortiq] npu: lane {h}: rows {row0}..{rows} of {rows}x{cols} in {:.2} s ({} MB on the DSP)",
                t0.elapsed().as_secs_f64(),
                BYTES.load(Ordering::Relaxed) / 1_000_000
            );
        }
        Some(Arc::new(Lane { handle: h, row0, gpr: t.gpr, tiles: t.tiles, shift: t.shift, row_max: t.row_max, w: buf }))
    })();
    map.insert(key, slot.clone());
    slot
}

struct Job {
    h: c_int,
    /// Second matrix of a pair (-1: none); for a batch, the vector count.
    h2_or_n: c_int,
    batch: bool,
    x: *const i8,
    y: *mut i32,
    done: mpsc::SyncSender<bool>,
}
unsafe impl Send for Job {}

fn worker() -> &'static Mutex<mpsc::Sender<Job>> {
    static W: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    W.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("cortiq-npu".into())
            .spawn(move || {
                let api = api().expect("npu api");
                let threads = env_f32("CMF_NPU_THREADS", 1.0) as c_int;
                let bthreads = env_f32("CMF_NPU_BATCH_THREADS", 2.0) as c_int;
                // The RPC's host side is short but latency-bound: on a little
                // core, behind the pool's workers, it doubles the round trip.
                if std::env::var("CMF_NPU_PIN").as_deref() != Ok("0") {
                    crate::pool::pin_thread_to_big_cores();
                }
                for j in rx {
                    let t = std::time::Instant::now();
                    let rc = unsafe {
                        if j.batch {
                            (api.gemm)(j.h, j.h2_or_n, j.x, j.y, bthreads)
                        } else {
                            (api.run)(j.h, j.h2_or_n, j.x, j.y, threads)
                        }
                    };
                    PROF_RPC.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                    if rc != 0 {
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            eprintln!("[cortiq] npu: call failed: {}", unsafe { CStr::from_ptr((api.err)()).to_string_lossy() })
                        });
                    }
                    let _ = j.done.send(rc == 0);
                }
            })
            .expect("npu thread");
        Mutex::new(tx)
    })
}

/// A product in flight on the DSP; `finish` waits for it.
pub struct Pending {
    x: Vec<i8>,
    y: Vec<i32>,
    rx: mpsc::Receiver<bool>,
    done: Option<bool>,
}

/// The module's input order: per group, for j in 0..4 the four even inputs
/// of block j, then its four odd ones.
fn reorder(dst: &mut [i8], src: &[i8]) {
    for (d, s) in dst.chunks_exact_mut(8).zip(src.chunks_exact(8)) {
        d[0] = s[0];
        d[1] = s[2];
        d[2] = s[4];
        d[3] = s[6];
        d[4] = s[1];
        d[5] = s[3];
        d[6] = s[5];
        d[7] = s[7];
    }
}

fn submit(h: c_int, h2_or_n: c_int, batch: bool, x: Vec<i8>, ylen: usize) -> Option<Pending> {
    let (tx, rx) = mpsc::sync_channel(1);
    let mut p = Pending { x, y: vec![0i32; ylen], rx, done: None };
    let job = Job { h, h2_or_n, batch, x: p.x.as_ptr(), y: p.y.as_mut_ptr(), done: tx };
    worker().lock().unwrap().send(job).ok()?;
    Some(p)
}

/// Start the matvec of `xq` (int8, column order) on `a` — and on `b` too, a
/// second matrix of the same shape, in the same call — and return at once.
/// The result is `a`'s `tiles*32` values, then `b`'s.
pub fn start(a: &Lane, b: Option<&Lane>, xq: &[i8]) -> Option<Pending> {
    debug_assert_eq!(xq.len(), a.gpr * 32);
    if b.is_some_and(|b| b.tiles != a.tiles || b.gpr != a.gpr || b.shift != a.shift) {
        return None;
    }
    let mut x = vec![0i8; xq.len()];
    reorder(&mut x, xq);
    submit(a.handle, b.map_or(-1, |b| b.handle), false, x, a.tiles * 32 * if b.is_some() { 2 } else { 1 })
}

/// Start the product of `n` vectors (`xq` holds them one after another,
/// column order). The result is `[n][tiles*32]`.
pub fn start_batch(a: &Lane, n: usize, xq: &[i8]) -> Option<Pending> {
    debug_assert_eq!(xq.len(), n * a.gpr * 32);
    let mut x = vec![0i8; xq.len()];
    reorder(&mut x, xq);
    submit(a.handle, n as c_int, true, x, n * a.tiles * 32)
}

impl Pending {
    /// Block until the DSP has answered (`finish` then returns at once).
    pub fn wait(&mut self) {
        if self.done.is_none() {
            self.done = Some(self.rx.recv().unwrap_or(false));
        }
    }

    /// Wait for the DSP. `None`: the call failed — the caller computes the
    /// product on the CPU.
    pub fn finish(mut self) -> Option<Vec<i32>> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        self.wait();
        if self.done == Some(true) {
            Some(std::mem::take(&mut self.y))
        } else {
            FALLBACKS.fetch_add(1, Ordering::Relaxed);
            None
        }
    }
}

/// `calls/fallbacks` since the process started.
pub fn stats() -> (u64, u64) {
    (CALLS.load(Ordering::Relaxed), FALLBACKS.load(Ordering::Relaxed))
}

// ---- the trials: is the NPU worth it on this device? ---------------------

const TRIAL_SKIP: usize = 1;
const TRIAL_SAMPLES: usize = 4;
/// Matrices a trial may build lanes for before its verdict.
const TRIAL_KEYS: usize = 6;

/// What a call should do while its use's verdict is open.
pub enum Arm {
    /// Take the NPU split (and time it when `timed`).
    Npu { timed: bool },
    /// Take the CPU alone and time it.
    Cpu,
}

/// Which arm this call runs. After the verdict every call is `Npu`.
pub fn arm(mode: Mode) -> Arm {
    let t = &TRIALS[mode as usize];
    if t.verdict.load(Ordering::Relaxed) == 1 {
        return Arm::Npu { timed: false };
    }
    if t.turn.fetch_add(1, Ordering::Relaxed) % 2 == 0 { Arm::Npu { timed: true } } else { Arm::Cpu }
}

fn median(v: &[u64]) -> u64 {
    let mut s = v[TRIAL_SKIP..].to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

/// Record a trial sample (decode: microseconds of a fused call; a batch:
/// nanoseconds per activation vector per million weights); decides once
/// both arms have enough.
///
/// The NPU must beat the CPU by a margin (`CMF_NPU_MARGIN`, 0.95): a tie is
/// not worth a second copy of the weights and a busy DSP. `CMF_NPU_PROBE=0`
/// skips the trials and uses the NPU regardless.
pub fn trial_sample(mode: Mode, npu: bool, us: u64) {
    let t = &TRIALS[mode as usize];
    if t.verdict.load(Ordering::Relaxed) != 0 {
        return;
    }
    if std::env::var("CMF_NPU_PROBE").as_deref() == Ok("0") {
        t.verdict.store(1, Ordering::Relaxed);
        return;
    }
    let mut s = t.samples.lock().unwrap();
    if npu { s.0.push(us) } else { s.1.push(us) }
    let need = TRIAL_SKIP + TRIAL_SAMPLES;
    if s.0.len() < need || s.1.len() < need {
        return;
    }
    let (n, c) = (median(&s.0), median(&s.1));
    *t.medians.lock().unwrap() = (n, c);
    let margin = env_f32("CMF_NPU_MARGIN", 0.95);
    let use_npu = (n as f32) < (c as f32) * margin;
    t.verdict.store(if use_npu { 1 } else { 2 }, Ordering::Relaxed);
    drop(s);
    save_verdicts();
    eprintln!(
        "[cortiq] npu trial ({}): {n} with the NPU against {c} on the CPU -> {}",
        if mode == Mode::Decode { "decode, us per gate+up" } else { "prompt, ns per vector per million weights" },
        if use_npu { "NPU on" } else { "CPU is faster here, NPU off for it" }
    );
}

fn verdict_name(mode: Mode) -> &'static str {
    match TRIALS[mode as usize].verdict.load(Ordering::Relaxed) {
        1 => "npu",
        2 => "cpu",
        _ => "untested",
    }
}

/// The lane's state for a status line or an app's settings screen.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// The caller (or `CMF_NPU=1`) asked for the NPU.
    pub requested: bool,
    /// The DSP module came up on this device.
    pub available: bool,
    /// Decode: "untested", "npu" or "cpu", and the medians (microseconds of
    /// a fused gate+up with the NPU / on the CPU).
    pub verdict: &'static str,
    pub npu_us: u64,
    pub cpu_us: u64,
    /// The prompt: the same, per activation vector of a chunk.
    pub batch_verdict: &'static str,
    pub batch_npu_us: u64,
    pub batch_cpu_us: u64,
    pub lanes: usize,
    pub weights_mb: u64,
    pub calls: u64,
    pub fallbacks: u64,
    /// Why the module is not available, when it is not.
    pub error: String,
}

pub fn status() -> Status {
    let requested = wanted();
    let available = requested && api().is_some();
    let (npu_us, cpu_us) = *TRIALS[0].medians.lock().unwrap();
    let (batch_npu_us, batch_cpu_us) = *TRIALS[1].medians.lock().unwrap();
    let (calls, fallbacks) = stats();
    Status {
        requested,
        available,
        verdict: verdict_name(Mode::Decode),
        npu_us,
        cpu_us,
        batch_verdict: verdict_name(Mode::Batch),
        batch_npu_us,
        batch_cpu_us,
        lanes: lanes().lock().unwrap().values().filter(|s| s.is_some()).count(),
        weights_mb: BYTES.load(Ordering::Relaxed) / 1_000_000,
        calls,
        fallbacks,
        error: if available { String::new() } else { LAST_ERROR.lock().unwrap().clone() },
    }
}

static PROF_RPC: AtomicU64 = AtomicU64::new(0);
static PROF: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static PROF_N: AtomicU64 = AtomicU64::new(0);

/// `CMF_NPU_PROF=1`: per fused call — the split and hand-off, the CPU's
/// share, the wait for the DSP after it, the whole call, and the RPC's wall
/// time on the lane's thread — printed every 600 calls.
pub fn prof(pre_us: u64, cpu_us: u64, wait_us: u64, total_us: u64) {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("CMF_NPU_PROF").is_ok()) {
        return;
    }
    for (a, v) in PROF.iter().zip([pre_us, cpu_us, wait_us, total_us]) {
        a.fetch_add(v, Ordering::Relaxed);
    }
    let n = PROF_N.fetch_add(1, Ordering::Relaxed) + 1;
    if n % 600 == 0 {
        let v: Vec<u64> = PROF.iter().map(|a| a.swap(0, Ordering::Relaxed) / 600).collect();
        eprintln!(
            "[cortiq] npu prof: pre {} us, cpu share {} us, wait {} us, whole {} us; rpc {} us",
            v[0], v[1], v[2], v[3],
            PROF_RPC.swap(0, Ordering::Relaxed) / 600
        );
    }
}

/// `CMF_NPU_SYNC=1`: wait for the DSP before starting `gate` (no overlap) —
/// separates the two costs from what they do to each other.
pub fn sync_mode() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_NPU_SYNC").is_ok())
}

/// Whether the decode split was asked for (`CMF_NPU_DECODE=1`).
pub fn decode_wanted() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_NPU_DECODE").as_deref() == Ok("1"))
}

/// Bytes of weights the lanes may copy to the DSP.
///
/// The copies are shared memory the system cannot reclaim, and the model
/// itself lives in the page cache: on a phone with 2 GB available, 640 MB of
/// lanes evicted the model's pages and decode fell from 11.8 to 4.6
/// tokens/s. So the default is a fifth of what is available when the first
/// lane is built, at most 512 MB; `CMF_NPU_MB` sets it outright.
fn budget_bytes() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| {
        if let Some(mb) = std::env::var("CMF_NPU_MB").ok().and_then(|v| v.parse::<f64>().ok()) {
            return (mb * 1e6) as u64;
        }
        let avail_kb = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|m| {
                m.lines()
                    .find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse::<u64>().ok()))
            })
            .unwrap_or(0);
        (avail_kb * 1024 / 5).min(512_000_000)
    })
}

// ---- the verdicts are kept between runs -----------------------------------

fn verdict_path() -> std::path::PathBuf {
    crate::gpu::cache_dir_pub().join("cortiq-npu-probe.tsv")
}

/// Load the verdicts a previous process reached (same engine version): a
/// prompt chunk calls each matrix once, so a short prompt would otherwise
/// end before its trial does. `CMF_NPU_PROBE=1` measures again.
fn load_verdicts() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var("CMF_NPU_PROBE").is_ok() {
            return;
        }
        let Ok(text) = std::fs::read_to_string(verdict_path()) else { return };
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 5 || f[0] != env!("CARGO_PKG_VERSION") {
                continue;
            }
            let (Ok(m), Ok(v), Ok(n), Ok(c)) = (f[1].parse::<usize>(), f[2].parse::<u32>(), f[3].parse::<u64>(), f[4].parse::<u64>()) else {
                continue;
            };
            if m < 2 && (v == 1 || v == 2) {
                TRIALS[m].verdict.store(v, Ordering::Relaxed);
                *TRIALS[m].medians.lock().unwrap() = (n, c);
            }
        }
    });
}

fn save_verdicts() {
    let mut out = String::new();
    for (m, t) in TRIALS.iter().enumerate() {
        let v = t.verdict.load(Ordering::Relaxed);
        if v != 0 {
            let (n, c) = *t.medians.lock().unwrap();
            out.push_str(&format!("{}\t{m}\t{v}\t{n}\t{c}\n", env!("CARGO_PKG_VERSION")));
        }
    }
    let _ = std::fs::write(verdict_path(), out);
}
