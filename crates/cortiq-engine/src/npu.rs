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
//! Whether the lane is used is measured per process (`arm`/`trial_sample`):
//! ten fused calls with the NPU against ten on the CPU, and the NPU stays
//! only when it wins by a margin.
//!
//! Snapdragon 778G (Hexagon V68), a 7680x2048 matrix: 0.69 ms on the DSP
//! with one HVX thread (it is memory-bound: more threads add nothing),
//! 1.12 ms with the FastRPC round trip; the CPU's four big cores take
//! 0.99 ms for one such matrix and ~1.5 ms for the fused pair.

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
    run: unsafe extern "C" fn(c_int, *const i8, *mut i32, c_int) -> c_int,
    err: unsafe extern "C" fn() -> *const c_char,
}

static ENABLED: AtomicBool = AtomicBool::new(false);
/// 0 unknown, 1 use the NPU, 2 the CPU is faster here.
static VERDICT: AtomicU32 = AtomicU32::new(0);
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
            let (Some(init), Some(alloc), Some(free), Some(add), Some(run), Some(err)) =
                (sym("cq_npu_init"), sym("cq_npu_alloc"), sym("cq_npu_free"), sym("cq_npu_add"), sym("cq_npu_run"), sym("cq_npu_error"))
            else {
                return fail(format!("{lib} is not a cortiq-npu module"));
            };
            let init: unsafe extern "C" fn(*const c_char) -> c_int = std::mem::transmute(init);
            let api = Api {
                alloc: std::mem::transmute(alloc),
                free: std::mem::transmute(free),
                add: std::mem::transmute(add),
                run: std::mem::transmute(run),
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

/// A matrix (its rows from `row0` on) in the module's tiled layout, as
/// `QTensor::npu_fill_tiles` leaves it.
pub struct Tiled {
    pub tiles: usize,
    pub gpr: usize,
    pub shift: u32,
    /// Each row's largest group scale; the DSP's scales are Q15 ratios of it.
    pub row_max: Vec<f32>,
}

/// A gate/up pair's NPU share: rows `row0..` of both matrices, one buffer
/// (gate's tiles, then up's), one call.
pub struct Lane {
    handle: c_int,
    /// First row the DSP computes; rows before it are the CPU's.
    pub row0: usize,
    pub gpr: usize,
    /// Tiles per matrix.
    pub tiles: usize,
    pub shift: u32,
    pub gate_max: Vec<f32>,
    pub up_max: Vec<f32>,
    /// The tiled buffer (shared with the DSP), for the host's outlier terms.
    w: *const u8,
}
unsafe impl Send for Lane {}
unsafe impl Sync for Lane {}

impl Lane {
    /// Level (code - 8) and scale of weight (i, j) of matrix `m` (0 gate,
    /// 1 up), `i` counted from `row0`, read from the tiled copy.
    #[inline]
    pub fn weight(&self, m: usize, i: usize, j: usize) -> (f32, f32) {
        let (g, k) = (j / 32, j % 32);
        let (t, lane) = (m * self.tiles + i / 32, i % 32);
        let rec = unsafe { self.w.add((t * self.gpr + g) * REC) };
        let byte = unsafe { *rec.add((k / 8) * 128 + lane * 4 + (k % 8) / 2) };
        let n = if k & 1 == 0 { byte & 0x0F } else { byte >> 4 };
        let q = unsafe { (rec.add(512 + lane * 4) as *const i32).read_unaligned() } >> 16;
        let mx = if m == 0 { self.gate_max[i] } else { self.up_max[i] };
        ((n as i32 - 8) as f32, mx * q as f32 * (1.0 / 32767.0))
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

/// The lane for this gate/up pair, built at first use. `None`: the NPU is
/// off, the pair is not a layout the module takes or too small to be worth a
/// call, the budget is spent, or the DSP refused it.
pub fn lane(gate: &QTensor, up: &QTensor) -> Option<Arc<Lane>> {
    if VERDICT.load(Ordering::Relaxed) == 2 {
        return None;
    }
    let api = api()?;
    let key = gate.npu_key()? ^ up.npu_key()?.rotate_left(17);
    let mut map = lanes().lock().unwrap();
    if let Some(slot) = map.get(&key) {
        return slot.clone();
    }
    // Until the trial has spoken, one pair is enough to measure with.
    if VERDICT.load(Ordering::Relaxed) == 0 && map.values().any(|s| s.is_some()) {
        return None;
    }
    let (rows, cols) = (up.rows(), up.cols());
    // The DSP's share of the rows: where its two matvecs and the round trip
    // take as long as the CPU's fused kernel on the rest (`CMF_NPU_SHARE`).
    let share = env_f32("CMF_NPU_SHARE", 0.4).clamp(0.05, 1.0);
    let row0 = (((rows as f32) * (1.0 - share)) as usize / 32) * 32;
    let tiles = (rows - row0).div_ceil(32);
    let gpr = cols / 32;
    let min = (env_f32("CMF_NPU_MIN_M", 2.0) * 1e6) as u64;
    let budget = (env_f32("CMF_NPU_MB", 1024.0) * 1e6) as u64;
    let bytes = (2 * tiles * gpr * REC) as u64;
    if gate.rows() != rows
        || gate.cols() != cols
        || cols % 32 != 0
        || ((rows * cols) as u64) < min
        || tiles == 0
        || BYTES.load(Ordering::Relaxed) + bytes > budget
    {
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
        let all = unsafe { std::slice::from_raw_parts_mut(buf, bytes as usize) };
        let (bg, bu) = all.split_at_mut(tiles * gpr * REC);
        let (Some(tg), Some(tu)) = (gate.npu_fill_tiles(bg, row0), up.npu_fill_tiles(bu, row0)) else {
            unsafe { (api.free)(buf as *mut c_void) };
            return None;
        };
        let h = unsafe { (api.add)(buf as *mut c_void, (2 * tiles) as c_int, gpr as c_int, tg.shift as c_int) };
        if h < 0 {
            eprintln!("[cortiq] npu: {rows}x{cols} refused: {}", unsafe { CStr::from_ptr((api.err)()).to_string_lossy() });
            unsafe { (api.free)(buf as *mut c_void) };
            return None;
        }
        BYTES.fetch_add(bytes, Ordering::Relaxed);
        if std::env::var("CMF_NPU_LOG").is_ok() {
            eprintln!(
                "[cortiq] npu: lane {h}: rows {row0}..{rows} of a {rows}x{cols} pair in {:.2} s ({} MB on the DSP)",
                t0.elapsed().as_secs_f64(),
                BYTES.load(Ordering::Relaxed) / 1_000_000
            );
        }
        Some(Arc::new(Lane { handle: h, row0, gpr, tiles, shift: tg.shift, gate_max: tg.row_max, up_max: tu.row_max, w: buf }))
    })();
    map.insert(key, slot.clone());
    slot
}

struct Job {
    handle: c_int,
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
                // The RPC's host side is short but latency-bound: on a little
                // core, behind the pool's workers, it doubles the round trip.
                if std::env::var("CMF_NPU_PIN").as_deref() != Ok("0") {
                    crate::pool::pin_thread_to_big_cores();
                }
                for j in rx {
                    let t = std::time::Instant::now();
                    let ok = unsafe { (api.run)(j.handle, j.x, j.y, threads) } == 0;
                    PROF_RPC.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                    if !ok {
                        static SAID: std::sync::Once = std::sync::Once::new();
                        SAID.call_once(|| {
                            eprintln!("[cortiq] npu: call failed: {}", unsafe { CStr::from_ptr((api.err)()).to_string_lossy() })
                        });
                    }
                    let _ = j.done.send(ok);
                }
            })
            .expect("npu thread");
        Mutex::new(tx)
    })
}

/// A matvec in flight on the DSP; `finish` waits for it.
pub struct Pending {
    x: Vec<i8>,
    y: Vec<i32>,
    rx: mpsc::Receiver<bool>,
    done: Option<bool>,
}

/// Start the matvec of `xq` (int8, in column order) and return at once.
pub fn start(lane: &Lane, xq: &[i8]) -> Option<Pending> {
    debug_assert_eq!(xq.len(), lane.gpr * 32);
    // The module's input order: per group, for j in 0..4 the four even
    // inputs of block j, then its four odd ones.
    let mut x = vec![0i8; xq.len()];
    for (dst, src) in x.chunks_exact_mut(8).zip(xq.chunks_exact(8)) {
        dst[0] = src[0];
        dst[1] = src[2];
        dst[2] = src[4];
        dst[3] = src[6];
        dst[4] = src[1];
        dst[5] = src[3];
        dst[6] = src[5];
        dst[7] = src[7];
    }
    let mut p = Pending { x, y: vec![0i32; 2 * lane.tiles * 32], rx: mpsc::sync_channel(1).1, done: None };
    let (tx, rx) = mpsc::sync_channel(1);
    p.rx = rx;
    let job = Job { handle: lane.handle, x: p.x.as_ptr(), y: p.y.as_mut_ptr(), done: tx };
    worker().lock().unwrap().send(job).ok()?;
    Some(p)
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

// ---- the trial: is the NPU worth it on this device? ----------------------

const TRIAL_SKIP: usize = 2;
const TRIAL_SAMPLES: usize = 10;
static TRIAL: Mutex<(Vec<u64>, Vec<u64>)> = Mutex::new((Vec::new(), Vec::new()));
static TRIAL_TURN: AtomicU32 = AtomicU32::new(0);
static MEDIANS: Mutex<(u64, u64)> = Mutex::new((0, 0));

/// What a fused call should do while the verdict is open.
pub enum Arm {
    /// Take the NPU path (and time it when `timed`).
    Npu { timed: bool },
    /// Take the CPU path and time it.
    Cpu,
}

/// Which arm this call runs. After the verdict every call is `Npu`.
pub fn arm() -> Arm {
    if VERDICT.load(Ordering::Relaxed) == 1 {
        return Arm::Npu { timed: false };
    }
    if TRIAL_TURN.fetch_add(1, Ordering::Relaxed) % 2 == 0 { Arm::Npu { timed: true } } else { Arm::Cpu }
}

fn median(v: &[u64]) -> u64 {
    let mut s = v[TRIAL_SKIP..].to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

/// Record a trial sample; decides once both arms have enough.
///
/// The NPU must beat the CPU by a margin (`CMF_NPU_MARGIN`, 0.92): a tie is
/// not worth a second copy of the weights and a busy DSP. `CMF_NPU_PROBE=0`
/// skips the trial and uses the NPU regardless.
pub fn trial_sample(npu: bool, us: u64) {
    if VERDICT.load(Ordering::Relaxed) != 0 {
        return;
    }
    if std::env::var("CMF_NPU_PROBE").as_deref() == Ok("0") {
        VERDICT.store(1, Ordering::Relaxed);
        return;
    }
    let mut t = TRIAL.lock().unwrap();
    if npu { t.0.push(us) } else { t.1.push(us) }
    let need = TRIAL_SKIP + TRIAL_SAMPLES;
    if t.0.len() < need || t.1.len() < need {
        return;
    }
    let (n, c) = (median(&t.0), median(&t.1));
    *MEDIANS.lock().unwrap() = (n, c);
    let margin = env_f32("CMF_NPU_MARGIN", 0.92);
    let use_npu = (n as f32) < (c as f32) * margin;
    VERDICT.store(if use_npu { 1 } else { 2 }, Ordering::Relaxed);
    eprintln!(
        "[cortiq] npu trial: gate+up {n} us with the NPU against {c} us on the CPU -> {}",
        if use_npu { "NPU lane on" } else { "CPU is faster here, NPU lane off" }
    );
}

/// The lane's state for a status line or an app's settings screen.
#[derive(Clone, Debug, Default)]
pub struct Status {
    /// The caller (or `CMF_NPU=1`) asked for the NPU.
    pub requested: bool,
    /// The backend came up on this device.
    pub available: bool,
    /// "untested", "npu" or "cpu".
    pub verdict: &'static str,
    /// Median microseconds of a fused gate+up with the NPU / on the CPU.
    pub npu_us: u64,
    pub cpu_us: u64,
    pub lanes: usize,
    pub weights_mb: u64,
    pub calls: u64,
    pub fallbacks: u64,
    /// Why the backend is not available, when it is not.
    pub error: String,
}

pub fn status() -> Status {
    let requested = wanted();
    let available = requested && api().is_some();
    let (npu_us, cpu_us) = *MEDIANS.lock().unwrap();
    let (calls, fallbacks) = stats();
    Status {
        requested,
        available,
        verdict: match VERDICT.load(Ordering::Relaxed) {
            1 => "npu",
            2 => "cpu",
            _ => "untested",
        },
        npu_us,
        cpu_us,
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
