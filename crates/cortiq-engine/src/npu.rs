//! NPU lane (prototype, Android): the FFN's `up` matvec on the Hexagon NPU
//! while the CPU computes `gate`.
//!
//! The NPU is reached through `libcortiq_npu.so` (`npu/cortiq_npu.cpp`, a thin
//! C ABI over Qualcomm's QNN HTP backend), loaded with `dlopen` only when
//! `CMF_NPU=1`. A matrix becomes one QNN graph: a 1x1 convolution with int8
//! weights quantized per output row and 16-bit activations.
//!
//! Measured on a Snapdragon 778G (Hexagon V68): the NPU streams 8-bit weights
//! at 11-14 G weights/s against the CPU's ~21 G for q4tp, and keeps 11 G while
//! the CPU generates — so the gain is from running the two together, not from
//! moving the work. V68 refuses 4-bit weights, hence int8.
//!
//! RESULT on that phone (LFM3-1.7B q4tp, 24 lanes of 7680x2048): correct —
//! the greedy text is the CPU's, 1.3% of calls saturate and fall back — but
//! SLOWER: 9.2 tokens/s against 12.1 on the CPU alone. Per fused call the
//! CPU's `gate` takes 0.99 ms and the NPU's `up` returns after 2.06 ms, while
//! the CPU's fused gate+up kernel does both in ~1.5 ms. Giving the NPU only
//! part of the rows would balance at ~1.2 ms, about +9% a token at best, so
//! the lane stays off by default. It is kept for Hexagon v73+ (HMX, 4-bit
//! weights), where the NPU is several times faster and this split may pay;
//! nothing here has been run on such a chip.
//!
//! Fixed point: the activation is scaled by its own max into u16; the output
//! range is `alpha * max row norm`, static per graph. A saturated output
//! falls back to the CPU for that call and halves the lane's input gain.

use crate::qtensor::QTensor;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};

struct Api {
    add: unsafe extern "C" fn(u32, u32, *const i8, *const f32, f32) -> c_int,
    run: unsafe extern "C" fn(c_int, *const u16, *mut u16) -> c_int,
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

/// Where `libcortiq_npu.so`, the QNN libraries and the Hexagon skeletons
/// live (an app's native library directory). Must be called before the first
/// use; without it the loader's default search path is used.
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
            // The DSP side loads its skeleton from this list; the vendor
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
            let (Some(init), Some(add), Some(run), Some(err)) = (sym("cq_npu_init"), sym("cq_npu_add"), sym("cq_npu_run"), sym("cq_npu_error")) else {
                return fail(format!("{lib} is not a cortiq-npu shim"));
            };
            let init: unsafe extern "C" fn(*const c_char) -> c_int = std::mem::transmute(init);
            let api = Api { add: std::mem::transmute(add), run: std::mem::transmute(run), err: std::mem::transmute(err) };
            let qnn = std::env::var("CMF_NPU_QNN").ok().or_else(|| dir.as_ref().map(|d| format!("{d}/libQnnHtp.so")));
            let qnn = qnn.and_then(|s| CString::new(s).ok());
            if init(qnn.as_ref().map_or(std::ptr::null(), |c| c.as_ptr())) != 0 {
                return fail(format!("QNN HTP did not come up: {}", CStr::from_ptr((api.err)()).to_string_lossy()));
            }
            eprintln!("[cortiq] npu: QNN HTP backend up ({lib})");
            Some(api)
        }
    })
    .as_ref()
}

/// One matrix on the NPU.
pub struct Lane {
    handle: c_int,
    rows: usize,
    cols: usize,
    out_scale: f32,
    /// Input gain in 1/65536 units (<= 65536): halved on saturation.
    gain: AtomicU32,
    quiet: AtomicU32,
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

/// The lane for this matrix, built at first use. `None`: the NPU is off, the
/// matrix is too small to be worth a call, the budget is spent, or the
/// backend refused it.
pub fn lane(t: &QTensor) -> Option<Arc<Lane>> {
    if VERDICT.load(Ordering::Relaxed) == 2 {
        return None;
    }
    let api = api()?;
    let key = t.npu_key()?;
    let mut map = lanes().lock().unwrap();
    if let Some(slot) = map.get(&key) {
        return slot.clone();
    }
    // Until the trial has spoken, one matrix is enough to measure with:
    // every lane costs a graph build and its weights on the DSP.
    if VERDICT.load(Ordering::Relaxed) == 0 && map.values().any(|s| s.is_some()) {
        return None;
    }
    let (rows, cols) = (t.rows(), t.cols());
    let min = env_f32("CMF_NPU_MIN_M", 2.0) as u64 * 1_000_000;
    let budget = env_f32("CMF_NPU_MB", 1024.0) as u64 * 1_000_000;
    let size = (rows * cols) as u64;
    if size < min || BYTES.load(Ordering::Relaxed) + size > budget {
        map.insert(key, None);
        return None;
    }
    let t0 = std::time::Instant::now();
    let mut w = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];
    let mut row = vec![0f32; cols];
    let mut max_norm = 0f64;
    for r in 0..rows {
        t.row_f32(r, &mut row);
        let mut mx = 0f32;
        let mut n = 0f64;
        for &v in &row {
            mx = mx.max(v.abs());
            n += (v as f64) * (v as f64);
        }
        max_norm = max_norm.max(n.sqrt());
        let s = if mx > 0.0 { mx / 127.0 } else { 1e-12 };
        scales[r] = s;
        let inv = 1.0 / s;
        for (d, &v) in w[r * cols..(r + 1) * cols].iter_mut().zip(&row) {
            *d = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
    let alpha = env_f32("CMF_NPU_ALPHA", 0.5);
    let out_scale = (alpha as f64 * max_norm / 32768.0) as f32;
    let h = unsafe { (api.add)(rows as u32, cols as u32, w.as_ptr(), scales.as_ptr(), out_scale) };
    let slot = if h < 0 {
        eprintln!("[cortiq] npu: {rows}x{cols} refused: {}", unsafe { CStr::from_ptr((api.err)()).to_string_lossy() });
        None
    } else {
        BYTES.fetch_add(size, Ordering::Relaxed);
        eprintln!(
            "[cortiq] npu: lane {h} {rows}x{cols} in {:.2} s ({} MB on the NPU)",
            t0.elapsed().as_secs_f64(),
            BYTES.load(Ordering::Relaxed) / 1_000_000
        );
        Some(Arc::new(Lane { handle: h, rows, cols, out_scale, gain: AtomicU32::new(65536), quiet: AtomicU32::new(0) }))
    };
    map.insert(key, slot.clone());
    slot
}

struct Job {
    handle: c_int,
    x: *const u16,
    y: *mut u16,
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
                for j in rx {
                    let ok = unsafe { (api.run)(j.handle, j.x, j.y) } == 0;
                    let _ = j.done.send(ok);
                }
            })
            .expect("npu thread");
        Mutex::new(tx)
    })
}

/// A matvec in flight on the NPU; `finish` waits for it.
pub struct Pending<'a> {
    lane: &'a Lane,
    xq: Vec<u16>,
    yq: Vec<u16>,
    scale: f32,
    rx: mpsc::Receiver<bool>,
}

/// Start `y = W x` on the NPU and return at once.
pub fn start<'a>(lane: &'a Lane, x: &[f32]) -> Option<Pending<'a>> {
    debug_assert_eq!(x.len(), lane.cols);
    let mx = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    if !(mx > 0.0) || !mx.is_finite() {
        return None;
    }
    let gain = lane.gain.load(Ordering::Relaxed) as f32 / 65536.0;
    let k = gain * 32767.0 / mx;
    let xq: Vec<u16> = x.iter().map(|&v| ((v * k).round() as i32 + 32768).clamp(0, 65535) as u16).collect();
    let mut p = Pending { lane, xq, yq: vec![0u16; lane.rows], scale: lane.out_scale * mx / gain, rx: mpsc::sync_channel(1).1 };
    let (tx, rx) = mpsc::sync_channel(1);
    p.rx = rx;
    let job = Job { handle: lane.handle, x: p.xq.as_ptr(), y: p.yq.as_mut_ptr(), done: tx };
    worker().lock().unwrap().send(job).ok()?;
    Some(p)
}

impl Pending<'_> {
    /// Wait for the NPU and write the result. `false`: the output saturated
    /// or the call failed — the caller computes it on the CPU.
    pub fn finish(self, y: &mut [f32]) -> bool {
        CALLS.fetch_add(1, Ordering::Relaxed);
        let ok = self.rx.recv().unwrap_or(false);
        let mut peak = 0u16;
        for (d, &q) in y.iter_mut().zip(&self.yq) {
            let v = q as i32 - 32768;
            peak = peak.max(v.unsigned_abs() as u16);
            *d = v as f32 * self.scale;
        }
        let lane = self.lane;
        if !ok || peak >= 32767 {
            let g = lane.gain.load(Ordering::Relaxed);
            lane.gain.store((g / 2).max(64), Ordering::Relaxed);
            lane.quiet.store(0, Ordering::Relaxed);
            FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // A long quiet stretch well inside the range earns the gain back.
        if peak < 8192 {
            if lane.quiet.fetch_add(1, Ordering::Relaxed) >= 64 {
                let g = lane.gain.load(Ordering::Relaxed);
                lane.gain.store((g * 2).min(65536), Ordering::Relaxed);
                lane.quiet.store(0, Ordering::Relaxed);
            }
        } else {
            lane.quiet.store(0, Ordering::Relaxed);
        }
        true
    }
}

/// `calls/fallbacks` since the process started, for a run's closing line.
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
