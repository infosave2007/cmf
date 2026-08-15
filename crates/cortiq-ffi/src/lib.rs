//! C ABI over the CMF runtime — the embedding surface for mobile apps
//! (Android JNI / iOS / desktop FFI). Design rules:
//! - opaque handle, every call goes through a Mutex (the engine is
//!   single-stream; callers may invoke from any thread, one at a time);
//! - no panics across the boundary (catch_unwind on every entry);
//! - errors are a thread-local UTF-8 string behind `cortiq_last_error`;
//! - streaming via a C callback returning `true` to continue — early
//!   stop is first-class, matching the engine's own TokenCallback.
// The entry points take raw pointers from a foreign caller by design;
// each one NULL-checks before dereferencing. Marking them `unsafe`
// would change nothing for C callers and only obscure the Rust tests.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};

use cortiq_core::CmfModel;
use cortiq_engine::{Pipeline, SamplerConfig};

struct Ctx {
    /// Kept so the network split can prove identity (`dir_hash`) and
    /// geometry to a peer without reopening the file.
    model: Arc<CmfModel>,
    pipeline: Mutex<Pipeline>,
    /// Clone of the pipeline's cancel flag — reachable while the
    /// pipeline mutex is held by a running generation.
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Sticky `enable_thinking` for reasoning-model chat templates
    /// (Qwen3/3.5): `None` leaves it undefined so the template picks its own
    /// default; `Some(false)` makes the model answer directly instead of
    /// emitting a `<think>` block. Set through `cortiq_set_options`.
    enable_thinking: Mutex<Option<bool>>,
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<CString> =
        std::cell::RefCell::new(CString::new("").unwrap());
}

thread_local! {
    /// Return buffer for JSON getters (`cortiq_peer_stats`). Per-thread
    /// so two callers cannot free each other's string.
    static LAST_JSON: std::cell::RefCell<CString> =
        std::cell::RefCell::new(CString::new("{}").unwrap());
}

/// Park a JSON answer in this thread's buffer and hand out its pointer.
/// Valid until the next JSON getter call on the same thread.
fn set_last_json(s: &str) -> *const c_char {
    LAST_JSON.with(|j| {
        *j.borrow_mut() = CString::new(s.replace('\0', " ")).unwrap_or_default();
        j.borrow().as_ptr()
    })
}

fn set_error(msg: &str) {
    let clean = msg.replace('\0', " ");
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(clean).unwrap());
}

/// UTF-8 description of the most recent failure ON THIS THREAD.
/// Valid until the next failing call from the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Engine version as a static UTF-8 string.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_version() -> *const c_char {
    static V: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    V.as_ptr() as *const c_char
}

/// Open a `.cmf` file and build the pipeline. Returns an opaque handle,
/// or NULL (see `cortiq_last_error`). The file is memory-mapped: keep it
/// on storage for the handle's lifetime.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_load(path: *const c_char) -> *mut c_void {
    let result = catch_unwind(|| {
        if path.is_null() {
            set_error("path is NULL");
            return std::ptr::null_mut();
        }
        let path = match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(p) => p,
            Err(_) => {
                set_error("path is not valid UTF-8");
                return std::ptr::null_mut();
            }
        };
        let model = match CmfModel::open_sharded(path) {
            Ok(m) => Arc::new(m),
            Err(e) => {
                set_error(&format!("open: {e}"));
                return std::ptr::null_mut();
            }
        };
        let pipeline = match Pipeline::from_model(&model, SamplerConfig::default()) {
            Ok(p) => p,
            Err(e) => {
                set_error(&format!("pipeline: {e}"));
                return std::ptr::null_mut();
            }
        };
        Box::into_raw(Box::new(Ctx {
            model,
            cancel: pipeline.cancel.clone(),
            pipeline: Mutex::new(pipeline),
            enable_thinking: Mutex::new(None),
        })) as *mut c_void
    });
    result.unwrap_or_else(|_| {
        set_error("panic during load");
        std::ptr::null_mut()
    })
}

/// Globally enable or disable the discrete GPU (Vulkan/DX12/Metal) graph.
/// Must be called before `cortiq_load` to take effect.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_set_gpu(enable: bool) {
    cortiq_engine::pipeline::GLOBAL_USE_GPU.store(enable, std::sync::atomic::Ordering::Relaxed);
}

/// True when this build carries a working GPU backend AND the device
/// can bring an adapter up (Vulkan on Android, Metal on iOS/macOS).
/// Distinguishes "GPU off" from "GPU impossible": a CPU-only library
/// returns false here while `cortiq_set_gpu` still accepts the flag.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_gpu_available() -> bool {
    cortiq_engine::gpu::backend_available()
}

/// Pin the worker-pool size from the embedder instead of the
/// process-wide `CMF_THREADS` environment variable. 0 restores the
/// automatic choice (env, then big-core topology). Call before
/// `cortiq_load` — the pool is sized once per load.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_set_threads(n: i32) {
    cortiq_engine::pool::FORCED_THREADS
        .store(n.max(0) as usize, std::sync::atomic::Ordering::Relaxed);
}

/// Kernel thread ids of the current worker pool (Android/Linux) — what
/// ADPF's PerformanceHintManager needs to report work durations to the
/// scheduler. Copies up to `cap` ids into `out`, returns the total
/// worker count (call again with a larger buffer if it exceeds `cap`).
/// Returns 0 on platforms without stable kernel tids or before a load.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_worker_tids(out: *mut i32, cap: i32) -> i32 {
    let tids = match cortiq_engine::pool::WORKER_TIDS.lock() {
        Ok(t) => t.clone(),
        Err(_) => return 0,
    };
    if !out.is_null() && cap > 0 {
        let n = tids.len().min(cap as usize);
        unsafe { std::ptr::copy_nonoverlapping(tids.as_ptr(), out, n) };
    }
    tids.len() as i32
}

/// Cancel the generation currently running on this handle (safe from
/// any thread — this is the point: `cortiq_chat*` blocks its caller).
/// The engine checks the flag at every prefill chunk and decode step
/// and finishes with `finish_reason: "cancelled"`. No-op when nothing
/// runs; the flag clears itself once honoured.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_cancel(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let ctx = unsafe { &*(handle as *const Ctx) };
    ctx.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// One-line execution summary as JSON, for status/About surfaces:
/// `{"simd":"neon","threads":4,"gpu_backend":true}`. `threads` is the
/// REAL worker-pool resolution (forced > CMF_THREADS > topology) —
/// the number the pool will actually spawn, valid before AND after
/// load. The returned pointer is a process-lifetime C string.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_execution_info() -> *const c_char {
    use std::sync::OnceLock;
    static INFO: OnceLock<std::ffi::CString> = OnceLock::new();
    INFO.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        let simd = "neon";
        #[cfg(target_arch = "x86_64")]
        let simd = "avx2";
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        let simd = "none";
        let threads = cortiq_engine::pool::Pool::effective_threads().max(1);
        let gpu = cortiq_engine::gpu::backend_available();
        std::ffi::CString::new(format!(
            "{{\"simd\":\"{simd}\",\"threads\":{threads},\"gpu_backend\":{gpu}}}"
        ))
        .expect("no interior NUL")
    })
    .as_ptr()
}

/// Release the handle. NULL is a no-op. Do not use the handle afterwards.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(handle as *mut Ctx) });
    }));
}

/// Streaming token callback: `token` is a NUL-terminated UTF-8 piece
/// (valid only during the call); return `true` to continue generating.
pub type CortiqTokenCb = Option<extern "C" fn(token: *const c_char, user: *mut c_void) -> bool>;

enum GenInput {
    Chat(String),
    Raw(String),
    History(Vec<(String, String)>),
}

fn run_generate(
    handle: *mut c_void,
    prompt: *const c_char,
    max_tokens: u32,
    chat: bool,
    cb: CortiqTokenCb,
    user: *mut c_void,
) -> i32 {
    if handle.is_null() {
        set_error("handle is NULL");
        return -1;
    }
    if prompt.is_null() {
        set_error("prompt is NULL");
        return -1;
    }
    let prompt = match unsafe { CStr::from_ptr(prompt) }.to_str() {
        Ok(p) => p.to_string(),
        Err(_) => {
            set_error("prompt is not valid UTF-8");
            return -1;
        }
    };
    let input = if chat {
        GenInput::Chat(prompt)
    } else {
        GenInput::Raw(prompt)
    };
    run_generate_ids(handle, input, max_tokens, cb, user)
}

fn run_generate_ids(
    handle: *mut c_void,
    input: GenInput,
    max_tokens: u32,
    cb: CortiqTokenCb,
    user: *mut c_void,
) -> i32 {
    let ctx = unsafe { &*(handle as *const Ctx) };
    let mut pipeline = match ctx.pipeline.lock() {
        Ok(g) => g,
        Err(_) => {
            set_error("pipeline mutex poisoned");
            return -1;
        }
    };
    // The raw pointer travels into the engine callback; the callback
    // contract (called synchronously on this thread) makes that sound.
    struct UserPtr(*mut c_void);
    unsafe impl Send for UserPtr {}
    impl UserPtr {
        // Accessor keeps the closure capturing &UserPtr — 2021 disjoint
        // capture would otherwise grab the raw pointer field itself.
        fn get(&self) -> *mut c_void {
            self.0
        }
    }
    let user = UserPtr(user);
    let on_token: Option<cortiq_engine::TokenCallback> = cb.map(|f| {
        Box::new(move |piece: &str| -> bool {
            match CString::new(piece.replace('\0', " ")) {
                Ok(c) => f(c.as_ptr(), user.get()),
                Err(_) => true,
            }
        }) as cortiq_engine::TokenCallback
    });
    let thinking = match ctx.enable_thinking.lock() {
        Ok(g) => *g,
        Err(_) => None,
    };
    let ids = match input {
        GenInput::Chat(prompt) => {
            let history = vec![("user".to_string(), prompt)];
            pipeline
                .tokenizer
                .apply_chat_template_opts(&history, thinking)
        }
        GenInput::Raw(prompt) => pipeline
            .tokenizer
            .with_bos(pipeline.tokenizer.encode(&prompt)),
        GenInput::History(history) => pipeline
            .tokenizer
            .apply_chat_template_opts(&history, thinking),
    };
    // A configured peer routes the whole generation through the layer
    // split. Same ids, same callback, same result shape — the caller
    // cannot tell except by the speed and the model it can now run.
    if peer_configured() {
        return match generate_over_peer(ctx, &mut pipeline, &ids, max_tokens as usize, on_token) {
            Ok(n) => n,
            Err(e) => {
                set_error(&format!("peer generate: {e}"));
                -1
            }
        };
    }
    match pipeline.generate_from_ids(&ids, max_tokens as usize, None, on_token) {
        Ok(res) => res.tokens_generated as i32,
        Err(e) => {
            set_error(&format!("generate: {e}"));
            -1
        }
    }
}

/// Partial sampler options as JSON — absent fields keep their current
/// values. Accepted keys: temperature, top_p, top_k,
/// repetition_penalty, min_p, seed, greedy (true = argmax: temperature
/// pinned to 0), enable_thinking (false makes reasoning models —
/// Qwen3/3.5 — answer directly with no `<think>` block; true re-enables it;
/// absent/null keeps the current value). Applies to every subsequent generate
/// on this handle. Returns 0, or −1 (`cortiq_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_set_options(handle: *mut c_void, options_json: *const c_char) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || options_json.is_null() {
            set_error("handle or options is NULL");
            return -1;
        }
        let json = match unsafe { CStr::from_ptr(options_json) }.to_str() {
            Ok(j) => j,
            Err(_) => {
                set_error("options is not valid UTF-8");
                return -1;
            }
        };
        #[derive(serde::Deserialize)]
        struct Opts {
            temperature: Option<f32>,
            top_p: Option<f32>,
            top_k: Option<u32>,
            repetition_penalty: Option<f32>,
            min_p: Option<f32>,
            seed: Option<u64>,
            greedy: Option<bool>,
            // Absent or `null` leaves the sticky value untouched (serde folds a
            // JSON `null` into the outer `None`); `true`/`false` pins it. To go
            // back to the template default, reload the handle.
            enable_thinking: Option<Option<bool>>,
        }
        let opts: Opts = match serde_json::from_str(json) {
            Ok(o) => o,
            Err(e) => {
                set_error(&format!("options: {e}"));
                return -1;
            }
        };
        let ctx = unsafe { &*(handle as *const Ctx) };
        let mut pipeline = match ctx.pipeline.lock() {
            Ok(g) => g,
            Err(_) => {
                set_error("pipeline mutex poisoned");
                return -1;
            }
        };
        let mut next = pipeline.sampler_config.clone();
        if let Some(v) = opts.temperature {
            if !v.is_finite() || v < 0.0 {
                set_error("temperature must be finite and >= 0");
                return -1;
            }
            next.temperature = v;
        }
        if let Some(v) = opts.top_p {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                set_error("top_p must be finite and between 0 and 1");
                return -1;
            }
            next.top_p = v;
        }
        if let Some(v) = opts.top_k {
            next.top_k = v;
        }
        if let Some(v) = opts.repetition_penalty {
            if !v.is_finite() || v <= 0.0 {
                set_error("repetition_penalty must be finite and > 0");
                return -1;
            }
            next.repetition_penalty = v;
        }
        if let Some(v) = opts.min_p {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                set_error("min_p must be finite and between 0 and 1");
                return -1;
            }
            next.min_p = v;
        }
        if opts.seed.is_some() {
            next.seed = opts.seed;
        }
        if opts.greedy == Some(true) {
            next.temperature = 0.0;
        }
        pipeline.set_sampler_config(next);
        drop(pipeline);
        if let Some(v) = opts.enable_thinking
            && let Ok(mut g) = ctx.enable_thinking.lock()
        {
            *g = v;
        }
        0
    }))
    .unwrap_or_else(|_| {
        set_error("panic during set_options");
        -1
    })
}

/// Multi-turn chat: `messages_json` is `[{"role": "...", "content":
/// "..."}, ...]` rendered through the file's own chat template — the
/// canonical way to carry a conversation (roles the template knows:
/// typically system / user / assistant). Same streaming/return contract
/// as `cortiq_chat`.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_chat_messages(
    handle: *mut c_void,
    messages_json: *const c_char,
    max_tokens: u32,
    cb: CortiqTokenCb,
    user: *mut c_void,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if handle.is_null() || messages_json.is_null() {
            set_error("handle or messages is NULL");
            return -1;
        }
        let json = match unsafe { CStr::from_ptr(messages_json) }.to_str() {
            Ok(j) => j,
            Err(_) => {
                set_error("messages is not valid UTF-8");
                return -1;
            }
        };
        #[derive(serde::Deserialize)]
        struct Msg {
            role: String,
            content: String,
        }
        let msgs: Vec<Msg> = match serde_json::from_str(json) {
            Ok(m) => m,
            Err(e) => {
                set_error(&format!("messages: {e}"));
                return -1;
            }
        };
        if msgs.is_empty() {
            set_error("messages is empty");
            return -1;
        }
        let history: Vec<(String, String)> =
            msgs.into_iter().map(|m| (m.role, m.content)).collect();
        run_generate_ids(handle, GenInput::History(history), max_tokens, cb, user)
    }))
    .unwrap_or_else(|_| {
        set_error("panic during generate");
        -1
    })
}

/// One chat turn: the file's own chat template wraps the prompt (models
/// without a template fall back to plain completion). Tokens stream
/// through `cb`; returns the generated-token count, or −1
/// (`cortiq_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_chat(
    handle: *mut c_void,
    prompt: *const c_char,
    max_tokens: u32,
    cb: CortiqTokenCb,
    user: *mut c_void,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        run_generate(handle, prompt, max_tokens, true, cb, user)
    }))
    .unwrap_or_else(|_| {
        set_error("panic during generate");
        -1
    })
}

/// Raw completion: the prompt goes to the model verbatim (plus the
/// tokenizer's BOS contract). Same streaming/return contract as
/// `cortiq_chat`.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_complete(
    handle: *mut c_void,
    prompt: *const c_char,
    max_tokens: u32,
    cb: CortiqTokenCb,
    user: *mut c_void,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        run_generate(handle, prompt, max_tokens, false, cb, user)
    }))
    .unwrap_or_else(|_| {
        set_error("panic during generate");
        -1
    })
}

/// Text → image (Lumina-Image 2.0 packed `.cmf`). Renders `prompt` at
/// `width`×`height` (multiples of 16) with `steps` denoising steps and
/// CFG `guidance` (≤1 disables CFG and halves the work — the right
/// default on phones), writing `height·width·3` interleaved RGB8 bytes
/// into caller-allocated `out_rgb`. `progress` (nullable) fires after
/// each denoising step with (step, total, user). The model file is
/// opened per call (mmap — cheap); weights stream from the map, so
/// peak RSS stays far below the file size. GPU (Metal / Vulkan via
/// `cortiq_set_gpu`) is probed against the CPU per process — enabling
/// it never makes generation slower. Returns 0 on success, -1 on
/// error (`cortiq_last_error` has the message).
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_imagine(
    model_path: *const c_char,
    prompt: *const c_char,
    width: u32,
    height: u32,
    steps: u32,
    guidance: f32,
    seed: u64,
    out_rgb: *mut u8,
    progress: Option<extern "C" fn(step: u32, total: u32, user: *mut c_void)>,
    user: *mut c_void,
) -> i32 {
    if model_path.is_null() || prompt.is_null() || out_rgb.is_null() {
        set_error("null argument");
        return -1;
    }
    let (path, prompt) = unsafe {
        let Ok(p) = CStr::from_ptr(model_path).to_str() else {
            set_error("model_path is not UTF-8");
            return -1;
        };
        let Ok(t) = CStr::from_ptr(prompt).to_str() else {
            set_error("prompt is not UTF-8");
            return -1;
        };
        (p.to_string(), t.to_string())
    };
    let params = cortiq_engine::imagegen::GenParams {
        height: height as usize,
        width: width as usize,
        steps: steps.max(1) as usize,
        guidance_scale: guidance,
        seed,
        ..Default::default()
    };
    let user_addr = user as usize; // callbacks may hop pool threads
    let img = match cortiq_engine::imagegen::generate(
        std::path::Path::new(&path),
        &prompt,
        &params,
        |i, n| {
            if let Some(cb) = progress {
                cb(i as u32, n as u32, user_addr as *mut c_void);
            }
        },
    ) {
        Ok(v) => v,
        Err(e) => {
            set_error(&e);
            return -1;
        }
    };
    // [3, h, w] planar f32 in [0,1] → interleaved RGB8.
    let plane = (height as usize) * (width as usize);
    let dst = unsafe { std::slice::from_raw_parts_mut(out_rgb, plane * 3) };
    for p in 0..plane {
        for ch in 0..3 {
            dst[p * 3 + ch] = (img[ch * plane + p].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ABI functions are plain Rust calls in-crate: exercise the
    /// error paths without a model file.
    #[test]
    fn null_arguments_error_cleanly() {
        assert!(cortiq_load(std::ptr::null()).is_null());
        let err = unsafe { CStr::from_ptr(cortiq_last_error()) };
        assert!(!err.to_bytes().is_empty());
        assert_eq!(
            cortiq_chat(
                std::ptr::null_mut(),
                std::ptr::null(),
                8,
                None,
                std::ptr::null_mut()
            ),
            -1
        );
        cortiq_free(std::ptr::null_mut());
        let v = unsafe { CStr::from_ptr(cortiq_version()) };
        assert!(v.to_str().unwrap().starts_with("0."));
    }
}

// ── Network: the worker as a library, and the peer as a setting ──────
//
// Phase 4b of the multi-device roadmap needs the worker to BE a library,
// not only a binary: iOS cannot spawn one, and an Android app that ships
// a `.so` should not have to ship a second executable next to it. Both
// entry points are additive — a caller that never touches them sees the
// old ABI unchanged.

/// `cortiq worker` as a call: hold this device's model and serve layer
/// spans to a coordinator. Runs on a background thread and returns
/// immediately; the listener lives until the process exits.
///
/// JSON: `{"model":"/path/x.cmf","listen":"0.0.0.0:9911","token":"secret"}`
/// `token` is REQUIRED for any address beyond loopback — the worker
/// refuses otherwise rather than serving a stranger's coordinator.
/// Returns 0 once the socket is bound and listening, −1 with
/// `cortiq_last_error` if the bind or the model open failed.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_worker_start(config_json: *const c_char) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(cfg) = read_json(config_json, "worker config") else {
            return -1;
        };
        let model = match cfg.get("model").and_then(|v| v.as_str()) {
            Some(m) => m.to_string(),
            None => {
                set_error("worker config: \"model\" (path to the .cmf) is required");
                return -1;
            }
        };
        let listen = cfg
            .get("listen")
            .and_then(|v| v.as_str())
            .unwrap_or("127.0.0.1:9911")
            .to_string();
        let token = cfg
            .get("token")
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // Bind here, on the caller's thread, so a busy port or a missing
        // file is an error the app can show — not a silent dead thread.
        let probe = std::net::TcpListener::bind(&listen);
        match probe {
            Ok(l) => drop(l),
            Err(e) => {
                set_error(&format!("worker listen {listen}: {e}"));
                return -1;
            }
        }
        if !std::path::Path::new(&model).exists() {
            set_error(&format!("worker model {model}: no such file"));
            return -1;
        }
        let cfg = cortiq_net::WorkerConfig {
            model_path: model,
            listen: listen.clone(),
            token,
        };
        std::thread::Builder::new()
            .name("cortiq-worker".into())
            .spawn(move || {
                if let Err(e) = cortiq_net::worker_serve(cfg) {
                    eprintln!("cortiq worker: {e}");
                }
            })
            .map(|_| 0)
            .unwrap_or_else(|e| {
                set_error(&format!("worker thread: {e}"));
                -1
            })
    }))
    .unwrap_or_else(|_| {
        set_error("panic in cortiq_worker_start");
        -1
    })
}

/// Point this process at a `cortiq worker` holding the tail of the SAME
/// model: every later generate runs as a layer split.
///
/// JSON: `{"addr":"192.168.1.5:9911","token":"secret","split":0,
///         "dtype":"f16","head":true}`
/// - `split` — first layer the peer runs; 0 = it runs all of them and
///   this side keeps only the tokenizer. Default: half the stack.
/// - `head` — the peer also owns lm_head and the sampler and answers
///   token ids. On a phone that is worth ~29 ms of a 73 ms token,
///   because the head does not shrink as you move layers away.
/// - `dtype` — `f32` reproduces the local text bit for bit; `f16` halves
///   the wire and legally changes it.
///
/// Pass `null` or `{}` to clear and go back to local-only. Returns 0, or
/// −1 with `cortiq_last_error`. The connection itself is made lazily on
/// the next generate, so a peer that is not up yet fails THERE, with a
/// message, instead of at configuration time.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_set_peer(config_json: *const c_char) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if config_json.is_null() {
            clear_peer();
            return 0;
        }
        let Some(cfg) = read_json(config_json, "peer config") else {
            return -1;
        };
        let Some(addr) = cfg.get("addr").and_then(|v| v.as_str()) else {
            clear_peer();
            return 0;
        };
        let dtype = cfg.get("dtype").and_then(|v| v.as_str()).unwrap_or("f32");
        if dtype != "f32" && dtype != "f16" {
            set_error(&format!("peer config: dtype {dtype:?} is not f32 or f16"));
            return -1;
        }
        let want = PeerCfg {
            addr: addr.to_string(),
            token: cfg
                .get("token")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            split: cfg
                .get("split")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize),
            f16: dtype == "f16",
            head: cfg.get("head").and_then(|v| v.as_bool()).unwrap_or(false),
            run_ahead: cfg
                .get("run_ahead")
                .and_then(|v| v.as_u64())
                .unwrap_or(1)
                .clamp(1, 64) as u32,
        };
        match PEER.lock() {
            Ok(mut g) => {
                // A changed target invalidates the live connection; the
                // next generate dials the new one.
                if g.as_ref().map(|c| c.cfg.clone()) != Some(want.clone()) {
                    *g = Some(PeerState {
                        cfg: want,
                        seg: None,
                    });
                }
                0
            }
            Err(_) => {
                set_error("peer mutex poisoned");
                -1
            }
        }
    }))
    .unwrap_or_else(|_| {
        set_error("panic in cortiq_set_peer");
        -1
    })
}

#[derive(Clone, PartialEq)]
struct PeerCfg {
    addr: String,
    token: String,
    split: Option<usize>,
    f16: bool,
    head: bool,
    run_ahead: u32,
}

struct PeerState {
    cfg: PeerCfg,
    /// Kept across generates so cross-turn KV reuse survives a chat turn.
    seg: Option<cortiq_net::RemoteSegment>,
}

static PEER: Mutex<Option<PeerState>> = Mutex::new(None);

fn clear_peer() {
    if let Ok(mut g) = PEER.lock() {
        *g = None;
    }
}

fn peer_configured() -> bool {
    PEER.lock().map(|g| g.is_some()).unwrap_or(false)
}

fn read_json(p: *const c_char, what: &str) -> Option<serde_json::Value> {
    if p.is_null() {
        set_error(&format!("{what}: NULL"));
        return None;
    }
    let s = match unsafe { CStr::from_ptr(p) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error(&format!("{what}: not valid UTF-8"));
            return None;
        }
    };
    match serde_json::from_str(s) {
        Ok(v) => Some(v),
        Err(e) => {
            set_error(&format!("{what}: {e}"));
            None
        }
    }
}

/// Run one generation over the configured peer. The segment is cached
/// between calls: a chat turn that extends the previous one reuses both
/// KVs, which is the whole point of holding the connection open.
fn generate_over_peer(
    ctx: &Ctx,
    pipeline: &mut Pipeline,
    ids: &[u32],
    max_tokens: usize,
    on_token: Option<cortiq_engine::TokenCallback>,
) -> Result<i32, String> {
    let mut guard = PEER.lock().map_err(|_| "peer mutex poisoned".to_string())?;
    let state = guard.as_mut().ok_or("no peer configured")?;
    let nl = pipeline.num_layers;
    let split = state.cfg.split.unwrap_or(nl / 2).min(nl - 1);
    if state.seg.is_none() {
        let dtype = if state.cfg.f16 {
            cortiq_net::WireDtype::F16
        } else {
            cortiq_net::WireDtype::F32
        };
        let spec = cortiq_net::SessionSpec {
            skill: None,
            task: None,
            o1: None,
            head: state.cfg.head,
            run_ahead: state.cfg.run_ahead,
            sampler: if state.cfg.head {
                Some(
                    serde_json::to_string(&pipeline.sampler_config)
                        .map_err(|e| format!("sampler config: {e}"))?,
                )
            } else {
                None
            },
        };
        state.seg = Some(cortiq_net::RemoteSegment::connect(
            &state.cfg.addr,
            &state.cfg.token,
            ctx.model.dir_hash(),
            &ctx.model.arch().arch_name,
            nl,
            pipeline.hidden_size,
            split,
            nl - 1,
            dtype,
            &spec,
        )?);
    }
    let seg = state.seg.as_mut().expect("connected above");
    match cortiq_net::generate_split(pipeline, seg, ids, max_tokens, None, on_token) {
        Ok((res, _stats)) => Ok(res.tokens_generated as i32),
        Err(e) => {
            // A broken wire must not leave a half-session behind: drop
            // the segment so the next call redials instead of inheriting
            // a KV the peer no longer has.
            state.seg = None;
            Err(e)
        }
    }
}

/// What the configured peer is worth right now, as JSON — thermal state,
/// mains power, the clock its fastest core is actually running at, free
/// memory. `{}` when no peer is configured. The string is valid until
/// the next call from this thread.
///
/// A phone's budget is not a constant: the same worker served a span at
/// 22.6 ms and then 42.4 ms with no temperature change, purely because
/// the governor let the clock fall. An app that plans once and never
/// looks again will keep a plan that stopped being true.
#[unsafe(no_mangle)]
pub extern "C" fn cortiq_peer_stats() -> *const c_char {
    let json = catch_unwind(AssertUnwindSafe(|| {
        let mut guard = PEER.lock().ok()?;
        let state = guard.as_mut()?;
        let seg = state.seg.as_mut()?;
        match seg.stats() {
            Ok(s) => serde_json::to_string(&s).ok(),
            Err(_) => None,
        }
    }))
    .ok()
    .flatten()
    .unwrap_or_else(|| "{}".to_string());
    set_last_json(&json)
}
