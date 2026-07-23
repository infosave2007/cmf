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
    pipeline: Mutex<Pipeline>,
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
            && let Ok(mut g) = ctx.enable_thinking.lock() {
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
