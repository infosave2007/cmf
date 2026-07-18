//! End-to-end through the C ABI against a real model. Skips (honestly,
//! with a message) when the local model file is absent — CI stays
//! hermetic, developer machines exercise the full path.
use std::ffi::{c_char, c_void, CStr, CString};

extern "C" fn collect(token: *const c_char, user: *mut c_void) -> bool {
    let s = unsafe { CStr::from_ptr(token) }.to_string_lossy().into_owned();
    let out = unsafe { &mut *(user as *mut String) };
    out.push_str(&s);
    true
}

#[test]
fn chat_through_the_abi() {
    let path = std::env::var("CORTIQ_FFI_TEST_MODEL")
        .unwrap_or_else(|_| "/Users/oleg/Documents/cortiq-bot/qwen25-05b-q8.cmf".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("ffi e2e skipped: no model at {path} (set CORTIQ_FFI_TEST_MODEL)");
        return;
    }
    let cpath = CString::new(path).unwrap();
    let h = cortiq_ffi::cortiq_load(cpath.as_ptr());
    assert!(!h.is_null(), "load failed: {:?}", unsafe {
        CStr::from_ptr(cortiq_ffi::cortiq_last_error())
    });
    let mut text = String::new();
    let prompt = CString::new("What is the capital of France? Answer briefly.").unwrap();
    let n = cortiq_ffi::cortiq_chat(
        h,
        prompt.as_ptr(),
        16,
        Some(collect),
        &mut text as *mut String as *mut c_void,
    );
    assert!(n > 0, "generate failed: {:?}", unsafe {
        CStr::from_ptr(cortiq_ffi::cortiq_last_error())
    });
    assert!(text.contains("Paris"), "unexpected answer: {text}");

    // Options: greedy + fixed seed must be accepted and survive to the
    // next generate; bad JSON must fail without touching the handle.
    let opts = CString::new(r#"{"greedy": true, "seed": 7}"#).unwrap();
    assert_eq!(cortiq_ffi::cortiq_set_options(h, opts.as_ptr()), 0);
    let bad = CString::new("{nope").unwrap();
    assert_eq!(cortiq_ffi::cortiq_set_options(h, bad.as_ptr()), -1);

    // Multi-turn: the second user turn only makes sense if the template
    // carried the first exchange.
    let msgs = CString::new(
        r#"[{"role": "user", "content": "My name is Oleg. Remember it."},
            {"role": "assistant", "content": "Understood, Oleg!"},
            {"role": "user", "content": "What is my name? Answer with just the name."}]"#,
    )
    .unwrap();
    let mut text2 = String::new();
    let n = cortiq_ffi::cortiq_chat_messages(
        h,
        msgs.as_ptr(),
        16,
        Some(collect),
        &mut text2 as *mut String as *mut c_void,
    );
    assert!(n > 0, "multiturn failed: {:?}", unsafe {
        CStr::from_ptr(cortiq_ffi::cortiq_last_error())
    });
    assert!(text2.contains("Oleg"), "history lost: {text2}");
    cortiq_ffi::cortiq_free(h);
}
