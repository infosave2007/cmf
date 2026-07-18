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
    cortiq_ffi::cortiq_free(h);
}
