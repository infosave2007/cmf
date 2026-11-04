//! A chat turn through the C ABI on a file with a `lookup` record: a key
//! question is the table's text with nothing generated, the decision is in
//! `cortiq_last_route`, an earlier turn's key answers a follow-up, a general
//! question generates, a raw completion is not routed. CPU (`CMF_GPU=0`),
//! synthetic GDN + bounded genome.

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS, lookup_entries};
use std::ffi::{CStr, CString, c_char, c_void};

extern "C" fn collect(token: *const c_char, user: *mut c_void) -> bool {
    let piece = unsafe { CStr::from_ptr(token) }.to_string_lossy();
    unsafe { &mut *(user as *mut String) }.push_str(&piece);
    true
}

fn last_route(h: *mut c_void) -> serde_json::Value {
    let route = unsafe { CStr::from_ptr(cortiq_ffi::cortiq_last_route(h)) };
    serde_json::from_str(route.to_str().unwrap()).unwrap()
}

fn chat(h: *mut c_void, prompt: &str) -> (i32, String) {
    let prompt = CString::new(prompt).unwrap();
    let mut text = String::new();
    let n = cortiq_ffi::cortiq_chat(
        h,
        prompt.as_ptr(),
        3,
        Some(collect),
        &mut text as *mut String as *mut c_void,
    );
    (n, text)
}

fn messages(h: *mut c_void, messages: serde_json::Value) -> (i32, String) {
    let json = CString::new(messages.to_string()).unwrap();
    let mut text = String::new();
    let n = cortiq_ffi::cortiq_chat_messages(
        h,
        json.as_ptr(),
        3,
        Some(collect),
        &mut text as *mut String as *mut c_void,
    );
    (n, text)
}

#[test]
fn chat_answers_from_the_table_and_reports_the_record() {
    // SAFETY: set before any pipeline of this test binary exists.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-ffi-lookup-{}", std::process::id()));
    let files = knowledge_synth::write_lookup_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let path = CString::new(files.f1.to_str().unwrap()).unwrap();
    let h = cortiq_ffi::cortiq_load(path.as_ptr());
    assert!(!h.is_null(), "load failed: {:?}", unsafe {
        CStr::from_ptr(cortiq_ffi::cortiq_last_error())
    });
    assert_eq!(last_route(h), serde_json::json!({}));
    let entries = lookup_entries();

    // The record answers word for word, nothing is generated.
    let (n, text) = chat(h, SKILL_TEXTS[4]);
    assert_eq!(n, 0);
    assert_eq!(text, entries[4].1["fields"]["family"].as_str().unwrap());
    let route = last_route(h);
    assert_eq!(route["target"], SKILL_ID, "{route}");
    assert_eq!(route["lookup_hit"], true, "{route}");
    assert_eq!(route["lookup_key"], "календулы", "{route}");
    assert_eq!(route["lookup_turn"], 0, "{route}");

    // The same through the messages entry point.
    let (n, again) = messages(
        h,
        serde_json::json!([{"role": "user", "content": SKILL_TEXTS[4]}]),
    );
    assert_eq!((n, again.as_str()), (0, text.as_str()));

    // A follow-up that names no plant answers from the turn that named one.
    let followed = [
        "Какое семейство у этого растения?",
        "Чем опасно это растение?",
        "Какие лечебные свойства у этого растения?",
        "Как заваривать это растение?",
    ]
    .iter()
    .any(|follow| {
        let (n, text) = messages(
            h,
            serde_json::json!([
                {"role": "user", "content": SKILL_TEXTS[2]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": follow},
            ]),
        );
        let route = last_route(h);
        route["lookup_hit"] == true
            && route["lookup_entry"] == 2
            && route["lookup_turn"] == 1
            && n == 0
            && !text.is_empty()
    });
    assert!(followed, "no follow-up answered from the earlier turn");

    // A general question generates on the backbone.
    let (n, _) = chat(h, GENERAL_TEXTS[0]);
    assert!(n > 0, "generate failed: {:?}", unsafe {
        CStr::from_ptr(cortiq_ffi::cortiq_last_error())
    });
    let route = last_route(h);
    assert_eq!(route["target"], "backbone", "{route}");
    assert_eq!(route["lookup_hit"], false, "{route}");

    // A raw completion is not routed.
    let raw = CString::new(SKILL_TEXTS[4]).unwrap();
    let n = cortiq_ffi::cortiq_complete(h, raw.as_ptr(), 2, None, std::ptr::null_mut());
    assert!(n > 0, "complete failed: {:?}", unsafe {
        CStr::from_ptr(cortiq_ffi::cortiq_last_error())
    });
    assert_eq!(last_route(h), serde_json::json!({}));

    // A reload after the free opens the tables again.
    cortiq_ffi::cortiq_free(h);
    let h = cortiq_ffi::cortiq_load(path.as_ptr());
    assert!(!h.is_null());
    assert_eq!(chat(h, SKILL_TEXTS[4]), (0, text));
    cortiq_ffi::cortiq_free(h);
    let _ = std::fs::remove_dir_all(&dir);
}
