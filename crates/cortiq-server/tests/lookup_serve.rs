//! `serve` with a `lookup` record (lookup spec §3): the chat endpoint
//! routes on the last user message, a key question is answered from the
//! table as the assistant message (no generation, no lane), the decision
//! in `x_cortiq_route` carries `lookup_key` / `field`, `context` mode
//! generates on the backbone slot with the card prepended, `off` ignores
//! the table, streaming ships the answer in the SSE shape, and
//! `/v1/completions` takes the same path on a single cmf-im-v1 user turn.
//! Conversation memory: a plant named in an earlier user turn answers a
//! later question that names none (`lookup_turn`), the latest named
//! plant wins, the window is `MEMORY_TURNS` user turns. The routing
//! policy `key_first`: a plant prompt the φ router sends to the backbone
//! is answered from the table on a strong key of the LAST message
//! (`decided_by: "key_first"`), never under `router_and_key`, never on a
//! one-word key or a key named only in an earlier turn. CPU
//! (`CMF_GPU=0`), synthetic GDN + bounded genome.

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use axum::body::Body;
use axum::http::Request;
use cortiq_core::CmfModel;
use cortiq_engine::lookup::LookupMode;
use cortiq_engine::{CortiqRuntime, Pipeline, SamplerConfig};
use cortiq_server::{AppState, PipelinePool, SkillRouter, build_router};
use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS, lookup_entries};
use std::sync::Arc;
use tower::ServiceExt;

const NO_KEY_TEXT: &str = "Какие лечебные свойства у мяты перечной?";

fn setup(tag: &str) -> (std::path::PathBuf, Arc<CmfModel>) {
    // SAFETY: set before any pipeline of this test binary exists; every
    // test sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-serve-lookup-{tag}-{}", std::process::id()));
    let files = knowledge_synth::write_lookup_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let model = Arc::new(CmfModel::open(&files.f1).unwrap());
    (dir, model)
}

fn make_app(model: &Arc<CmfModel>, mode: LookupMode) -> (axum::Router, Arc<AppState>, Arc<SkillRouter>) {
    let backbone = Pipeline::from_model(model, SamplerConfig::default()).unwrap();
    let probe = Pipeline::from_model(model, SamplerConfig::default()).unwrap();
    let m = model.clone();
    let router = Arc::new(
        SkillRouter::new(
            model.clone(),
            probe,
            Box::new(move |id: &str| {
                Pipeline::from_model_with_skill(&m, SamplerConfig::default(), Some(id))
                    .map_err(|e| e.to_string())
            }),
            false,
        )
        .with_lookup_mode(mode),
    );
    let state = Arc::new(AppState {
        runtime: CortiqRuntime::new(model.clone()),
        tokenizer: backbone.tokenizer.clone(),
        slots: PipelinePool::new(vec![backbone]),
        remote: None,
        routing: Some(router.clone()),
    });
    (build_router(state.clone()), state, router)
}

async fn post(app: &axum::Router, path: &str, body: serde_json::Value) -> (u16, String) {
    let resp = app
        .clone()
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status().as_u16();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn chat(app: &axum::Router, messages: serde_json::Value) -> serde_json::Value {
    let (status, body) = post(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "cortiq",
            "messages": messages,
            "max_tokens": 3,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).unwrap()
}

fn user(text: &str) -> serde_json::Value {
    serde_json::json!([{"role": "user", "content": text}])
}

#[tokio::test]
async fn chat_answers_from_the_table_and_reports_the_key() {
    let (dir, model) = setup("answer");
    let entries = lookup_entries();
    let (app, state, router) = make_app(&model, LookupMode::Answer);
    assert!(router.has_lookups());
    assert_eq!(router.lookup_ids(), vec![SKILL_ID.to_string()]);
    assert_eq!(router.lookup_mode(), LookupMode::Answer);

    // A key question with a field: the table's field text, no generation.
    let v = chat(&app, user(SKILL_TEXTS[4])).await;
    let family = entries[4].1["fields"]["family"].as_str().unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], family, "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["completion_tokens"], 0);
    let r = &v["x_cortiq_route"];
    assert_eq!(r["target"], SKILL_ID, "{v}");
    assert_eq!(r["lookup_hit"], true);
    assert_eq!(r["lookup_key"], "календулы");
    assert_eq!(r["lookup_entry"], 4);
    assert_eq!(r["lookup_lang"], "ru");
    assert_eq!(r["field"], "family");
    assert_eq!(r["lookup_mode"], "answer");
    assert_eq!(r["lookup_match"], "exact");
    assert_eq!(r["lookup_turn"], 0);
    assert!(r["reason"].as_str().unwrap().contains("lookup key"), "{v}");
    assert!(router.loaded_lanes().is_empty(), "a lookup record has no lane");
    {
        let slot = state.slots.acquire().await;
        assert!(slot.pipe.kv_prefix.is_empty(), "the backbone slot never ran");
    }
    // The LAST user message decides; no field named → the whole card.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": GENERAL_TEXTS[1]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": SKILL_TEXTS[2]},
        ]),
    )
    .await;
    assert_eq!(
        v["choices"][0]["message"]["content"],
        entries[2].1["card"],
        "{v}"
    );
    assert!(v["x_cortiq_route"]["field"].is_null(), "{v}");
    // A general question: the backbone slot generates.
    let v = chat(&app, user(GENERAL_TEXTS[0])).await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert_eq!(v["x_cortiq_route"]["lookup_hit"], false);
    assert!(v["x_cortiq_route"].get("lookup_key").is_none());
    {
        let slot = state.slots.acquire().await;
        assert!(!slot.pipe.kv_prefix.is_empty(), "the backbone slot served it");
    }
    // An in-domain question without a key: the backbone, whatever the
    // router said first.
    let v = chat(&app, user(NO_KEY_TEXT)).await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert_eq!(v["x_cortiq_route"]["lookup_hit"], false);
    assert!(router.loaded_lanes().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn streaming_ships_the_table_answer() {
    let (dir, model) = setup("stream");
    let (app, _state, _router) = make_app(&model, LookupMode::Answer);
    let family = lookup_entries()[4].1["fields"]["family"]
        .as_str()
        .unwrap()
        .to_string();
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "cortiq",
            "messages": user(SKILL_TEXTS[4]),
            "max_tokens": 3,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let chunks: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(chunks.len(), 4, "{body}");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], family);
    assert_eq!(chunks[2]["x_cortiq_route"]["lookup_key"], "календулы");
    assert_eq!(chunks[2]["x_cortiq_route"]["field"], "family");
    assert_eq!(chunks[2]["usage"]["completion_tokens"], 0);
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
    assert!(body.trim_end().ends_with("data: [DONE]"), "{body}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn context_and_off_modes_run_the_backbone_slot() {
    let (dir, model) = setup("modes");
    let family = lookup_entries()[4].1["fields"]["family"]
        .as_str()
        .unwrap()
        .to_string();
    // context: the backbone generates with the card in front; the target
    // and the key are reported; no lane is loaded.
    let (app, state, router) = make_app(&model, LookupMode::Context);
    let v = chat(&app, user(SKILL_TEXTS[4])).await;
    let r = &v["x_cortiq_route"];
    assert_eq!(r["target"], SKILL_ID, "{v}");
    assert_eq!(r["lookup_hit"], true);
    assert_eq!(r["lookup_mode"], "context");
    assert_eq!(r["field"], "family");
    assert_ne!(v["choices"][0]["message"]["content"], family);
    assert!(router.loaded_lanes().is_empty());
    {
        let slot = state.slots.acquire().await;
        assert!(!slot.pipe.kv_prefix.is_empty(), "the backbone slot served it");
    }
    // off: the backbone runs the plain message.
    let (app, _state, router) = make_app(&model, LookupMode::Off);
    let v = chat(&app, user(SKILL_TEXTS[4])).await;
    let r = &v["x_cortiq_route"];
    assert_eq!(r["target"], "backbone", "{v}");
    assert_eq!(r["lookup_hit"], false);
    assert_eq!(r["lookup_mode"], "off");
    assert!(r["reason"].as_str().unwrap().contains("lookup mode off"));
    assert!(router.loaded_lanes().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn completions_take_the_lookup_path_on_a_single_user_turn() {
    let (dir, model) = setup("completions");
    let family = lookup_entries()[4].1["fields"]["family"]
        .as_str()
        .unwrap()
        .to_string();
    let (app, _state, router) = make_app(&model, LookupMode::Answer);
    let completion = |prompt: String| {
        let app = app.clone();
        async move {
            let (status, body) = post(
                &app,
                "/v1/completions",
                serde_json::json!({"model": "cortiq", "prompt": prompt, "max_tokens": 2, "temperature": 0.0}),
            )
            .await;
            assert_eq!(status, 200, "{body}");
            serde_json::from_str::<serde_json::Value>(&body).unwrap()
        }
    };
    let v = completion(cortiq_engine::router::render_cmf_im_v1(SKILL_TEXTS[4])).await;
    assert_eq!(v["choices"][0]["text"], family, "{v}");
    assert_eq!(v["x_cortiq_route"]["target"], SKILL_ID);
    assert_eq!(v["x_cortiq_route"]["lookup_key"], "календулы");
    assert_eq!(v["usage"]["completion_tokens"], 0);
    // A raw prompt is not a user turn: the backbone, no lookup.
    let v = completion(SKILL_TEXTS[4].to_string()).await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert!(v["x_cortiq_route"].get("lookup_hit").is_none());
    assert_ne!(v["choices"][0]["text"], family);
    assert!(router.loaded_lanes().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Follow-up questions that name no plant. The synthetic router (φ of
/// the last message) must send them to the table for the memory to be
/// consulted at all; the test picks the ones it does and requires at
/// least one.
const FOLLOW_UPS: &[&str] = &[
    "Какое семейство у этого растения?",
    "Чем опасно это растение?",
    "Какие лечебные свойства у этого растения?",
    "Как заваривать это растение?",
];

#[tokio::test]
async fn memory_answers_from_the_plant_named_in_an_earlier_turn() {
    let (dir, model) = setup("memory");
    let entries = lookup_entries();
    let (app, _state, router) = make_app(&model, LookupMode::Answer);
    let table = router.lookup_table(SKILL_ID).unwrap().expect("table");
    // The follow-ups hold no key of their own (exact or stem).
    for f in FOLLOW_UPS {
        assert!(table.find_key(f).is_none(), "{f:?} has a key");
    }
    let routed: Vec<&str> = FOLLOW_UPS
        .iter()
        .copied()
        .filter(|f| router.decide(f).target == cortiq_engine::router::RouteTarget::Skill(SKILL_ID.into()))
        .collect();
    assert!(!routed.is_empty(), "the synthetic router sends every follow-up to the backbone");
    let follow = routed[0];
    let family2 = entries[2].1["fields"]["family"].as_str().unwrap();
    let want_field = cortiq_engine::lookup::select_field(follow);
    let expect2 = match want_field {
        Some(f) => entries[2].1["fields"][f].as_str().unwrap().to_string(),
        None => entries[2].1["card"].as_str().unwrap().to_string(),
    };

    // 1. The plant is named one turn back: the key comes from there, the
    //    field from the question asked now.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": SKILL_TEXTS[2]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": follow},
        ]),
    )
    .await;
    let r = &v["x_cortiq_route"];
    assert_eq!(r["target"], SKILL_ID, "{v}");
    assert_eq!(r["lookup_hit"], true, "{v}");
    assert_eq!(r["lookup_entry"], 2, "{v}");
    assert_eq!(r["lookup_turn"], 1, "{v}");
    assert_eq!(r["lookup_match"], "exact");
    assert_eq!(r["field"], serde_json::json!(want_field), "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], expect2, "{v}");
    assert!(r["reason"].as_str().unwrap().contains("1 turns back"), "{v}");

    // 2. An unrelated turn without a key in between: still that plant,
    //    two turns back.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": SKILL_TEXTS[2]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Спасибо, понятно."},
            {"role": "assistant", "content": "Пожалуйста."},
            {"role": "user", "content": follow},
        ]),
    )
    .await;
    let r = &v["x_cortiq_route"];
    assert_eq!((r["lookup_hit"].as_bool(), r["lookup_entry"].as_u64()), (Some(true), Some(2)), "{v}");
    assert_eq!(r["lookup_turn"], 2, "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], expect2, "{v}");

    // 3. A new plant in the latest turn overrides the earlier one.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": SKILL_TEXTS[2]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": SKILL_TEXTS[4]},
        ]),
    )
    .await;
    let r = &v["x_cortiq_route"];
    assert_eq!(r["lookup_entry"], 4, "{v}");
    assert_eq!(r["lookup_turn"], 0, "{v}");
    assert_eq!(r["lookup_key"], "календулы");
    assert_eq!(
        v["choices"][0]["message"]["content"],
        entries[4].1["fields"]["family"].as_str().unwrap(),
        "{v}"
    );
    // The most recent named plant wins even when an older turn names
    // another: the field still comes from the last message.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": SKILL_TEXTS[4]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": SKILL_TEXTS[2]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": follow},
        ]),
    )
    .await;
    let r = &v["x_cortiq_route"];
    assert_eq!((r["lookup_entry"].as_u64(), r["lookup_turn"].as_u64()), (Some(2), Some(1)), "{v}");
    assert_eq!(v["choices"][0]["message"]["content"], expect2, "{v}");
    let _ = family2;

    // 4. The window: a plant named MEMORY_TURNS user turns back (6 turns
    //    in all, the key in the oldest) is remembered; one turn further
    //    back is not — the backbone runs.
    let filler = "Спасибо, понятно.";
    let mut msgs = vec![serde_json::json!({"role": "user", "content": SKILL_TEXTS[2]})];
    for _ in 0..(cortiq_engine::lookup::MEMORY_TURNS - 2) {
        msgs.push(serde_json::json!({"role": "assistant", "content": "ok"}));
        msgs.push(serde_json::json!({"role": "user", "content": filler}));
    }
    msgs.push(serde_json::json!({"role": "assistant", "content": "ok"}));
    msgs.push(serde_json::json!({"role": "user", "content": follow}));
    let n_user = msgs.iter().filter(|m| m["role"] == "user").count();
    assert_eq!(n_user, cortiq_engine::lookup::MEMORY_TURNS);
    let v = chat(&app, serde_json::Value::Array(msgs.clone())).await;
    let r = &v["x_cortiq_route"];
    assert_eq!(r["lookup_hit"], true, "{v}");
    assert_eq!(r["lookup_turn"], cortiq_engine::lookup::MEMORY_TURNS - 1, "{v}");
    // One more filler turn pushes the plant out of the window.
    let mut beyond = msgs.clone();
    let last = beyond.pop().unwrap();
    beyond.push(serde_json::json!({"role": "assistant", "content": "ok"}));
    beyond.push(serde_json::json!({"role": "user", "content": filler}));
    beyond.push(serde_json::json!({"role": "assistant", "content": "ok"}));
    beyond.push(last);
    let v = chat(&app, serde_json::Value::Array(beyond)).await;
    let r = &v["x_cortiq_route"];
    assert_eq!(r["target"], "backbone", "{v}");
    assert_eq!(r["lookup_hit"], false, "{v}");
    assert_eq!(r["decided_target"], SKILL_ID, "the router chose the table; no key in the window");

    // 5. An inflected name in the last turn answers through the stem
    //    index when the router sends it to the table.
    let inflected = "Чем полезен отвар шалфея лекарственного?";
    assert_eq!(table.find_key(inflected).map(|k| (k.entry, k.via)), Some((2, cortiq_engine::lookup::MatchVia::Stem)));
    if router.decide(inflected).target == cortiq_engine::router::RouteTarget::Skill(SKILL_ID.into()) {
        let v = chat(&app, user(inflected)).await;
        let r = &v["x_cortiq_route"];
        assert_eq!(r["lookup_entry"], 2, "{v}");
        assert_eq!(r["lookup_match"], "stem", "{v}");
        assert_eq!(r["lookup_key"], "шалфея лекарственного");
        // `полезен` names no field: the whole card.
        assert_eq!(v["choices"][0]["message"]["content"], entries[2].1["card"], "{v}");
    }
    assert!(router.loaded_lanes().is_empty(), "a lookup record has no lane");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Plant-naming prompts in ENGLISH framing (the synthetic router's
/// backbone class) with a STRONG key, and the entry each names.
const STRONG_KEY_GENERAL: &[(&str, usize)] = &[
    ("Write a Rust function that returns the family of Matricaria chamomilla.", 0),
    ("What is the capital of France, and where does pot marigold grow?", 4),
];
/// A one-word key (`calendula`) in English framing.
const ONE_WORD_KEY_GENERAL: &str = "Say what calendula is made of.";

#[tokio::test]
async fn key_first_answers_backbone_routed_plant_prompts() {
    let (dir, model) = setup("keyfirst");
    let entries = lookup_entries();
    // The same F1 switched to key_first by a header-only update.
    let fk = dir.join("f1-key-first.cmf");
    std::fs::copy(dir.join("f1.cmf"), &fk).unwrap();
    CmfModel::update_header_append(&fk, |h| {
        h.skills[0].lookup.as_mut().unwrap().policy = Some("key_first".into());
    })
    .unwrap();
    let mk = Arc::new(CmfModel::open(&fk).unwrap());
    let (app1, _s1, r1) = make_app(&model, LookupMode::Answer);
    let (appk, statek, rk) = make_app(&mk, LookupMode::Answer);

    for &(text, e) in STRONG_KEY_GENERAL {
        // router_and_key: the φ router sends it to the backbone, which
        // generates (precondition + the recall miss key_first is for).
        let v = chat(&app1, user(text)).await;
        let r = &v["x_cortiq_route"];
        assert_eq!(r["target"], "backbone", "precondition — {text:?}: {v}");
        assert!(r.get("decided_by").is_none(), "{v}");
        assert_eq!(r["lookup_hit"], false, "{v}");
        // key_first: the table answers, nothing is generated.
        let v = chat(&appk, user(text)).await;
        let r = &v["x_cortiq_route"];
        assert_eq!(r["target"], SKILL_ID, "{text:?}: {v}");
        assert_eq!(r["decided_target"], SKILL_ID);
        assert_eq!(r["decided_by"], "key_first", "{v}");
        assert_eq!(r["lookup_hit"], true);
        assert_eq!(r["lookup_entry"], e);
        assert_eq!(r["lookup_lang"], "en");
        assert!(r["lookup_key_words"].as_u64().unwrap() >= 2);
        assert!(r["reason"].as_str().unwrap().starts_with("key_first: lookup 'herbs'"), "{v}");
        assert_eq!(v["usage"]["completion_tokens"], 0);
        let card = &entries[e].2;
        let want = match r["field"].as_str() {
            Some(f) => card["fields"][f].as_str().unwrap(),
            None => card["card"].as_str().unwrap(),
        };
        assert_eq!(v["choices"][0]["message"]["content"], want, "{v}");
    }
    assert!(rk.loaded_lanes().is_empty() && r1.loaded_lanes().is_empty());
    {
        let slot = statek.slots.acquire().await;
        assert!(slot.pipe.kv_prefix.is_empty(), "the backbone slot never ran on key_first");
    }
    // A one-word key and a general prompt: the backbone, as the router said.
    for text in [ONE_WORD_KEY_GENERAL, GENERAL_TEXTS[0]] {
        let v = chat(&appk, user(text)).await;
        let r = &v["x_cortiq_route"];
        assert_eq!(r["target"], "backbone", "{text:?}: {v}");
        assert_eq!(r["lookup_hit"], false);
        assert!(r.get("decided_by").is_none(), "{v}");
    }
    // A strong key only in an EARLIER user turn: the last message decides.
    let v = chat(
        &appk,
        serde_json::json!([
            {"role": "user", "content": STRONG_KEY_GENERAL[0].0},
            {"role": "assistant", "content": "Asteraceae"},
            {"role": "user", "content": GENERAL_TEXTS[0]},
        ]),
    )
    .await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert_eq!(v["x_cortiq_route"]["lookup_hit"], false);
    // /v1/completions: the same pre-pass on a single cmf-im-v1 user turn.
    let (text, e) = STRONG_KEY_GENERAL[1];
    let (status, body) = post(
        &appk,
        "/v1/completions",
        serde_json::json!({
            "model": "cortiq",
            "prompt": cortiq_engine::router::render_cmf_im_v1(text),
            "max_tokens": 2,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["x_cortiq_route"]["decided_by"], "key_first", "{v}");
    assert_eq!(v["x_cortiq_route"]["lookup_entry"], e);
    assert_eq!(v["usage"]["completion_tokens"], 0);
    // Context mode: the backbone slot generates with the card in front.
    let (appc, statec, _) = make_app(&mk, LookupMode::Context);
    let v = chat(&appc, user(STRONG_KEY_GENERAL[0].0)).await;
    let r = &v["x_cortiq_route"];
    assert_eq!((r["target"].as_str(), r["decided_by"].as_str(), r["lookup_mode"].as_str()), (Some(SKILL_ID), Some("key_first"), Some("context")), "{v}");
    {
        let slot = statec.slots.acquire().await;
        assert!(!slot.pipe.kv_prefix.is_empty(), "the backbone slot served it");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
