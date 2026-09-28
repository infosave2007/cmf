//! `serve` under router policy v2: per-request routing on the last user
//! message, a lazily loaded lane per routed skill, the backbone on the
//! server's own slots, the decision in the response (`x_cortiq_route`),
//! and no slot ever switching its overlay — each lane keeps its own
//! prefix-reuse state. CPU (`CMF_GPU=0`), synthetic GDN + bounded genome.

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;
#[path = "../../cortiq-engine/tests/common/knowledge_synth.rs"]
mod knowledge_synth;

use axum::body::Body;
use axum::http::Request;
use cortiq_core::CmfModel;
use cortiq_engine::router::RouteTarget;
use cortiq_engine::{CortiqRuntime, Pipeline, SamplerConfig};
use cortiq_server::{AppState, PipelinePool, SkillRouter, build_router};
use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS};
use std::sync::Arc;
use tower::ServiceExt;

fn setup(tag: &str) -> (std::path::PathBuf, Arc<CmfModel>) {
    // SAFETY: set before any pipeline of this test binary exists; every
    // test sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-serve-route-{tag}-{}", std::process::id()));
    let files = knowledge_synth::write_knowledge_pair(
        &dir,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        "active",
    );
    let model = Arc::new(CmfModel::open(&files.f1).unwrap());
    (dir, model)
}

fn skill_router(model: &Arc<CmfModel>) -> SkillRouter {
    let probe = Pipeline::from_model(model, SamplerConfig::default()).unwrap();
    let m = model.clone();
    SkillRouter::new(
        model.clone(),
        probe,
        Box::new(move |id: &str| {
            Pipeline::from_model_with_skill(&m, SamplerConfig::default(), Some(id))
                .map_err(|e| e.to_string())
        }),
        false,
    )
}

/// Unit level: the decision picks the lane; a skill lane is created once,
/// on its first request, holding a pipeline loaded WITH the skill.
#[tokio::test]
async fn skill_router_picks_a_lane_per_request() {
    let (dir, model) = setup("unit");
    let r = skill_router(&model);
    assert_eq!(r.routable_skills(), vec![SKILL_ID.to_string()]);

    let d = r.decide(GENERAL_TEXTS[0]);
    assert_eq!(d.target, RouteTarget::Backbone, "{}", d.reason);
    assert!(
        r.lane(&d.target).unwrap().is_none(),
        "the backbone uses the server's slots"
    );
    assert!(
        r.loaded_lanes().is_empty(),
        "no skill lane before a skill request"
    );

    let d = r.decide(SKILL_TEXTS[1]);
    assert_eq!(
        d.target,
        RouteTarget::Skill(SKILL_ID.into()),
        "{}",
        d.reason
    );
    let lane = r.lane(&d.target).unwrap().expect("skill lane");
    assert_eq!(r.loaded_lanes(), vec![SKILL_ID.to_string()]);
    let again = r.lane(&d.target).unwrap().expect("skill lane");
    assert!(Arc::ptr_eq(&lane, &again), "the lane is created once");
    let idx = model.header.skills.iter().position(|s| s.id == SKILL_ID);
    let slot = lane.acquire().await;
    assert_eq!(
        slot.pipe.active_skill(),
        idx,
        "the lane pipeline carries the skill"
    );
    drop(slot);
    let _ = std::fs::remove_dir_all(&dir);
}

async fn chat(app: &axum::Router, messages: serde_json::Value) -> serde_json::Value {
    let body = serde_json::json!({
        "model": "cortiq",
        "messages": messages,
        "max_tokens": 3,
        "temperature": 0.0,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status().is_success(), "status {}", resp.status());
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// HTTP level: the chat endpoint routes on the LAST user message, reports
/// the decision, runs the skill on its lane and the backbone on the
/// server's slot — and the backbone slot's reuse key survives a skill
/// request (routing never resets a generation slot, nothing switches).
#[tokio::test]
async fn chat_endpoint_routes_per_request_and_reports_the_decision() {
    let (dir, model) = setup("http");
    let backbone = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let router = Arc::new(skill_router(&model));
    let state = Arc::new(AppState {
        runtime: CortiqRuntime::new(model.clone()),
        tokenizer: backbone.tokenizer.clone(),
        slots: PipelinePool::new(vec![backbone]),
        remote: None,
        routing: Some(router.clone()),
    });
    let app = build_router(state.clone());

    // General question → backbone slot.
    let v = chat(
        &app,
        serde_json::json!([{"role": "user", "content": GENERAL_TEXTS[0]}]),
    )
    .await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    for k in ["novelty", "e_base", "e_skill"] {
        assert!(v["x_cortiq_route"][k].is_number(), "{k}: {v}");
    }
    assert!(router.loaded_lanes().is_empty());
    let held = {
        let slot = state.slots.acquire().await;
        assert_eq!(slot.pipe.active_skill(), None);
        slot.pipe.kv_prefix.len()
    };
    assert!(
        held > 0,
        "the backbone slot keeps a reuse key after its turn"
    );

    // Earlier messages are history: the LAST user message decides.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": GENERAL_TEXTS[1]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": SKILL_TEXTS[2]},
        ]),
    )
    .await;
    assert_eq!(v["x_cortiq_route"]["target"], SKILL_ID, "{v}");
    assert_eq!(router.loaded_lanes(), vec![SKILL_ID.to_string()]);
    {
        let slot = state.slots.acquire().await;
        assert_eq!(
            slot.pipe.active_skill(),
            None,
            "the backbone slot never switched"
        );
        assert_eq!(
            slot.pipe.kv_prefix.len(),
            held,
            "a skill request reset the backbone slot"
        );
    }
    let lane = router
        .lane(&RouteTarget::Skill(SKILL_ID.into()))
        .unwrap()
        .unwrap();
    {
        let slot = lane.acquire().await;
        assert!(slot.pipe.active_skill().is_some());
        assert!(
            !slot.pipe.kv_prefix.is_empty(),
            "the skill lane served the request"
        );
    }

    // A system/assistant-only tail does not route on non-user text.
    let v = chat(
        &app,
        serde_json::json!([
            {"role": "user", "content": GENERAL_TEXTS[2]},
            {"role": "assistant", "content": SKILL_TEXTS[0]},
        ]),
    )
    .await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn last_user_text_takes_the_last_user_turn() {
    let msgs = [
        ("system", "s".to_string()),
        ("user", "first".to_string()),
        ("assistant", "a".to_string()),
        ("user", "second".to_string()),
        ("tool", "t".to_string()),
    ];
    let got = cortiq_server::route::last_user_text(msgs.iter().map(|(r, t)| (*r, t.clone())));
    assert_eq!(got.as_deref(), Some("second"));
    assert_eq!(
        cortiq_server::route::last_user_text(std::iter::once(("system", "x".to_string()))),
        None
    );
}

async fn completion(app: &axum::Router, prompt: &str) -> serde_json::Value {
    let body = serde_json::json!({
        "model": "cortiq",
        "prompt": prompt,
        "max_tokens": 2,
        "temperature": 0.0,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::post("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(resp.status().is_success(), "status {}", resp.status());
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// R5/PHI-2: `/v1/completions` routes only a prompt that is exactly one
/// cmf-im-v1 user turn (on that turn's text; the skill then generates in
/// its own contract). A raw prompt — even a pure skill question — and a
/// client-rendered transcript run the backbone with the reason in the
/// response, and never load a skill lane.
#[tokio::test]
async fn completions_route_only_a_single_cmf_im_v1_user_turn() {
    let (dir, model) = setup("completions");
    let backbone = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let router = Arc::new(skill_router(&model));
    let state = Arc::new(AppState {
        runtime: CortiqRuntime::new(model.clone()),
        tokenizer: backbone.tokenizer.clone(),
        slots: PipelinePool::new(vec![backbone]),
        remote: None,
        routing: Some(router.clone()),
    });
    let app = build_router(state.clone());

    // Raw skill question: no user turn → backbone.
    let v = completion(&app, SKILL_TEXTS[0]).await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert!(
        v["x_cortiq_route"]["reason"]
            .as_str()
            .unwrap()
            .contains("no user turn"),
        "{v}"
    );
    // A rendered two-turn transcript ending in the skill question → backbone
    // (no φ over the whole history).
    let transcript = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n\
         <|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        GENERAL_TEXTS[0], SKILL_TEXTS[1]
    );
    let v = completion(&app, &transcript).await;
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert!(router.loaded_lanes().is_empty(), "a skill lane was loaded");
    // Exactly one cmf-im-v1 user turn: routed on its text.
    let one = cortiq_engine::router::render_cmf_im_v1(SKILL_TEXTS[1]);
    let v = completion(&app, &one).await;
    assert_eq!(v["x_cortiq_route"]["target"], SKILL_ID, "{v}");
    assert_eq!(router.loaded_lanes(), vec![SKILL_ID.to_string()]);
    let one = cortiq_engine::router::render_cmf_im_v1(GENERAL_TEXTS[2]);
    assert_eq!(
        completion(&app, &one).await["x_cortiq_route"]["target"],
        "backbone"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// PHI-2 (chat side): the serve decision checks the skill's contract
/// against the chat frame. A tokenizer whose ChatML fallback lacks the
/// markers is not the cmf-im-v1 frame → the backbone runs.
#[test]
fn chat_frame_gates_the_skill_contract() {
    let (dir, model) = setup("frame");
    let r = skill_router(&model);
    use cortiq_engine::router::PromptFrame;
    let d = r.decide_framed(SKILL_TEXTS[0], PromptFrame::CmfImV1);
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()), "{}", d.reason);
    let d = r.decide_framed(SKILL_TEXTS[0], PromptFrame::Other);
    assert_eq!(d.target, RouteTarget::Backbone);
    assert!(d.reason.contains("not satisfied"), "{}", d.reason);
    // The synthetic vocabulary carries both markers: its ChatML fallback
    // IS cmf-im-v1; the bare byte-level tokenizer's is not.
    let p = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    assert_eq!(
        cortiq_engine::router::chat_frame(&p.tokenizer),
        PromptFrame::CmfImV1
    );
    assert_eq!(
        cortiq_engine::router::chat_frame(&cortiq_engine::tokenizer::Tokenizer::byte_level()),
        PromptFrame::Other
    );
    let _ = std::fs::remove_dir_all(&dir);
}
