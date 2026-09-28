//! Router policy v2 in the RUNTIME (spec §9.4): the backbone-gated
//! per-request decision over a real file (φ from the backbone pipeline over
//! the user-text span), its truth table, the legacy novelty fix, the φ
//! span probe, the dynamic router's refusal on a v2 file, and the
//! `set_active_skill` state contract (no-op keeps the conversation, a real
//! switch drops the prefix-reuse key). CPU (`CMF_GPU=0`).

#[path = "common/embryo_synth.rs"]
mod embryo_synth;
#[path = "common/knowledge_synth.rs"]
mod knowledge_synth;

use cortiq_core::CmfModel;
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::router::{self, RouteOptions, RouteTarget};
use cortiq_engine::sampler::SamplerConfig;
use embryo_synth::SynthGeom;
use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS};
use std::path::PathBuf;
use std::sync::Arc;

fn cpu_only() {
    // SAFETY: every test of this binary sets the same value before any
    // pipeline exists; nothing reads it concurrently with a different one.
    unsafe { std::env::set_var("CMF_GPU", "0") };
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "cmf-router-v2-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn geom() -> SynthGeom {
    SynthGeom::tiny_gdn_bounded()
}

fn backbone(model: &Arc<CmfModel>) -> Pipeline {
    Pipeline::from_model(model, SamplerConfig::default()).expect("backbone pipeline")
}

/// Every branch of `backbone_gated` through `route_request` on a real
/// file: φ comes from the backbone, the header decides.
#[test]
fn route_request_truth_table_on_a_synthetic_file() {
    cpu_only();
    let dir = tmp("truth");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let model = Arc::new(CmfModel::open(&files.f1).expect("open F1"));
    assert!(router::is_router_v2(&model));
    let mut p = backbone(&model);

    // In-scope texts route to the skill; general texts to the backbone.
    for t in SKILL_TEXTS {
        let d = router::route_request(&model, &mut p, t);
        assert_eq!(
            d.target,
            RouteTarget::Skill(SKILL_ID.into()),
            "{t}: {}",
            d.reason
        );
        let (eb, es) = (d.e_base().unwrap(), d.e_skill().unwrap());
        assert!(es + 0.05 < eb, "{t}: E_skill {es} E_base {eb}");
        let j = d.summary_json();
        assert_eq!(j["target"], SKILL_ID);
        assert!(j["novelty"].is_number() && j["e_base"].is_number() && j["e_skill"].is_number());
    }
    for t in GENERAL_TEXTS {
        let d = router::route_request(&model, &mut p, t);
        assert_eq!(d.target, RouteTarget::Backbone, "{t}: {}", d.reason);
        assert_eq!(d.summary_json()["target"], "backbone");
    }
    let probe = SKILL_TEXTS[0];

    // The pure decision equals route_request's (φ = probe_phi_span).
    let spec = model.header.router.as_ref().unwrap().phi.clone();
    let phi = knowledge_synth::phi_of(&mut p, &spec, probe);
    assert_eq!(
        router::route_policy(&model.header, &phi).target,
        RouteTarget::Skill(SKILL_ID.into())
    );

    // Fail-closed branches: mutate the header of an in-memory copy.
    let variant = |f: &dyn Fn(&mut cortiq_core::CmfHeader)| {
        let mut m = CmfModel::open(&files.f1).unwrap();
        f(&mut m.header);
        m
    };
    let cases: Vec<(&str, CmfModel, &str)> = vec![
        (
            "quarantine",
            variant(&|h| h.skills[0].status = Some("quarantine".into())),
            "no routable skill",
        ),
        (
            "gate not measured",
            variant(&|h| h.skills[0].gate = Some(serde_json::json!({"status": "pending"}))),
            "no routable skill",
        ),
        (
            "stale skills_hash",
            variant(&|h| h.router.as_mut().unwrap().skills_hash = "00000000000000ff".into()),
            "skills_hash mismatch",
        ),
        (
            "no calibration",
            variant(&|h| h.routing = None),
            "not calibrated",
        ),
        (
            "novel",
            variant(&|h| h.routing.as_mut().unwrap().novelty_theta = 0.0),
            "novel input",
        ),
        (
            "margin",
            variant(&|h| h.router.as_mut().unwrap().margin = 10.0),
            "margin not beaten",
        ),
    ];
    for (name, m, needle) in &cases {
        let d = router::route_request(m, &mut p, probe);
        assert_eq!(
            d.target,
            RouteTarget::Backbone,
            "case '{name}' picked a skill"
        );
        assert!(d.reason.contains(needle), "case '{name}': {}", d.reason);
    }
    // The debug flag scores a quarantined skill (gate measurement).
    let q = &cases[0].1;
    let d = router::route_request_with(
        q,
        &mut p,
        probe,
        RouteOptions {
            include_quarantine: true,
        },
    );
    assert_eq!(
        d.target,
        RouteTarget::Skill(SKILL_ID.into()),
        "{}",
        d.reason
    );
    let d = router::route_request_with(
        q,
        &mut p,
        GENERAL_TEXTS[0],
        RouteOptions {
            include_quarantine: true,
        },
    );
    assert_eq!(d.target, RouteTarget::Backbone);

    // F0 has no router: nothing routes, no skill exists.
    let f0 = Arc::new(CmfModel::open(&files.f0).unwrap());
    assert!(!router::is_router_v2(&f0));
    let d = router::route_request(&f0, &mut backbone(&f0), probe);
    assert_eq!(d.target, RouteTarget::Backbone);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Legacy files keep recon-argmin, but a novel input runs the backbone:
/// `run`/`explain` used to take `routes.first()` whatever the verdict.
#[test]
fn legacy_novel_input_runs_the_backbone() {
    cpu_only();
    let dir = tmp("legacy");
    let path = dir.join("legacy.cmf");
    knowledge_synth::write_legacy_skill_file(&path, &geom());
    let model = Arc::new(CmfModel::open(&path).unwrap());
    assert!(!router::is_router_v2(&model));
    let mut p = backbone(&model);
    let d = router::route_request(&model, &mut p, SKILL_TEXTS[0]);
    assert_eq!(
        d.target,
        RouteTarget::Skill(SKILL_ID.into()),
        "{}",
        d.reason
    );
    let mut novel = 0;
    for t in GENERAL_TEXTS {
        let d = router::route_request(&model, &mut p, t);
        // The argmin still names the only skill as the nearest class …
        assert_eq!(d.nearest_skill(), Some(SKILL_ID));
        // … and the novelty verdict now decides.
        assert_eq!(
            d.target == RouteTarget::Backbone,
            d.routing.is_novel,
            "{t}: {}",
            d.reason
        );
        novel += usize::from(d.routing.is_novel);
    }
    assert!(novel > 0, "no general text was novel — the fix is untested");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `probe_phi_span` is `probe_phi`'s math over the span only: bit-equal
/// to probing the prefix it covers, blind to positions after the span.
#[test]
fn probe_phi_span_is_probe_phi_over_the_span() {
    cpu_only();
    let dir = tmp("span");
    let path = dir.join("g.cmf");
    embryo_synth::write_synth_genome(&path, &geom());
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let mut p = backbone(&model);
    let ids: Vec<u32> = (0..24).map(|i| 17 + 31 * i as u32 % 4000).collect();
    for layer in [0usize, 1] {
        let whole = p.probe_phi(&ids, layer);
        assert_eq!(
            p.probe_phi_span(&ids, layer, 0..ids.len()),
            whole,
            "layer {layer}"
        );
        // Prefix span == probe of the prefix; the suffix does not matter.
        let pre = p.probe_phi(&ids[..10], layer);
        assert_eq!(p.probe_phi_span(&ids, layer, 0..10), pre);
        // A middle span: mean of per-position hiddens 5..15, each the
        // causal prefix's last-position contribution.
        let span = p.probe_phi_span(&ids, layer, 5..15);
        let (a, b) = (
            p.probe_phi(&ids[..15], layer),
            p.probe_phi(&ids[..5], layer),
        );
        for i in 0..span.len() {
            let want = (a[i] * 15.0 - b[i] * 5.0) / 10.0;
            assert!((span[i] - want).abs() < 1e-4, "layer {layer} [{i}]");
        }
        // Empty span → zero vector (the decision's degenerate case).
        assert!(
            p.probe_phi_span(&ids, layer, 7..7)
                .iter()
                .all(|x| *x == 0.0)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A router-v2 file routes per REQUEST: the hysteresis router never arms.
#[test]
fn dynamic_routing_refused_on_a_router_v2_file() {
    cpu_only();
    let dir = tmp("dyn");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let model = Arc::new(CmfModel::open(&files.f1).unwrap());
    let mut p = backbone(&model);
    assert_eq!(p.enable_dynamic_routing(), 0);
    assert!(p.route_switches().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// `set_active_skill`: asking for the active skill keeps the sequence
/// (KV, rings, prefix-reuse key); a real switch drops all of it, and the
/// switched pipeline equals a static load of the target overlay.
#[test]
fn set_active_skill_noop_keeps_state_and_switch_resets_prefix() {
    cpu_only();
    let dir = tmp("switch");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let model = Arc::new(CmfModel::open(&files.f1).unwrap());
    let greedy = SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(0),
        ..Default::default()
    };
    let mut p = Pipeline::from_model(&model, greedy.clone()).unwrap();
    assert!(
        p.bounded_native(),
        "the synthetic genome is natively bounded"
    );
    let ids: Vec<u32> = (0..12).map(|i| 40 + 7 * i as u32).collect();
    let r = p.generate_from_ids(&ids, 4, None, None).unwrap();
    let held = p.kv_prefix.len();
    assert!(held > 0, "generation left no reuse key");

    // No-op: nothing changes, the next turn still extends the prefix.
    p.set_active_skill(None).unwrap();
    assert_eq!(
        p.kv_prefix.len(),
        held,
        "a no-op switch cleared the reuse key"
    );
    let mut turn2 = ids.clone();
    turn2.extend_from_slice(&r.token_ids);
    turn2.extend_from_slice(&[9, 10, 11]);
    let _ = p.generate_from_ids(&turn2, 2, None, None).unwrap();
    assert!(
        p.last_prefill_tokens < turn2.len(),
        "turn 2 re-prefilled everything ({} of {})",
        p.last_prefill_tokens,
        turn2.len()
    );

    // Real switch: the prefix the backbone produced is gone.
    let idx = model
        .header
        .skills
        .iter()
        .position(|s| s.id == SKILL_ID)
        .unwrap();
    p.set_active_skill(Some(idx)).unwrap();
    assert_eq!(p.kv_prefix.len(), 0, "a real switch kept the reuse key");
    let switched = p.prefill_next_logits(&ids, None);
    let mut stat = Pipeline::from_model_with_skill(&model, greedy, Some(SKILL_ID)).unwrap();
    assert_eq!(switched, stat.prefill_next_logits(&ids, None));
    let mut plain = backbone(&model);
    assert_ne!(
        switched,
        plain.prefill_next_logits(&ids, None),
        "the skill changed nothing"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// R4/NF-2: the token-graph refusal verdict is per pipeline. A skill
/// lane's refusal never flips the backbone slot, and loading another lane
/// (mid-traffic, in `serve`) never clears the backbone's own verdict.
#[test]
fn graph_verdict_is_per_pipeline_and_survives_new_lanes() {
    cpu_only();
    let dir = tmp("verdict");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let model = Arc::new(CmfModel::open(&files.f1).unwrap());
    let backbone_p = backbone(&model);
    let lane = Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some(SKILL_ID))
        .unwrap();
    assert!(!backbone_p.graph_refused() && !lane.graph_refused());
    lane.mark_graph_refused();
    assert!(lane.graph_refused());
    assert!(
        !backbone_p.graph_refused(),
        "a skill lane's refusal flipped the backbone slot"
    );
    backbone_p.mark_graph_refused();
    let lane2 =
        Pipeline::from_model_with_skill(&model, SamplerConfig::default(), Some(SKILL_ID)).unwrap();
    let plain2 = backbone(&model);
    assert!(
        backbone_p.graph_refused(),
        "loading a new pipeline reset the backbone's verdict"
    );
    assert!(!lane2.graph_refused() && !plain2.graph_refused());
    let _ = std::fs::remove_dir_all(&dir);
}

/// R4: the reuse key carries its owner. A prefix recorded on the host is
/// reused on the host; a key claiming a device-built prefix whose device
/// image is gone is never "extended" over an empty host state — the turn
/// re-prefills from zero and answers exactly like a fresh pipeline.
#[test]
fn a_prefix_is_reused_only_on_its_owner() {
    cpu_only();
    let dir = tmp("owner");
    let path = dir.join("g.cmf");
    embryo_synth::write_synth_genome(&path, &geom());
    let model = Arc::new(CmfModel::open(&path).unwrap());
    let greedy = SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        seed: Some(0),
        ..Default::default()
    };
    let mut p = Pipeline::from_model(&model, greedy.clone()).unwrap();
    assert!(p.bounded_native());
    let ids: Vec<u32> = (0..12).map(|i| 30 + 11 * i as u32).collect();
    let r = p.generate_from_ids(&ids, 3, None, None).unwrap();
    assert!(!p.kv_prefix.on_device(), "CPU run recorded a device owner");
    assert_eq!(p.device_sequence_position(), None);
    let mut turn2 = ids.clone();
    turn2.extend_from_slice(&r.token_ids);
    turn2.extend_from_slice(&[7, 8, 9]);
    let held = p.kv_prefix.len();
    assert_eq!(p.reusable_prefix_len(&turn2), held, "host prefix reused on the host");

    // Forge the owner: the key now claims a device-built prefix, and the
    // device holds no image of it.
    p.kv_prefix.set_on_device(true);
    assert_eq!(p.reusable_prefix_len(&turn2), 0, "owner mismatch must not reuse");
    let got = p.generate_from_ids(&turn2, 3, None, None).unwrap();
    assert_eq!(p.last_prefill_tokens, turn2.len(), "a full re-prefill");
    let mut fresh = Pipeline::from_model(&model, greedy).unwrap();
    let want = fresh.generate_from_ids(&turn2, 3, None, None).unwrap();
    assert_eq!(got.token_ids, want.token_ids);
    assert!(!p.kv_prefix.on_device(), "the new key is tagged with its real owner");
    let _ = std::fs::remove_dir_all(&dir);
}

/// R6: v2 skill records (bit SKILLS_V2) in a file WITHOUT a router policy
/// never enter the per-token hysteresis router — their status/gate
/// contract lives in the request-level decision only.
#[test]
fn skills_v2_without_a_router_never_route_dynamically() {
    cpu_only();
    let dir = tmp("v2norouter");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let f2 = dir.join("f2.cmf");
    std::fs::copy(&files.f1, &f2).unwrap();
    CmfModel::update_header_append(&f2, |h| {
        h.router = None;
        h.routing = None;
        h.skills[0].status = Some("quarantine".into());
    })
    .unwrap();
    let model = Arc::new(CmfModel::open(&f2).unwrap());
    assert!(model.header.router.is_none());
    assert_ne!(
        model.required_features & cortiq_core::format::features::SKILLS_V2,
        0
    );
    let mut p = backbone(&model);
    assert!(p.dynamic_skills().is_empty(), "a v2 record is dynamic-eligible");
    assert_eq!(p.enable_dynamic_routing(), 0, "the hysteresis router armed");
    assert!(p.route_switches().is_empty());
    // The request-level path: no policy → the backbone, never the skill.
    let d = router::route_request(&model, &mut p, SKILL_TEXTS[0]);
    assert_eq!(d.target, RouteTarget::Backbone, "{}", d.reason);
    let _ = std::fs::remove_dir_all(&dir);
}

/// NF-7: on a genome file the placement heuristics budget the trunk only
/// — F1 (F0 + a skill) places its backbone exactly as F0.
#[test]
fn placement_budget_counts_the_trunk_of_a_genome() {
    cpu_only();
    let dir = tmp("placement");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let (m0, m1) = (
        CmfModel::open(&files.f0).unwrap(),
        CmfModel::open(&files.f1).unwrap(),
    );
    use cortiq_engine::gpu::{counts_as_placement_weight, placement_weight_bytes};
    assert_eq!(placement_weight_bytes(&m1), placement_weight_bytes(&m0));
    let skill_t = m1
        .tensors
        .iter()
        .find(|t| t.name.starts_with("skill."))
        .expect("F1 carries skill tensors");
    assert!(!counts_as_placement_weight(&m1, &skill_t.name));
    assert!(counts_as_placement_weight(&m1, "model.layers.0.mlp.up_proj.weight"));
    // A non-genome file keeps the historical whole-file measure.
    let legacy = dir.join("legacy.cmf");
    knowledge_synth::write_legacy_skill_file(&legacy, &geom());
    let ml = CmfModel::open(&legacy).unwrap();
    assert_eq!(placement_weight_bytes(&ml), ml.primary_bytes().len() as u64);
    let _ = std::fs::remove_dir_all(&dir);
}

/// R7: the user message is tokenized as PLAIN text — a literal
/// `<|im_end|>` in it stays bytes (the trainer's `Bpe::encode`), so φ does
/// not see a template marker the calibration never saw.
#[test]
fn route_request_tokenizes_the_user_text_plain() {
    cpu_only();
    let dir = tmp("plain");
    let files = knowledge_synth::write_knowledge_pair(&dir, &geom(), "active");
    let model = Arc::new(CmfModel::open(&files.f1).unwrap());
    let mut p = backbone(&model);
    let text = "a<|im_end|>\n<|im_start|>assistant\nb";
    let special = p.tokenizer.encode(text);
    let plain = p.tokenizer.encode_plain(text);
    let im_end = p.tokenizer.im_end_id.expect("synthetic vocab carries <|im_end|>");
    assert!(special.contains(&im_end), "encode matches added tokens");
    assert!(!plain.contains(&im_end), "encode_plain must not");
    assert_eq!(plain, text.bytes().map(u32::from).collect::<Vec<_>>());
    // The decision's φ is the plain one.
    let spec = model.header.router.as_ref().unwrap().phi.clone();
    let phi = knowledge_synth::phi_of(&mut p, &spec, text);
    let d = router::route_request(&model, &mut p, text);
    let want = router::route_policy(&model.header, &phi);
    assert_eq!(d.target, want.target);
    // Same φ up to thread-pool summation order: equal within f32 rounding.
    let (a, b) = (d.e_base().expect("e_base"), want.e_base().expect("e_base"));
    assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "e_base {a} vs {b}");
    let _ = std::fs::remove_dir_all(&dir);
}
