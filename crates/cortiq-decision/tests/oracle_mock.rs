//! The oracle cascade against an in-test OpenRouter mock (spec §5, §6.5),
//! hermetic: every call goes to a local TCP server started by the test.
//!
//! * the request body for one choice question is byte for byte the v4 DeepSeek
//!   driver's: 9 ledger calls (3 per dataset, `fixtures/oracle`), by the sha256
//!   the ledgers recorded, both from `oracle::request_body` and on the wire
//!   through service + cascade (the mock replays the recorded answers);
//! * abstain → oracle with usage, cost and passthrough; gate-accepted questions
//!   never reach the oracle (0 mock hits, even with `cmf.oracle: true`);
//! * cache: a repeat and a paraphrase are cache answers with 0 calls;
//!   single flight: two identical parallel requests make one call;
//! * budget: a reservation above the rest makes no call; `max_calls`; the key's
//!   `oracle_budget_usd`;
//! * stop rules: HTTP 401/402/403, `unexpected_model`, a cost above the
//!   reservation, `max_errors` failures in a row; `POST /v1/admin/oracle
//!   {"enabled":true}` re-enables;
//! * bad answers: 200 with only `error`, invalid JSON, an option outside the
//!   contract, another finish reason, the deadline;
//! * consent: `oracle.enabled: false`, `cmf.oracle: false`, `oracle_allowed:
//!   false`, `default_per_request: false`, no key in the environment, the admin
//!   switch — 0 calls;
//! * PII redaction on by default (string leaves of a JSON state too);
//! * the key: read from the environment only (a child process with the variable
//!   set), its bytes never on disk or in the captured logs;
//! * the ledger and the stop survive a restart; an open reservation is charged.

#[path = "fixtures/oracle/support.rs"]
mod support;

use cortiq_decision::config::{Config, OracleConfig};
use cortiq_decision::learn;
use cortiq_decision::manifest::Rubric;
use cortiq_decision::metering::Usd;
use cortiq_decision::oracle::{self, process_env};
use cortiq_decision::protocol::{
    ModelRule, Question, QuestionKind, parse_request, validate_decisions_response, wire_questions,
};
use cortiq_decision::service::{Action, AdminCommand, Principal};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use support::*;

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn fixtures() -> Value {
    cortiq_decision::canonical::parse(include_bytes!("fixtures/oracle/deepseek_bodies.json"))
        .unwrap()
}

/// Texts the `topics` gate rejects (cruise words: out of every training pool),
/// pairwise cos φ_P < 0.97 so that none is a cache hit of another.
fn rejected() -> &'static [String] {
    static R: OnceLock<Vec<String>> = OnceLock::new();
    R.get_or_init(|| distinct_texts("cruise", 12, 7, "q", 0.97))
}

/// A dev text the `topics` gate accepts.
fn accepted() -> &'static str {
    static A: OnceLock<String> = OnceLock::new();
    A.get_or_init(|| {
        let mut cfg = Config::default();
        cfg.oracle.enabled = false;
        let st = Stand::new(&cfg);
        toy()
            .dev
            .iter()
            .map(|(t, _)| t.clone())
            .find(|t| st.decide(&topics_body(t)).unwrap().questions[0].action == Action::Local)
            .expect("a dev text the gate accepts")
    })
}

fn flags(d: &cortiq_decision::service::Decided, q: usize) -> Vec<String> {
    d.questions[q].flags.clone()
}

fn score_question() -> Value {
    json!({"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]})
}

// ------------------------------------------------------------------ bodies

#[test]
fn bodies_equal_the_driver_for_nine_ledger_calls() {
    let fx = fixtures();
    let cfg = OracleConfig::default();
    let max_price = cfg.max_price().unwrap();
    let mut n = 0;
    for (ds, d) in fx["datasets"].as_object().unwrap() {
        let q = Question {
            id: "task".into(),
            kind: QuestionKind::Choice,
            instructions: d["question"]["instructions"].clone(),
            criteria: Some(d["question"]["criteria"].clone()),
        };
        for r in d["rows"].as_array().unwrap() {
            let body = oracle::request_body(&cfg, &[&q], &r["text"]);
            assert_eq!(
                sha256_hex(&body),
                r["request_sha256"].as_str().unwrap(),
                "{ds} {} line {}",
                r["ledger"],
                r["line"]
            );
            assert_eq!(body.len() as u64, r["request_bytes"].as_u64().unwrap());
            let res = oracle::reservation_usd(body.len(), 64, max_price);
            assert_eq!(res, r["reserved_usd"].as_f64().unwrap(), "{ds} reservation");
            n += 1;
        }
    }
    assert_eq!(n, 9);
}

#[test]
fn a_rubric_stored_in_a_skill_manifest_rebuilds_the_driver_body() {
    // `cortiq decision learn` asks the oracle with the skill's rubric; as stored
    // (canonical JSON: sorted criteria plus `criteria_order`) it must rebuild the
    // driver's body for the same text, so the v4 dev ledgers can be reused.
    let fx = fixtures();
    let cfg = OracleConfig::default();
    for d in fx["datasets"].as_object().unwrap().values() {
        let q = &d["question"];
        let rubric = Rubric::new(
            q["instructions"].as_str().unwrap(),
            q["criteria"].as_object().unwrap().clone(),
        );
        let stored = cortiq_decision::canonical::vec_of(&rubric).unwrap();
        let back: Rubric = serde_json::from_slice(&stored).unwrap();
        let question = learn::question_of_rubric(&back);
        for r in d["rows"].as_array().unwrap() {
            let body = oracle::request_body(&cfg, &[&question], &r["text"]);
            assert_eq!(sha256_hex(&body), r["request_sha256"].as_str().unwrap());
        }
    }
}

#[test]
fn bodies_on_the_wire_equal_the_driver_and_replay_the_ledger() {
    let fx = fixtures();
    let mut recorded: HashMap<String, Value> = HashMap::new();
    let mut cases = Vec::new();
    for d in fx["datasets"].as_object().unwrap().values() {
        for r in d["rows"].as_array().unwrap() {
            recorded.insert(
                r["request_sha256"].as_str().unwrap().into(),
                r["oracle"].clone(),
            );
            cases.push((d["question"].clone(), r.clone()));
        }
    }
    // A ledger proxy: the answer recorded for the body's sha256, else 500.
    let mock = MockOracle::start(move |req| {
        let sha = sha256_hex(&req.body);
        let Some(o) = recorded.get(&sha) else {
            return raw_reply(500, r#"{"error":{"message":"unknown body"}}"#);
        };
        let u = &o["usage"];
        let body = serde_json::to_vec(&json!({
            "id": o["id"], "model": o["returned_model"], "provider": o["returned_provider"],
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant",
                          "content": json!({"task": o["choice"]}).to_string()}}],
            "usage": {"prompt_tokens": u["input_tokens"], "completion_tokens": u["output_tokens"],
                      "cost": u["cost"], "prompt_tokens_details": {"cached_tokens": u["cached_tokens"]}},
        }))
        .unwrap();
        MockReply {
            status: 200,
            body,
            delay: Duration::ZERO,
        }
    });
    // Default configuration (redaction on: none of these texts has PII); the
    // cache is off so that every row is a call.
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let st = Stand::new(&cfg);
    for (i, (q, r)) in cases.iter().enumerate() {
        let mut question = q.as_object().unwrap().clone();
        question.insert("type".into(), json!("choice"));
        let b = body(r["text"].clone(), json!({"task": question}), None);
        let d = st.decide(&b).unwrap();
        let o = &d.questions[0];
        assert_eq!(o.action, Action::Oracle, "row {i}");
        assert_eq!(
            d.response["answers"]["task"],
            json!({"type": "choice", "choice": r["oracle"]["choice"]})
        );
        let cost = r["oracle"]["usage"]["cost"].as_f64().unwrap();
        assert_eq!(
            d.response["cmf"]["usage"]["oracle"]["cost"].as_f64(),
            Some(cost)
        );
        assert_eq!(
            d.response["usage"]["cost"].as_f64(),
            Some(cost),
            "passthrough"
        );
        let sent = &mock.requests()[i];
        assert_eq!(
            sha256_hex(&sent.body),
            r["request_sha256"].as_str().unwrap()
        );
        assert_eq!(
            d.response["model"]
                .as_str()
                .map(|m| ModelRule::Cortiq.matches(m)),
            Some(true)
        );
    }
    assert_eq!(mock.hits(), 9);
}

// ------------------------------------------------------------------ cascade

#[test]
fn abstain_goes_to_the_oracle_with_usage_and_passthrough_cost() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let b = topics_body(&rejected()[0]);
    let d = st.decide(&b).unwrap();
    let q = &d.questions[0];
    assert_eq!(q.action, Action::Oracle);
    assert!(!q.certified);
    assert_eq!(q.decision_path, "escalate→oracle");
    assert_eq!(
        d.response["answers"]["task"],
        json!({"type": "choice", "choice": "travel"})
    );
    let cq = &d.response["cmf"]["questions"]["task"];
    assert_eq!(cq["source"], "oracle");
    assert_eq!(cq["gate"]["accepted"], false, "the local gate rejected it");
    let u = &d.response["cmf"]["usage"]["oracle"];
    assert_eq!(u["calls"], 1);
    assert_eq!(u["input_tokens"], 1200);
    assert_eq!(u["output_tokens"], 7);
    assert_eq!(u["cost"].as_f64(), Some(1.3e-5));
    assert_eq!(u["passthrough"], true);
    assert_eq!(d.response["usage"]["cost"].as_f64(), Some(1.3e-5));
    assert_eq!(d.record.oracle_calls, 1);
    // Oracle answers carry no probabilities (spec §4.7), so only the local
    // answers of this file are checked with the Jev validator port; the same
    // request with the oracle off abstains with a valid local distribution.
    let mut off = stand_config(&mock.url());
    off.oracle.enabled = false;
    let local = Stand::new(&off).decide(&b).unwrap();
    assert_eq!(local.questions[0].action, Action::Abstain);
    let req = parse_request(&b, &Default::default()).unwrap();
    validate_decisions_response(&local.response, &wire_questions(&req), &ModelRule::Cortiq)
        .unwrap();

    assert_eq!(mock.hits(), 1);
    let sent = &mock.requests()[0];
    assert_eq!(sent.path, "/chat/completions");
    assert_eq!(
        sent.header("authorization"),
        Some(format!("Bearer {TEST_KEY}").as_str())
    );
    assert_eq!(sent.header("content-type"), Some("application/json"));
    assert_eq!(sent.header("x-title"), Some("cortiq-decision"));
    assert_eq!(sent.state(), json!(rejected()[0]));

    let lines = st.ledger();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0]["status"], "reserved");
    assert_eq!(lines[1]["status"], "settled");
    assert_eq!(lines[0]["call_id"], lines[1]["call_id"]);
    assert_eq!(lines[1]["cost_usd"].as_f64(), Some(1.3e-5));
    assert_eq!(lines[1]["request_id"], json!(d.id));
    let res = lines[0]["reserved_usd"].as_f64().unwrap();
    let expect = oracle::reservation_usd(sent.body.len(), 64, (0.1, 0.5));
    assert_eq!(res, expect);
    let t = st.cascade.oracle().totals();
    assert_eq!(
        (t.calls, t.settled, t.spent, t.inflight),
        (1, 1, 1.3e-5, 0.0)
    );
}

#[test]
fn gate_accepted_questions_never_reach_the_oracle() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    for cmf in [None, Some(json!({"oracle": true}))] {
        let d = st
            .decide(&body(
                json!(accepted()),
                json!({"task": choice(&TOPICS)}),
                cmf,
            ))
            .unwrap();
        assert_eq!(d.questions[0].action, Action::Local);
    }
    assert_eq!(mock.hits(), 0);
    // An accepted and an untrained question: the call carries only the second.
    let d = st
        .decide(&body(
            json!(accepted()),
            json!({"a": choice(&TOPICS), "b": score_question()}),
            Some(json!({"oracle": true})),
        ))
        .unwrap();
    assert_eq!(d.questions[0].action, Action::Local);
    assert_eq!(d.questions[1].action, Action::Oracle);
    assert_eq!(mock.hits(), 1);
    let sent = mock.requests()[0].json();
    assert_eq!(
        sent["response_format"]["json_schema"]["schema"]["required"],
        json!(["b"])
    );
    assert!(
        sent["messages"][0]["content"]
            .as_str()
            .unwrap()
            .starts_with(oracle::SYSTEM_TYPED)
    );
}

#[test]
fn cache_answers_repeats_and_paraphrases_without_calls() {
    let a = "cruise ship cabin deck please";
    let p = "cruise ship cabin today";
    let c = cos(&phi_p(a), &phi_p(p));
    assert!((0.97..1.0).contains(&c), "paraphrase cos {c}");
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let d1 = st.decide(&topics_body(a)).unwrap();
    assert_eq!(d1.questions[0].action, Action::Oracle);
    let d2 = st.decide(&topics_body(a)).unwrap();
    assert_eq!(d2.questions[0].action, Action::Cache);
    assert_eq!(d2.questions[0].decision_path, "escalate→cache");
    assert_eq!(d2.response["answers"]["task"]["choice"], "travel");
    assert_eq!(d2.response["usage"]["cost"].as_f64(), Some(0.0));
    assert_eq!(d2.response["cmf"]["usage"]["oracle"]["calls"], 0);
    let d3 = st.decide(&topics_body(p)).unwrap();
    assert_eq!(d3.questions[0].action, Action::Cache, "paraphrase");
    assert_eq!(mock.hits(), 1);
    // A far text is a call; another skill scope (the same text asked with
    // other options) is a miss too.
    let far = &rejected()[1];
    assert!(cos(&phi_p(a), &phi_p(far)) < 0.97);
    assert_eq!(
        st.decide(&topics_body(far)).unwrap().questions[0].action,
        Action::Oracle
    );
    assert_eq!(mock.hits(), 2);
    let sub = body(
        json!(a),
        json!({"task": choice(&["billing", "travel"])}),
        None,
    );
    let d = st.decide(&sub).unwrap();
    assert_ne!(d.questions[0].action, Action::Cache);
    assert_eq!(
        st.cascade.cache_len(),
        2 + usize::from(d.questions[0].action == Action::Oracle)
    );

    // Cache off: the repeat is a call.
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let st2 = Stand::new(&cfg);
    let before = mock.hits();
    for _ in 0..2 {
        assert_eq!(
            st2.decide(&topics_body(a)).unwrap().questions[0].action,
            Action::Oracle
        );
    }
    assert_eq!(mock.hits(), before + 2);
}

#[test]
fn single_flight_joins_two_parallel_identical_requests() {
    let mock = MockOracle::start(|req| {
        let mut r = answer_reply(req, |_, o| pick(o, "travel"), 1.3e-5);
        r.delay = Duration::from_millis(800);
        r
    });
    let st = Stand::new(&stand_config(&mock.url()));
    let b = topics_body(&rejected()[2]);
    let (a1, a2) = std::thread::scope(|s| {
        let h1 = s.spawn(|| st.decide(&b).unwrap());
        std::thread::sleep(Duration::from_millis(250));
        let h2 = s.spawn(|| st.decide(&b).unwrap());
        (h1.join().unwrap(), h2.join().unwrap())
    });
    assert_eq!(mock.hits(), 1, "one call for two identical requests");
    let mut actions = [a1.questions[0].action, a2.questions[0].action];
    actions.sort_by_key(|a| format!("{a:?}"));
    assert_eq!(actions, [Action::Cache, Action::Oracle]);
    assert_eq!(a2.response["answers"]["task"]["choice"], "travel");
    assert_eq!(a1.response["answers"]["task"]["choice"], "travel");
    // A leader whose call fails fails its followers too.
    mock.set(|_| MockReply {
        status: 500,
        body: b"{}".to_vec(),
        delay: Duration::from_millis(800),
    });
    let b = topics_body(&rejected()[3]);
    let (f1, f2) = std::thread::scope(|s| {
        let h1 = s.spawn(|| st.decide(&b).unwrap());
        std::thread::sleep(Duration::from_millis(250));
        let h2 = s.spawn(|| st.decide(&b).unwrap());
        (h1.join().unwrap(), h2.join().unwrap())
    });
    assert_eq!(mock.hits(), 2);
    for f in [&f1, &f2] {
        assert_eq!(f.questions[0].action, Action::Abstain);
        assert!(flags(f, 0).contains(&"oracle_unavailable".to_string()));
    }
}

// ------------------------------------------------------------------ budget

#[test]
fn a_reservation_above_the_budget_left_makes_no_call() {
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.budget_usd = 1e-4; // below any reservation (≥ 4096·0.1/1e6)
    let st = Stand::new(&cfg);
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    assert_eq!(d.questions[0].action, Action::Abstain);
    assert_eq!(flags(&d, 0), vec!["budget"]);
    assert_eq!(d.questions[0].decision_path, "escalate→disabled");
    // An untrained question cannot abstain: 503.
    let e = st
        .decide(&body(json!("x"), json!({"u": score_question()}), None))
        .unwrap_err();
    assert_eq!(
        (e.status, e.reason.code()),
        (503, "ORACLE_BUDGET_EXHAUSTED")
    );
    assert_eq!(mock.hits(), 0);
    assert!(st.ledger().is_empty(), "no reservation was written");
}

#[test]
fn max_calls_and_the_key_budget_limit_calls() {
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.max_calls = 1;
    let st = Stand::new(&cfg);
    assert_eq!(
        st.decide(&topics_body(&rejected()[0])).unwrap().questions[0].action,
        Action::Oracle
    );
    let d = st.decide(&topics_body(&rejected()[1])).unwrap();
    assert_eq!(flags(&d, 0), vec!["budget"]);
    assert_eq!(mock.hits(), 1);

    // Per key: spent + reservation ≤ oracle_budget_usd.
    let st = Stand::new(&stand_config(&mock.url()));
    let mut p = Principal::open();
    p.account = "acme".into();
    p.key12 = Some("0123456789ab".into());
    p.oracle_budget_usd = Some(Usd::parse("0.0001").unwrap());
    let d = st.decide_as(&topics_body(&rejected()[2]), &p).unwrap();
    assert_eq!(flags(&d, 0), vec!["budget"]);
    p.oracle_budget_usd = Some(Usd::parse("1").unwrap());
    let d = st.decide_as(&topics_body(&rejected()[2]), &p).unwrap();
    assert_eq!(d.questions[0].action, Action::Oracle);
    assert_eq!(mock.hits(), 2);
    let settled = st
        .ledger()
        .into_iter()
        .find(|l| l["status"] == "settled")
        .unwrap();
    assert_eq!(settled["key_id"], "key:0123456789ab");
    assert_eq!(settled["account"], "acme");
}

// ------------------------------------------------------------------ stop rules

fn enable(st: &Stand) {
    st.svc
        .admin(&AdminCommand::OracleUpdate(json!({"enabled": true})))
        .unwrap();
}

#[test]
fn http_401_402_403_stop_the_oracle_until_the_admin_enables_it() {
    for status in [401u16, 402, 403] {
        let mock = MockOracle::start(move |_| raw_reply(status, r#"{"error":{"message":"no"}}"#));
        let st = Stand::new(&stand_config(&mock.url()));
        let d = st.decide(&topics_body(&rejected()[0])).unwrap();
        assert_eq!(d.questions[0].action, Action::Abstain);
        assert_eq!(flags(&d, 0), vec!["oracle_unavailable"]);
        assert_eq!(d.questions[0].decision_path, "escalate→oracle_unavailable");
        assert_eq!(
            st.oracle_state()["stop_reason"],
            json!(format!("http_{status}"))
        );
        let d = st.decide(&topics_body(&rejected()[1])).unwrap();
        assert_eq!(flags(&d, 0), vec!["stopped"]);
        assert_eq!(mock.hits(), 1, "stopped after {status}");
        let e = st
            .decide(&body(json!("x"), json!({"u": score_question()}), None))
            .unwrap_err();
        assert_eq!((e.status, e.reason.code()), (503, "ORACLE_DISABLED"));
        let status_json = st.svc.admin(&AdminCommand::OracleStatus).unwrap();
        assert_eq!(status_json["stop_reason"], json!(format!("http_{status}")));
        enable(&st);
        assert_eq!(st.oracle_state()["stop_reason"], Value::Null);
        mock.set(|req| answer_reply(req, |_, o| pick(o, "travel"), 1e-5));
        assert_eq!(
            st.decide(&topics_body(&rejected()[1])).unwrap().questions[0].action,
            Action::Oracle
        );
        assert_eq!(mock.hits(), 2);
        let last = st.ledger().pop().unwrap();
        assert_eq!(last["status"], "settled");
    }
}

#[test]
fn unexpected_model_and_a_cost_above_the_reservation_stop_the_oracle() {
    let mock = MockOracle::start(|req| {
        let content = verdicts(req, |_, o| pick(o, "travel")).to_string();
        MockReply {
            status: 200,
            body: completion(&content, json!(1e-5), "openai/other-model"),
            delay: Duration::ZERO,
        }
    });
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    assert_eq!(flags(&d, 0), vec!["oracle_unavailable"]);
    assert_eq!(st.oracle_state()["stop_reason"], "unexpected_model");
    let l = st.ledger();
    assert_eq!(l[1]["status"], "failed_billed");
    assert_eq!(l[1]["error"], "unexpected_model");
    assert_eq!(l[1]["cost_usd"].as_f64(), Some(1e-5));
    assert_eq!(
        d.response["usage"]["cost"].as_f64(),
        Some(0.0),
        "a failed call is not billed"
    );
    assert_eq!(
        flags(&st.decide(&topics_body(&rejected()[1])).unwrap(), 0),
        vec!["stopped"]
    );
    assert_eq!(mock.hits(), 1);

    // A cost above the reservation: the answer is used, then the oracle stops.
    let mock = MockOracle::start(|req| answer_reply(req, |_, o| pick(o, "travel"), 0.5));
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    assert_eq!(d.questions[0].action, Action::Oracle);
    assert_eq!(d.response["usage"]["cost"].as_f64(), Some(0.5));
    assert_eq!(st.oracle_state()["stop_reason"], "cost_above_reservation");
    assert_eq!(
        flags(&st.decide(&topics_body(&rejected()[1])).unwrap(), 0),
        vec!["stopped"]
    );
    assert_eq!(mock.hits(), 1);
}

#[test]
fn max_errors_in_a_row_stop_the_oracle() {
    let fail = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let f2 = fail.clone();
    let mock = MockOracle::start(move |req| {
        if f2.load(std::sync::atomic::Ordering::SeqCst) {
            raw_reply(500, r#"{"error":{"message":"upstream"}}"#)
        } else {
            answer_reply(req, |_, o| pick(o, "travel"), 1e-5)
        }
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.max_errors = 2;
    let st = Stand::new(&cfg);
    let set = |v: bool| fail.store(v, std::sync::atomic::Ordering::SeqCst);
    let r = rejected();
    assert_eq!(
        flags(&st.decide(&topics_body(&r[0])).unwrap(), 0),
        vec!["oracle_unavailable"]
    );
    set(false);
    assert_eq!(
        st.decide(&topics_body(&r[1])).unwrap().questions[0].action,
        Action::Oracle
    );
    set(true);
    assert_eq!(
        flags(&st.decide(&topics_body(&r[2])).unwrap(), 0),
        vec!["oracle_unavailable"]
    );
    assert_eq!(
        st.oracle_state(),
        Value::Null,
        "one failure after a success does not stop"
    );
    assert_eq!(
        flags(&st.decide(&topics_body(&r[3])).unwrap(), 0),
        vec!["oracle_unavailable"]
    );
    assert_eq!(st.oracle_state()["stop_reason"], "max_errors");
    assert_eq!(
        flags(&st.decide(&topics_body(&r[4])).unwrap(), 0),
        vec!["stopped"]
    );
    assert_eq!(mock.hits(), 4);
    let failed = st
        .ledger()
        .iter()
        .filter(|l| l["status"] == "failed_unknown_cost")
        .count();
    assert_eq!(failed, 3);
}

// ------------------------------------------------------------------ bad answers

#[test]
fn bad_answers_are_failures() {
    // 200 whose body is only an error: failure charged at the reservation.
    let mock = MockOracle::start(|_| {
        raw_reply(200, r#"{"error":{"code":502,"message":"provider down"}}"#)
    });
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    assert_eq!(d.questions[0].action, Action::Abstain);
    assert_eq!(flags(&d, 0), vec!["oracle_unavailable"]);
    let l = st.ledger();
    assert_eq!(l[1]["status"], "failed_unknown_cost");
    assert_eq!(l[1]["error"], "error_body");
    let res = l[0]["reserved_usd"].as_f64().unwrap();
    assert_eq!(st.cascade.oracle().totals().spent, res);
    // An untrained question with a failed call: 502.
    let e = st
        .decide(&body(json!("x"), json!({"u": score_question()}), None))
        .unwrap_err();
    assert_eq!((e.status, e.reason.code()), (502, "ORACLE_UNAVAILABLE"));

    // Invalid JSON content with a cost: a billed failure, not billed to the client.
    let mock = MockOracle::start(|_| MockReply {
        status: 200,
        body: completion("{\"task\": \"trav", json!(2e-5), ORACLE_MODEL),
        delay: Duration::ZERO,
    });
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    assert_eq!(flags(&d, 0), vec!["oracle_unavailable"]);
    assert_eq!(d.response["usage"]["cost"].as_f64(), Some(0.0));
    assert_eq!(d.response["cmf"]["usage"]["oracle"]["calls"], 0);
    let l = st.ledger();
    assert_eq!(
        (l[1]["status"].as_str(), l[1]["error"].as_str()),
        (Some("failed_billed"), Some("invalid_json"))
    );
    assert_eq!(st.cascade.oracle().totals().spent, 2e-5);

    // An option outside the contract, another finish reason.
    for (content, code) in [
        (r#"{"task":"nowhere"}"#, "choice_outside_contract"),
        (r#"{"task":"travel","extra":1}"#, "question_ids_mismatch"),
    ] {
        let c = content.to_string();
        let mock = MockOracle::start(move |_| MockReply {
            status: 200,
            body: completion(&c, json!(1e-5), ORACLE_MODEL),
            delay: Duration::ZERO,
        });
        let st = Stand::new(&stand_config(&mock.url()));
        let d = st.decide(&topics_body(&rejected()[0])).unwrap();
        assert_eq!(flags(&d, 0), vec!["oracle_unavailable"], "{code}");
        assert_eq!(st.ledger()[1]["error"], code);
    }
    let mock = MockOracle::start(|req| {
        let content = verdicts(req, |_, o| pick(o, "travel")).to_string();
        let mut v: Value =
            serde_json::from_slice(&completion(&content, json!(1e-5), ORACLE_MODEL)).unwrap();
        v["choices"][0]["finish_reason"] = json!("length");
        MockReply {
            status: 200,
            body: serde_json::to_vec(&v).unwrap(),
            delay: Duration::ZERO,
        }
    });
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    assert_eq!(flags(&d, 0), vec!["oracle_unavailable"]);
    assert_eq!(st.ledger()[1]["error"], "finish_length");
}

#[test]
fn the_deadline_bounds_a_call() {
    let mock = MockOracle::start(|req| {
        let mut r = answer_reply(req, |_, o| pick(o, "travel"), 1e-5);
        r.delay = Duration::from_secs(3);
        r
    });
    let mut cfg = stand_config(&mock.url());
    cfg.oracle.deadline_s = 0.5;
    let st = Stand::new(&cfg);
    let t0 = Instant::now();
    let d = st.decide(&topics_body(&rejected()[0])).unwrap();
    let took = t0.elapsed();
    assert!(took < Duration::from_millis(2500), "the call took {took:?}");
    assert_eq!(flags(&d, 0), vec!["oracle_unavailable"]);
    let l = st.ledger();
    assert_eq!(l[1]["status"], "failed_unknown_cost");
    assert!(
        l[1]["error"].as_str().unwrap().starts_with("transport_")
            || l[1]["error"].as_str().unwrap().starts_with("read_"),
        "{}",
        l[1]["error"]
    );
    let e = st
        .decide(&body(json!("x"), json!({"u": score_question()}), None))
        .unwrap_err();
    assert_eq!(e.status, 502);
}

// ------------------------------------------------------------------ consent

#[test]
fn consent_switches_make_no_call() {
    let mock = MockOracle::answering("travel");
    let trained =
        |cmf: Option<Value>| body(json!(rejected()[0]), json!({"task": choice(&TOPICS)}), cmf);
    let untrained = |cmf: Option<Value>| body(json!("x"), json!({"u": score_question()}), cmf);

    let mut off = stand_config(&mock.url());
    off.oracle.enabled = false;
    let st = Stand::new(&off);
    assert_eq!(
        flags(&st.decide(&trained(None)).unwrap(), 0),
        vec!["oracle_disabled"]
    );
    assert_eq!(st.decide(&untrained(None)).unwrap_err().status, 422);

    let st = Stand::new(&stand_config(&mock.url()));
    let no = Some(json!({"oracle": false}));
    assert_eq!(
        flags(&st.decide(&trained(no.clone())).unwrap(), 0),
        vec!["consent_off"]
    );
    assert_eq!(st.decide(&untrained(no)).unwrap_err().status, 422);
    let mut p = Principal::open();
    p.oracle_allowed = false;
    assert_eq!(
        flags(&st.decide_as(&trained(None), &p).unwrap(), 0),
        vec!["consent_off"]
    );

    let mut per = stand_config(&mock.url());
    per.oracle.default_per_request = false;
    let st = Stand::new(&per);
    assert_eq!(
        flags(&st.decide(&trained(None)).unwrap(), 0),
        vec!["consent_off"]
    );

    // No key in the environment.
    let st = Stand::open(
        tempfile::tempdir().unwrap(),
        &stand_config(&mock.url()),
        no_key_lookup(),
    );
    assert_eq!(
        flags(&st.decide(&trained(None)).unwrap(), 0),
        vec!["oracle_disabled"]
    );
    let status = st.svc.admin(&AdminCommand::OracleStatus).unwrap();
    assert_eq!(status["key_present"], false);

    // The admin switch.
    let st = Stand::new(&stand_config(&mock.url()));
    st.svc
        .admin(&AdminCommand::OracleUpdate(json!({"enabled": false})))
        .unwrap();
    assert_eq!(
        flags(&st.decide(&trained(None)).unwrap(), 0),
        vec!["oracle_disabled"]
    );
    assert_eq!(mock.hits(), 0);
    enable(&st);
    assert_eq!(
        st.decide(&trained(None)).unwrap().questions[0].action,
        Action::Oracle
    );
    assert_eq!(mock.hits(), 1);
}

// ------------------------------------------------------------------ PII

#[test]
fn pii_is_redacted_by_default() {
    let text = "cruise ship yacht harbor mail john.doe@example.com call +15551234567";
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&topics_body(text)).unwrap();
    assert_eq!(d.questions[0].action, Action::Oracle, "the gate rejects it");
    assert_eq!(flags(&d, 0), vec!["pii_redacted"]);
    let sent = mock.requests()[0].state();
    assert_eq!(
        sent,
        json!("cruise ship yacht harbor mail [REDACTED] call [REDACTED]")
    );
    assert!(!String::from_utf8_lossy(&mock.requests()[0].body).contains("john.doe"));

    // Consent to egress, or redaction off: sent as is.
    let mut cfg = stand_config(&mock.url());
    cfg.cache.enabled = false;
    let st = Stand::new(&cfg);
    let d = st
        .decide(&body(
            json!(text),
            json!({"task": choice(&TOPICS)}),
            Some(json!({"allow_pii_egress": true})),
        ))
        .unwrap();
    assert!(flags(&d, 0).is_empty());
    assert_eq!(mock.requests()[1].state(), json!(text));
    let mut raw = stand_config(&mock.url());
    raw.oracle.redact_pii = false;
    let st = Stand::new(&raw);
    st.decide(&topics_body(text)).unwrap();
    assert_eq!(mock.requests()[2].state(), json!(text));

    // A JSON state: string leaves are redacted, keys kept.
    let st = Stand::new(&stand_config(&mock.url()));
    let state = json!({"from": "a.b@c.de", "body": "cruise ship yacht"});
    let d = st
        .decide(&body(state, json!({"u": score_question()}), None))
        .unwrap();
    assert_eq!(flags(&d, 0), vec!["pii_redacted"]);
    assert_eq!(
        mock.requests()[3].state(),
        json!({"body": "cruise ship yacht", "from": "[REDACTED]"})
    );
}

// ------------------------------------------------------------------ the key

mod capture {
    //! A process-wide tracing subscriber that keeps every event and span field.
    use std::fmt::Write as _;
    use std::sync::Mutex;
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    pub static LOG: Mutex<String> = Mutex::new(String::new());

    struct V;
    impl Visit for V {
        fn record_debug(&mut self, f: &Field, v: &dyn std::fmt::Debug) {
            let _ = write!(LOG.lock().unwrap(), "{}={:?} ", f.name(), v);
        }
    }

    pub struct Capture;
    impl Subscriber for Capture {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, a: &Attributes<'_>) -> Id {
            a.record(&mut V);
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, r: &Record<'_>) {
            r.record(&mut V);
        }
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, e: &Event<'_>) {
            let _ = write!(LOG.lock().unwrap(), "\n[{}] ", e.metadata().level());
            e.record(&mut V);
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }
}

/// Runs in a child process with the key in its environment (see
/// `the_key_comes_only_from_the_environment`).
#[test]
#[ignore = "child process of the_key_comes_only_from_the_environment"]
fn child_key_from_the_environment() {
    let Ok(key) = std::env::var(KEY_ENV) else {
        println!("CHILD SKIPPED: {KEY_ENV} not set");
        return;
    };
    tracing::subscriber::set_global_default(capture::Capture).unwrap();
    let flip = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let f2 = flip.clone();
    let mock =
        MockOracle::start(
            move |req| match f2.fetch_add(1, std::sync::atomic::Ordering::SeqCst) {
                0 => answer_reply(req, |_, o| pick(o, "travel"), 1e-5),
                1 => raw_reply(500, r#"{"error":{"message":"boom"}}"#),
                _ => raw_reply(401, r#"{"error":{"message":"bad key"}}"#),
            },
        );
    let cfg = stand_config(&mock.url());
    let dir = tempfile::tempdir().unwrap();
    let st = Stand::open(
        dir,
        &cfg,
        process_env(), // the real environment
    );
    let r = rejected();
    assert_eq!(
        st.decide(&topics_body(&r[0])).unwrap().questions[0].action,
        Action::Oracle
    );
    assert_eq!(
        flags(&st.decide(&topics_body(&r[1])).unwrap(), 0),
        vec!["oracle_unavailable"]
    );
    assert_eq!(
        flags(&st.decide(&topics_body(&r[2])).unwrap(), 0),
        vec!["oracle_unavailable"]
    );
    assert_eq!(
        flags(&st.decide(&topics_body(&r[3])).unwrap(), 0),
        vec!["stopped"]
    );
    for req in mock.requests() {
        assert_eq!(
            req.header("authorization"),
            Some(format!("Bearer {key}").as_str())
        );
    }
    let status = st
        .svc
        .admin(&AdminCommand::OracleStatus)
        .unwrap()
        .to_string();
    let learning = st.svc.admin(&AdminCommand::Learning).unwrap().to_string();
    let debug = format!("{:?} {:?}", st.cascade, st.svc);
    for (what, text) in [
        ("status", &status),
        ("learning", &learning),
        ("debug", &debug),
    ] {
        assert!(!text.contains(&key), "the key is in {what}");
    }
    let on_disk = bytes_on_disk(st.state.root(), key.as_bytes());
    assert!(on_disk.is_empty(), "the key is on disk: {on_disk:?}");
    let log = capture::LOG.lock().unwrap().clone();
    assert!(
        log.contains("oracle stopped"),
        "the logs were captured: {log}"
    );
    assert!(!log.contains(&key), "the key is in the logs");
    println!("CAPTURED LOGS BEGIN{log}\nCAPTURED LOGS END");
    println!(
        "CHILD OK {} files scanned",
        files_under(st.state.root()).len()
    );
}

#[test]
fn the_key_comes_only_from_the_environment() {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "--ignored",
            "--exact",
            "child_key_from_the_environment",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(KEY_ENV, TEST_KEY)
        .env("CMF_GPU", "0")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "child failed:\n{stdout}\n{stderr}");
    assert!(stdout.contains("CHILD OK"), "{stdout}");
    assert!(stdout.contains("CAPTURED LOGS BEGIN"));
    assert!(
        !stdout.contains(TEST_KEY) && !stderr.contains(TEST_KEY),
        "the key leaked into the output"
    );
    // Without the variable the oracle is disabled (the key is not read from
    // anywhere else, the configuration included).
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    assert!(std::env::var(KEY_ENV).is_err());
    let st = Stand::open(tempfile::tempdir().unwrap(), &cfg, process_env());
    assert_eq!(
        flags(&st.decide(&topics_body(&rejected()[0])).unwrap(), 0),
        vec!["oracle_disabled"]
    );
    assert_eq!(mock.hits(), 0);
    assert!(!cfg.to_value().to_string().contains(TEST_KEY));
}

// ------------------------------------------------------------------ multi-type, restart, admin

#[test]
fn multitype_request_is_answered_through_the_mock() {
    let fx: Value = serde_json::from_slice(
        &std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jev/multitype.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let req = fx["request"].clone();
    let b = serde_json::to_vec(&req).unwrap();
    let mock = MockOracle::start(|r| answer_reply(r, |_, o| pick(o, "billing"), 3e-5));
    let st = Stand::new(&stand_config(&mock.url()));
    let d = st.decide(&b).unwrap();
    assert_eq!(mock.hits(), 1, "one call for three questions");
    let sent = mock.requests()[0].json();
    assert_eq!(
        sent["response_format"]["json_schema"]["schema"]["required"],
        json!(["team", "urgency", "refund"])
    );
    assert_eq!(sent["max_tokens"], 192);
    let a = &d.response["answers"];
    assert_eq!(a["team"], json!({"type": "choice", "choice": "billing"}));
    assert_eq!(a["urgency"]["score"], 0);
    assert_eq!(
        a["urgency"]["legend"]["2"],
        "Critical outage or immediate danger"
    );
    assert_eq!(
        a["refund"],
        json!({"type": "noul", "noul": 1, "value_semantics": "boolean_verdict_not_probability"})
    );
    for q in ["team", "urgency", "refund"] {
        assert_eq!(d.response["cmf"]["questions"][q]["action"], "oracle");
    }
    assert_eq!(d.response["usage"]["cost"].as_f64(), Some(3e-5));
    // Without the oracle: 422 with a reason per question.
    let mut off = stand_config(&mock.url());
    off.oracle.enabled = false;
    let e = Stand::new(&off).decide(&b).unwrap_err();
    assert_eq!(e.status, 422);
    let details = e.details.as_ref().unwrap();
    let qs = details["questions"].as_object().unwrap();
    assert_eq!(
        qs.keys().collect::<Vec<_>>(),
        vec!["team", "urgency", "refund"]
    );
}

#[test]
fn the_ledger_and_the_stop_survive_a_restart() {
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    let st = Stand::new(&cfg);
    st.decide(&topics_body(&rejected()[0])).unwrap();
    let spent = st.cascade.oracle().totals().spent;
    mock.set(|_| raw_reply(403, "{}"));
    st.decide(&topics_body(&rejected()[1])).unwrap();
    let spent2 = st.cascade.oracle().totals().spent;
    assert!(spent2 > spent);
    // A reservation left open by a crash.
    let ledger = st.state.oracle_ledger_path();
    let st = {
        let dir_state = st;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&ledger)
            .unwrap();
        use std::io::Write;
        writeln!(f, "{}", json!({"status": "reserved", "call_id": "oc-crash", "key_id": "account:anonymous", "reserved_usd": 0.25})).unwrap();
        dir_state.restart(&cfg)
    };
    let t = st.cascade.oracle().totals();
    assert_eq!(t.spent, spent2 + 0.25, "the open reservation is charged");
    assert_eq!(t.calls, 3);
    assert_eq!(st.oracle_state()["stop_reason"], "http_403");
    mock.set(|req| answer_reply(req, |_, o| pick(o, "travel"), 1e-5));
    assert_eq!(
        flags(&st.decide(&topics_body(&rejected()[2])).unwrap(), 0),
        vec!["stopped"]
    );
    let closed = st
        .ledger()
        .into_iter()
        .find(|l| l["call_id"] == "oc-crash" && l["status"] == "failed_unknown_cost");
    assert!(closed.is_some(), "the open reservation was closed at start");
    // Restarting again charges nothing twice.
    let st = st.restart(&cfg);
    assert_eq!(st.cascade.oracle().totals().spent, spent2 + 0.25);
}

#[test]
fn admin_oracle_status_and_limits() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let s = st.svc.admin(&AdminCommand::OracleStatus).unwrap();
    assert_eq!(s["enabled"], true);
    assert_eq!(s["configured"], true);
    assert_eq!(s["key_present"], true);
    assert_eq!(s["key_env"], KEY_ENV);
    assert_eq!(s["budget_usd"].as_f64(), Some(1.0));
    assert!(!s.to_string().contains(TEST_KEY));
    let e = st
        .svc
        .admin(&AdminCommand::OracleUpdate(json!({"budget_usd": 5.0})))
        .unwrap_err();
    assert_eq!(e.status, 400, "a budget above the configuration");
    let e = st
        .svc
        .admin(&AdminCommand::OracleUpdate(json!({"bogus": 1})))
        .unwrap_err();
    assert_eq!(e.status, 400);
    let s = st
        .svc
        .admin(&AdminCommand::OracleUpdate(
            json!({"budget_usd": 0.0, "max_calls": 5}),
        ))
        .unwrap();
    assert_eq!(
        (s["budget_usd"].as_f64(), s["max_calls"].as_u64()),
        (Some(0.0), Some(5))
    );
    assert_eq!(
        flags(&st.decide(&topics_body(&rejected()[0])).unwrap(), 0),
        vec!["budget"]
    );
    assert_eq!(mock.hits(), 0);
    assert_eq!(st.oracle_state()["budget_usd"].as_f64(), Some(0.0));
    let s = st
        .svc
        .admin(&AdminCommand::OracleUpdate(json!({"budget_usd": null})))
        .unwrap();
    assert_eq!(s["budget_usd"].as_f64(), Some(1.0));
}
