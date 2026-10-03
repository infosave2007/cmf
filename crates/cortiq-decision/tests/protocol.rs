//! The decisions protocol (spec §4.4, §4.7, §4.8, §6.3), hermetic:
//!
//! * a table of request cases: every limit on both sides of its bound, all
//!   three question types, OpenRouter's optional fields accepted, unknown keys
//!   and duplicate keys refused, criteria order kept, the model id rules, the
//!   413 bound;
//! * the error envelope (OpenRouter shape with cortiq-router reason codes);
//! * the Rust port of `openrouter_bench.validate_oracle_response` accepts the
//!   20 stored Jev answers per dataset (`tests/fixtures/jev/*.json`, answer
//!   objects copied verbatim from the ledgers) and the stored multi-type Jev
//!   answer, and refuses each broken variant with the Python message;
//! * `round = 2` writes every stored Jev answer back byte for byte (hundredths,
//!   0 and 1 as integers);
//! * Jev's confidence equals `(N·p_max − 1)/(N − 1)` of its own (quantised)
//!   probabilities within the envelope measured on 4,678 answers.

use cortiq_decision::answer::{self, Rounding};
use cortiq_decision::canonical;
use cortiq_decision::protocol::{
    self, ApiError, CAPACITY_MARKER, CmfOptions, ModelRef, ModelRule, Profile, QuestionKind,
    Reason, RequestLimits, SYSTEMONE_MODEL_ALIASES, SYSTEMONE_MODEL_ID, State, parse_feedback,
    parse_request, parse_systemone_request, validate_decisions_response,
};
use serde_json::{Map, Value, json};
use std::path::PathBuf;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jev")
}

fn load(name: &str) -> Value {
    let bytes = std::fs::read(fixtures().join(name)).unwrap();
    canonical::parse(&bytes).unwrap()
}

fn choice_q(n: usize) -> Value {
    let mut c = Map::new();
    for i in 0..n {
        c.insert(format!("opt{i}"), json!(format!("option {i}")));
    }
    json!({"type": "choice", "instructions": "Pick one.", "criteria": c})
}

fn base() -> Value {
    json!({
        "model": "cortiq/decision",
        "state": "I still have not received my new card",
        "questions": {"task": choice_q(3)},
    })
}

fn with(mut v: Value, path: &[&str], x: Value) -> Value {
    let mut cur = &mut v;
    for p in &path[..path.len() - 1] {
        cur = cur.get_mut(*p).unwrap();
    }
    cur.as_object_mut()
        .unwrap()
        .insert(path[path.len() - 1].to_string(), x);
    v
}

fn without(mut v: Value, path: &[&str]) -> Value {
    let mut cur = &mut v;
    for p in &path[..path.len() - 1] {
        cur = cur.get_mut(*p).unwrap();
    }
    cur.as_object_mut().unwrap().remove(path[path.len() - 1]);
    v
}

enum Expect {
    Ok,
    Err(u16, Reason),
}

struct Case {
    name: &'static str,
    body: Vec<u8>,
    expect: Expect,
}

fn case(name: &'static str, v: Value, expect: Expect) -> Case {
    Case {
        name,
        body: serde_json::to_vec(&v).unwrap(),
        expect,
    }
}

fn raw(name: &'static str, body: &str, expect: Expect) -> Case {
    Case {
        name,
        body: body.as_bytes().to_vec(),
        expect,
    }
}

const BAD: Expect = Expect::Err(400, Reason::InvalidRequest);

fn cases() -> Vec<Case> {
    use Expect::Ok as OK;
    let questions = |n: usize| -> Value {
        let mut m = Map::new();
        for i in 0..n {
            m.insert(format!("q{i}"), choice_q(2));
        }
        Value::Object(m)
    };
    let big_state = "x".repeat(32 * 1024);
    let mut levels10 = Vec::new();
    for i in 0..10 {
        levels10.push(json!(format!("level {i}")));
    }
    let mut levels11 = levels10.clone();
    levels11.push(json!("level 10"));
    vec![
        // ---- accepted shapes
        case("choice with a string state", base(), OK),
        case(
            "object state",
            with(base(), &["state"], json!({"ticket": "charged twice"})),
            OK,
        ),
        case(
            "array state",
            with(base(), &["state"], json!(["a", {"b": 1}])),
            OK,
        ),
        case(
            "OpenRouter optional fields",
            with(
                with(
                    with(
                        with(base(), &["provider"], json!({"sort": "price"})),
                        &["user"],
                        json!("u-1"),
                    ),
                    &["session_id"],
                    json!("s-1"),
                ),
                &["trace"],
                json!({"trace_id": "t", "span_name": "x"}),
            ),
            OK,
        ),
        case(
            "OpenRouter optional fields null",
            with(
                with(base(), &["user"], Value::Null),
                &["provider"],
                Value::Null,
            ),
            OK,
        ),
        case(
            "user of 256 characters",
            with(base(), &["user"], json!("é".repeat(256))),
            OK,
        ),
        case(
            "session_id of 256 characters",
            with(base(), &["session_id"], json!("é".repeat(256))),
            OK,
        ),
        case(
            "score with 2 levels",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": "How urgent?", "criteria": ["low", "high"]}),
            ),
            OK,
        ),
        case(
            "score with 10 levels",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": "How?", "criteria": levels10}),
            ),
            OK,
        ),
        case(
            "score levels as objects",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": {"q": "How?"}, "criteria": [{"what": "low", "examples": ["a"]}, ["high"]]}),
            ),
            OK,
        ),
        case(
            "noul without criteria",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "noul", "instructions": "Refund asked?"}),
            ),
            OK,
        ),
        case(
            "noul with true and false",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "noul", "instructions": "Refund?", "criteria": {"true": "yes", "false": null}}),
            ),
            OK,
        ),
        case(
            "noul with true only",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "noul", "instructions": "Refund?", "criteria": {"true": "yes"}}),
            ),
            OK,
        ),
        case(
            "choice with 255 options",
            with(base(), &["questions", "task"], choice_q(255)),
            OK,
        ),
        case(
            "option id of 256 bytes",
            with(
                base(),
                &["questions", "task", "criteria", &"k".repeat(256)],
                json!("long id"),
            ),
            OK,
        ),
        case(
            "option descriptions null, object, array",
            with(
                with(
                    with(
                        base(),
                        &["questions", "task", "criteria", "opt0"],
                        Value::Null,
                    ),
                    &["questions", "task", "criteria", "opt1"],
                    json!({"what": "x"}),
                ),
                &["questions", "task", "criteria", "opt2"],
                json!(["x", "y"]),
            ),
            OK,
        ),
        case(
            "description of 24,000 bytes",
            with(
                base(),
                &["questions", "task", "criteria", "opt0"],
                json!("d".repeat(24_000)),
            ),
            OK,
        ),
        case(
            "32 questions",
            with(base(), &["questions"], questions(32)),
            OK,
        ),
        case(
            "question id of 128 characters",
            with(
                base(),
                &["questions"],
                json!({ "ж".repeat(128): choice_q(2) }),
            ),
            OK,
        ),
        case(
            "state of 32 KiB",
            with(base(), &["state"], json!(big_state)),
            OK,
        ),
        case(
            "pinned model",
            with(base(), &["model"], json!("cortiq/decision@0123456789ab")),
            OK,
        ),
        case(
            "every cmf option",
            with(
                base(),
                &["cmf"],
                json!({"skill": "banking77", "oracle": true, "allow_pii_egress": true, "round": 2, "explain": true, "profile": "quality-first"}),
            ),
            OK,
        ),
        case("cmf null", with(base(), &["cmf"], Value::Null), OK),
        case(
            "cmf round null",
            with(base(), &["cmf"], json!({"round": null})),
            OK,
        ),
        // ---- 400: shape
        raw("not JSON", "{\"model\":", BAD),
        raw("not UTF-8", "\u{0}", BAD),
        case("top level array", json!([base()]), BAD),
        case(
            "unknown top-level key",
            with(base(), &["temperature"], json!(0)),
            BAD,
        ),
        case(
            "unknown cmf key",
            with(base(), &["cmf"], json!({"allow_oracle": true})),
            BAD,
        ),
        case(
            "unknown question key",
            with(base(), &["questions", "task", "examples"], json!([])),
            BAD,
        ),
        raw(
            "duplicate criteria keys",
            r#"{"model":"cortiq/decision","state":"s","questions":{"t":{"type":"choice","instructions":"i","criteria":{"a":"1","b":"2","a":"3"}}}}"#,
            BAD,
        ),
        raw(
            "duplicate top-level key",
            r#"{"model":"cortiq/decision","state":"s","state":"t","questions":{"t":{"type":"noul","instructions":"i"}}}"#,
            BAD,
        ),
        raw(
            "duplicate question ids",
            r#"{"model":"cortiq/decision","state":"s","questions":{"t":{"type":"noul","instructions":"i"},"t":{"type":"noul","instructions":"j"}}}"#,
            BAD,
        ),
        case("state missing", without(base(), &["state"]), BAD),
        // An empty state is a state-less request since 0.8.8 (DESIGN A19:
        // each question's instructions are its input), no longer a 400.
        case("empty state", with(base(), &["state"], json!("")), OK),
        case(
            "empty object state",
            with(base(), &["state"], json!({})),
            OK,
        ),
        case("empty array state", with(base(), &["state"], json!([])), OK),
        case("null state", with(base(), &["state"], Value::Null), OK),
        // A state-less request's instructions are its text: the state's
        // limit is theirs; the same instructions under a state are not.
        case(
            "state-less instructions over 32 KiB",
            with(
                with(base(), &["state"], json!({})),
                &["questions", "task", "instructions"],
                json!("x".repeat(32 * 1024 + 1)),
            ),
            BAD,
        ),
        case(
            "instructions over 32 KiB under a state",
            with(
                base(),
                &["questions", "task", "instructions"],
                json!("x".repeat(32 * 1024 + 1)),
            ),
            OK,
        ),
        case("numeric state", with(base(), &["state"], json!(42)), BAD),
        case(
            "state over 32 KiB",
            with(base(), &["state"], json!("x".repeat(32 * 1024 + 1))),
            BAD,
        ),
        case(
            "object state over 32 KiB as canonical JSON",
            with(base(), &["state"], json!({"t": "x".repeat(32 * 1024 - 7)})),
            BAD,
        ),
        case("questions missing", without(base(), &["questions"]), BAD),
        case("no question", with(base(), &["questions"], json!({})), BAD),
        case(
            "33 questions",
            with(base(), &["questions"], questions(33)),
            BAD,
        ),
        case(
            "questions as array",
            with(base(), &["questions"], json!([choice_q(2)])),
            BAD,
        ),
        case(
            "empty question id",
            with(base(), &["questions"], json!({"": choice_q(2)})),
            BAD,
        ),
        case(
            "question id of 129 characters",
            with(
                base(),
                &["questions"],
                json!({ "q".repeat(129): choice_q(2) }),
            ),
            BAD,
        ),
        case(
            "type missing",
            without(base(), &["questions", "task", "type"]),
            BAD,
        ),
        case(
            "unknown type",
            with(base(), &["questions", "task", "type"], json!("multi")),
            BAD,
        ),
        case(
            "instructions missing",
            without(base(), &["questions", "task", "instructions"]),
            BAD,
        ),
        case(
            "numeric instructions",
            with(base(), &["questions", "task", "instructions"], json!(1)),
            BAD,
        ),
        case(
            "choice with 1 option",
            with(base(), &["questions", "task"], choice_q(1)),
            BAD,
        ),
        case(
            "choice with 256 options",
            with(base(), &["questions", "task"], choice_q(256)),
            BAD,
        ),
        case(
            "choice criteria as array",
            with(
                base(),
                &["questions", "task", "criteria"],
                json!(["a", "b"]),
            ),
            BAD,
        ),
        case(
            "empty option id",
            with(base(), &["questions", "task", "criteria", ""], json!("x")),
            BAD,
        ),
        case(
            "option id of 257 bytes",
            with(
                base(),
                &["questions", "task", "criteria", &"k".repeat(257)],
                json!("x"),
            ),
            BAD,
        ),
        case(
            "description of 24,001 bytes",
            with(
                base(),
                &["questions", "task", "criteria", "opt0"],
                json!("d".repeat(24_001)),
            ),
            BAD,
        ),
        case(
            "numeric description",
            with(base(), &["questions", "task", "criteria", "opt0"], json!(3)),
            BAD,
        ),
        case(
            "score with 1 level",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": "i", "criteria": ["only"]}),
            ),
            BAD,
        ),
        case(
            "score with 11 levels",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": "i", "criteria": levels11}),
            ),
            BAD,
        ),
        case(
            "score level null",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": "i", "criteria": ["a", null]}),
            ),
            BAD,
        ),
        case(
            "score without criteria",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "score", "instructions": "i"}),
            ),
            BAD,
        ),
        case(
            "noul criteria with another key",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "noul", "instructions": "i", "criteria": {"true": "y", "maybe": "m"}}),
            ),
            BAD,
        ),
        case(
            "noul criteria as array",
            with(
                base(),
                &["questions", "task"],
                json!({"type": "noul", "instructions": "i", "criteria": ["y", "n"]}),
            ),
            BAD,
        ),
        case(
            "cmf round 3",
            with(base(), &["cmf"], json!({"round": 3})),
            BAD,
        ),
        case(
            "cmf round as string",
            with(base(), &["cmf"], json!({"round": "2"})),
            BAD,
        ),
        case(
            "cmf unknown profile",
            with(base(), &["cmf"], json!({"profile": "fast"})),
            BAD,
        ),
        case(
            "cmf oracle not boolean",
            with(base(), &["cmf"], json!({"oracle": "yes"})),
            BAD,
        ),
        case("cmf as array", with(base(), &["cmf"], json!([])), BAD),
        case(
            "user of 257 characters",
            with(base(), &["user"], json!("u".repeat(257))),
            BAD,
        ),
        case(
            "session_id of 257 characters",
            with(base(), &["session_id"], json!("s".repeat(257))),
            BAD,
        ),
        case(
            "numeric session_id",
            with(base(), &["session_id"], json!(5)),
            BAD,
        ),
        case(
            "provider as string",
            with(base(), &["provider"], json!("openai")),
            BAD,
        ),
        case("trace as array", with(base(), &["trace"], json!([])), BAD),
        case("model missing", without(base(), &["model"]), BAD),
        case("numeric model", with(base(), &["model"], json!(1)), BAD),
        // ---- 404: model ids
        case(
            "Jev model name",
            with(base(), &["model"], json!("typesafe/jev-1.13")),
            Expect::Err(404, Reason::ModelNotFound),
        ),
        case(
            "v1 model name",
            with(base(), &["model"], json!("cortiq/decision-v1")),
            Expect::Err(404, Reason::ModelNotFound),
        ),
        case(
            "pinned model with uppercase hex",
            with(base(), &["model"], json!("cortiq/decision@0123456789AB")),
            Expect::Err(404, Reason::ModelNotFound),
        ),
        case(
            "pinned model with 11 hex",
            with(base(), &["model"], json!("cortiq/decision@0123456789a")),
            Expect::Err(404, Reason::ModelNotFound),
        ),
        case(
            "pinned model with 32 hex",
            with(
                base(),
                &["model"],
                json!("cortiq/decision@0123456789abcdef0123456789abcdef"),
            ),
            Expect::Err(404, Reason::ModelNotFound),
        ),
        // ---- 413
        case(
            "body over 1 MiB",
            with(base(), &["user"], json!("u".repeat(1 << 20))),
            Expect::Err(413, Reason::PayloadTooLarge),
        ),
    ]
}

/// Both sides of the body limit: a body of exactly `limits.body_bytes` is
/// parsed, one byte more is 413 (the padding sits in `trace`, which is only
/// checked for being an object).
#[test]
fn body_of_exactly_the_limit_is_accepted_one_byte_more_is_413() {
    let limits = RequestLimits::default();
    let body = |pad: usize| -> Vec<u8> {
        serde_json::to_vec(&with(base(), &["trace"], json!({"pad": "p".repeat(pad)}))).unwrap()
    };
    let pad = limits.body_bytes - body(0).len();
    let exact = body(pad);
    assert_eq!(exact.len(), limits.body_bytes);
    assert!(parse_request(&exact, &limits).is_ok());
    let over = body(pad + 1);
    assert_eq!(over.len(), limits.body_bytes + 1);
    let e = parse_request(&over, &limits).unwrap_err();
    assert_eq!((e.status, e.reason), (413, Reason::PayloadTooLarge));
}

#[test]
fn request_cases_table() {
    let limits = RequestLimits::default();
    let cases = cases();
    assert!(cases.len() >= 40, "{} cases", cases.len());
    let mut failures = Vec::new();
    for c in &cases {
        let got = parse_request(&c.body, &limits);
        let ok = match (&c.expect, &got) {
            (Expect::Ok, Ok(_)) => true,
            (Expect::Err(status, reason), Err(e)) => e.status == *status && e.reason == *reason,
            _ => false,
        };
        if !ok {
            failures.push(format!(
                "{}: expected {}, got {:?}",
                c.name,
                match &c.expect {
                    Expect::Ok => "ok".to_string(),
                    Expect::Err(s, r) => format!("{s} {}", r.code()),
                },
                got.as_ref().map(|_| "ok").map_err(|e| e.to_string())
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    println!("{} protocol cases pass", cases.len());
}

#[test]
fn parsed_fields_keep_order_and_values() {
    let mut crit = Map::new();
    for k in ["zeta", "alpha", "mid", "Beta"] {
        crit.insert(k.into(), json!(format!("about {k}")));
    }
    let body = json!({
        "model": "cortiq/decision@0123456789ab",
        "state": {"ticket": "charged twice", "a": [1, 2.5]},
        "questions": {
            "second": {"type": "choice", "instructions": "Which?", "criteria": crit},
            "first": {"type": "score", "instructions": "How?", "criteria": ["lo", "mid", "hi"]},
            "third": {"type": "noul", "instructions": "Refund?"},
        },
        "cmf": {"skill": "banking77", "oracle": false, "round": 2, "explain": true, "profile": "cost-saver"},
        "user": "u", "session_id": "s",
    });
    let r = parse_request(
        &serde_json::to_vec(&body).unwrap(),
        &RequestLimits::default(),
    )
    .unwrap();
    assert_eq!(r.model, ModelRef::Pinned("0123456789ab".into()));
    assert!(!r.state.is_text());
    assert_eq!(r.state_text, r#"{"a":[1,2.5],"ticket":"charged twice"}"#);
    let ids: Vec<&str> = r.questions.iter().map(|q| q.id.as_str()).collect();
    assert_eq!(
        ids,
        ["second", "first", "third"],
        "question order is the request's"
    );
    assert_eq!(r.questions[0].options(), ["zeta", "alpha", "mid", "Beta"]);
    assert_eq!(r.questions[1].kind, QuestionKind::Score);
    assert_eq!(r.questions[1].levels().len(), 3);
    assert_eq!(r.questions[2].criteria, None);
    assert_eq!(
        r.cmf,
        CmfOptions {
            skill: Some("banking77".into()),
            oracle: Some(false),
            allow_pii_egress: false,
            round: Some(Rounding::Hundredths),
            explain: true,
            profile: Profile::CostSaver,
        }
    );
    assert_eq!(r.user.as_deref(), Some("u"));
    // The contract keeps the criteria order (the oracle's enum order).
    let c = r.questions[0].contract();
    let keys: Vec<&String> = c["criteria"].as_object().unwrap().keys().collect();
    assert_eq!(keys, ["zeta", "alpha", "mid", "Beta"]);
    // A string state is used as is.
    let r = parse_request(
        &serde_json::to_vec(&base()).unwrap(),
        &RequestLimits::default(),
    )
    .unwrap();
    assert_eq!(
        r.state,
        State::Text("I still have not received my new card".into())
    );
    assert_eq!(r.model, ModelRef::Latest);
    assert_eq!(r.cmf, CmfOptions::default());
    // Configured limits apply.
    let tight = RequestLimits {
        body_bytes: 1 << 20,
        state_bytes: 8,
        questions: 1,
    };
    let e = parse_request(&serde_json::to_vec(&base()).unwrap(), &tight).unwrap_err();
    assert_eq!((e.status, e.reason), (400, Reason::InvalidRequest));
}

#[test]
fn systemone_parser_keeps_the_native_boundary_strict() {
    let limits = RequestLimits::default();
    // This is the smallest useful System One body: the adapter accepts an
    // omitted transport model, null state and omitted instructions, while the
    // native decisions surface intentionally does not.
    let body = json!({
        "state": null,
        "questions": {"is_billing": {"type": "noul"}},
    });
    let bytes = serde_json::to_vec(&body).unwrap();
    let r = parse_systemone_request(&bytes, &limits).unwrap();
    assert_eq!(r.model, ModelRef::Latest);
    assert_eq!(r.state, State::Json(Value::Null));
    assert_eq!(r.state_text, "null");
    assert_eq!(r.questions[0].instructions, Value::Null);
    let native = parse_request(&bytes, &limits).unwrap_err();
    assert_eq!(
        (native.status, native.reason),
        (400, Reason::InvalidRequest)
    );

    // Every documented adapter alias maps to the locally served CMF model;
    // callers never get an answer that claims to be Jev. `default` is the
    // Decision Index kit's http engine placeholder (0.8.8).
    for model in SYSTEMONE_MODEL_ALIASES.iter().copied().chain([
        "typesafe/jev-1.13",
        SYSTEMONE_MODEL_ID,
        "default",
    ]) {
        let r = parse_systemone_request(
            &serde_json::to_vec(&json!({
                "model": model,
                "state": "refund request",
                "questions": {"task": {"type": "noul", "instructions": null}},
            }))
            .unwrap(),
            &limits,
        )
        .unwrap();
        assert_eq!(r.model, ModelRef::Latest, "{model}");
    }

    // ... on the System One dialect only.
    let e = parse_request(
        &serde_json::to_vec(&json!({
            "model": "default",
            "state": "refund request",
            "questions": {"task": {"type": "noul", "instructions": "refund?"}},
        }))
        .unwrap(),
        &limits,
    )
    .unwrap_err();
    assert_eq!((e.status, e.reason), (404, Reason::ModelNotFound));

    let e = parse_systemone_request(
        &serde_json::to_vec(&json!({
            "model": "typesafe/jev-2",
            "state": "refund request",
            "questions": {"task": {"type": "noul"}},
        }))
        .unwrap(),
        &limits,
    )
    .unwrap_err();
    assert_eq!((e.status, e.reason), (404, Reason::ModelNotFound));

    // Score legends remain the official non-null array contract even at the
    // compatibility boundary.
    let e = parse_systemone_request(
        &serde_json::to_vec(&json!({
            "model": "jev-latest",
            "state": "refund request",
            "questions": {"urgency": {"type": "score", "criteria": ["low", null]}},
        }))
        .unwrap(),
        &limits,
    )
    .unwrap_err();
    assert_eq!((e.status, e.reason), (400, Reason::InvalidRequest));
}

/// State-less detection (DESIGN A19): an empty state (`""`, `{}`, `[]`,
/// `null`) on either surface makes each question read its instructions (a
/// string as is, an object as canonical JSON); a question without
/// instructions, and every question under a non-empty state, read the state.
#[test]
fn an_empty_state_makes_the_instructions_the_input() {
    let limits = RequestLimits::default();
    let body = |state: Value| {
        json!({
            "model": "cortiq/decision",
            "state": state,
            "questions": {
                "a": {"type": "choice", "instructions": "Classify: my card is late", "criteria": {"A": "x", "B": "y"}},
                "b": {"type": "choice", "instructions": {"text": "lost pin", "lang": "en"}, "criteria": {"A": "x", "B": "y"}},
            },
        })
    };
    for state in [json!(""), json!({}), json!([]), Value::Null] {
        let bytes = serde_json::to_vec(&body(state.clone())).unwrap();
        for r in [
            parse_request(&bytes, &limits).unwrap(),
            parse_systemone_request(&bytes, &limits).unwrap(),
        ] {
            assert!(r.state.is_empty() && r.is_stateless(), "{state}");
            let (a, b) = (&r.questions[0], &r.questions[1]);
            assert!(r.reads_instructions(a) && r.reads_instructions(b));
            assert_eq!(r.input_text(a), "Classify: my card is late");
            assert_eq!(r.input_text(b), r#"{"lang":"en","text":"lost pin"}"#);
            // The state-less contract is the criteria alone.
            assert_eq!(a.contract_sha256_as(true), b.contract_sha256_as(true));
            assert_ne!(a.contract_sha256_as(true), a.contract_sha256());
        }
    }
    for state in [json!("s"), json!({"t": ""}), json!([0])] {
        let r = parse_request(&serde_json::to_vec(&body(state)).unwrap(), &limits).unwrap();
        assert!(!r.is_stateless());
        assert!(!r.reads_instructions(&r.questions[0]));
        assert_eq!(r.input_text(&r.questions[0]), r.state_text);
    }
    // System One without instructions: the state's text, as before.
    let bare = json!({"state": {}, "questions": {"n": {"type": "noul"}}});
    let r = parse_systemone_request(&serde_json::to_vec(&bare).unwrap(), &limits).unwrap();
    assert!(r.is_stateless() && !r.reads_instructions(&r.questions[0]));
    assert_eq!(r.input_text(&r.questions[0]), "{}");
}

/// The lead-in line (DESIGN C1): a state-less question whose instructions'
/// first line ends with ':' and continue with text reads that text alone (the
/// Decision Index rows "<lead-in>:\n<text>"); anything else reads the whole
/// instructions, and the contract keys never change with it.
#[test]
fn a_lead_in_line_is_not_part_of_the_local_input() {
    use cortiq_decision::protocol::after_lead_in;
    assert_eq!(
        after_lead_in("Classify the banking intent of this user request:\nmy card is late"),
        Some("my card is late")
    );
    assert_eq!(after_lead_in("Q: \r\n  two\nlines \n"), Some("two\nlines"));
    for whole in [
        "Classify: my card is late",
        "Classify this:\n   \n",
        "Which one?\nmy card is late",
        "no lead-in at all",
        "",
    ] {
        assert_eq!(after_lead_in(whole), None, "{whole:?}");
    }
    let limits = RequestLimits::default();
    let body = |instructions: &str| {
        json!({
            "state": {},
            "questions": {"q1": {"type": "choice", "instructions": instructions,
                                 "criteria": {"A": "card arrival", "B": "exchange rate"}}},
        })
    };
    let lead = body("Classify the banking intent of this user request:\nmy card is late");
    let bare = body("Classify: my card is late");
    let parse = |v: &Value| {
        parse_systemone_request(&serde_json::to_vec(v).unwrap(), &limits).unwrap()
    };
    let (a, b) = (parse(&lead), parse(&bare));
    assert_eq!(a.input_text(&a.questions[0]), "my card is late");
    assert_eq!(b.input_text(&b.questions[0]), "Classify: my card is late");
    // The contract (state-less: the criteria) is that of the question.
    assert_eq!(
        a.questions[0].contract_sha256_as(true),
        b.questions[0].contract_sha256_as(true)
    );
    // Under a state the instructions are not the input at all.
    let mut stateful = lead.clone();
    stateful["state"] = json!("the state");
    let r = parse(&stateful);
    assert_eq!(r.input_text(&r.questions[0]), "the state");
}

/// Capacity errors (DESIGN A21) keep their status and reason and carry the
/// marker the Decision Index kit reads, with `details.capacity`.
#[test]
fn capacity_errors_carry_the_marker() {
    let limits = RequestLimits {
        body_bytes: 64 * 1024,
        state_bytes: 64,
        questions: 2,
    };
    let check = |v: Value, status: u16, reason: Reason| {
        let e = parse_request(&serde_json::to_vec(&v).unwrap(), &limits).unwrap_err();
        assert_eq!((e.status, e.reason), (status, reason), "{e}");
        assert!(e.message.contains(CAPACITY_MARKER), "{e}");
        assert!(e.is_capacity());
        e
    };
    check(
        with(base(), &["state"], json!("s".repeat(65))),
        400,
        Reason::InvalidRequest,
    );
    let e = check(
        with(
            with(base(), &["state"], json!({})),
            &["questions", "task", "instructions"],
            json!("i".repeat(65)),
        ),
        400,
        Reason::InvalidRequest,
    );
    assert_eq!(e.details.unwrap()["field"], "questions.task.instructions");
    check(
        with(
            base(),
            &["questions", "task", "criteria", "opt0"],
            json!("d".repeat(24_001)),
        ),
        400,
        Reason::InvalidRequest,
    );
    check(
        with(base(), &["questions", "task"], choice_q(256)),
        400,
        Reason::InvalidRequest,
    );
    let mut three = Map::new();
    for i in 0..3 {
        three.insert(format!("q{i}"), choice_q(2));
    }
    check(
        with(base(), &["questions"], Value::Object(three)),
        400,
        Reason::InvalidRequest,
    );
    check(
        with(base(), &["user"], json!("u".repeat(70 * 1024))),
        413,
        Reason::PayloadTooLarge,
    );
    // Not capacity: a malformed request keeps its plain message.
    let e = parse_request(
        &serde_json::to_vec(&with(base(), &["questions", "task"], choice_q(1))).unwrap(),
        &limits,
    )
    .unwrap_err();
    assert!(!e.is_capacity() && !e.message.contains(CAPACITY_MARKER));
}

#[test]
fn duplicate_key_error_names_the_field() {
    let body = r#"{"model":"cortiq/decision","state":"s","questions":{"t":{"type":"choice","instructions":"i","criteria":{"a":"1","a":"2"}}}}"#;
    let e = parse_request(body.as_bytes(), &RequestLimits::default()).unwrap_err();
    let d = e.details.as_deref().unwrap();
    assert_eq!(d["field"], "questions.t.criteria");
    assert_eq!(d["key"], "a");
    assert!(e.message.contains("duplicate key 'a'"), "{}", e.message);
}

#[test]
fn malformed_json_is_named_once() {
    // A body cut after a comma (a request truncated in transit).
    let body = r#"{"model":"cortiq/decision","state":"s","questions":{"t":{"type":"choice","instructions":"i","#;
    let e = parse_request(body.as_bytes(), &RequestLimits::default()).unwrap_err();
    assert_eq!((e.status, e.reason), (400, Reason::InvalidRequest));
    assert_eq!(
        e.message,
        format!(
            "the body is not valid JSON: expected a string key at byte {}",
            body.len()
        )
    );
    let e = parse_request(b"{\"a\":1} x", &RequestLimits::default()).unwrap_err();
    assert_eq!(
        e.message,
        "the body is not valid JSON: trailing characters at byte 8"
    );
    let e = parse_request(b"\"\xff\"", &RequestLimits::default()).unwrap_err();
    assert!(
        e.message.starts_with("the body is not UTF-8 JSON: "),
        "{}",
        e.message
    );
}

#[test]
fn error_envelope_has_the_openrouter_shape() {
    let e = ApiError::new(Reason::RateLimited, "slow down")
        .with_retry_after(12)
        .with_detail("rate_per_min", json!(60));
    let b = e.body("cmf-dec-1-abc");
    assert_eq!(
        b,
        json!({"error": {"code": 429, "message": "slow down", "metadata": {
            "reason": "RATE_LIMITED", "retriable": true, "request_id": "cmf-dec-1-abc",
            "details": {"rate_per_min": 60}}}})
    );
    assert_eq!(e.retry_after, Some(12));
    let table = [
        (Reason::InvalidRequest, 400, false),
        (Reason::Unauthorized, 401, false),
        (Reason::QuotaExceeded, 402, false),
        (Reason::ModelNotFound, 404, false),
        (Reason::AdminDisabled, 404, false),
        (Reason::PayloadTooLarge, 413, false),
        (Reason::UnsupportedQuestion, 422, false),
        (Reason::RateLimited, 429, true),
        (Reason::Overloaded, 429, true),
        (Reason::Internal, 500, true),
        (Reason::OracleUnavailable, 502, true),
        (Reason::OracleBudgetExhausted, 503, false),
        (Reason::OracleDisabled, 503, false),
    ];
    for (r, status, retriable) in table {
        let e = ApiError::new(r, "x");
        assert_eq!((e.status, e.retriable()), (status, retriable), "{r:?}");
        assert_eq!(e.body("id")["error"]["metadata"]["reason"], r.code());
        assert_eq!(e.body("id")["error"]["metadata"]["details"], json!({}));
    }
    let nf = ApiError::not_found("no such decision");
    assert_eq!((nf.status, nf.reason.code()), (404, "INVALID_REQUEST"));
}

#[test]
fn feedback_body_rules() {
    let l = RequestLimits::default();
    let f = parse_feedback(
        br#"{"id":"cmf-dec-1-x","question":"task","label":"card_arrival"}"#,
        &l,
    )
    .unwrap();
    assert_eq!(
        (f.id.as_str(), f.question.as_str(), f.label.as_str()),
        ("cmf-dec-1-x", "task", "card_arrival")
    );
    for bad in [
        r#"{"id":"x","question":"q"}"#,
        r#"{"id":"x","question":"q","label":""}"#,
        r#"{"id":"x","question":"q","label":"l","extra":1}"#,
        r#"{"id":1,"question":"q","label":"l"}"#,
        r#"[]"#,
    ] {
        let e = parse_feedback(bad.as_bytes(), &l).unwrap_err();
        assert_eq!(e.status, 400, "{bad}");
    }
}

// ------------------------------------------------------------------ Jev fixtures

const DATASETS: [&str; 3] = ["banking77", "clinc150", "massive"];

fn questions_of(fx: &Value) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("task".into(), fx["question"].clone());
    m
}

#[test]
fn stored_jev_answers_pass_the_validator() {
    for ds in DATASETS {
        let fx = load(&format!("{ds}.json"));
        let qs = questions_of(&fx);
        let responses = fx["responses"].as_array().unwrap();
        assert_eq!(responses.len(), 20, "{ds}");
        for r in responses {
            let resp = &r["response"];
            validate_decisions_response(resp, &qs, &ModelRule::Jev)
                .unwrap_or_else(|e| panic!("{ds} index {}: {e}", r["index"]));
            // Not a Cortiq answer.
            assert!(validate_decisions_response(resp, &qs, &ModelRule::Cortiq).is_err());
        }
    }
    // The stored multi-type Jev call (choice + score + noul).
    let mt = load("multitype.json");
    let qs = mt["request"]["questions"].as_object().unwrap().clone();
    validate_decisions_response(&mt["jev_response"], &qs, &ModelRule::Jev).unwrap();
}

fn mutated(resp: &Value, f: impl FnOnce(&mut Value)) -> Value {
    let mut v = resp.clone();
    f(&mut v);
    v
}

#[test]
fn validator_refuses_broken_answers_with_the_python_messages() {
    let fx = load("banking77.json");
    let qs = questions_of(&fx);
    let resp = fx["responses"][12]["response"].clone();
    validate_decisions_response(&resp, &qs, &ModelRule::Jev).unwrap();
    let first_key = |v: &Value| -> String {
        v["answers"]["task"]["probabilities"]
            .as_object()
            .unwrap()
            .keys()
            .next()
            .unwrap()
            .clone()
    };
    let k0 = first_key(&resp);
    let choice = resp["answers"]["task"]["choice"]
        .as_str()
        .unwrap()
        .to_string();
    let cases: Vec<(&str, Value)> = vec![
        (
            "unexpected oracle model",
            mutated(&resp, |v| v["model"] = json!("typesafe/jev-2")),
        ),
        ("unexpected oracle model", json!([1])),
        (
            "oracle question IDs mismatch",
            mutated(&resp, |v| {
                v["answers"]["extra"] = json!({"type": "noul", "noul": 1});
            }),
        ),
        (
            "oracle answer type mismatch",
            mutated(&resp, |v| v["answers"]["task"]["type"] = json!("score")),
        ),
        (
            "invalid oracle probabilities",
            mutated(&resp, |v| {
                v["answers"]["task"]["probabilities"]
                    .as_object_mut()
                    .unwrap()
                    .remove(&k0);
            }),
        ),
        (
            "invalid oracle probabilities",
            mutated(&resp, |v| {
                v["answers"]["task"]["probabilities"][&k0] = json!(true)
            }),
        ),
        (
            "invalid oracle distribution/confidence",
            mutated(&resp, |v| {
                v["answers"]["task"]["probabilities"][&choice] = json!(0.5)
            }),
        ),
        (
            "invalid oracle distribution/confidence",
            mutated(&resp, |v| v["answers"]["task"]["confidence"] = json!(1.5)),
        ),
        (
            "invalid oracle choice",
            mutated(&resp, |v| {
                let other = if k0 == choice {
                    "age_limit"
                } else {
                    k0.as_str()
                };
                v["answers"]["task"]["choice"] = json!(other);
            }),
        ),
        (
            "invalid oracle cost",
            mutated(&resp, |v| v["usage"]["cost"] = json!(-1)),
        ),
        (
            "invalid oracle cost",
            mutated(&resp, |v| {
                v["usage"].as_object_mut().unwrap().remove("cost");
            }),
        ),
        (
            "invalid oracle token usage",
            mutated(&resp, |v| v["usage"]["input_tokens"] = json!(4372.0)),
        ),
    ];
    for (want, v) in cases {
        assert_eq!(
            validate_decisions_response(&v, &qs, &ModelRule::Jev),
            Err(want.to_string())
        );
    }
    // Score and noul paths on the multi-type answer.
    let mt = load("multitype.json");
    let mq = mt["request"]["questions"].as_object().unwrap().clone();
    let j = &mt["jev_response"];
    let score_cases: Vec<(&str, Value)> = vec![
        (
            "invalid noul",
            mutated(j, |v| v["answers"]["refund"]["noul"] = json!(1.5)),
        ),
        (
            "invalid oracle score",
            mutated(j, |v| v["answers"]["urgency"]["score"] = json!(3)),
        ),
        (
            "invalid oracle legend",
            mutated(j, |v| {
                v["answers"]["urgency"]["legend"]
                    .as_object_mut()
                    .unwrap()
                    .remove("2");
            }),
        ),
        (
            "oracle score inconsistent with distribution",
            mutated(j, |v| v["answers"]["urgency"]["score"] = json!(2)),
        ),
    ];
    for (want, v) in score_cases {
        assert_eq!(
            validate_decisions_response(&v, &mq, &ModelRule::Jev),
            Err(want.to_string())
        );
    }
}

/// `(option, p)` of an answer in its key order, as f32.
fn probs_f32(a: &Value) -> Vec<(String, f32)> {
    a["probabilities"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap() as f32))
        .collect()
}

/// A stored Jev number with its float noise removed: a float becomes the f64
/// nearest to its hundredth (`k/100`), an integer stays an integer.
fn nearest_hundredth(v: &Value) -> Value {
    match v {
        Value::Number(n) if n.is_f64() => {
            let x = n.as_f64().unwrap();
            json!((x * 100.0).round() / 100.0)
        }
        other => other.clone(),
    }
}

/// Limitation (the data allows no stronger check): the fixtures hold Jev's
/// answers only, already quantised to hundredths, and Jev's own confidence; its
/// raw probabilities are not stored. So this feeds Jev's quantised numbers back
/// through `round: 2` and shows the formatting and idempotence of the rounding
/// (0 and 1 as integers, the shortest text of each hundredth) byte for byte,
/// not the quantisation of raw probabilities (`answer::tests::
/// round2_matches_jev_forms` covers rounding from raw values).
#[test]
fn round2_reproduces_the_stored_jev_answers_byte_for_byte() {
    let mut n = 0;
    let mut values = 0;
    let mut noisy = Vec::new();
    for ds in DATASETS {
        let fx = load(&format!("{ds}.json"));
        for r in fx["responses"].as_array().unwrap() {
            let jev = &r["response"]["answers"]["task"];
            let probs = probs_f32(jev);
            let opts: Vec<(&str, f32)> = probs.iter().map(|(k, p)| (k.as_str(), *p)).collect();
            let conf = jev["confidence"].as_f64().unwrap() as f32;
            let ours = answer::choice_answer(
                &opts,
                jev["choice"].as_str().unwrap(),
                conf,
                Rounding::Hundredths,
            );
            // Key order of probabilities is the request's (here: Jev's).
            let ok: Vec<&String> = ours["probabilities"].as_object().unwrap().keys().collect();
            let jk: Vec<&String> = jev["probabilities"].as_object().unwrap().keys().collect();
            assert_eq!(ok, jk);
            // Number by number: the same text wherever Jev wrote the nearest
            // double of its hundredth (and 0/1 as integers); the same hundredth
            // where Jev's own arithmetic left float noise (e.g. 0.9400000000000001).
            let mut pairs: Vec<(String, &Value, &Value)> = jev["probabilities"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v, &ours["probabilities"][k]))
                .collect();
            pairs.push(("confidence".into(), &jev["confidence"], &ours["confidence"]));
            for (k, j, o) in pairs {
                values += 1;
                if canonical::to_string(&nearest_hundredth(j)) == canonical::to_string(j) {
                    assert_eq!(
                        canonical::to_string(o),
                        canonical::to_string(j),
                        "{ds} index {} {k}",
                        r["index"]
                    );
                } else {
                    assert_eq!(o, &nearest_hundredth(j), "{ds} index {} {k}", r["index"]);
                    noisy.push(format!("{ds}#{} {k}={j}", r["index"]));
                }
            }
            n += 1;
        }
    }
    assert_eq!(n, 60);
    println!(
        "round=2 reproduced {n} Jev answers: {} of {values} numbers byte for byte, {} with Jev float noise and the same hundredth: {noisy:?}",
        values - noisy.len(),
        noisy.len()
    );
    assert!(noisy.len() * 1000 < values, "noise is rare in Jev's output");
}

#[test]
fn jev_confidence_is_the_n_pmax_formula() {
    let mut worst = 0.0f64;
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for ds in DATASETS {
        let fx = load(&format!("{ds}.json"));
        for r in fx["responses"].as_array().unwrap() {
            let jev = &r["response"]["answers"]["task"];
            let probs = probs_f32(jev);
            let pmax = probs.iter().map(|(_, p)| *p).fold(0.0f32, f32::max);
            let ours = answer::round2(answer::answer_confidence(pmax, probs.len()));
            let err = (ours.as_f64().unwrap() - jev["confidence"].as_f64().unwrap()).abs();
            worst = worst.max(err);
            sum += err;
            n += 1;
        }
    }
    let mae = sum / n as f64;
    println!(
        "Jev confidence vs (N·p_max−1)/(N−1) on {n} stored answers: MAE {mae:.4}, max {worst:.4}"
    );
    // Jev computes it from unrounded p_max; the fixtures carry p_max rounded to
    // hundredths, which bounds the gap (measured on 4,678 answers: MAE
    // 0.004–0.012, max 0.029).
    assert!(worst <= 0.03, "max {worst}");
    assert!(mae <= 0.012, "MAE {mae}");
}

#[test]
fn local_answer_shapes_pass_the_validator() {
    // Softmax-like distributions over 3 and 77 options, exact and rounded.
    for n in [3usize, 77] {
        let mut ps: Vec<f32> = (0..n).map(|i| (-(i as f32) * 0.37).exp()).collect();
        let s: f32 = ps.iter().sum();
        for p in &mut ps {
            *p /= s;
        }
        let ids: Vec<String> = (0..n).map(|i| format!("label_{i}")).collect();
        let opts: Vec<(&str, f32)> = ids
            .iter()
            .map(String::as_str)
            .zip(ps.iter().copied())
            .collect();
        let mut crit = Map::new();
        for id in &ids {
            crit.insert(id.clone(), json!("d"));
        }
        let qs: Map<String, Value> = [(
            "task".to_string(),
            json!({"type": "choice", "instructions": "i", "criteria": crit}),
        )]
        .into_iter()
        .collect();
        for rounding in [Rounding::Exact, Rounding::Hundredths] {
            let conf = answer::answer_confidence(ps[0], n);
            let a = answer::choice_answer(&opts, &ids[0], conf, rounding);
            let resp = json!({
                "id": protocol::new_request_id(1),
                "model": "cortiq/decision@0123456789ab",
                "provider": "Cortiq",
                "answers": {"task": a},
                "usage": {"input_tokens": 10, "output_tokens": n, "cost": 0.0},
            });
            validate_decisions_response(&resp, &qs, &ModelRule::Cortiq)
                .unwrap_or_else(|e| panic!("n {n} {rounding:?}: {e}"));
        }
    }
}
