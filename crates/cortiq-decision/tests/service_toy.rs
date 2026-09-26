//! `DecisionService` on the toy model (spec §4, §6.1), hermetic.
//!
//! The toy file has two skills built from the toy encoder
//! (`tests/fixtures/toy/encoder`): `topics` {Weather, billing, cards, travel}
//! and `shop` {billing, cards, food}.
//!
//! * `decide` is bit-equal to the batch evaluator (`eval.rs`, `cortiq decide
//!   --input`) on every dev row: choice, p_top, confidence, novelty, margin,
//!   accepted, certified, the five smallest errors; the answers pass the
//!   validator port and carry Jev's confidence formula;
//! * matching: exact, subset (decided over L only), superset, untrained,
//!   forced (`cmf.skill`), ambiguous; the stored multi-type request → 422 with a
//!   reason per question, and 200 with a mock oracle;
//! * escalation with a mock [`Escalator`]: gate-accepted questions never reach
//!   it (even with `cmf.oracle: true`), consent and configuration switches, the
//!   refusal and failure codes (422/502/503) and flags, cache and oracle
//!   answers, passthrough cost;
//! * metering (spec §4.9): input tokens over the whole request, output tokens,
//!   processed tokens with truncation, prices exact and 0 by default;
//! * keys, rate window, quotas and credit; `keys.json` holds no raw key; admin
//!   guard;
//! * usage ledger: append, replay, snapshot + tail, month files, a cut partial
//!   line, a refused corrupt line, the background flusher;
//! * the state directory `LOCK` is exclusive; `CURRENT` round-trips.

use cortiq_decision::answer::OracleAnswer;
use cortiq_decision::build::{self, TrainOptions};
use cortiq_decision::canonical;
use cortiq_decision::config::Config;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::eval::{EvalInput, Evaluator, jev_confidence};
use cortiq_decision::keys::{self, KeyStore, NewKey};
use cortiq_decision::ledger::{Actions, UsageLedger, UsageRecord, month_of};
use cortiq_decision::matching::MatchKind;
use cortiq_decision::metering::{self, Usd};
use cortiq_decision::protocol::{
    ApiError, FeedbackRequest, ModelRule, Reason, validate_decisions_response, wire_questions,
};
use cortiq_decision::resonance::decide;
use cortiq_decision::service::{
    Action, AdminCommand, Decided, DecisionService, Escalation, EscalationResult, Escalator,
    LoadedModel, ModelHandle, Observation, OracleUsage, Principal, RefusalReason, Resolution,
    Resolved,
};
use cortiq_decision::statedir::{Current, StateDir};
use serde_json::{Map, Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const EPOCH: u64 = 1_790_000_000;
const TOPICS: [&str; 4] = ["Weather", "billing", "cards", "travel"];
const SHOP: [&str; 3] = ["billing", "cards", "food"];

// ------------------------------------------------------------------ toy model

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn pool(label: &str) -> [&'static str; 8] {
    match label {
        "Weather" => [
            "rain",
            "snow",
            "forecast",
            "sunny",
            "wind",
            "storm",
            "cloudy",
            "temperature",
        ],
        "billing" => [
            "invoice",
            "charge",
            "refund",
            "payment",
            "bill",
            "fee",
            "receipt",
            "statement",
        ],
        "cards" => [
            "card",
            "pin",
            "atm",
            "contactless",
            "debit",
            "credit",
            "freeze",
            "replace",
        ],
        "travel" => [
            "flight", "hotel", "airport", "booking", "passport", "luggage", "visa", "train",
        ],
        "food" => [
            "pizza", "salad", "bread", "cheese", "soup", "rice", "apple", "coffee",
        ],
        _ => unreachable!(),
    }
}

const FILLER: [&str; 8] = [
    "please", "help", "my", "the", "today", "need", "about", "with",
];

fn synth(labels: &[&str], per_label: usize, seed: u64, tag: &str) -> Vec<(String, String)> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    for i in 0..per_label {
        for label in labels {
            let p = pool(label);
            let mut words = Vec::new();
            for _ in 0..2 + rng.below(3) {
                words.push(p[rng.below(p.len())]);
            }
            for _ in 0..1 + rng.below(2) {
                words.push(FILLER[rng.below(FILLER.len())]);
            }
            let r = rng.below(words.len());
            words.swap(0, r);
            out.push((format!("{} {tag}{i}", words.join(" ")), label.to_string()));
        }
    }
    out
}

fn jsonl(rows: &[(String, String)]) -> String {
    rows.iter()
        .map(|(t, l)| json!({"text": t, "label": l}).to_string() + "\n")
        .collect()
}

fn write(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    p
}

struct Toy {
    _dir: tempfile::TempDir,
    path: PathBuf,
    dev: Vec<(String, String)>,
}

fn skill_opts(
    dir: &Path,
    id: &str,
    labels: &[&str],
    seed: u64,
) -> (TrainOptions, Vec<(String, String)>) {
    let train = synth(labels, 30, seed, &format!("{id}t"));
    let cal = synth(labels, 80, seed + 1, &format!("{id}c"));
    let dev = synth(labels, 12, seed + 2, &format!("{id}d"));
    let mut crit = Map::new();
    for l in labels {
        crit.insert(l.to_string(), json!(format!("The message is about {l}.")));
    }
    let q = json!({"instructions": "Which topic is the message about?", "criteria": crit});
    let mut o = TrainOptions::new(
        id,
        vec![write(dir, &format!("{id}-train.jsonl"), &jsonl(&train))],
    );
    o.calibration = Some(write(dir, &format!("{id}-cal.jsonl"), &jsonl(&cal)));
    o.dev = Some(write(dir, &format!("{id}-dev.jsonl"), &jsonl(&dev)));
    o.question = Some(write(dir, &format!("{id}-q.json"), &q.to_string()));
    o.threads = 2;
    o.created_unix = Some(EPOCH);
    (o, dev)
}

fn toy() -> &'static Toy {
    static TOY: OnceLock<Toy> = OnceLock::new();
    TOY.get_or_init(|| {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let enc = d.join("enc.cmf");
        let export = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder");
        build::init_encoder(&export, &enc, Some(EPOCH)).expect("init toy encoder");
        let (o1, dev) = skill_opts(d, "topics", &TOPICS, 11);
        let s1 = d.join("s1.cmf");
        build::train(&enc, &o1, &s1).expect("train topics");
        let (o2, _) = skill_opts(d, "shop", &SHOP, 21);
        let path = d.join("toy.cmf");
        build::add_skill(&s1, &o2, &path).expect("add shop");
        Toy {
            _dir: dir,
            path,
            dev,
        }
    })
}

fn loaded() -> LoadedModel {
    LoadedModel::new(DecisionModel::open(&toy().path, Verify::Full).unwrap()).unwrap()
}

fn service_with(cfg: Config, esc: Option<Arc<dyn Escalator>>) -> DecisionService {
    let handle = Arc::new(ModelHandle::new(loaded()));
    DecisionService::open(handle, cfg, esc)
        .unwrap()
        .with_loopback(true)
}

fn service() -> DecisionService {
    service_with(Config::default(), None)
}

fn oracle_cfg() -> Config {
    let mut c = Config::default();
    c.oracle.enabled = true;
    c
}

fn choice(labels: &[&str]) -> Value {
    let mut c = Map::new();
    for l in labels {
        c.insert(l.to_string(), json!(format!("about {l}")));
    }
    json!({"type": "choice", "instructions": "Which topic?", "criteria": c})
}

fn body(state: Value, questions: Value, cmf: Option<Value>) -> Vec<u8> {
    let mut v = json!({"model": "cortiq/decision", "state": state, "questions": questions});
    if let Some(c) = cmf {
        v["cmf"] = c;
    }
    serde_json::to_vec(&v).unwrap()
}

fn run(svc: &DecisionService, b: &[u8]) -> Result<Decided, ApiError> {
    svc.decide_body(b, &Principal::open())
}

// ------------------------------------------------------------------ mock escalator

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Answer,
    Cache,
    Refuse(RefusalReason),
    Fail,
    WrongOption,
}

/// What the mock saw of one pending question: id, match kind, skill, unknown
/// options, whether it carried a local decision.
type Seen = (String, MatchKind, Option<String>, Vec<String>, bool);

struct Mock {
    mode: Mutex<Mode>,
    calls: AtomicUsize,
    observed: AtomicUsize,
    /// Per call: (question id, match kind, skill, unknown options, had a local decision).
    seen: Mutex<Vec<Vec<Seen>>>,
    cost: f64,
}

impl Mock {
    fn new(mode: Mode) -> Arc<Self> {
        Arc::new(Self {
            mode: Mutex::new(mode),
            calls: AtomicUsize::new(0),
            observed: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
            cost: 0.00002,
        })
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Escalator for Mock {
    fn escalate(&self, e: &Escalation<'_>) -> EscalationResult {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.seen.lock().unwrap().push(
            e.pending
                .iter()
                .map(|p| {
                    (
                        p.question.id.clone(),
                        p.matched.kind,
                        p.matched.skill.clone(),
                        p.matched.unknown.clone(),
                        p.local.is_some(),
                    )
                })
                .collect(),
        );
        assert_eq!(
            e.features.phi_p.len() + e.features.phi_h.len(),
            e.model.model().signal_dim()
        );
        let mode = *self.mode.lock().unwrap();
        let verdict = |q: &cortiq_decision::protocol::Question| match q.kind {
            cortiq_decision::protocol::QuestionKind::Choice => {
                OracleAnswer::Choice(q.options().last().unwrap().to_string())
            }
            cortiq_decision::protocol::QuestionKind::Score => OracleAnswer::Score(1),
            cortiq_decision::protocol::QuestionKind::Noul => OracleAnswer::Noul(true),
        };
        let resolved = e
            .pending
            .iter()
            .map(|p| {
                Resolved::new(match mode {
                    Mode::Answer => Resolution::Oracle(verdict(p.question)),
                    Mode::Cache => Resolution::Cache(verdict(p.question)),
                    Mode::Refuse(r) => Resolution::Refused(r),
                    Mode::Fail => Resolution::Failed("mock failure".into()),
                    Mode::WrongOption => Resolution::Oracle(OracleAnswer::Choice("nope".into())),
                })
            })
            .collect();
        let usage = if mode == Mode::Answer || mode == Mode::WrongOption {
            OracleUsage {
                calls: 1,
                input_tokens: 1234,
                output_tokens: 5,
                cost: Usd::from_f64(self.cost).unwrap(),
            }
        } else {
            OracleUsage::default()
        };
        EscalationResult { resolved, usage }
    }

    fn feedback(&self, f: &FeedbackRequest, p: &Principal) -> Result<Value, ApiError> {
        Ok(json!({"id": f.id, "account": p.account, "accepted": true}))
    }

    fn admin(&self, c: &AdminCommand) -> Result<Value, ApiError> {
        Ok(json!({"command": format!("{c:?}")}))
    }

    fn observe(&self, o: &Observation<'_>) {
        assert!(!o.questions.is_empty());
        self.observed.fetch_add(1, Ordering::SeqCst);
    }
}

/// A dev text the `topics` gate accepts and a text it rejects (found with a
/// service without oracle).
fn accepted_and_rejected() -> (String, String) {
    static PAIR: OnceLock<(String, String)> = OnceLock::new();
    PAIR.get_or_init(find_accepted_and_rejected).clone()
}

fn find_accepted_and_rejected() -> (String, String) {
    let svc = &service();
    let q = json!({"task": choice(&TOPICS)});
    let mut acc = None;
    for (t, _) in &toy().dev {
        let d = run(svc, &body(json!(t), q.clone(), None)).unwrap();
        if d.questions[0].action == Action::Local {
            acc = Some(t.clone());
            break;
        }
    }
    let mut rej = None;
    for t in [
        "xylophone quantum zebra",
        "the the the the",
        "pizza salad bread cheese",
        "please help",
        "zzz qqq",
    ] {
        let d = run(svc, &body(json!(t), q.clone(), None)).unwrap();
        if d.questions[0].action == Action::Abstain {
            rej = Some(t.to_string());
            break;
        }
    }
    (
        acc.expect("a dev text the gate accepts"),
        rej.expect("a text the gate rejects"),
    )
}

// ------------------------------------------------------------------ bit equality with eval.rs

#[test]
fn decide_is_bit_equal_to_the_batch_evaluator() {
    let svc = service();
    let model = DecisionModel::open(&toy().path, Verify::Light).unwrap();
    let ev = Evaluator::new(&model, "topics").unwrap();
    // Options in reverse task order: the decision must not depend on it.
    let mut opts = TOPICS.to_vec();
    opts.reverse();
    let qs = json!({"task": choice(&opts)});
    let questions = wire_questions(
        &cortiq_decision::protocol::parse_request(
            &body(json!("x"), qs.clone(), None),
            &Default::default(),
        )
        .unwrap(),
    );
    let (mut local, mut abstain) = (0, 0);
    for (i, (text, label)) in toy().dev.iter().enumerate() {
        let row = ev
            .row(
                i,
                &EvalInput {
                    text: text.clone(),
                    label: Some(label.clone()),
                },
            )
            .unwrap();
        let d = run(&svc, &body(json!(text), qs.clone(), None)).unwrap();
        let o = &d.questions[0];
        assert_eq!(o.matched.kind, MatchKind::Exact);
        let l = o.local.as_ref().unwrap();
        let bits = |x: f32| x.to_bits();
        assert_eq!(l.choice, row.choice, "row {i}");
        assert_eq!(bits(l.decision.p_top), bits(row.p_top), "row {i}");
        assert_eq!(bits(l.confidence), bits(row.confidence), "row {i}");
        assert_eq!(bits(l.decision.novelty), bits(row.novelty), "row {i}");
        assert_eq!(bits(l.decision.margin), bits(row.margin), "row {i}");
        assert_eq!(l.gate_accepted, row.accepted, "row {i}");
        assert_eq!(l.accepted, row.accepted, "row {i} (balanced profile)");
        assert_eq!(l.certified, row.certified, "row {i}");
        let top5: Vec<(String, u32)> = l
            .ranked_errors(5)
            .into_iter()
            .map(|(k, e)| (k.to_string(), e.to_bits()))
            .collect();
        let want: Vec<(String, u32)> = row
            .errors_top5
            .iter()
            .map(|(k, e)| (k.clone(), e.to_bits()))
            .collect();
        assert_eq!(top5, want, "row {i}");
        // The JSON answer: every option in request order, the choice, Jev's
        // confidence; the validator port accepts it.
        let a = &d.response["answers"]["task"];
        let keys: Vec<&String> = a["probabilities"].as_object().unwrap().keys().collect();
        assert_eq!(keys, opts);
        assert_eq!(a["choice"], json!(row.choice));
        let n = opts.len() as f32;
        let conf = ((n * row.p_top - 1.0) / (n - 1.0)).max(0.0);
        assert_eq!(a["confidence"], cortiq_decision::eval::f32_json(conf));
        assert_eq!(
            conf.to_bits(),
            jev_confidence(row.p_top, 4).max(0.0).to_bits()
        );
        for (k, v) in a["probabilities"].as_object().unwrap() {
            let p = l.probability(k).unwrap();
            assert_eq!(v, &cortiq_decision::eval::f32_json(p));
        }
        validate_decisions_response(&d.response, &questions, &ModelRule::Cortiq).unwrap();
        let q = &d.response["cmf"]["questions"]["task"];
        assert_eq!(q["match"], "exact");
        assert_eq!(q["skill"], "topics");
        assert_eq!(q["certified"], json!(row.certified));
        assert_eq!(q["gate"]["accepted"], json!(row.accepted));
        assert_eq!(
            q["gate"]["p_top"],
            cortiq_decision::eval::f32_json(row.p_top)
        );
        match o.action {
            Action::Local => local += 1,
            Action::Abstain => abstain += 1,
            _ => unreachable!(),
        }
        assert_eq!(o.action == Action::Local, row.accepted);
    }
    println!(
        "{} dev rows bit-equal to eval.rs ({local} local, {abstain} abstain)",
        toy().dev.len()
    );
    assert!(local > 0);
}

#[test]
fn round2_answers_are_hundredths_and_still_valid() {
    let svc = service();
    let qs = json!({"task": choice(&TOPICS)});
    let req = cortiq_decision::protocol::parse_request(
        &body(json!(toy().dev[0].0), qs.clone(), Some(json!({"round": 2}))),
        &Default::default(),
    )
    .unwrap();
    let d = svc.decide(&req, &Principal::open()).unwrap();
    let a = &d.response["answers"]["task"];
    for v in a["probabilities"].as_object().unwrap().values() {
        let x = v.as_f64().unwrap();
        assert!((x * 100.0 - (x * 100.0).round()).abs() < 1e-9, "{v}");
        if x == 0.0 || x == 1.0 {
            assert!(v.is_u64(), "0 and 1 are integers: {v}");
        }
    }
    validate_decisions_response(&d.response, &wire_questions(&req), &ModelRule::Cortiq).unwrap();
    // The configured default applies when the request says nothing.
    let mut cfg = Config::default();
    cfg.response.round = Some(2);
    let svc2 = service_with(cfg, None);
    let d2 = run(&svc2, &body(json!(toy().dev[0].0), qs.clone(), None)).unwrap();
    assert_eq!(d2.response["answers"], d.response["answers"]);
    // cmf.round null overrides the configured rounding.
    let d3 = run(
        &svc2,
        &body(json!(toy().dev[0].0), qs, Some(json!({"round": null}))),
    )
    .unwrap();
    assert_eq!(
        d3.response["answers"],
        run(
            &svc,
            &body(
                json!(toy().dev[0].0),
                json!({"task": choice(&TOPICS)}),
                None
            )
        )
        .unwrap()
        .response["answers"]
    );
}

// ------------------------------------------------------------------ matching

#[test]
fn subset_is_decided_over_its_options_only() {
    let svc = service();
    let model = DecisionModel::open(&toy().path, Verify::Light).unwrap();
    let ev = Evaluator::new(&model, "topics").unwrap();
    let s = ev.scorer();
    let sub = ["travel", "Weather"];
    for (text, _) in toy().dev.iter().take(12) {
        let d = run(
            &svc,
            &body(json!(text), json!({"task": choice(&sub)}), None),
        )
        .unwrap();
        let o = &d.questions[0];
        assert_eq!(o.matched.kind, MatchKind::Subset);
        assert_eq!(o.matched.skill.as_deref(), Some("topics"));
        let l = o.local.as_ref().unwrap();
        assert!(!l.certified && !o.certified, "a subset is never certified");
        // Reference: the errors of all candidates, restricted to L in candidate order.
        let x = ev.encoder().signal(text);
        let all = s.errors(&x).unwrap();
        let idx: Vec<usize> = s
            .labels()
            .iter()
            .enumerate()
            .filter(|(_, lab)| sub.contains(&lab.as_str()))
            .map(|(i, _)| i)
            .collect();
        let e: Vec<f32> = idx.iter().map(|&i| all[i]).collect();
        let st: Vec<_> = idx.iter().map(|&i| s.stats()[i]).collect();
        let want = decide(&e, &st, s.gate().temperature).unwrap();
        assert_eq!(l.decision, want);
        let a = &d.response["answers"]["task"];
        assert_eq!(a["probabilities"].as_object().unwrap().len(), 2);
        let expected_path = if o.action == Action::Local {
            "router:uncertified_subset"
        } else {
            "escalate→disabled"
        };
        assert_eq!(
            d.response["cmf"]["questions"]["task"]["decision_path"],
            expected_path
        );
    }
}

#[test]
fn superset_untrained_forced_and_ambiguous() {
    let svc = service();
    let text = json!(toy().dev[0].0);
    // Superset without an oracle: 422 with the unknown option named.
    let mut sup = TOPICS.to_vec();
    sup.push("zz_new");
    let e = run(
        &svc,
        &body(text.clone(), json!({"task": choice(&sup)}), None),
    )
    .unwrap_err();
    assert_eq!((e.status, e.reason), (422, Reason::UnsupportedQuestion));
    let q = &e.details.as_deref().unwrap()["questions"]["task"];
    assert_eq!(q["match"], "superset");
    assert_eq!(q["skill"], "topics");
    assert!(q["reason"].as_str().unwrap().contains("zz_new"));
    // Untrained options.
    let e = run(
        &svc,
        &body(text.clone(), json!({"task": choice(&["x", "y"])}), None),
    )
    .unwrap_err();
    assert_eq!(e.status, 422);
    assert_eq!(
        e.details.as_deref().unwrap()["questions"]["task"]["match"],
        "untrained"
    );
    // Ambiguous: {billing, cards} is a subset of both skills.
    let e = run(
        &svc,
        &body(
            text.clone(),
            json!({"task": choice(&["billing", "cards"])}),
            None,
        ),
    )
    .unwrap_err();
    let r = e.details.as_deref().unwrap()["questions"]["task"]["reason"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(r.contains("ambiguous"), "{r}");
    // Forced: cmf.skill picks the skill.
    for skill in ["topics", "shop"] {
        let d = run(
            &svc,
            &body(
                text.clone(),
                json!({"task": choice(&["billing", "cards"])}),
                Some(json!({"skill": skill})),
            ),
        )
        .unwrap();
        assert_eq!(d.questions[0].matched.kind, MatchKind::Subset);
        assert_eq!(d.questions[0].matched.skill.as_deref(), Some(skill));
    }
    // Exact on the second skill without forcing.
    let d = run(
        &svc,
        &body(
            text.clone(),
            json!({"task": choice(&["food", "cards", "billing"])}),
            None,
        ),
    )
    .unwrap();
    assert_eq!(d.questions[0].matched.kind, MatchKind::Exact);
    assert_eq!(d.questions[0].matched.skill.as_deref(), Some("shop"));
    // Forced onto a skill the options do not fit → untrained (422).
    let e = run(
        &svc,
        &body(
            text.clone(),
            json!({"task": choice(&["food", "cards"])}),
            Some(json!({"skill": "topics"})),
        ),
    )
    .unwrap_err();
    assert_eq!(e.status, 422);
    // Unknown forced skill → 400.
    let e = run(
        &svc,
        &body(
            text.clone(),
            json!({"task": choice(&TOPICS)}),
            Some(json!({"skill": "nope"})),
        ),
    )
    .unwrap_err();
    assert_eq!((e.status, e.reason), (400, Reason::InvalidRequest));
    // Several questions of one skill share the errors; an object state is not certified.
    let d = run(
        &svc,
        &body(
            json!({"message": toy().dev[0].0}),
            json!({"a": choice(&TOPICS), "b": choice(&["cards", "travel"])}),
            None,
        ),
    )
    .unwrap();
    assert!(d.questions.iter().all(|o| !o.certified));
    assert_eq!(d.questions[0].matched.kind, MatchKind::Exact);
    // A pinned model id must name the served model.
    let sha12 = svc.handle().current().sha12().to_string();
    let mut v: Value =
        serde_json::from_slice(&body(text.clone(), json!({"task": choice(&TOPICS)}), None))
            .unwrap();
    v["model"] = json!(format!("cortiq/decision@{sha12}"));
    assert!(run(&svc, &serde_json::to_vec(&v).unwrap()).is_ok());
    v["model"] = json!("cortiq/decision@000000000000");
    let e = run(&svc, &serde_json::to_vec(&v).unwrap()).unwrap_err();
    assert_eq!((e.status, e.reason), (404, Reason::ModelNotFound));
}

#[test]
fn multitype_request_is_422_without_oracle_and_200_with_a_mock() {
    let mt: Value = canonical::parse(
        &std::fs::read(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jev/multitype.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let b = serde_json::to_vec(&mt["request"]).unwrap();
    let e = run(&service(), &b).unwrap_err();
    assert_eq!((e.status, e.reason), (422, Reason::UnsupportedQuestion));
    let d = e.details.as_deref().unwrap();
    assert_eq!(d["questions"]["team"]["match"], "untrained");
    assert!(
        d["questions"]["urgency"]["reason"]
            .as_str()
            .unwrap()
            .contains("score")
    );
    assert!(
        d["questions"]["refund"]["reason"]
            .as_str()
            .unwrap()
            .contains("noul")
    );
    // With a mock oracle: 200, typed answers.
    let mock = Mock::new(Mode::Answer);
    let svc = service_with(oracle_cfg(), Some(mock.clone()));
    let d = run(&svc, &b).unwrap();
    assert_eq!(mock.calls(), 1, "one call for all undetermined questions");
    let a = &d.response["answers"];
    assert_eq!(a["team"], json!({"type": "choice", "choice": "account"}));
    assert_eq!(a["urgency"]["type"], "score");
    assert_eq!(a["urgency"]["score"], 1);
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
        assert_eq!(d.response["cmf"]["questions"][q]["certified"], false);
    }
    // Output tokens: one per oracle answer.
    assert_eq!(d.metered.output_tokens, 3);
    assert_eq!(mock.observed.load(Ordering::SeqCst), 1);
}

// ------------------------------------------------------------------ escalation

#[test]
fn gate_accepted_questions_never_reach_the_oracle() {
    let mock = Mock::new(Mode::Answer);
    let svc = service_with(oracle_cfg(), Some(mock.clone()));
    let (acc, _) = accepted_and_rejected();
    let calls0 = mock.calls();
    for cmf in [None, Some(json!({"oracle": true}))] {
        let d = run(
            &svc,
            &body(json!(acc), json!({"task": choice(&TOPICS)}), cmf),
        )
        .unwrap();
        assert_eq!(d.questions[0].action, Action::Local);
        assert_eq!(d.metered.oracle.calls, 0);
    }
    assert_eq!(
        mock.calls(),
        calls0,
        "an accepted question is never escalated"
    );
}

/// `decide_local` (the shadow mode of `serve --shadow-of`) is exactly the local
/// decision `decide` starts from, for every profile, and nothing else: no
/// escalation, no observation, no usage.
#[test]
fn decide_local_is_the_local_decision_and_nothing_else() {
    let mock = Mock::new(Mode::Answer);
    let svc = service_with(oracle_cfg(), Some(mock.clone()));
    let plain = service();
    let (acc, rej) = accepted_and_rejected();
    let (calls0, observed0) = (mock.calls(), mock.observed.load(Ordering::SeqCst));
    for text in [&acc, &rej] {
        for profile in ["balanced", "quality-first", "cost-saver"] {
            let b = body(
                json!(text),
                json!({"task": choice(&TOPICS)}),
                Some(json!({"profile": profile, "oracle": true})),
            );
            let req = cortiq_decision::protocol::parse_request(&b, &svc.limits()).unwrap();
            let lo = svc.decide_local(&req).unwrap();
            let d = run(&plain, &b).unwrap();
            assert_eq!(lo.questions.len(), 1);
            let (m, local) = &lo.questions[0];
            assert_eq!(m, &d.questions[0].matched);
            assert_eq!(local.as_ref(), d.questions[0].local.as_ref(), "{profile}");
            assert_eq!(
                local.as_ref().unwrap().accepted,
                d.questions[0].action == Action::Local
            );
            assert_eq!(lo.timings.oracle, std::time::Duration::ZERO);
            assert_eq!(lo.model, d.response["model"].as_str().unwrap());
        }
    }
    // An untrained question has no local decision and is not an error.
    let b = body(json!(rej), json!({"task": choice(&["x", "y"])}), None);
    let req = cortiq_decision::protocol::parse_request(&b, &svc.limits()).unwrap();
    let lo = svc.decide_local(&req).unwrap();
    assert!(lo.questions[0].1.is_none());
    assert_eq!(mock.calls(), calls0, "decide_local never escalates");
    assert_eq!(mock.observed.load(Ordering::SeqCst), observed0);
    assert_eq!(svc.totals(&Principal::open().account).decisions, 0);
    assert_eq!(svc.inflight(), 0);
}

#[test]
fn abstentions_escalate_only_with_consent() {
    let mock = Mock::new(Mode::Answer);
    let svc = service_with(oracle_cfg(), Some(mock.clone()));
    let (_, rej) = accepted_and_rejected();
    let calls0 = mock.calls();
    let q = json!({"task": choice(&TOPICS)});
    // Consent (default_per_request true): the oracle answers.
    let d = run(&svc, &body(json!(rej), q.clone(), None)).unwrap();
    assert_eq!(mock.calls(), calls0 + 1);
    let o = &d.questions[0];
    assert_eq!(o.action, Action::Oracle);
    assert_eq!(
        d.response["answers"]["task"],
        json!({"type": "choice", "choice": "travel"})
    );
    let cq = &d.response["cmf"]["questions"]["task"];
    assert_eq!(cq["source"], "oracle");
    assert_eq!(cq["decision_path"], "escalate→oracle");
    assert_eq!(cq["confident"], true);
    assert_eq!(
        cq["certified"], false,
        "an oracle answer is never certified"
    );
    assert!(cq["gate"]["accepted"] == json!(false));
    let seen = mock.seen.lock().unwrap().last().unwrap().clone();
    assert_eq!(seen[0].1, MatchKind::Exact);
    assert!(seen[0].4, "a trained question carries its local decision");
    // Passthrough: the oracle's cost is the client's (prices are 0).
    assert_eq!(d.response["usage"]["cost"], json!(0.00002));
    assert_eq!(d.response["cmf"]["usage"]["oracle"]["calls"], 1);
    assert_eq!(d.response["cmf"]["usage"]["oracle"]["input_tokens"], 1234);
    assert_eq!(d.response["cmf"]["usage"]["local"]["cost"], json!(0.0));
    assert_eq!(d.record.cost_oracle_usd, 0.00002);
    assert_eq!(d.record.oracle_calls, 1);
    // cmf.oracle false: no call, abstain with consent_off.
    let d = run(
        &svc,
        &body(json!(rej), q.clone(), Some(json!({"oracle": false}))),
    )
    .unwrap();
    assert_eq!(mock.calls(), calls0 + 1);
    assert_eq!(d.questions[0].action, Action::Abstain);
    assert_eq!(d.questions[0].flags, ["consent_off"]);
    assert_eq!(
        d.response["cmf"]["questions"]["task"]["decision_path"],
        "escalate→disabled"
    );
    assert!(d.response["answers"]["task"]["probabilities"].is_object());
    // A key without oracle_allowed: no call.
    let mut p = Principal::open();
    p.oracle_allowed = false;
    let d = svc
        .decide_body(&body(json!(rej), q.clone(), None), &p)
        .unwrap();
    assert_eq!(d.questions[0].flags, ["consent_off"]);
    assert_eq!(mock.calls(), calls0 + 1);
    // oracle.enabled false: no call.
    let svc_off = service_with(Config::default(), Some(mock.clone()));
    let d = run(
        &svc_off,
        &body(json!(rej), q.clone(), Some(json!({"oracle": true}))),
    )
    .unwrap();
    assert_eq!(d.questions[0].flags, ["oracle_disabled"]);
    assert_eq!(mock.calls(), calls0 + 1);
    // default_per_request false and no cmf.oracle: no call.
    let mut cfg = oracle_cfg();
    cfg.oracle.default_per_request = false;
    let svc_opt_in = service_with(cfg, Some(mock.clone()));
    let d = run(&svc_opt_in, &body(json!(rej), q.clone(), None)).unwrap();
    assert_eq!(d.questions[0].flags, ["consent_off"]);
    assert_eq!(mock.calls(), calls0 + 1);
    let d = run(
        &svc_opt_in,
        &body(json!(rej), q, Some(json!({"oracle": true}))),
    )
    .unwrap();
    assert_eq!(d.questions[0].action, Action::Oracle);
    assert_eq!(mock.calls(), calls0 + 2);
}

#[test]
fn refusals_failures_and_cache() {
    let mock = Mock::new(Mode::Cache);
    let svc = service_with(oracle_cfg(), Some(mock.clone()));
    let (_, rej) = accepted_and_rejected();
    let trained = json!({"task": choice(&TOPICS)});
    let untrained = json!({"task": choice(&["x", "y"])});
    // Cache: action cache, no oracle cost.
    let d = run(&svc, &body(json!(rej), trained.clone(), None)).unwrap();
    assert_eq!(d.questions[0].action, Action::Cache);
    assert_eq!(
        d.response["cmf"]["questions"]["task"]["decision_path"],
        "escalate→cache"
    );
    assert_eq!(d.response["usage"]["cost"], json!(0.0));
    assert_eq!(d.record.cache_hits, 1);
    let d = run(&svc, &body(json!(rej), untrained.clone(), None)).unwrap();
    assert_eq!(d.questions[0].action, Action::Cache);
    let table = [
        (
            Mode::Refuse(RefusalReason::Budget),
            503,
            Reason::OracleBudgetExhausted,
            "budget",
        ),
        (
            Mode::Refuse(RefusalReason::Stopped),
            503,
            Reason::OracleDisabled,
            "stopped",
        ),
        (
            Mode::Refuse(RefusalReason::OracleDisabled),
            422,
            Reason::UnsupportedQuestion,
            "oracle_disabled",
        ),
        (
            Mode::Fail,
            502,
            Reason::OracleUnavailable,
            "oracle_unavailable",
        ),
        (
            Mode::WrongOption,
            502,
            Reason::OracleUnavailable,
            "oracle_unavailable",
        ),
    ];
    for (mode, status, reason, flag) in table {
        *mock.mode.lock().unwrap() = mode;
        let e = run(&svc, &body(json!(rej), untrained.clone(), None)).unwrap_err();
        assert_eq!((e.status, e.reason), (status, reason), "{mode:?}");
        // A trained question never errors: it abstains with the flag.
        let d = run(&svc, &body(json!(rej), trained.clone(), None)).unwrap();
        assert_eq!(d.questions[0].action, Action::Abstain, "{mode:?}");
        assert_eq!(d.questions[0].flags, [flag], "{mode:?}");
        assert_eq!(
            d.response["usage"]["cost"],
            json!(0.0),
            "{mode:?}: failed calls are not billed"
        );
    }
    // Superset questions reach the escalator with their skill and unknown options.
    *mock.mode.lock().unwrap() = Mode::Answer;
    let mut sup = TOPICS.to_vec();
    sup.push("zz_new");
    let d = run(&svc, &body(json!(rej), json!({"task": choice(&sup)}), None)).unwrap();
    assert_eq!(d.questions[0].action, Action::Oracle);
    let seen = mock.seen.lock().unwrap().last().unwrap().clone();
    assert_eq!(seen[0].1, MatchKind::Superset);
    assert_eq!(seen[0].2.as_deref(), Some("topics"));
    assert_eq!(seen[0].3, ["zz_new"]);
    assert!(!seen[0].4);
    // Feedback and admin go to the escalator; without one they are 404.
    let fb = br#"{"id":"cmf-dec-1-x","question":"task","label":"cards"}"#;
    assert_eq!(
        svc.feedback(fb, &Principal::open()).unwrap()["accepted"],
        true
    );
    assert_eq!(
        service()
            .feedback(fb, &Principal::open())
            .unwrap_err()
            .status,
        404
    );
    assert!(svc.admin(&AdminCommand::Generations).is_ok());
    assert_eq!(
        service().admin(&AdminCommand::Learning).unwrap_err().status,
        404
    );
}

#[test]
fn profiles_change_the_gate() {
    let svc = service();
    let mut n_bal = 0;
    for (t, _) in &toy().dev {
        let run_p = |p: &str| {
            run(
                &svc,
                &body(
                    json!(t),
                    json!({"task": choice(&TOPICS)}),
                    Some(json!({"profile": p})),
                ),
            )
            .unwrap()
        };
        let bal = run_p("balanced");
        let qf = run_p("quality-first");
        let cs = run_p("cost-saver");
        let (b, q, c) = (
            bal.questions[0].local.as_ref().unwrap(),
            qf.questions[0].local.as_ref().unwrap(),
            cs.questions[0].local.as_ref().unwrap(),
        );
        // Same decision, different gates.
        assert_eq!(b.decision, q.decision);
        assert!(
            !q.accepted || b.accepted,
            "quality-first accepts a subset of balanced"
        );
        if q.accepted {
            assert!(
                q.decision.margin >= 0.08 && q.decision.novelty <= q.gate.novelty_theta.min(0.5)
            );
        }
        assert_eq!(c.accepted, !c.decision.is_novel(c.gate.novelty_theta));
        assert!(!c.certified && !cs.questions[0].certified);
        assert_eq!(b.certified, q.certified);
        n_bal += usize::from(b.accepted);
    }
    assert!(n_bal > 0);
}

#[test]
fn router_extras_follow_the_router_formulas() {
    let mut cfg = Config::default();
    cfg.task_complexity.insert("cards".into(), 0.9);
    cfg.routing_tiers
        .insert("low".into(), "my-small-model".into());
    cfg.routing_tiers
        .insert("medium".into(), "my-mid-model".into());
    cfg.routing_tiers
        .insert("high".into(), "my-big-model".into());
    let svc = service_with(cfg.clone(), None);
    let text = &toy().dev[2].0;
    let d = run(
        &svc,
        &body(
            json!(text),
            json!({"task": choice(&TOPICS)}),
            Some(json!({"explain": true})),
        ),
    )
    .unwrap();
    let o = &d.questions[0];
    let l = o.local.as_ref().unwrap();
    let q = &d.response["cmf"]["questions"]["task"];
    let words = text.split_whitespace().count();
    let base = cfg
        .task_complexity
        .get(l.choice.as_deref().unwrap())
        .copied()
        .unwrap_or(0.4);
    let w = cfg.complexity_weights;
    let amb = 1.0 - l.decision.p_top;
    let mrg = 1.0 - (l.decision.margin * 8.0).clamp(0.0, 1.0);
    let len = (words as f32 / 40.0).clamp(0.0, 1.0);
    let score = (w.base * base
        + w.ambiguity * amb
        + w.novelty * l.decision.novelty
        + w.margin * mrg
        + w.length * len)
        .clamp(0.0, 1.0);
    assert_eq!(
        q["complexity"]["score"],
        cortiq_decision::eval::f32_json(score)
    );
    let tier = if score <= 0.33 {
        "low"
    } else if score <= 0.66 {
        "medium"
    } else {
        "high"
    };
    assert_eq!(q["complexity"]["tier"], tier);
    assert_eq!(q["routing"]["target"], cfg.routing_tiers[tier].as_str());
    // explain: all errors, the top-1/top-2 line and the path.
    assert_eq!(q["errors"].as_object().unwrap().len(), 4);
    let ex = q["explanation"]["top1_vs_top2"].as_str().unwrap();
    assert!(ex.contains(" leads ") && ex.ends_with(" score"), "{ex}");
    assert_eq!(q["explanation"]["decision_path"], q["decision_path"]);
    assert_eq!(q["confident"], json!(o.action == Action::Local));
    // Without explain: the five smallest errors (4 labels here) and no explanation.
    let d = run(
        &svc,
        &body(json!(text), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    assert!(
        d.response["cmf"]["questions"]["task"]
            .get("explanation")
            .is_none()
    );
    // No routing without routing_tiers.
    let d = run(
        &service(),
        &body(json!(text), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    assert!(
        d.response["cmf"]["questions"]["task"]
            .get("routing")
            .is_none()
    );
}

// ------------------------------------------------------------------ metering

#[test]
fn metering_counts_the_whole_request_like_jev() {
    let svc = service();
    let model = svc.handle().current();
    let tok = model.encoder().encoder().tokenizer();
    let t = |s: &str| tok.encode_pieces(s).len() as u64;
    let text = toy().dev[3].0.clone();
    let mut crit = Map::new();
    crit.insert("Weather".into(), json!("rain snow forecast"));
    crit.insert(
        "billing".into(),
        json!({"what": "invoice charge", "examples": ["refund please"]}),
    );
    crit.insert("cards".into(), Value::Null);
    crit.insert("travel".into(), json!(["flight", "hotel booking"]));
    let instructions = json!({"task": "Which topic is it?", "note": "one only"});
    let q = json!({"type": "choice", "instructions": instructions, "criteria": crit});
    let d = run(&svc, &body(json!(text), json!({"task": q}), None)).unwrap();
    // The state and the instructions as their text (canonical JSON for an
    // object), criteria keys as text and criteria values as canonical JSON (a
    // string with its quotes, null as `null`).
    let expected = t(&text)
        + t(&canonical::to_string(&instructions))
        + t("Weather")
        + t(r#""rain snow forecast""#)
        + t("billing")
        + t(&canonical::to_string(&crit["billing"]))
        + t("cards")
        + t("null")
        + t("travel")
        + t(&canonical::to_string(&crit["travel"]));
    assert!(t(r#""rain snow forecast""#) == t("rain snow forecast") + 2);
    assert_eq!(d.metered.input_tokens, expected);
    assert_eq!(d.response["usage"]["input_tokens"], expected);
    assert_eq!(d.metered.output_tokens, 4, "one per probability");
    let max = tok.max_length() as u64;
    let processed = (t(&text) + 2).min(max);
    assert_eq!(d.metered.processed_tokens, processed);
    assert_eq!(
        d.response["cmf"]["usage"]["local"]["processed_tokens"],
        processed
    );
    // Prices default to 0.
    assert_eq!(d.response["usage"]["cost"], json!(0.0));
    // A state longer than the encoder's window: input counts every token,
    // processed stops at max_length.
    let long: String = std::iter::repeat_n("rain card hotel invoice", 20)
        .collect::<Vec<_>>()
        .join(" ");
    assert!(t(&long) + 2 > max);
    let d = run(
        &svc,
        &body(json!(long), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    assert_eq!(d.metered.processed_tokens, max);
    let crit_tokens: u64 = TOPICS
        .iter()
        .map(|l| t(l) + t(&format!("\"about {l}\"")))
        .sum();
    assert_eq!(
        d.metered.input_tokens,
        t(&long) + t("Which topic?") + crit_tokens
    );
    // Two questions: both contracts count, the state once.
    let d2 = run(
        &svc,
        &body(
            json!(text),
            json!({"a": choice(&TOPICS), "b": choice(&TOPICS)}),
            None,
        ),
    )
    .unwrap();
    assert_eq!(
        d2.metered.input_tokens,
        t(&text) + 2 * (t("Which topic?") + crit_tokens)
    );
    assert_eq!(d2.metered.output_tokens, 8);
    // Prices: exact decimal arithmetic.
    let mut cfg = Config::default();
    cfg.pricing.input_usd_per_1m = "0.042".into();
    cfg.pricing.output_usd_per_1m = "0.5".into();
    cfg.pricing.request_usd = "0.0001".into();
    let svc = service_with(cfg, None);
    let d = run(
        &svc,
        &body(json!(text), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    let want = Usd::parse("0.042")
        .unwrap()
        .per_million(d.metered.input_tokens)
        .unwrap()
        .checked_add(Usd::parse("0.5").unwrap().per_million(4).unwrap())
        .unwrap()
        .checked_add(Usd::parse("0.0001").unwrap())
        .unwrap();
    assert_eq!(d.metered.cost.total, want);
    assert_eq!(d.response["usage"]["cost"], json!(want.to_f64()));
    // /v1/models prices per token as strings.
    let m = svc.models_json();
    let p = &m["data"][0]["pricing"];
    assert_eq!(p["prompt"], "0.000000042");
    assert_eq!(p["completion"], "0.0000005");
    assert_eq!(p["request"], "0.0001");
    assert_eq!(p["image"], "0");
    assert_eq!(
        metering::answer_output_tokens(&json!({"type": "noul", "noul": 1})),
        1
    );
}

#[test]
fn listings_and_usage() {
    let svc = service();
    let m = svc.models_json();
    let e = &m["data"][0];
    assert_eq!(e["id"], "cortiq/decision");
    assert_eq!(e["output_modalities"], json!(["decisions"]));
    assert_eq!(
        e["pricing"],
        json!({"prompt": "0", "completion": "0", "request": "0", "image": "0"})
    );
    let skills = e["cmf"]["skills"].as_array().unwrap();
    assert_eq!(skills.len(), 2);
    assert_eq!(skills[0]["id"], "topics");
    assert_eq!(skills[0]["labels"], 4);
    let s = svc.skills_json();
    assert_eq!(
        s["skills"][1]["labels"],
        json!(["billing", "cards", "food"])
    );
    let one = svc.skill_json("shop").unwrap();
    assert_eq!(
        one["rubric"]["criteria"]["food"],
        "The message is about food."
    );
    assert_eq!(svc.skill_json("nope").unwrap_err().status, 404);
    assert_eq!(svc.healthz_json()["status"], "ok");
    // Usage: one account only.
    let mut a = Principal::open();
    a.account = "acct_a".into();
    let mut b = Principal::open();
    b.account = "acct_b".into();
    for _ in 0..2 {
        svc.decide_body(
            &body(json!("rain today"), json!({"task": choice(&TOPICS)}), None),
            &a,
        )
        .unwrap();
    }
    svc.decide_body(
        &body(
            json!("rain today"),
            json!({"x": choice(&TOPICS), "y": choice(&SHOP)}),
            None,
        ),
        &b,
    )
    .unwrap();
    let ua = svc.usage_json(&a);
    assert_eq!(ua["account"], "acct_a");
    assert_eq!(ua["usage"]["requests"], 2);
    assert_eq!(ua["usage"]["decisions"], 2);
    assert_eq!(svc.usage_json(&b)["usage"]["decisions"], 2);
    assert_eq!(svc.admin_usage()["accounts"].as_object().unwrap().len(), 2);
}

// ------------------------------------------------------------------ keys, limits, quotas

#[test]
fn keys_auth_rate_quotas_and_admin() {
    let dir = tempfile::tempdir().unwrap();
    let state = StateDir::open(dir.path().join("state")).unwrap();
    let store = Arc::new(KeyStore::open(state.keys_path(), "cortiq_").unwrap());
    let plans = Config::default().auth.plans;
    let now = keys::now_unix();
    let dev = store
        .create(
            &NewKey {
                plan: Some("developer".into()),
                account: Some("acme".into()),
                label: Some("ci".into()),
                ..Default::default()
            },
            &plans,
            now,
        )
        .unwrap();
    assert_eq!(dev.record.rate_per_min, 120);
    assert_eq!(dev.record.decision_quota, 100_000);
    assert_eq!(dev.record.expires, None);
    assert!(!dev.record.oracle_allowed);
    let starter = store.create(&NewKey::default(), &plans, now).unwrap();
    assert_eq!(starter.record.plan, "starter");
    assert_eq!(starter.record.expires, Some(now + 30 * 86_400));
    assert!(starter.record.account.starts_with("acct_"));
    let limited = store
        .create(
            &NewKey {
                account: Some("tiny".into()),
                rate_per_min: Some(2),
                decision_quota: Some(2),
                ..Default::default()
            },
            &plans,
            now,
        )
        .unwrap();
    // keys.json holds hashes only.
    let text = std::fs::read_to_string(state.keys_path()).unwrap();
    for k in [&dev, &starter, &limited] {
        assert!(k.raw.starts_with("cortiq_") && k.raw.len() == 47);
        assert!(!text.contains(&k.raw), "raw key in keys.json");
        assert!(!text.contains(&k.raw[7..]), "raw key hex in keys.json");
        assert!(text.contains(&keys::hash_key(&k.raw)));
    }
    assert!(
        !format!("{dev:?}").contains(&dev.raw),
        "Debug never prints the key"
    );
    // Router import: a config key (raw, hashed on import) and an expired MySQL row.
    let config = keys::read_router_keys(
        b"[[api_keys]]\nkey = \"router-live-key\"\naccount = \"legacy\"\nrate_per_min = 60\n",
        keys::ImportFormat::RouterToml,
        now,
    )
    .unwrap();
    let rows = keys::read_router_keys(
        format!(
            r#"[{{"key_hash":"{}","account":"old","plan":"pro","active":"1","rate_per_min":"600","decision_quota":"1000000","expires_at":1000,"created_at":"5"}}]"#,
            keys::hash_key("router-old-key")
        )
        .as_bytes(),
        keys::ImportFormat::MysqlJson,
        now,
    )
    .unwrap();
    let imported = store.import_router_keys(&config, now).unwrap();
    assert_eq!((imported.imported, imported.unchanged), (1, 0));
    let imported = store.import_router_keys(&rows, now).unwrap();
    assert_eq!((imported.imported, imported.imported_expired), (1, 1));
    let again = store.import_router_keys(&config, now).unwrap();
    assert_eq!(
        (again.imported, again.unchanged, again.written),
        (0, 1, false)
    );

    let svc = service_with(Config::default(), None)
        .with_keys(store.clone())
        .with_loopback(false);
    // 401: missing, invalid, expired, revoked.
    let e = svc.authenticate(None, None).unwrap_err();
    assert_eq!((e.status, e.reason), (401, Reason::Unauthorized));
    let e = svc
        .authenticate(Some("Bearer cortiq_nope"), None)
        .unwrap_err();
    assert_eq!(e.status, 401);
    let e = svc.authenticate(None, Some("router-old-key")).unwrap_err();
    assert!(e.message.contains("expired"), "{}", e.message);
    // Bearer, bare Authorization and x-api-key are accepted.
    let p = svc
        .authenticate(Some(&format!("Bearer {}", dev.raw)), None)
        .unwrap();
    assert_eq!((p.account.as_str(), p.plan.as_str()), ("acme", "developer"));
    assert_eq!(p.key12.as_deref(), Some(&dev.record.hash[..12]));
    assert_eq!(
        svc.authenticate(Some(&dev.raw), None).unwrap().account,
        "acme"
    );
    assert_eq!(
        svc.authenticate(None, Some(&dev.raw)).unwrap().account,
        "acme"
    );
    assert_eq!(
        svc.authenticate(None, Some("router-live-key"))
            .unwrap()
            .account,
        "legacy"
    );
    assert_eq!(store.revoke_account("legacy").unwrap(), 1);
    let e = svc.authenticate(None, Some("router-live-key")).unwrap_err();
    assert!(e.message.contains("revoked"), "{}", e.message);
    // Another writer (the CLI) adds a key: a reload picks it up.
    let cli = KeyStore::open(state.keys_path(), "cortiq_").unwrap();
    let late = cli.create(&NewKey::default(), &plans, now).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    store.reload().unwrap();
    assert!(svc.authenticate(None, Some(&late.raw)).is_ok());
    // Rate window: 2 per minute.
    let tiny = svc.authenticate(None, Some(&limited.raw)).unwrap();
    let t0 = 1_790_000_040u64;
    svc.admit_at(&tiny, t0).unwrap();
    svc.admit_at(&tiny, t0 + 1).unwrap();
    let e = svc.admit_at(&tiny, t0 + 2).unwrap_err();
    assert_eq!((e.status, e.reason), (429, Reason::RateLimited));
    assert_eq!(e.retry_after, Some(60 - (t0 + 2) % 60));
    svc.admit_at(&tiny, t0 + 60).unwrap();
    // Decision quota 2: two answered questions, then 402.
    svc.decide_body(
        &body(
            json!("rain"),
            json!({"a": choice(&TOPICS), "b": choice(&SHOP)}),
            None,
        ),
        &tiny,
    )
    .unwrap();
    let e = svc.admit_at(&tiny, t0 + 120).unwrap_err();
    assert_eq!((e.status, e.reason), (402, Reason::QuotaExceeded));
    assert_eq!(e.details.as_deref().unwrap()["quota"], "decision");
    // Token quota and credit.
    let mut p2 = Principal::open();
    p2.account = "tok".into();
    p2.token_quota = 5;
    svc.admit_at(&p2, t0).unwrap();
    svc.decide_body(
        &body(json!("rain"), json!({"a": choice(&TOPICS)}), None),
        &p2,
    )
    .unwrap();
    assert_eq!(
        svc.admit_at(&p2, t0)
            .unwrap_err()
            .details
            .as_deref()
            .unwrap()["quota"],
        "token"
    );
    let mut cfg = Config::default();
    cfg.pricing.request_usd = "0.001".into();
    let paid = service_with(cfg, None);
    let mut p3 = Principal::open();
    p3.account = "credit".into();
    p3.credit_usd = Some(Usd::parse("0.002").unwrap());
    for _ in 0..2 {
        paid.admit_at(&p3, t0).unwrap();
        paid.decide_body(
            &body(json!("rain"), json!({"a": choice(&TOPICS)}), None),
            &p3,
        )
        .unwrap();
    }
    let e = paid.admit_at(&p3, t0).unwrap_err();
    assert_eq!(e.details.as_deref().unwrap()["quota"], "credit");
    // In flight.
    let mut cfg = Config::default();
    cfg.limits.max_inflight = 1;
    let busy = service_with(cfg, None);
    let g = busy.enter().unwrap();
    let e = busy.enter().unwrap_err();
    assert_eq!(
        (e.status, e.reason, e.retry_after),
        (429, Reason::Overloaded, Some(1))
    );
    drop(g);
    assert!(busy.enter().is_ok());
    // Open mode only without keys and without require.
    let open = service_with(Config::default(), None).with_loopback(true);
    assert!(open.authenticate(None, None).unwrap().is_open());
    let closed = service_with(Config::default(), None).with_loopback(false);
    assert_eq!(closed.authenticate(None, None).unwrap_err().status, 401);
    let keyed = service_with(Config::default(), None)
        .with_keys(store.clone())
        .with_loopback(true);
    assert_eq!(
        keyed.authenticate(None, None).unwrap_err().status,
        401,
        "keys close the open mode"
    );
    // Admin guard and key admin.
    assert_eq!(
        svc.admin_authorize(Some("x")).unwrap_err().reason,
        Reason::AdminDisabled
    );
    let admin = service_with(Config::default(), None)
        .with_keys(store.clone())
        .with_admin_token(Some("s3cret".into()));
    assert_eq!(
        admin.admin_authorize(Some("s3cre")).unwrap_err().status,
        401
    );
    assert_eq!(admin.admin_authorize(None).unwrap_err().status, 401);
    admin.admin_authorize(Some("s3cret")).unwrap();
    let created = admin
        .admin_create_key(
            br#"{"plan":"pro","account":"bigco","oracle_allowed":true,"credit_usd":"5"}"#,
        )
        .unwrap();
    let raw = created["key"].as_str().unwrap().to_string();
    assert_eq!(created["rate_per_min"], 600);
    assert_eq!(created["oracle_allowed"], true);
    let pb = admin.authenticate(None, Some(&raw)).unwrap();
    assert!(pb.oracle_allowed && pb.credit_usd == Some(Usd::parse("5").unwrap()));
    let listing = admin.admin_list_keys().unwrap();
    let listed = listing["keys"].as_array().unwrap();
    assert!(
        listed
            .iter()
            .all(|k| k.get("key").is_none() && k["hash12"].as_str().unwrap().len() == 12)
    );
    assert!(!listing.to_string().contains(&raw));
    assert_eq!(
        admin
            .admin_create_key(br#"{"plan":"gold"}"#)
            .unwrap_err()
            .status,
        400
    );
    assert_eq!(
        admin
            .admin_create_key(br#"{"bogus":1}"#)
            .unwrap_err()
            .status,
        400
    );
    let h12 = &keys::hash_key(&raw)[..12];
    assert_eq!(admin.admin_revoke_hash(h12).unwrap()["revoked"], 1);
    assert_eq!(
        admin.authenticate(None, Some(&raw)).unwrap_err().status,
        401
    );
    assert_eq!(admin.admin_revoke_account("acme").unwrap()["revoked"], 1);
    assert!(
        !std::fs::read_to_string(state.keys_path())
            .unwrap()
            .contains(&raw)
    );
}

// ------------------------------------------------------------------ usage ledger

fn rec(ts: u64, account: &str, n: u64, cost: f64) -> UsageRecord {
    UsageRecord {
        ts,
        id: format!("cmf-dec-{ts}-{n}"),
        account: account.into(),
        key12: Some("0123456789ab".into()),
        model: "cortiq/decision@0123456789ab".into(),
        generation: 0,
        input_tokens: 100 + n,
        output_tokens: 4,
        cost_usd: cost,
        cost_local_usd: cost,
        cost_oracle_usd: 0.0,
        oracle_calls: n % 2,
        cache_hits: 0,
        questions: 1,
        actions: Actions {
            local: 1,
            ..Default::default()
        },
    }
}

#[test]
fn usage_ledger_append_replay_snapshot_are_exact() {
    let dir = tempfile::tempdir().unwrap();
    let usage = dir.path().join("usage");
    // Costs whose f64 sum would drift: 0.1 + 0.2 ≠ 0.3 in binary.
    let costs = [0.1, 0.2, 0.000183624, 1.7472e-5, 0.3, 2.5e-9, 0.1];
    let months = [1_788_000_000u64, 1_790_500_000, 1_793_000_000];
    let live = {
        let l = UsageLedger::open(&usage).unwrap();
        for (i, c) in costs.iter().enumerate() {
            let ts = months[i * months.len() / costs.len()];
            l.append(&rec(ts, if i % 3 == 0 { "a" } else { "b" }, i as u64, *c))
                .unwrap();
        }
        assert_eq!(l.pending(), costs.len());
        l.flush().unwrap();
        assert_eq!(l.pending(), 0);
        l.all_totals()
    };
    // Exact decimal sums: 0.1 + 0.000017472 + 0.1 and 0.2 + 0.000183624 + 0.3 + 0.0000000025.
    assert_eq!(live["a"].requests, 3);
    assert_eq!(live["a"].cost_usd.to_string(), "0.200017472");
    assert_eq!(live["b"].cost_usd.to_string(), "0.5001836265");
    assert_eq!(live["b"].decisions, 4);
    // Month files by UTC month of ts.
    let mut files: Vec<String> = std::fs::read_dir(&usage)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    files.sort();
    let want: Vec<String> = months
        .iter()
        .map(|&t| format!("{}.jsonl", month_of(t)))
        .collect();
    assert_eq!(files, want);
    // Every line is a record in the spec's field order, without text.
    let line = std::fs::read_to_string(usage.join(&files[0])).unwrap();
    let first = line.lines().next().unwrap();
    assert!(first.starts_with(r#"{"ts":"#), "{first}");
    let keys: Vec<String> = canonical::parse(first.as_bytes())
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(
        keys,
        [
            "ts",
            "id",
            "account",
            "key12",
            "model",
            "generation",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "cost_local_usd",
            "cost_oracle_usd",
            "oracle_calls",
            "cache_hits",
            "questions",
            "actions"
        ]
    );
    // Replay from scratch.
    let l = UsageLedger::open(&usage).unwrap();
    assert_eq!(l.all_totals(), live);
    assert_eq!(l.replayed(), costs.len() as u64);
    // Snapshot, then a tail.
    l.snapshot().unwrap();
    l.append(&rec(months[2], "a", 99, 0.2)).unwrap();
    l.flush().unwrap();
    let after = l.all_totals();
    drop(l);
    let l = UsageLedger::open(&usage).unwrap();
    assert_eq!(
        l.replayed(),
        1,
        "only the tail after the snapshot is replayed"
    );
    assert_eq!(l.all_totals(), after);
    // The same totals without the snapshot.
    std::fs::remove_file(usage.join("totals.json")).unwrap();
    let l2 = UsageLedger::open(&usage).unwrap();
    assert_eq!(l2.replayed(), costs.len() as u64 + 1);
    assert_eq!(l2.all_totals(), after);
    drop((l, l2));
    // A crash in the middle of a line: the partial line is cut, totals intact.
    let last = usage.join(files.last().unwrap());
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(&last)
        .unwrap();
    use std::io::Write;
    f.write_all(br#"{"ts":1793000000,"id":"cmf-dec-part"#)
        .unwrap();
    drop(f);
    let l3 = UsageLedger::open(&usage).unwrap();
    assert_eq!(l3.all_totals(), after);
    assert!(
        l3.warnings().iter().any(|w| w.contains("partial")),
        "{:?}",
        l3.warnings()
    );
    l3.append(&rec(months[2], "b", 7, 0.1)).unwrap();
    l3.flush().unwrap();
    drop(l3);
    let l4 = UsageLedger::open(&usage).unwrap();
    assert_eq!(l4.all_totals()["b"].requests, after["b"].requests + 1);
    drop(l4);
    // A corrupt complete line refuses the open (billing data is never skipped).
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(&last)
        .unwrap();
    f.write_all(b"{\"not\":\"a record\"}\n").unwrap();
    drop(f);
    std::fs::remove_file(usage.join("totals.json")).ok();
    assert!(UsageLedger::open(&usage).is_err());
    // A negative cost is refused at append.
    let l5 = UsageLedger::open(dir.path().join("u2")).unwrap();
    assert!(l5.append(&rec(months[0], "a", 1, -1.0)).is_err());
}

#[test]
fn background_flusher_writes_within_a_second_and_on_stop() {
    let dir = tempfile::tempdir().unwrap();
    let l = Arc::new(UsageLedger::open(dir.path()).unwrap());
    let f = l.start_flusher(std::time::Duration::from_millis(50));
    l.append(&rec(1_790_500_000, "a", 1, 0.5)).unwrap();
    let t = std::time::Instant::now();
    while l.pending() > 0 {
        assert!(
            t.elapsed() < std::time::Duration::from_secs(5),
            "not flushed"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    l.append(&rec(1_790_500_001, "a", 2, 0.25)).unwrap();
    f.stop().unwrap();
    assert_eq!(l.pending(), 0);
    assert!(
        dir.path().join("totals.json").exists(),
        "a snapshot at stop"
    );
    let r = UsageLedger::open(dir.path()).unwrap();
    assert_eq!(r.totals("a").cost_usd.to_string(), "0.75");
}

#[test]
fn service_bills_successful_requests_only() {
    let dir = tempfile::tempdir().unwrap();
    let ledger = Arc::new(UsageLedger::open(dir.path()).unwrap());
    let mock = Mock::new(Mode::Answer);
    let svc = service_with(oracle_cfg(), Some(mock.clone())).with_ledger(ledger.clone());
    let (acc, rej) = accepted_and_rejected();
    let before = ledger.totals("anonymous");
    run(
        &svc,
        &body(json!(acc), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    run(
        &svc,
        &body(json!(rej), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    assert!(
        run(
            &svc,
            &body(
                json!(acc),
                json!({"task": choice(&["x", "y"])}),
                Some(json!({"oracle": false}))
            )
        )
        .is_err()
    );
    let t = ledger.totals("anonymous");
    assert_eq!(t.requests - before.requests, 2, "errors are not billed");
    assert_eq!(t.actions.local - before.actions.local, 1);
    assert_eq!(t.actions.oracle - before.actions.oracle, 1);
    assert_eq!(t.oracle_calls - before.oracle_calls, 1);
    ledger.close().unwrap();
    let replay = UsageLedger::open(dir.path()).unwrap();
    assert_eq!(replay.totals("anonymous"), t);
    let text: String = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| std::fs::read_to_string(e.unwrap().path()).unwrap_or_default())
        .collect();
    assert!(
        !text.contains(&acc) && !text.contains(&rej),
        "no text in the ledger"
    );
}

// ------------------------------------------------------------------ state dir

#[test]
fn state_lock_is_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let s = StateDir::open(dir.path().join("st")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(s.root()).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }
    assert!(s.usage_dir().is_dir() && s.generations_dir().is_dir());
    let lock = s.lock(false).unwrap();
    let e = s.lock(false).unwrap_err();
    let msg = format!("{e:#}");
    assert!(
        msg.contains(&std::process::id().to_string()) && msg.contains("--break-lock"),
        "{msg}"
    );
    // Another handle on the same directory is refused too.
    let s2 = StateDir::open(s.root()).unwrap();
    assert!(s2.lock(false).is_err());
    drop(lock);
    assert!(!s.lock_path().exists());
    let l2 = s2.lock(false).unwrap();
    // --break-lock takes it over; the old guard does not remove the new lock.
    let l3 = s.lock(true).unwrap();
    drop(l2);
    assert!(
        s.lock_path().exists(),
        "the broken guard must not remove the new lock"
    );
    assert!(s.lock(false).is_err());
    drop(l3);
    assert!(!s.lock_path().exists());
    // CURRENT and generations.
    assert_eq!(s.read_current().unwrap(), None);
    let c = Current {
        generation: 3,
        sha256: "ab".repeat(32),
    };
    s.write_current(&c).unwrap();
    assert_eq!(s.read_current().unwrap(), Some(c));
    assert_eq!(
        std::fs::read_to_string(s.current_path()).unwrap(),
        format!("g000003 {}\n", "ab".repeat(32))
    );
    assert_eq!(s.current_overlay().unwrap(), Some(s.generation_path(3)));
    std::fs::write(s.generation_path(3), b"x").unwrap();
    std::fs::write(s.generation_path(1), b"x").unwrap();
    std::fs::write(s.generations_dir().join("junk.cmf"), b"x").unwrap();
    let g: Vec<u64> = s
        .generations()
        .unwrap()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(g, [1, 3]);
    assert!(
        s.write_current(&Current {
            generation: 1,
            sha256: "XYZ".into()
        })
        .is_err()
    );
    std::fs::write(s.current_path(), "g3 zz").unwrap();
    assert!(s.read_current().is_err());
}

#[test]
fn promotion_swaps_the_model_for_new_requests() {
    let svc = service();
    let h = svc.handle().clone();
    let old = h.current();
    let derived = old
        .derive(DecisionModel::open(&toy().path, Verify::Light).unwrap())
        .unwrap();
    assert!(
        Arc::ptr_eq(derived.encoder(), old.encoder()),
        "a generation shares the encoder"
    );
    let prev = h.promote(derived);
    assert!(Arc::ptr_eq(&prev, &old));
    assert!(!Arc::ptr_eq(&h.current(), &old));
    let d = run(
        &svc,
        &body(json!("rain"), json!({"task": choice(&TOPICS)}), None),
    )
    .unwrap();
    assert_eq!(d.response["model"], json!(old.name()));
    assert_eq!(d.record.model, old.name());
}
