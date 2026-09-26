//! Self-learning, generations and offline pre-training on the toy files (spec
//! §5.7–§5.11, §5.14), hermetic (the oracle is an in-test mock server).
//!
//! * 25 oracle answers of a label promote a challenger (generation 1, written,
//!   opened with the overlay loader, made `CURRENT`); the next similar request is
//!   decided locally; the recertified gate equals a certification from scratch;
//!   the task is the fit of its train rows and its learned rows; every other task
//!   is untouched (0 isolation violations);
//! * a holdout regression (25 wrong feedback labels) refuses the challenger;
//! * a certified gate that loses its qualifying tau refuses the challenger (the
//!   `trap` file);
//! * cold start: a label only the oracle knows becomes a task after 25 answers,
//!   and its wins are not certified;
//! * a second label (a cold start) is promoted in the same process after a
//!   first promotion (refit) of the skill, on top of it (generation 2, parent 1);
//! * activating an inactive task that regresses the holdout is refused;
//! * rollback to the base and forward, restart restores the served generation,
//!   the cache and the buffer from `learn.log` (a cut tail is dropped); the
//!   offline rollback of the CLI; a promotion after a rollback;
//! * two synchronous runs write the same generation bytes; the background worker
//!   promotes off the request thread; feedback is scoped to its account;
//! * `cortiq decision learn` (library side): the oracle only for abstentions,
//!   answers reused from ledgers by body sha256, promotions with the holdout
//!   gate, the other skills byte for byte, learned rows in the rows blob, and
//!   every promotion undone when the certified gate is lost; `calls` counts
//!   the calls sent (refusals apart); live calls without a key are refused.

#[path = "fixtures/oracle/support.rs"]
mod support;

use cortiq_decision::build;
use cortiq_decision::container::{DecisionModel, Verify};
use cortiq_decision::eval::Evaluator;
use cortiq_decision::generation;
use cortiq_decision::learn::{self, OfflineOptions, SkillBook};
use cortiq_decision::manifest::{Gate, TaskOrigin, TaskState, sha256_hex};
use cortiq_decision::oracle;
use cortiq_decision::rows::{Row, Source, Split};
use cortiq_decision::service::{Action, AdminCommand, LoadedModel, Principal};
use cortiq_decision::statedir::StateDir;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use support::*;

/// 25 cruise texts, pairwise cos φ_P < 0.97 (no cache hit among them).
fn lesson() -> &'static [String] {
    static L: OnceLock<Vec<String>> = OnceLock::new();
    L.get_or_init(|| distinct_texts("cruise", 25, 7, "q", 0.97))
}

/// Cruise texts never taught.
fn fresh() -> &'static [String] {
    static F: OnceLock<Vec<String>> = OnceLock::new();
    F.get_or_init(|| distinct_texts("cruise", 8, 99, "z", 0.995))
}

fn served(st: &Stand) -> Arc<LoadedModel> {
    st.handle.current()
}

fn teach(st: &Stand, texts: &[String]) -> Vec<Action> {
    texts
        .iter()
        .map(|t| st.decide(&topics_body(t)).unwrap().questions[0].action)
        .collect()
}

type TaskRow = (String, u64, u64, Option<String>, Option<String>, TaskState);

/// The tasks of every skill: (skill, i, k, mean sha, basis sha, state).
fn task_table(m: &DecisionModel) -> Vec<TaskRow> {
    let mut v = Vec::new();
    for s in m.skills() {
        for t in &s.manifest.tasks {
            v.push((
                s.id().to_string(),
                t.i,
                t.k,
                t.mean_sha256.clone(),
                t.basis_sha256.clone(),
                t.state,
            ));
        }
    }
    v
}

fn learning(st: &Stand) -> Value {
    st.svc.admin(&AdminCommand::Learning).unwrap()
}

/// The topology of task `i` refitted from its stored rows (train, base learned,
/// `rows.learned`) equals the served one bit for bit.
fn assert_fit_reproduces(m: &DecisionModel, skill: &str, i: usize) {
    let base = m.rows(skill).unwrap();
    let learned = m.rows_learned(skill).unwrap();
    let mut rows: Vec<&Row> = base
        .rows
        .iter()
        .filter(|r| r.task as usize == i && r.split == Split::Train)
        .collect();
    rows.extend(
        base.rows
            .iter()
            .filter(|r| r.task as usize == i && r.split == Split::Learned),
    );
    if let Some(l) = &learned {
        rows.extend(l.rows.iter().filter(|r| r.task as usize == i));
    }
    let k = m.skill(skill).unwrap().manifest.recipe.k_max as usize;
    let f = build::fit_task_rows(&rows, base.dim_h, k).unwrap();
    let t = m.topology(skill, i).unwrap().unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&f.topology.mean),
        bits(&t.mean),
        "{skill} task {i} mean"
    );
    assert_eq!(
        bits(&f.topology.basis),
        bits(&t.basis),
        "{skill} task {i} basis"
    );
    let rec = &m.skill(skill).unwrap().manifest.tasks[i];
    assert_eq!(
        (rec.err_mean, rec.err_std, rec.n_train),
        (f.err_mean, f.err_std, rows.len() as u64)
    );
}

/// The served gate equals a certification computed from scratch (full scorer
/// errors of every calibration row): the reused columns are bit-exact.
fn assert_gate_from_scratch(m: &DecisionModel, skill: &str) {
    let st = learn::calib_from_model(m, skill, 2).unwrap();
    let book = SkillBook::from_model(m, skill, None).unwrap();
    let cert = book.certify(&st).unwrap();
    assert_eq!(
        Gate::from_certification(&cert),
        m.skill(skill).unwrap().manifest.gate
    );
}

// ------------------------------------------------------------------ promotion

#[test]
fn twenty_five_oracle_answers_promote_and_the_next_similar_request_is_local() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let before = DecisionModel::open(&toy().path, Verify::Full).unwrap();
    // Before: every fresh cruise text is rejected by the gate.
    let mut off = stand_config(&mock.url());
    off.oracle.enabled = false;
    let plain = Stand::new(&off);
    for t in fresh() {
        assert_eq!(
            plain.decide(&topics_body(t)).unwrap().questions[0].action,
            Action::Abstain
        );
    }

    let acts = teach(&st, lesson());
    assert!(acts.iter().all(|a| *a == Action::Oracle), "{acts:?}");
    assert_eq!(mock.hits(), 25);
    let l = learning(&st);
    assert_eq!(
        (l["attempts"].as_u64(), l["promotions"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(l["isolation_violations"], 0);
    let rec = &l["recent"][0];
    assert_eq!(
        (rec["outcome"].as_str(), rec["kind"].as_str()),
        (Some("promoted"), Some("promote"))
    );
    assert_eq!(rec["holdout"]["passed"], true);
    assert_eq!(rec["gate_after"]["certified"], true);
    assert_eq!(rec["n_fit"], 55);

    // Served generation 1, CURRENT, the overlay loader.
    let now = served(&st);
    assert_eq!(now.generation(), 1);
    let cur = st.state.read_current().unwrap().unwrap();
    assert_eq!(cur.generation, 1);
    let gpath = st.state.generation_path(1);
    assert_eq!(cur.sha256, generation::file_sha256(&gpath).unwrap());
    let reopened =
        DecisionModel::open_with_overlay(&toy().path, Some(&gpath), Verify::Full).unwrap();
    assert_eq!(reopened.model_sha(), now.model_sha());
    let om = reopened.overlay_manifest().unwrap();
    assert_eq!((om.parent, om.events.len()), (0, 1));
    assert_eq!(
        (om.events[0].label.as_str(), om.events[0].kind.as_str()),
        ("travel", "promote")
    );
    assert_eq!(om.skills.keys().collect::<Vec<_>>(), vec!["topics"]);

    // Isolation: only topics task 3 changed.
    let m = now.model();
    assert!(learn::isolation_violations(&before, m, "topics", 3).is_empty());
    let (a, b) = (task_table(&before), task_table(m));
    let changed: Vec<_> = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x != y)
        .map(|(x, _)| (x.0.clone(), x.1))
        .collect();
    assert_eq!(changed, vec![("topics".to_string(), 3)]);
    // The detector reports a change it is not told about.
    assert_eq!(
        learn::isolation_violations(&before, m, "topics", 0).len(),
        1
    );

    // Learned rows: the 25 answers, task 3, source oracle.
    let learned = m.rows_learned("topics").unwrap().unwrap();
    assert_eq!(learned.rows.len(), 25);
    assert!(learned.rows.iter().all(|r| r.task == 3
        && r.split == Split::Learned
        && r.source == Source::Oracle
        && r.weight == 1.0));
    assert_fit_reproduces(m, "topics", 3);
    assert_fit_reproduces(m, "topics", 0);
    assert_gate_from_scratch(m, "topics");
    assert_eq!(m.skill("topics").unwrap().manifest.taxonomy_version, 1);

    // The next similar requests are decided locally (no call).
    let hits = mock.hits();
    for t in fresh() {
        let d = st.decide(&topics_body(t)).unwrap();
        assert_eq!(d.questions[0].action, Action::Local, "{t}");
        assert_eq!(d.response["answers"]["task"]["choice"], "travel");
        assert_eq!(d.response["cmf"]["generation"], 1);
    }
    assert_eq!(mock.hits(), hits);
    // A request pinned to the old generation is refused.
    let mut v: Value = serde_json::from_slice(&topics_body(&fresh()[0])).unwrap();
    v["model"] = json!(format!("cortiq/decision@{}", &before.model_sha()[..12]));
    assert_eq!(
        st.decide(&serde_json::to_vec(&v).unwrap())
            .unwrap_err()
            .status,
        404
    );
    assert_eq!(st.cascade.label_counts("topics", "travel"), (25, 0));
}

#[test]
fn a_holdout_regression_refuses_the_challenger() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let bills = distinct_texts("billing", 40, 5, "b", 0.995);
    let sha0 = served(&st).model_sha().to_string();
    let mut ids = Vec::new();
    for t in &bills {
        let d = st.decide(&topics_body(t)).unwrap();
        if d.questions[0].action == Action::Local {
            ids.push(d.id.clone());
        }
    }
    assert!(ids.len() >= 25, "{} billing texts accepted", ids.len());
    let fb =
        |id: &str, label: &str| format!(r#"{{"id":"{id}","question":"task","label":"{label}"}}"#);
    // Feedback is scoped: another account does not find the decision; the
    // label must be an option; an unknown id is 404.
    let mut other = Principal::open();
    other.account = "someone-else".into();
    let open = Principal::open();
    assert_eq!(
        st.svc
            .feedback(fb(&ids[0], "cards").as_bytes(), &other)
            .unwrap_err()
            .status,
        404
    );
    assert_eq!(
        st.svc
            .feedback(fb(&ids[0], "food").as_bytes(), &open)
            .unwrap_err()
            .status,
        400
    );
    assert_eq!(
        st.svc
            .feedback(fb("cmf-dec-1-x", "cards").as_bytes(), &open)
            .unwrap_err()
            .status,
        404
    );
    let mut last = Value::Null;
    for (n, id) in ids.iter().take(25).enumerate() {
        let r = st.svc.feedback(fb(id, "cards").as_bytes(), &open).unwrap();
        assert_eq!(r["accepted"], true, "feedback {n}");
        assert_eq!(r["weight"], 3.0);
        assert_eq!(r["cold_start"], false);
        last = r;
    }
    // The decision is consumed by its feedback.
    assert_eq!(
        st.svc
            .feedback(fb(&ids[0], "cards").as_bytes(), &open)
            .unwrap_err()
            .status,
        404
    );
    let l = &last["learning"];
    assert_eq!(l["outcome"], "rejected");
    assert_eq!(l["reason"], "holdout_regression");
    let h = &l["holdout"];
    assert_eq!(h["passed"], false);
    assert!(
        h["challenger"]["macro"].as_f64().unwrap() + 1e-4
            < h["champion"]["macro"].as_f64().unwrap()
    );
    // The champion stays; nothing was written; the counter restarts, the
    // examples stay (weight 3, client feedback).
    assert_eq!(served(&st).model_sha(), sha0);
    assert!(st.state.generations().unwrap().is_empty());
    assert!(st.state.read_current().unwrap().is_none());
    assert_eq!(st.cascade.label_counts("topics", "cards"), (25, 0));
    assert_eq!(learning(&st)["rejections"], 1);
    assert_eq!(mock.hits(), 0);
}

#[test]
fn a_lost_certified_gate_refuses_the_challenger() {
    let trap_model = DecisionModel::open(&trap().path, Verify::Full).unwrap();
    assert!(trap_model.skill("trap").unwrap().manifest.gate.certified);
    let mock = MockOracle::answering("travel");
    let st = Stand::open_on(
        &trap().path,
        tempfile::tempdir().unwrap(),
        &stand_config(&mock.url()),
        test_key_lookup(),
    );
    let acts = teach(&st, lesson());
    assert!(acts.iter().all(|a| *a == Action::Oracle));
    let rec = &learning(&st)["recent"][0];
    assert_eq!(rec["outcome"], "rejected");
    assert_eq!(rec["reason"], "gate_lost");
    assert_eq!(
        rec["holdout"]["passed"], true,
        "the holdout alone would promote it"
    );
    assert_eq!(rec["gate_before"]["certified"], true);
    assert_eq!(rec["gate_after"]["certified"], false);
    assert!(st.state.generations().unwrap().is_empty());
    assert_eq!(served(&st).generation(), 0);
}

#[test]
fn cold_start_turns_an_oracle_label_into_a_task_after_25_answers() {
    let mock = MockOracle::answering("cruise");
    let st = Stand::new(&stand_config(&mock.url()));
    let before = DecisionModel::open(&toy().path, Verify::Full).unwrap();
    let mut l5 = TOPICS.to_vec();
    l5.push("cruise");
    let ask = |t: &str| body(json!(t), json!({"task": choice(&l5)}), None);
    for (n, t) in lesson().iter().enumerate() {
        let d = st.decide(&ask(t)).unwrap();
        assert_eq!(d.questions[0].action, Action::Oracle);
        assert_eq!(d.response["cmf"]["questions"]["task"]["match"], "superset");
        if n == 23 {
            let l = learning(&st);
            assert_eq!(
                l["quarantine"],
                json!([{"skill": "topics", "label": "cruise", "examples": 24, "new": 24}])
            );
            assert_eq!(l["promotions"], 0);
        }
    }
    let l = learning(&st);
    assert_eq!(
        (l["promotions"].as_u64(), l["cold_starts"].as_u64()),
        (Some(1), Some(1))
    );
    assert_eq!(l["quarantine"], json!([]));
    let now = served(&st);
    let m = now.model();
    let sk = &m.skill("topics").unwrap().manifest;
    assert_eq!(
        sk.labels,
        vec!["Weather", "billing", "cards", "travel", "cruise"]
    );
    let t = &sk.tasks[4];
    assert_eq!(
        (t.origin, t.state, t.n_train),
        (TaskOrigin::ColdStart, TaskState::Active, 25)
    );
    assert_eq!(sk.taxonomy_version, 2);
    assert!(learn::isolation_violations(&before, m, "topics", 4).is_empty());
    assert_fit_reproduces(m, "topics", 4);
    assert_gate_from_scratch(m, "topics");
    // Now an exact match: decided locally, never certified (origin cold_start).
    for t in fresh() {
        let d = st.decide(&ask(t)).unwrap();
        let q = &d.questions[0];
        assert_eq!(d.response["cmf"]["questions"]["task"]["match"], "exact");
        assert_eq!(q.action, Action::Local);
        assert_eq!(d.response["answers"]["task"]["choice"], "cruise");
        assert!(!q.certified);
    }
    // The four-label question is a subset now (not certified).
    let d = st.decide(&topics_body(&toy().dev[0].0)).unwrap();
    assert_eq!(d.response["cmf"]["questions"]["task"]["match"], "subset");
}

/// Two labels of one skill promoted one after the other in the same process
/// (no restart, no rollback in between): a refit of `travel`, then a cold start
/// of `cuisine` on top of it (generation 2, parent 1).
#[test]
fn a_second_label_is_promoted_in_the_same_process_after_a_first_promotion() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    let before = DecisionModel::open(&toy().path, Verify::Full).unwrap();
    teach(&st, lesson());
    let g1 = served(&st);
    assert_eq!(g1.generation(), 1);

    mock.set(|req| answer_reply(req, |_, opts| pick(opts, "cuisine"), 1.3e-5));
    let mut l5 = TOPICS.to_vec();
    l5.push("cuisine");
    let ask = |t: &str| body(json!(t), json!({"task": choice(&l5)}), None);
    let food = distinct_texts("food", 25, 17, "f", 0.97);
    for t in &food {
        let d = st.decide(&ask(t)).unwrap();
        assert_eq!(d.questions[0].action, Action::Oracle, "{t}");
    }
    assert_eq!(mock.hits(), 50);
    let l = learning(&st);
    assert_eq!(
        (
            l["attempts"].as_u64(),
            l["promotions"].as_u64(),
            l["cold_starts"].as_u64(),
            l["errors"].as_u64(),
        ),
        (Some(2), Some(2), Some(1), Some(0)),
        "{l}"
    );
    assert_eq!(l["isolation_violations"], 0);
    let rec = l["recent"].as_array().unwrap().last().unwrap();
    assert_eq!(
        (
            rec["outcome"].as_str(),
            rec["kind"].as_str(),
            rec["generation"].as_u64()
        ),
        (Some("promoted"), Some("cold_start"), Some(2)),
        "{rec}"
    );

    let now = served(&st);
    assert_eq!(now.generation(), 2);
    let m = now.model();
    let om = m.overlay_manifest().unwrap();
    assert_eq!((om.generation, om.parent), (2, 1));
    let events: Vec<(&str, &str)> = om
        .events
        .iter()
        .map(|e| (e.label.as_str(), e.kind.as_str()))
        .collect();
    assert_eq!(
        events,
        vec![("travel", "promote"), ("cuisine", "cold_start")]
    );
    let gpath = st.state.generation_path(2);
    let reopened =
        DecisionModel::open_with_overlay(&toy().path, Some(&gpath), Verify::Full).unwrap();
    assert_eq!(reopened.model_sha(), now.model_sha());

    // Generation 1 → 2 changed only the new task; the refit of generation 1 is
    // carried unchanged; base → 2 changed task 3 and added task 4.
    assert!(learn::isolation_violations(g1.model(), m, "topics", 4).is_empty());
    let (a, b) = (task_table(&before), task_table(m));
    let key = |r: &TaskRow| (r.0.clone(), r.1);
    let changed: Vec<_> = b.iter().filter(|y| !a.contains(y)).map(key).collect();
    assert_eq!(
        changed,
        vec![("topics".to_string(), 3), ("topics".to_string(), 4)]
    );
    assert_eq!(b.len(), a.len() + 1);
    let sk = &m.skill("topics").unwrap().manifest;
    assert_eq!(
        sk.labels,
        vec!["Weather", "billing", "cards", "travel", "cuisine"]
    );
    assert_eq!(
        (sk.tasks[4].origin, sk.tasks[4].state, sk.tasks[4].n_train),
        (TaskOrigin::ColdStart, TaskState::Active, 25)
    );
    let learned = m.rows_learned("topics").unwrap().unwrap();
    assert_eq!(
        learned.rows.iter().map(|r| r.task).collect::<Vec<_>>(),
        [vec![3u32; 25], vec![4u32; 25]].concat()
    );
    assert_fit_reproduces(m, "topics", 3);
    assert_fit_reproduces(m, "topics", 4);
    assert_gate_from_scratch(m, "topics");

    // The state on disk is the state in memory.
    let sha = now.model_sha().to_string();
    let st = st.restart(&stand_config(&mock.url()));
    assert_eq!(served(&st).model_sha(), sha);
}

/// Activating an inactive data task (a label the skill already has) is gated
/// on the holdout like a refit: `zc1` of the trap file taught with 25 travel
/// texts takes travel's holdout rows and is refused.
#[test]
fn an_activation_that_regresses_the_holdout_is_refused() {
    let mock = MockOracle::answering("zc1");
    let st = Stand::open_on(
        &trap().path,
        tempfile::tempdir().unwrap(),
        &stand_config(&mock.url()),
        test_key_lookup(),
    );
    let trap_model = DecisionModel::open(&trap().path, Verify::Full).unwrap();
    let sk = &trap_model.skill("trap").unwrap().manifest;
    assert_eq!(
        sk.tasks[sk.task_of("zc1").unwrap()].state,
        TaskState::Inactive
    );
    // The eight-label question is a superset (zc1..zc4 are not trained): every
    // text goes to the oracle, which answers zc1.
    let labels = [
        "Weather", "billing", "cards", "travel", "zc1", "zc2", "zc3", "zc4",
    ];
    let ask = |t: &str| body(json!(t), json!({"task": choice(&labels)}), None);
    for t in distinct_texts("travel", 25, 41, "v", 0.97) {
        let d = st.decide(&ask(&t)).unwrap();
        assert_eq!(d.questions[0].action, Action::Oracle, "{t}");
    }
    assert_eq!(mock.hits(), 25);
    let lj = learning(&st);
    assert_eq!(lj["attempts"], 1, "{lj}");
    let l = &lj["recent"][0];
    assert_eq!(l["kind"], "activate", "{l}");
    assert_eq!(l["outcome"], "rejected", "{l}");
    assert_eq!(l["reason"], "holdout_regression", "{l}");
    let h = &l["holdout"];
    assert_eq!(
        (h["gated"].as_bool(), h["passed"].as_bool()),
        (Some(true), Some(false))
    );
    assert!(
        h["challenger"]["macro"].as_f64().unwrap() + 1e-4
            < h["champion"]["macro"].as_f64().unwrap()
    );
    assert!(st.state.generations().unwrap().is_empty());
    assert_eq!(served(&st).generation(), 0);
    assert_eq!(lj["rejections"], 1);
}

// ------------------------------------------------------------------ generations

#[test]
fn rollback_and_restart_restore_the_served_state() {
    let mock = MockOracle::answering("travel");
    let cfg = stand_config(&mock.url());
    let st = Stand::new(&cfg);
    teach(&st, lesson());
    assert_eq!(served(&st).generation(), 1);
    let buffer = st.cascade.buffer_len();
    assert_eq!((st.cascade.cache_len(), buffer), (25, 25));

    let r = st
        .svc
        .admin(&AdminCommand::Rollback { generation: 0 })
        .unwrap();
    assert_eq!(r["generation"], 0);
    assert_eq!(served(&st).generation(), 0);
    assert_eq!(
        st.state.read_current().unwrap().unwrap().sha256,
        served(&st).model_sha()
    );
    let d = st.decide(&topics_body(&fresh()[0])).unwrap();
    assert_ne!(
        d.questions[0].action,
        Action::Local,
        "the base rejects it again"
    );
    let g = st.svc.admin(&AdminCommand::Generations).unwrap();
    assert_eq!(g["current"], 0);
    assert_eq!(g["generations"][0]["generation"], 1);
    assert_eq!(g["generations"][0]["current"], false);
    assert_eq!(
        st.cascade.label_counts("topics", "travel"),
        (25, 0),
        "buffer kept, counters reset"
    );
    assert_eq!(
        st.svc
            .admin(&AdminCommand::Rollback { generation: 7 })
            .unwrap_err()
            .status,
        404
    );

    st.svc
        .admin(&AdminCommand::Rollback { generation: 1 })
        .unwrap();
    assert_eq!(
        st.decide(&topics_body(&fresh()[1])).unwrap().questions[0].action,
        Action::Local
    );

    // Restart: CURRENT, the cache and the buffer come back from disk.
    let cache_now = st.cascade.cache_len();
    let st = st.restart(&cfg);
    let hits = mock.hits();
    assert_eq!(served(&st).generation(), 1);
    assert_eq!(st.cascade.cache_len(), cache_now);
    assert_eq!(st.cascade.buffer_len(), buffer);
    assert_eq!(
        st.decide(&topics_body(&fresh()[2])).unwrap().questions[0].action,
        Action::Local
    );
    // Back on the base, a taught text is answered from the restored cache.
    st.svc
        .admin(&AdminCommand::Rollback { generation: 0 })
        .unwrap();
    let d = st.decide(&topics_body(&lesson()[3])).unwrap();
    assert_eq!(d.questions[0].action, Action::Cache);
    assert_eq!(mock.hits(), hits);

    // A cut learn.log tail is dropped at start.
    let log = st.state.learn_log_path();
    let good = std::fs::metadata(&log).unwrap().len();
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
        f.write_all(b"CDLG\x10\x00\x00\x00\x02garbage").unwrap();
    }
    let st = st.restart(&cfg);
    assert_eq!(std::fs::metadata(&log).unwrap().len(), good);
    assert_eq!(st.cascade.buffer_len(), buffer);
    assert_eq!(served(&st).generation(), 0);

    // The CLI's rollback without a server.
    let state_root = st.state.root().to_path_buf();
    let Stand { dir, .. } = st;
    let state = StateDir::open(&state_root).unwrap();
    let cur = generation::rollback_state(&state, 1, Some(&toy().path)).unwrap();
    assert_eq!(cur.generation, 1);
    let st = Stand::open(dir, &cfg, test_key_lookup());
    assert_eq!(served(&st).generation(), 1);
    assert_eq!(st.cascade.label_counts("topics", "travel"), (25, 0));
    let Stand { dir: _dir, .. } = st;
    let cur = generation::rollback_state(&state, 0, None).unwrap();
    assert_eq!(cur.generation, 0);
    assert_eq!(
        cur.sha256,
        DecisionModel::open(&toy().path, Verify::Light)
            .unwrap()
            .base_model_sha()
    );
}

#[test]
fn a_promotion_after_a_rollback_takes_the_next_number_and_keeps_the_buffer() {
    let mock = MockOracle::answering("travel");
    let st = Stand::new(&stand_config(&mock.url()));
    teach(&st, lesson());
    st.svc
        .admin(&AdminCommand::Rollback { generation: 0 })
        .unwrap();
    // 25 new answers: the refit takes every pending example (the 25 of the
    // rolled-back generation are pending again) and writes generation 2 on
    // parent 0.
    let lesson_phi: Vec<Vec<f32>> = lesson().iter().map(|t| phi_p(t)).collect();
    let more: Vec<String> = distinct_texts("cruise", 80, 1234, "m", 0.97)
        .into_iter()
        .filter(|t| {
            let p = phi_p(t);
            lesson_phi.iter().all(|l| cos(l, &p) < 0.97)
        })
        .take(25)
        .collect();
    assert_eq!(more.len(), 25);
    let acts = teach(&st, &more);
    assert!(acts.iter().all(|a| *a == Action::Oracle), "{acts:?}");
    let l = learning(&st);
    let rec = l["recent"].as_array().unwrap().last().unwrap();
    assert_eq!(rec["outcome"], "promoted", "{rec}");
    assert_eq!(rec["generation"], 2);
    assert_eq!(rec["pending"], 50);
    assert_eq!(rec["n_fit"], 80);
    let om = served(&st).model().overlay_manifest().unwrap().clone();
    assert_eq!((om.generation, om.parent), (2, 0));
    assert_eq!(
        om.events.len(),
        1,
        "generation 2 carries the lineage of its parent (0)"
    );
    assert_eq!(st.state.generations().unwrap().len(), 2);
}

#[test]
fn two_synchronous_runs_write_the_same_generation() {
    let run = || {
        let mock = MockOracle::answering("travel");
        let st = Stand::new(&stand_config(&mock.url()));
        teach(&st, lesson());
        let p = st.state.generation_path(1);
        (
            generation::file_sha256(&p).unwrap(),
            std::fs::read(&p).unwrap().len(),
        )
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b);
}

#[test]
fn background_learning_promotes_off_the_request_thread() {
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.learning.synchronous = false;
    let st = Stand::new(&cfg);
    teach(&st, lesson());
    assert!(st.cascade.wait_idle(Duration::from_secs(120)));
    assert_eq!(served(&st).generation(), 1);
    assert_eq!(learning(&st)["promotions"], 1);
    assert_eq!(
        st.decide(&topics_body(&fresh()[0])).unwrap().questions[0].action,
        Action::Local
    );
}

#[test]
fn learning_disabled_keeps_no_example() {
    let mock = MockOracle::answering("travel");
    let mut cfg = stand_config(&mock.url());
    cfg.learning.enabled = false;
    let st = Stand::new(&cfg);
    teach(&st, lesson());
    assert_eq!(st.cascade.buffer_len(), 0);
    assert_eq!(served(&st).generation(), 0);
    let d = st.decide(&topics_body(&accepted_dev()[0])).unwrap();
    assert_eq!(d.questions[0].action, Action::Local);
    let fb = format!(r#"{{"id":"{}","question":"task","label":"cards"}}"#, d.id);
    assert_eq!(
        st.svc
            .feedback(fb.as_bytes(), &Principal::open())
            .unwrap_err()
            .status,
        404
    );
}

/// Dev texts the `topics` gate accepts.
fn accepted_dev() -> &'static [String] {
    static A: OnceLock<Vec<String>> = OnceLock::new();
    A.get_or_init(|| {
        let mut off = cortiq_decision::config::Config::default();
        off.oracle.enabled = false;
        let st = Stand::new(&off);
        toy()
            .dev
            .iter()
            .map(|(t, _)| t.clone())
            .filter(|t| st.decide(&topics_body(t)).unwrap().questions[0].action == Action::Local)
            .take(10)
            .collect()
    })
}

// ------------------------------------------------------------------ offline

fn offline_opts(
    dir: &Path,
    mock: &MockOracle,
    traffic: &Path,
    answers: Vec<PathBuf>,
) -> OfflineOptions {
    let cfg = stand_config(&mock.url());
    let mut o = OfflineOptions::new(traffic, cfg.oracle);
    o.skill = Some("topics".into());
    o.answers = answers;
    o.ledger = Some(dir.join("oracle.jsonl"));
    o.threads = 2;
    o.created_unix = Some(EPOCH);
    o
}

fn jsonl_of(lines: &[Value]) -> String {
    lines.iter().map(|l| l.to_string() + "\n").collect()
}

#[test]
fn offline_learning_asks_only_about_abstentions_and_reuses_ledger_answers() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // Traffic: 10 accepted dev texts (half with a label), 30 cruise texts
    // (abstentions), one repeat.
    let accepted = accepted_dev();
    assert_eq!(accepted.len(), 10);
    let cruise = distinct_texts("cruise", 30, 7, "q", 0.97);
    let mut lines = Vec::new();
    for (i, t) in accepted.iter().enumerate() {
        lines.push(if i % 2 == 0 {
            json!({"text": t, "label": "x"})
        } else {
            json!({"text": t})
        });
    }
    for t in &cruise {
        lines.push(json!({"text": t}));
    }
    lines.push(json!({"text": cruise[0]}));
    let traffic = write(d, "traffic.jsonl", &jsonl_of(&lines));

    // A driver-format ledger answering the first 10 cruise texts.
    let input = DecisionModel::open(&toy().path, Verify::Full).unwrap();
    let q = learn::rubric_question(&input.skill("topics").unwrap().manifest).unwrap();
    let ocfg = stand_config("http://unused").oracle;
    let mut ledger = vec![json!({"record_type": "run"})];
    for t in &cruise[..10] {
        let b = oracle::request_body(&ocfg, &[&q], &json!(t));
        ledger.push(json!({"record_type": "oracle_call", "request_sha256": sha256_hex(&b), "oracle": {"choice": "travel"}}));
    }
    ledger.push(json!({"record_type": "oracle_call", "request_sha256": "0".repeat(64), "oracle": {"error": "http_500"}}));
    let answers = write(d, "answers.jsonl", &jsonl_of(&ledger));

    let mock = MockOracle::answering("travel");
    let opts = offline_opts(d, &mock, &traffic, vec![answers.clone()]);
    let out = d.join("learned.cmf");
    let rep = learn::learn_offline(&toy().path, &opts, test_key_lookup(), &out).unwrap();
    assert_eq!(rep.texts, 41);
    assert_eq!(rep.labelled, 5);
    assert_eq!(rep.accepted, 10);
    assert_eq!(rep.abstained, 31);
    assert_eq!(rep.answers_reused, 10);
    assert_eq!(rep.live_calls, 20);
    assert_eq!(
        mock.hits(),
        20,
        "the oracle is asked only about abstentions without a stored answer"
    );
    assert_eq!((rep.examples, rep.duplicates, rep.unanswered), (30, 1, 0));
    assert_eq!(rep.promoted_labels, vec!["travel"]);
    assert!(rep.rejected_labels.is_empty());
    assert!(!rep.rolled_back);
    assert!(rep.gate_after.certified);
    // Every body the mock saw is the driver's for its text.
    for (req, t) in mock.requests().iter().zip(&cruise[10..]) {
        assert_eq!(req.body, oracle::request_body(&ocfg, &[&q], &json!(t)));
    }

    // The output: self-contained; the encoder and the other skill byte for byte.
    let m = DecisionModel::open(&out, Verify::Full).unwrap();
    assert_eq!(m.generation(), 0);
    assert!(m.overlay().is_none());
    assert_eq!(
        m.skill("shop").unwrap().sha256,
        input.skill("shop").unwrap().sha256
    );
    for e in input.base().tensors.iter().filter(|e| {
        e.name.starts_with("decision.encoder.") || e.name.starts_with("decision.skill.shop.")
    }) {
        assert_eq!(
            m.tensor_bytes(&e.name).unwrap(),
            input.tensor_bytes(&e.name).unwrap(),
            "{}",
            e.name
        );
    }
    for (x, y) in task_table(&input).iter().zip(&task_table(&m)) {
        if (x.0.as_str(), x.1) != ("topics", 3) {
            assert_eq!(x, y);
        }
    }
    let sk = &m.skill("topics").unwrap().manifest;
    let l = sk.learned.as_ref().unwrap();
    assert_eq!((l.calls, l.answers_reused), (20, 10));
    assert_eq!(l.promoted_labels, vec!["travel"]);
    assert_eq!(
        l.traffic_sha256,
        sha256_hex(&std::fs::read(&traffic).unwrap())
    );
    assert_eq!(l.oracle_model, "deepseek/deepseek-v4.1-flash");
    assert_eq!(l.gate_after, sk.gate);
    assert_eq!(l.gate_before, input.skill("topics").unwrap().manifest.gate);
    assert_eq!(sk.rows.n_learned, 30);
    let rows = m.rows("topics").unwrap();
    let learned: Vec<&Row> = rows
        .rows
        .iter()
        .filter(|r| r.split == Split::Learned)
        .collect();
    assert_eq!(learned.len(), 30);
    assert!(
        learned
            .iter()
            .all(|r| r.task == 3 && r.source == Source::Oracle)
    );
    assert_eq!(sk.tasks[3].n_train, 60);
    // The build rows come first, unchanged; the learned rows follow them.
    let base_rows = input.rows("topics").unwrap();
    assert_eq!(&rows.rows[..base_rows.rows.len()], &base_rows.rows[..]);
    assert_fit_reproduces(&m, "topics", 3);
    assert_gate_from_scratch(&m, "topics");
    // The pre-trained file decides new cruise texts locally.
    let ev = Evaluator::new(&m, "topics").unwrap();
    for t in fresh() {
        let td = ev.decide_text(t).unwrap();
        assert!(ev.scorer().accepted(&td.decision), "{t}");
    }
    // An existing output is refused; a second run gives the same bytes.
    assert!(learn::learn_offline(&toy().path, &opts, test_key_lookup(), &out).is_err());
    let mock2 = MockOracle::answering("travel");
    let d2 = tempfile::tempdir().unwrap();
    let opts2 = offline_opts(d2.path(), &mock2, &traffic, vec![answers]);
    let out2 = d2.path().join("learned.cmf");
    let rep2 = learn::learn_offline(&toy().path, &opts2, test_key_lookup(), &out2).unwrap();
    assert_eq!(rep2.out.sha256, rep.out.sha256);
    // Its reservation ledger: 20 reserved + 20 settled lines, no key.
    let ledger = std::fs::read_to_string(d.join("oracle.jsonl")).unwrap();
    assert_eq!(ledger.lines().count(), 40);
    assert!(!ledger.contains(TEST_KEY));
    // The pre-trained file keeps learning online: its learned rows are served.
    let st = Stand::open_on(
        &out,
        tempfile::tempdir().unwrap(),
        &stand_config(&mock.url()),
        test_key_lookup(),
    );
    assert_eq!(
        st.decide(&topics_body(&fresh()[0])).unwrap().questions[0].action,
        Action::Local
    );
}

#[test]
fn offline_learning_undoes_every_promotion_when_the_certified_gate_is_lost() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let cruise = distinct_texts("cruise", 30, 7, "q", 0.97);
    let lines: Vec<Value> = cruise.iter().map(|t| json!({"text": t})).collect();
    let traffic = write(d, "traffic.jsonl", &jsonl_of(&lines));
    let mock = MockOracle::answering("travel");
    let mut opts = offline_opts(d, &mock, &traffic, Vec::new());
    opts.skill = None; // the only skill of the file
    let out = d.join("trap-learned.cmf");
    let rep = learn::learn_offline(&trap().path, &opts, test_key_lookup(), &out).unwrap();
    assert_eq!(rep.skill, "trap");
    assert_eq!(rep.abstained, 30);
    assert!(rep.rolled_back, "{:?}", rep.labels);
    assert!(rep.promoted_labels.is_empty());
    assert_eq!(rep.rejected_labels, vec!["travel"]);
    assert_eq!(rep.labels[0].reason.as_deref(), Some("gate_lost"));
    assert_eq!(rep.labels[0].holdout.as_ref().map(|h| h.passed), Some(true));
    let input = DecisionModel::open(&trap().path, Verify::Full).unwrap();
    let m = DecisionModel::open(&out, Verify::Full).unwrap();
    assert_eq!(task_table(&input), task_table(&m), "every task as before");
    let sk = &m.skill("trap").unwrap().manifest;
    assert_eq!(sk.gate, input.skill("trap").unwrap().manifest.gate);
    assert_eq!(sk.rows.n_learned, 0);
    let l = sk.learned.as_ref().unwrap();
    assert_eq!(l.gate_before, l.gate_after);
    assert_eq!(l.rejected_labels, vec!["travel"]);
    assert!(l.promoted_labels.is_empty());
}

#[test]
fn offline_learning_without_live_calls_uses_the_ledgers_only() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let cruise = distinct_texts("cruise", 5, 7, "q", 0.97);
    let lines: Vec<Value> = cruise.iter().map(|t| json!({"text": t})).collect();
    let traffic = write(d, "traffic.jsonl", &jsonl_of(&lines));
    let mock = MockOracle::answering("travel");
    let mut opts = offline_opts(d, &mock, &traffic, Vec::new());
    opts.oracle.enabled = false;
    opts.ledger = None;
    let rep =
        learn::learn_offline(&toy().path, &opts, test_key_lookup(), &d.join("o.cmf")).unwrap();
    assert_eq!((rep.abstained, rep.unanswered, rep.live_calls), (5, 5, 0));
    assert_eq!(mock.hits(), 0);
    assert!(rep.promoted_labels.is_empty());
    // Live calls without a reservation ledger are refused up front.
    let mut o2 = offline_opts(d, &mock, &traffic, Vec::new());
    o2.ledger = None;
    assert!(learn::learn_offline(&toy().path, &o2, test_key_lookup(), &d.join("o2.cmf")).is_err());
    // Live calls enabled without a key in the environment: refused up front,
    // before the ledger or the output exists.
    let o3 = offline_opts(d, &mock, &traffic, Vec::new());
    let e = learn::learn_offline(&toy().path, &o3, no_key_lookup(), &d.join("o3.cmf")).unwrap_err();
    assert!(format!("{e:#}").contains(KEY_ENV), "{e:#}");
    assert!(!d.join("o3.cmf").exists());
    assert!(!d.join("oracle.jsonl").exists());
    assert_eq!(mock.hits(), 0);
}

/// `live_calls` (and `learned.calls`) count the calls sent; calls the budget
/// refuses are reported apart and their texts are unanswered.
#[test]
fn offline_learning_counts_only_the_calls_sent() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let cruise = distinct_texts("cruise", 5, 7, "q", 0.97);
    let lines: Vec<Value> = cruise.iter().map(|t| json!({"text": t})).collect();
    let traffic = write(d, "traffic.jsonl", &jsonl_of(&lines));
    let mock = MockOracle::answering("travel");
    let mut opts = offline_opts(d, &mock, &traffic, Vec::new());
    opts.oracle.max_calls = 3;
    let out = d.join("o.cmf");
    let rep = learn::learn_offline(&toy().path, &opts, test_key_lookup(), &out).unwrap();
    assert_eq!(mock.hits(), 3);
    assert_eq!(
        (
            rep.abstained,
            rep.live_calls,
            rep.failed_calls,
            rep.unanswered
        ),
        (5, 3, 0, 2)
    );
    assert_eq!(
        rep.refused_calls,
        std::collections::BTreeMap::from([("budget".to_string(), 2)])
    );
    assert_eq!(rep.to_json()["refused_calls"]["budget"], 2);
    let m = DecisionModel::open(&out, Verify::Full).unwrap();
    let l = m.skill("topics").unwrap().manifest.learned.clone().unwrap();
    assert_eq!((l.calls, l.answers_reused), (3, 0));
    // The ledger holds the 3 calls sent (reserved + settled), nothing refused.
    let ledger = std::fs::read_to_string(d.join("oracle.jsonl")).unwrap();
    assert_eq!(ledger.lines().count(), 6);
}
