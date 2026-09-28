//! The `lookup` record END TO END through the `cortiq` binary (lookup spec
//! §2–§4) on the synthetic GDN + bounded genome of tests/knowledge_cli.rs:
//!
//! entries.jsonl (5 entries, RU + EN keys incl. declensions and the Latin
//! binomial, RU + EN cards with fields)
//!   → `lookup-build F0 → F1` (quarantined record, base untouched)
//!   → `route-fit F1` on two prompt classes (RU questions naming the
//!     entries vs EN general prompts: capitals, code, arithmetic)
//!   → `genome-verify F0 F1` passes (G1)
//!   → `route-eval` on a fresh general set as the gate file →
//!     `skill-gate --status active` → auto-routable
//!   → `run F1` on a key question: the field text from the table, nothing
//!     generated; on a general question: the backbone; EN keys / EN slot
//!     through the pinned record; the parenthesised binomial
//!   → `dump-logits F0` vs `dump-logits F1 --route auto` +
//!     `logits-compare`: every backbone record (general prompts and an
//!     in-domain prompt without a key) bit-identical to F0 (G2)
//!   → `probe-utility F1 --route auto --json`: per-row `route` /
//!     `lookup_hit` / `lookup_key` / `field`, the answer is the table's text
//!   → `serve F1`: `/v1/chat/completions` answers from the table with
//!     `x_cortiq_route.{lookup_key, field}`, a general question runs the
//!     backbone.
//!
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;

use cortiq_core::{CmfModel, GenomeInfo};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const SKILL_ID: &str = "herbs";

fn cortiq(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_LOOKUP_MODE")
        .output()
        .expect("spawn cortiq")
}

fn ok(args: &[&str]) -> (String, String) {
    let out = cortiq(args);
    let (so, se) = (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    );
    assert!(
        out.status.success(),
        "cortiq {args:?} failed\nstdout:\n{so}\nstderr:\n{se}"
    );
    (so, se)
}

fn json(args: &[&str]) -> serde_json::Value {
    let (so, _) = ok(args);
    serde_json::from_str(&so).unwrap_or_else(|e| panic!("cortiq {args:?}: not JSON ({e}):\n{so}"))
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn sha_file(p: &Path) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(std::fs::read(p).unwrap())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn write_prompts(path: &Path, rows: &[(String, &str, &str)]) {
    let text: String = rows
        .iter()
        .map(|(p, lang, src)| {
            serde_json::json!({"prompt": p, "lang": lang, "src": src, "expect": []}).to_string() + "\n"
        })
        .collect();
    std::fs::write(path, text).unwrap();
}

// ── the reference: 5 entries, RU + EN keys, RU + EN cards with fields ──

struct Entry {
    keys: &'static [&'static str],
    ru: (&'static str, &'static [(&'static str, &'static str)]),
    en: (&'static str, &'static [(&'static str, &'static str)]),
}

const ENTRIES: &[Entry] = &[
    Entry {
        keys: &["Пихта бальзамическая", "пихты бальзамической", "Abies balsamea", "balsam fir"],
        ru: (
            "Пихта бальзамическая (Abies balsamea) — хвойное дерево семейства Сосновые.",
            &[("family", "Сосновые (Pinaceae)"), ("parts", "хвоя, смола, почки"), ("uses", "бальзам при простуде")],
        ),
        en: (
            "Balsam fir (Abies balsamea), a conifer of the family Pinaceae.",
            &[("family", "Pinaceae"), ("parts", "needles, resin, buds"), ("uses", "balsam for colds")],
        ),
    },
    Entry {
        keys: &["Ромашка аптечная", "ромашки аптечной", "Matricaria chamomilla", "chamomile"],
        ru: (
            "Ромашка аптечная (Matricaria chamomilla) — однолетник семейства Астровые.",
            &[("family", "Астровые (Asteraceae)"), ("parts", "цветочные корзинки"), ("uses", "противовоспалительное, спазмолитик")],
        ),
        en: (
            "Chamomile (Matricaria chamomilla), an annual of the family Asteraceae.",
            &[("family", "Asteraceae"), ("parts", "flower heads"), ("uses", "anti-inflammatory, antispasmodic")],
        ),
    },
    Entry {
        keys: &["Зверобой продырявленный", "зверобоя продырявленного", "Hypericum perforatum", "St John's wort"],
        ru: (
            "Зверобой продырявленный (Hypericum perforatum) — многолетник семейства Зверобойные.",
            &[("family", "Зверобойные (Hypericaceae)"), ("parts", "верхушки побегов с цветками"), ("safety", "фотосенсибилизация, взаимодействие с лекарствами")],
        ),
        en: (
            "St John's wort (Hypericum perforatum), a perennial of the family Hypericaceae.",
            &[("family", "Hypericaceae"), ("parts", "flowering tops"), ("safety", "photosensitivity, drug interactions")],
        ),
    },
    Entry {
        keys: &["Шалфей лекарственный", "шалфея лекарственного", "Salvia officinalis", "common sage"],
        ru: (
            "Шалфей лекарственный (Salvia officinalis) — полукустарник семейства Яснотковые.",
            &[("family", "Яснотковые (Lamiaceae)"), ("parts", "листья"), ("uses", "полоскания при ангине")],
        ),
        en: (
            "Common sage (Salvia officinalis), a subshrub of the family Lamiaceae.",
            &[("family", "Lamiaceae"), ("parts", "leaves"), ("uses", "gargles for sore throat")],
        ),
    },
    Entry {
        keys: &["Календула лекарственная", "календулы лекарственной", "Calendula officinalis", "pot marigold"],
        ru: (
            "Календула лекарственная (Calendula officinalis) — однолетник семейства Астровые.",
            &[("family", "Астровые (Asteraceae)"), ("parts", "цветки"), ("uses", "ранозаживляющее")],
        ),
        en: (
            "Pot marigold (Calendula officinalis), an annual of the family Asteraceae.",
            &[("family", "Asteraceae"), ("parts", "flowers"), ("uses", "wound healing")],
        ),
    },
];

fn entries_jsonl() -> String {
    let card = |(text, fields): (&str, &[(&str, &str)])| {
        let f: serde_json::Map<String, serde_json::Value> = fields
            .iter()
            .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
            .collect();
        serde_json::json!({"card": text, "fields": f})
    };
    ENTRIES
        .iter()
        .map(|e| serde_json::json!({"keys": e.keys, "ru": card(e.ru), "en": card(e.en)}).to_string() + "\n")
        .collect()
}

fn field(e: usize, lang: &str, name: &str) -> &'static str {
    let (_, fields) = if lang == "ru" { ENTRIES[e].ru } else { ENTRIES[e].en };
    fields.iter().find(|(k, _)| *k == name).map(|(_, v)| *v).unwrap()
}

/// The declined RU key of every entry (the form the questions use).
fn ru_declined(e: usize) -> &'static str {
    ENTRIES[e].keys[1]
}

// ── prompt classes for route-fit (subject-major: the held-out 20 % are
// unseen subjects under every template) ──

const SKILL_TEMPLATES: &[&str] = &[
    "Какие лечебные свойства у {}?",
    "Чем полезен отвар {}?",
    "Как заваривать настой {}?",
    "Какое семейство у {}?",
    "Чем опасна передозировка {}?",
    "Какие части {} применяют?",
];

const GENERAL_TEMPLATES: &[&str] = &[
    "What is {}?",
    "Explain {} in two sentences.",
    "Tell me about {}.",
    "Give a short answer: {}.",
    "Write one line about {}.",
];

const EXTRA_HERBS: &[&str] = &[
    "валерианы лекарственной",
    "болиголова пятнистого",
    "мяты перечной",
    "крапивы двудомной",
    "подорожника большого",
];

const TOPICS: &[&str] = &[
    "the capital of France",
    "a Rust function that returns the maximum element",
    "why Earth has seasons",
    "the value of 17 * 19 + 23",
    "the capital of Japan",
    "why a hash table offers constant-time lookup",
    "the boiling point of water at sea level",
    "how binary search works",
    "the largest planet of the solar system",
    "what a prime number is",
];

const FRESH_TOPICS: &[&str] = &[
    "the capital of Italy",
    "a Python function that reverses a string",
    "the value of 12 * 12 - 7",
    "what a compiler does",
    "why the sky is blue",
    "the freezing point of water",
    "how a linked list works",
    "the speed of light",
];

fn expand(templates: &[&str], subjects: &[&str]) -> Vec<String> {
    subjects
        .iter()
        .flat_map(|h| templates.iter().map(move |t| t.replace("{}", h)))
        .collect()
}

fn skill_train() -> Vec<String> {
    let mut subjects: Vec<&str> = (0..ENTRIES.len()).map(ru_declined).collect();
    subjects.extend_from_slice(EXTRA_HERBS);
    expand(SKILL_TEMPLATES, &subjects)
}

fn general_train() -> Vec<String> {
    expand(GENERAL_TEMPLATES, TOPICS)
}

fn general_fresh() -> Vec<String> {
    FRESH_TOPICS
        .iter()
        .enumerate()
        .map(|(i, t)| GENERAL_TEMPLATES[i % GENERAL_TEMPLATES.len()].replace("{}", t))
        .collect()
}

/// An in-domain question whose subject is not in the table.
const NO_KEY_TEXT: &str = "Какие лечебные свойства у мяты перечной?";

fn family_question(e: usize) -> String {
    format!("Какое семейство у {}?", ru_declined(e))
}

// ── serve ──

struct Server {
    child: Child,
    port: u16,
    /// The server's stdout (the banner: routing policy, lookup records).
    output: Arc<Mutex<Vec<String>>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_server(model: &Path) -> Server {
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut child = Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(["serve", s(model), "--port", &port.to_string(), "--host", "127.0.0.1"])
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_LOOKUP_MODE")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn cortiq serve");
    let output = Arc::new(Mutex::new(Vec::new()));
    let sink = output.clone();
    let pipe = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(pipe).lines().map_while(Result::ok) {
            sink.lock().unwrap().push(line);
        }
    });
    let t0 = Instant::now();
    loop {
        if let Ok(r) = ureq::get(&format!("http://127.0.0.1:{port}/healthz")).call() {
            if r.status() == 200 {
                break;
            }
        }
        assert!(
            t0.elapsed() < Duration::from_secs(180),
            "server did not come up:\n{}",
            output.lock().unwrap().join("\n")
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    Server { child, port, output }
}

fn chat(port: u16, user: &str, max_tokens: u32) -> serde_json::Value {
    let body = serde_json::json!({
        "model": "cortiq",
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    });
    ureq::post(&format!("http://127.0.0.1:{port}/v1/chat/completions"))
        .timeout(Duration::from_secs(300))
        .send_json(body)
        .expect("chat request")
        .into_json()
        .expect("chat json")
}

#[test]
fn lookup_end_to_end() {
    // SAFETY: set before any pipeline of this process exists.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-lookup-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let f0 = dir.join("f0.cmf");
    embryo_synth::write_synth_genome_with(
        &f0,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        Some(GenomeInfo::birth("synth-genome", "pre_chat", "f32")),
    );
    let entries = dir.join("entries.jsonl");
    std::fs::write(&entries, entries_jsonl()).unwrap();
    let f1 = dir.join("f1.cmf");
    let sha0 = sha_file(&f0);
    let n_keys: usize = ENTRIES.iter().map(|e| e.keys.len()).sum();

    // ── 1. lookup-build ──
    let b = json(&[
        "lookup-build",
        s(&f0),
        "--entries",
        s(&entries),
        "--id",
        SKILL_ID,
        "--out",
        s(&f1),
        "--langs",
        "ru,en",
    ]);
    eprintln!("lookup-build: {b}");
    assert_eq!(b["entries"], ENTRIES.len(), "{b}");
    assert_eq!(b["keys"], n_keys, "{b}");
    assert_eq!(b["duplicates_dropped"], 0);
    assert_eq!(b["empty_keys_dropped"], 0);
    assert_eq!(b["unreachable_keys_dropped"], 0);
    assert_eq!(b["missing_slots"], 0);
    assert_eq!(b["langs"], serde_json::json!(["ru", "en"]));
    // The union of field names in first-seen order.
    assert_eq!(b["fields"], serde_json::json!(["family", "parts", "uses", "safety"]));
    assert_eq!(b["status"], "quarantine");
    assert_eq!(b["auto_routable"], false);
    assert_eq!(sha_file(&f0), sha0, "the base was modified");
    let m = CmfModel::open(&f1).unwrap();
    let rec = &m.header.skills[0];
    assert_eq!(rec.id, SKILL_ID);
    assert_eq!(rec.kind.as_deref(), Some("lookup"));
    assert_eq!(rec.status.as_deref(), Some("quarantine"));
    let info = rec.lookup.as_ref().unwrap();
    assert_eq!((info.entries, info.keys), (ENTRIES.len(), n_keys));
    assert_eq!(info.langs, vec!["ru", "en"]);
    assert_eq!(rec.bound.as_ref().unwrap().genome_id, "synth-genome");
    drop(m);
    assert_eq!(json(&["genome-verify", s(&f0), s(&f1)])["pass"], true, "G1 after lookup-build");

    // ── 2. route-fit: skill prompts (questions naming the entries) vs
    // general prompts (capitals, code, arithmetic) ──
    let (skp, genp, gene) = (
        dir.join("herbs-q.jsonl"),
        dir.join("general-calib.jsonl"),
        dir.join("general-eval.jsonl"),
    );
    let sk: Vec<(String, &str, &str)> = skill_train().into_iter().map(|p| (p, "ru", "herbs-train")).collect();
    let gt: Vec<(String, &str, &str)> = general_train().into_iter().map(|p| (p, "en", "general-calib")).collect();
    let gf: Vec<(String, &str, &str)> = general_fresh().into_iter().map(|p| (p, "en", "general-eval")).collect();
    write_prompts(&skp, &sk);
    write_prompts(&genp, &gt);
    write_prompts(&gene, &gf);
    // The synthetic byte-level φ of a random genome gives a tight in-scope
    // shell: with the default target-fpr 0.05 the (1−fpr) quantile θ
    // rejects one held-out prompt and 1–3 fitted templates on fitted
    // subjects (measured: rank 2…8 → 57–59 of 60). θ at the maximum
    // held-out novelty (`--target-fpr 0`) routes every fitted question
    // form; the general class still goes to the backbone (8/8 below).
    let before = std::fs::read(&f1).unwrap();
    let r = json(&[
        "route-fit",
        s(&f1),
        "--id",
        SKILL_ID,
        "--skill-prompts",
        s(&skp),
        "--general-prompts",
        s(&genp),
        "--phi-layer",
        "0",
        "--rank",
        "2",
        "--target-fpr",
        "0",
    ]);
    eprintln!(
        "route-fit: recall {} false_accept {} (CP95 {}) n_in {} n_general {} θ {} T {} skill {} base {}",
        r["in_scope_recall"],
        r["false_accept"],
        r["false_accept_upper95"],
        r["n_in"],
        r["n_general"],
        r["novelty_theta"],
        r["temperature"],
        r["skill_prompts"]["descriptor"],
        r["general_prompts"]["descriptor"]
    );
    // Held-out 20 % of each class (subject-major: unseen subjects).
    let (ho_sk, ho_gt) = (sk.len() / 5, gt.len() / 5);
    assert_eq!(r["skill_prompts"]["descriptor"]["n"], sk.len(), "{r}");
    assert_eq!(r["skill_prompts"]["descriptor"]["holdout"], ho_sk);
    assert_eq!(r["general_prompts"]["descriptor"]["n"], gt.len());
    assert_eq!(r["general_prompts"]["descriptor"]["holdout"], ho_gt);
    assert_eq!(r["n_in"], ho_sk);
    assert_eq!(r["n_general"], ho_gt);
    assert_eq!(r["in_scope_recall"], 1.0, "{r}");
    assert_eq!(r["false_accept"], 0.0, "{r}");
    assert_eq!(r["classes"], serde_json::json!([SKILL_ID]));
    assert_eq!(r["status"], "quarantine");
    assert_eq!(r["auto_routable"], false);
    assert_eq!(r["phi_backend"], "cpu");
    assert_eq!(r["runtime_check"]["decisions_equal"], 16, "{}", r["runtime_check"]);
    // Header-only append: bytes [128, old_len) unchanged.
    let old_len = r["old_len"].as_u64().unwrap() as usize;
    let after = std::fs::read(&f1).unwrap();
    assert_eq!(old_len, before.len());
    assert_eq!(&after[128..old_len], &before[128..old_len]);
    // ── 3. G1 ──
    assert_eq!(json(&["genome-verify", s(&f0), s(&f1)])["pass"], true, "G1 after route-fit");

    // ── 4. gate on a fresh general set → skill-gate active ──
    let g = json(&[
        "route-eval",
        s(&f1),
        "--prompts-jsonl",
        s(&gene),
        "--expect",
        "backbone",
        "--include-quarantine",
        "--json",
    ]);
    eprintln!("route-eval general-eval: {}", g);
    assert_eq!(g["n"], gf.len(), "{g}");
    assert_eq!(g["accepted"], gf.len(), "a general prompt reached the lookup skill: {g}");
    assert_eq!(g["status"], "measured");
    let gate = dir.join("gate.json");
    std::fs::write(&gate, serde_json::to_string(&g).unwrap()).unwrap();
    let sg = json(&["skill-gate", s(&f1), "--id", SKILL_ID, "--gate", s(&gate), "--status", "active"]);
    assert_eq!(sg["gate_measured"], true, "{sg}");
    assert_eq!(sg["auto_routable"], true, "{sg}");
    assert_eq!(sg["calibration_stale"], false);
    let m = CmfModel::open(&f1).unwrap();
    assert!(m.header.skills[0].is_auto_routable());
    assert_eq!(m.header.skills[0].status.as_deref(), Some("active"));
    drop(m);
    assert_eq!(json(&["genome-verify", s(&f0), s(&f1)])["pass"], true, "G1 after skill-gate");

    // ── 5. run: a key question is answered from the table ──
    let q1 = family_question(1);
    let (so, se) = ok(&["run", s(&f1), "-p", &q1, "-n", "4", "--greedy"]);
    assert!(se.contains(&format!("route: {SKILL_ID}")), "{se}");
    assert!(se.contains(&format!("lookup: {SKILL_ID}")), "{se}");
    assert!(se.contains("field family") && se.contains("mode answer"), "{se}");
    assert!(so.contains(field(1, "ru", "family")), "{so}");
    assert!(so.contains("answered from the table"), "{so}");
    assert!(!so.contains("tokens"), "something was generated:\n{so}");
    // Every entry through its declined RU key, field `family`.
    for e in 0..ENTRIES.len() {
        let q = family_question(e);
        let (so, se) = ok(&["run", s(&f1), "-p", &q, "-n", "4", "--greedy"]);
        assert!(se.contains(&format!("lookup: {SKILL_ID}")), "{q}: {se}");
        assert!(se.contains(&format!("entry {e}")), "{q}: {se}");
        assert!(so.contains(field(e, "ru", "family")), "{q}: {so}");
    }
    // A `parts` question (a fitted template; the parts rule fires before
    // `uses`). A template the router was never fitted on is "novel" under
    // the tight θ of the synthetic φ and runs the backbone — the pinned
    // record below takes those.
    let q = format!("Какие части {} применяют?", ru_declined(0));
    let (so, se) = ok(&["run", s(&f1), "-p", &q, "-n", "4", "--greedy"]);
    assert!(se.contains("field parts"), "{se}");
    assert!(so.contains(field(0, "ru", "parts")), "{so}");
    // A general question: the backbone, no lookup line.
    let (_, se) = ok(&["run", s(&f1), "-p", "What is the capital of France?", "-n", "2", "--greedy"]);
    assert!(se.contains("route: backbone"), "{se}");
    assert!(!se.contains("lookup:"), "{se}");
    // An in-domain question without a key: the backbone.
    let (so, se) = ok(&["run", s(&f1), "-p", NO_KEY_TEXT, "-n", "2", "--greedy"]);
    assert!(se.contains("route: backbone"), "{se}");
    assert!(!so.contains("answered from the table"), "{so}");
    // EN keys and the EN slot through the pinned record (an EN question
    // is outside the fitted RU class): `balsam fir` → family Pinaceae;
    // `St John's wort` with both `parts` and `used` → parts (rule order).
    let (so, se) = ok(&["run", s(&f1), "-p", "What family is balsam fir?", "--skill", SKILL_ID]);
    assert!(se.contains("pinned") && se.contains("lookup key"), "{se}");
    assert!(se.contains("field family"), "{se}");
    assert!(so.contains(field(0, "en", "family")), "{so}");
    let (so, se) = ok(&["run", s(&f1), "-p", "Which parts of St John's wort are used?", "--skill", SKILL_ID]);
    assert!(se.contains("field parts"), "{se}");
    assert!(so.contains(field(2, "en", "parts")), "{so}");
    // The parenthesised Latin binomial is tried first; `опасна` → safety.
    let q = "Чем опасна передозировка травы (Hypericum perforatum)?";
    let (so, se) = ok(&["run", s(&f1), "-p", q, "--skill", SKILL_ID]);
    assert!(se.contains("entry 2") && se.contains("field safety"), "{se}");
    assert!(so.contains(field(2, "ru", "safety")), "{so}");
    // --lookup-mode off: the backbone runs the key question.
    let (so, se) = ok(&["run", s(&f1), "-p", &q1, "-n", "2", "--greedy", "--lookup-mode", "off"]);
    assert!(se.contains("route: backbone") && se.contains("mode off"), "{se}");
    assert!(!so.contains("answered from the table"), "{so}");
    // F0 knows no table.
    let (so, se) = ok(&["run", s(&f0), "-p", &q1, "-n", "2", "--greedy"]);
    assert!(!se.contains("lookup:"), "{se}");
    assert!(!so.contains(field(1, "ru", "family")), "{so}");

    // ── 6. G2: general prompts bit-identical to F0 ──
    // mixed = general[i], skill[i] interleaved, then the no-key prompt.
    let mixed = dir.join("mixed.jsonl");
    let general_rows: Vec<String> = general_fresh().into_iter().take(ENTRIES.len()).collect();
    let mut rows: Vec<(String, &str, &str)> = Vec::new();
    for (e, g) in general_rows.iter().enumerate() {
        rows.push((g.clone(), "en", "general"));
        rows.push((family_question(e), "ru", "herbs"));
    }
    rows.push((NO_KEY_TEXT.to_string(), "ru", "nokey"));
    write_prompts(&mixed, &rows);
    let n_skill = ENTRIES.len();
    let n_bb = general_rows.len() + 1;
    let (d0, d1) = (dir.join("f0.bin"), dir.join("f1.bin"));
    let j0 = json(&["dump-logits", s(&f0), "--prompts-jsonl", s(&mixed), "--tokens", "3", "--out", s(&d0)]);
    assert_eq!(j0["route_mode"], "none", "{j0}");
    assert_eq!(j0["lookup_hits"], 0);
    let j1 = json(&[
        "dump-logits",
        s(&f1),
        "--prompts-jsonl",
        s(&mixed),
        "--tokens",
        "3",
        "--route",
        "auto",
        "--out",
        s(&d1),
    ]);
    assert_eq!(j1["route_mode"], "auto", "{j1}");
    assert_eq!(j1["lookup_mode"], "answer");
    assert_eq!(j1["lookup_hits"], n_skill, "{j1}");
    assert_eq!(j1["route_counts"]["backbone"], n_bb, "{j1}");
    assert_eq!(j1["route_counts"][SKILL_ID], n_skill);
    let c = json(&["logits-compare", s(&d0), s(&d1)]);
    eprintln!(
        "logits-compare: matched {} backbone {}/{} bit-identical, max_abs_diff {} g2_pass {}",
        c["matched"], c["b_backbone_bit_identical"], c["b_backbone_records"], c["max_abs_diff"], c["g2_pass"]
    );
    assert_eq!(c["matched"], n_bb + n_skill, "{c}");
    assert_eq!(c["b_backbone_records"], n_bb);
    assert_eq!(c["b_backbone_bit_identical"], n_bb, "{c}");
    assert_eq!(c["b_backbone_max_abs_diff"], 0.0);
    assert_eq!(c["g2_pass"], true, "{c}");
    // `answer` mode: the hit records carry the backbone's untouched view
    // of the plain prompt — bit-identical too.
    assert_eq!(c["max_abs_diff"], 0.0, "{c}");

    // ── 7. probe-utility --route auto: per-row route / lookup_hit ──
    let v = json(&[
        "probe-utility",
        s(&f1),
        "--prompts-jsonl",
        s(&mixed),
        "--max-tokens",
        "2",
        "--route",
        "auto",
        "--json",
    ]);
    let m = &v[0];
    assert_eq!(m["route_mode"], "auto", "{m}");
    assert_eq!(m["lookup_mode"], "answer");
    assert_eq!(m["lookup_hits"], n_skill, "{m}");
    // The key-miss rate (spec §5): the only miss can be the in-domain
    // no-key row (the router may pick the record; no key → backbone).
    let misses = m["lookup_misses"].as_u64().unwrap();
    assert!(misses <= 1, "a key question missed its key: {m}");
    assert_eq!(m["lookup_targets"], n_skill as u64 + misses, "{m}");
    assert_eq!(m["lookup_off"], 0);
    assert_eq!(m["decided_counts"][SKILL_ID], n_skill as u64 + misses);
    assert_eq!(m["route_counts"][SKILL_ID], n_skill);
    assert_eq!(m["route_counts"]["backbone"], n_bb);
    let prow = m["per_prompt"].as_array().unwrap();
    assert_eq!(prow.len(), rows.len());
    for (i, row) in prow.iter().enumerate() {
        let is_skill = i % 2 == 1 && i < 2 * n_skill;
        assert!(row.get("route").is_some(), "no `route` on row {i}: {row}");
        assert_eq!(row["lookup_hit"], is_skill, "{row}");
        assert_eq!(row["route"]["lookup_hit"], is_skill, "{row}");
        if is_skill {
            let e = i / 2;
            assert_eq!(row["route"]["target"], SKILL_ID, "{row}");
            assert_eq!(row["route"]["decided_target"], SKILL_ID);
            assert_eq!(row["route"]["lookup_entry"], e);
            assert_eq!(row["route"]["lookup_key"], ru_declined(e));
            assert_eq!(row["route"]["lookup_lang"], "ru");
            assert_eq!(row["route"]["field"], "family");
            assert_eq!(row["route"]["lookup_mode"], "answer");
            assert_eq!(row["finish_reason"], "lookup");
            assert_eq!(row["generated"], 0);
            assert_eq!(row["answer"], field(e, "ru", "family"));
        } else {
            assert_eq!(row["route"]["target"], "backbone", "{row}");
            assert_ne!(row["finish_reason"], "lookup");
        }
    }
    let last = &prow[prow.len() - 1];
    assert_eq!(last["label"], "ru:nokey");
    assert_eq!(last["route"]["target"], "backbone", "{last}");
    assert_eq!(last["lookup_hit"], false);
    if misses == 1 {
        assert_eq!(last["route"]["decided_target"], SKILL_ID, "{last}");
        assert!(last["route"]["reason"].as_str().unwrap().contains("no key in the message"), "{last}");
    }

    // ── 8. serve: the chat endpoint answers from the table ──
    let server = start_server(&f1);
    // The banner is written before the listener binds; the pipe reader is
    // a separate thread: give it a moment.
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        server.output.lock().unwrap().iter().any(|l| l.contains("Lookup records: [\"herbs\"]")),
        "serve did not announce the lookup record:\n{}",
        server.output.lock().unwrap().join("\n")
    );
    let v = chat(server.port, &family_question(4), 8);
    assert_eq!(v["choices"][0]["message"]["content"], field(4, "ru", "family"), "{v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["completion_tokens"], 0, "{v}");
    let xr = &v["x_cortiq_route"];
    assert_eq!(xr["target"], SKILL_ID, "{v}");
    assert_eq!(xr["lookup_hit"], true);
    assert_eq!(xr["lookup_key"], ru_declined(4));
    assert_eq!(xr["lookup_entry"], 4);
    assert_eq!(xr["field"], "family");
    let v = chat(server.port, "What is the capital of France?", 2);
    assert_eq!(v["x_cortiq_route"]["target"], "backbone", "{v}");
    assert_eq!(v["x_cortiq_route"]["lookup_hit"], false);
    assert!(v["x_cortiq_route"].get("lookup_key").is_none(), "{v}");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() > 0, "{v}");
    drop(server);
    let _ = std::fs::remove_dir_all(&dir);
}
