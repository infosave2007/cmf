//! `cortiq lookup-build` → `route-fit` end to end through the `cortiq`
//! binary on the synthetic GDN + bounded genome of tests/knowledge_cli.rs
//! (lookup spec §2, CMF_V2_SPEC §9.5.2 / §9.4):
//!
//! * `lookup-build F0 --entries e.jsonl --id herbs --out F1` — the table
//!   (keys normalised, hashed, sorted; duplicates dropped and reported;
//!   slots `entry · L + lang`, a missing language = empty card), the
//!   quarantined v2 record, `genome-verify F0 F1` (G1), every key found by
//!   binary search with `cortiq_core::key_hash`, the refusals leave the
//!   base and the output untouched;
//! * `route-fit F1 --id herbs --skill-prompts … --general-prompts …
//!   --phi-layer 0 --rank 2` on two separable prompt classes (Cyrillic vs
//!   Latin under the byte-level synthetic tokenizer) — `header.router`
//!   (phi = the cmf-im-v1 frame of the file's tokenizer), the record's
//!   `selection` with its held-out, `header.routing`, a `skills_hash`
//!   that binds, `measured` with the prompt-set sha256s, the lineage
//!   event, bytes `[128, old_len)` unchanged (G1 again);
//! * `route-eval` on DISJOINT sets: general → backbone (no false accept),
//!   in-scope → the skill; `skill-gate --status active` → auto-routable.
//!
//! CPU (`CMF_GPU=0`).

#[path = "../../cortiq-engine/tests/common/embryo_synth.rs"]
mod embryo_synth;

use cortiq_core::knowledge::{key_hash, lookup_leaf, lookup_tensor_name, read_u32_le, read_u64_le};
use cortiq_core::{CmfModel, GenomeInfo, TensorDtype};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SKILL_ID: &str = "herbs";

fn cortiq(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cortiq"))
        .args(args)
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
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

fn fails(args: &[&str]) -> String {
    let out = cortiq(args);
    assert!(!out.status.success(), "cortiq {args:?} unexpectedly succeeded");
    String::from_utf8_lossy(&out.stderr).into_owned()
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

fn write_prompts(path: &Path, prompts: &[String], lang: &str, src: &str) {
    let text: String = prompts
        .iter()
        .map(|p| serde_json::json!({"prompt": p, "lang": lang, "src": src, "expect": []}).to_string() + "\n")
        .collect();
    std::fs::write(path, text).unwrap();
}

const HERBS: &[&str] = &[
    "ромашки аптечной",
    "зверобоя продырявленного",
    "шалфея лекарственного",
    "валерианы лекарственной",
    "календулы лекарственной",
    "болиголова пятнистого",
    "мяты перечной",
    "крапивы двудомной",
    "подорожника большого",
    "тысячелистника обыкновенного",
];

const TOPICS: &[&str] = &[
    "the capital of France",
    "a Rust function that returns the maximum element",
    "why Earth has seasons",
    "the value of 17 * 19 + 23",
    "what water is made of",
    "why a hash table offers constant-time lookup",
    "the boiling point of water at sea level",
    "how binary search works",
    "the largest planet of the solar system",
    "what a prime number is",
];

const FRESH_HERBS: &[&str] = &[
    "мелиссы лекарственной",
    "липы сердцевидной",
    "душицы обыкновенной",
    "чабреца ползучего",
    "пустырника сердечного",
    "солодки голой",
    "алтея лекарственного",
    "череды трёхраздельной",
];

const FRESH_TOPICS: &[&str] = &[
    "the speed of light",
    "how photosynthesis works",
    "a Python function that reverses a string",
    "the value of 12 * 12 - 7",
    "what a compiler does",
    "why the sky is blue",
    "the freezing point of water",
    "how a linked list works",
];

/// `subjects × templates` in subject-major order (so the held-out tail
/// of a fit is unseen subjects under every template), and the fresh
/// subjects under the same templates (an in-distribution eval set).
fn prompt_sets(
    templates: &[&str],
    subjects: &[&str],
    fresh: &[&str],
) -> (Vec<String>, Vec<String>) {
    let train = subjects
        .iter()
        .flat_map(|h| templates.iter().map(move |t| t.replace("{}", h)))
        .collect();
    let fresh = fresh
        .iter()
        .enumerate()
        .map(|(i, h)| templates[i % templates.len()].replace("{}", h))
        .collect();
    (train, fresh)
}

/// 50 in-scope prompts (10 herbs × 5 templates) and 8 fresh ones.
fn skill_prompts() -> (Vec<String>, Vec<String>) {
    prompt_sets(
        &[
            "Какие лечебные свойства у {}?",
            "Чем полезен отвар {}?",
            "Как заваривать настой {}?",
            "Какое семейство у {}?",
            "Чем опасна передозировка {}?",
        ],
        HERBS,
        FRESH_HERBS,
    )
}

/// 50 general prompts (10 topics × 5 templates) and 8 fresh ones.
fn general_prompts() -> (Vec<String>, Vec<String>) {
    prompt_sets(
        &[
            "What is {}?",
            "Explain {} in two sentences.",
            "Tell me about {}.",
            "Give a short answer: {}.",
            "Write one line about {}.",
        ],
        TOPICS,
        FRESH_TOPICS,
    )
}

/// Entry 1 repeats its own key twice (punctuation, and the ru-wiki
/// spelling with stress marks — one key under cmf-key-v2) and carries a
/// 7-word composite key the runtime could never match; entry 2 shares
/// `balsam fir` with entry 0 (a cross-entry duplicate) and has no `en`.
const ENTRIES: &str = r#"{"keys": ["Пихта бальзамическая", "Abies balsamea", "balsam fir"], "ru": {"card": "Пихта бальзамическая — хвойное дерево семейства Сосновые.", "fields": {"family": "Сосновые (Pinaceae)", "parts": "хвоя, смола"}}, "en": {"card": "Balsam fir, a conifer of the family Pinaceae.", "fields": {"family": "Pinaceae"}}}
{"keys": ["Ромашка аптечная", "Matricaria chamomilla", "chamomile", "ромашка аптечная!", "Рома́шка апте́чная", "Matricaria chamomilla syn. Chamomilla recutita (L.) Rauschert"], "ru": {"card": "Ромашка аптечная — семейство Астровые.", "fields": {"family": "Астровые (Asteraceae)", "uses": "противовоспалительное"}}, "en": {"card": "Chamomile, family Asteraceae.", "fields": {"family": "Asteraceae"}}}

{"keys": ["Зверобой продырявленный", "Hypericum perforatum", "balsam fir", "!!!"], "id": "hypericum", "ru": {"card": "Зверобой продырявленный — семейство Зверобойные.", "fields": {"family": "Зверобойные (Hypericaceae)"}}}
"#;

/// `lookup-build` with the fixture's cross-entry duplicate resolved
/// explicitly (the default refuses it).
fn build_args<'a>(f0: &'a str, entries: &'a str, out: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec![
        "lookup-build",
        f0,
        "--entries",
        entries,
        "--id",
        SKILL_ID,
        "--out",
        out,
        "--on-duplicate",
        "first",
    ];
    a.extend_from_slice(extra);
    a
}

struct Fixture {
    dir: PathBuf,
    f0: PathBuf,
    entries: PathBuf,
}

fn fixture(tag: &str) -> Fixture {
    // SAFETY: set before any pipeline of this process exists.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!("cmf-lookup-build-cli-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f0 = dir.join("f0.cmf");
    embryo_synth::write_synth_genome_with(
        &f0,
        &embryo_synth::SynthGeom::tiny_gdn_bounded(),
        Some(GenomeInfo::birth("synth-genome", "pre_chat", "f32")),
    );
    let entries = dir.join("entries.jsonl");
    std::fs::write(&entries, ENTRIES).unwrap();
    Fixture { dir, f0, entries }
}

/// Binary search of `key` in the file's table → (entry, slot texts per lang).
fn lookup(m: &CmfModel, key: &str) -> Option<(u32, Vec<String>)> {
    let bytes = |leaf: &str| m.tensor_bytes(&lookup_tensor_name(SKILL_ID, leaf)).unwrap();
    let hashes = read_u64_le(bytes(lookup_leaf::KEYS_HASH));
    let entries = read_u32_le(bytes(lookup_leaf::KEYS_ENTRY));
    let off = read_u64_le(bytes(lookup_leaf::ENTRIES_OFF));
    let text = bytes(lookup_leaf::TEXT);
    let i = hashes.binary_search(&key_hash(key)).ok()?;
    let e = entries[i] as usize;
    let l = m.header.skills[0].lookup.as_ref().unwrap().langs.len();
    let slots = (0..l)
        .map(|lang| {
            let s = e * l + lang;
            String::from_utf8(text[off[s] as usize..off[s + 1] as usize].to_vec()).unwrap()
        })
        .collect();
    Some((e as u32, slots))
}

#[test]
fn lookup_build_route_fit_and_gates() {
    let fx = fixture("e2e");
    let f1 = fx.dir.join("f1.cmf");
    let sha0 = sha_file(&fx.f0);

    // ── lookup-build ──
    // The cross-entry duplicate is refused by default: nothing is written.
    let se = fails(&["lookup-build", s(&fx.f0), "--entries", s(&fx.entries), "--id", SKILL_ID, "--out", s(&f1), "--langs", "ru,en"]);
    assert!(se.contains("1 cross-entry duplicate key(s)") && se.contains("\"balsam fir\" (entries 0 and 2)"), "{se}");
    assert!(!f1.exists());
    assert!(std::fs::read_dir(&fx.dir).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().contains(".tmp")), "a temp file was left behind");
    let b = json(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--langs", "ru,en"]));
    assert_eq!(b["entries"], 3, "{b}");
    // 3 + 3 + 2 keys: "ромашка аптечная!" and the accented spelling repeat
    // their own entry's key, "balsam fir" repeats entry 0's, "!!!"
    // normalises to nothing, the 7-word composite key is unreachable.
    assert_eq!(b["keys"], 8);
    assert_eq!(b["key_norm"], "cmf-key-v2");
    assert_eq!(b["on_duplicate"], "first");
    assert_eq!(b["duplicates_dropped"], 3);
    assert_eq!(b["duplicates_same_entry"], 2);
    assert_eq!(b["duplicates_cross_entry"], 1);
    assert_eq!(b["empty_keys_dropped"], 1);
    assert_eq!(b["unreachable_keys_dropped"], 1, "{b}");
    let unr = &b["unreachable_keys"][0];
    assert_eq!(unr["entry"], 1);
    assert_eq!(unr["words"], 7, "{unr}");
    assert_eq!(unr["norm"], "matricaria chamomilla syn chamomilla recutita l rauschert");
    assert_eq!(b["langs"], serde_json::json!(["ru", "en"]));
    assert_eq!(b["fields"], serde_json::json!(["family", "parts", "uses"]));
    assert_eq!(b["status"], "quarantine");
    assert_eq!(b["missing_slots"], 1, "entry 2 has no en card");
    assert_eq!(b["missing_slot_list"], serde_json::json!([{"entry": 2, "lang": "en"}]));
    assert_eq!(b["auto_routable"], false);
    assert!(b["probe"].is_null() && b["drop_list"].is_null());
    let dups = b["duplicate_keys"].as_array().unwrap();
    assert_eq!(dups[0]["key"], "ромашка аптечная!");
    assert_eq!(dups[0]["entry"], 1);
    assert_eq!(dups[0]["first_entry"], 1);
    assert_eq!(dups[0]["same_entry"], true);
    assert_eq!(dups[1]["key"], "Рома́шка апте́чная");
    assert_eq!(dups[1]["norm"], "ромашка аптечная");
    assert_eq!(dups[1]["same_entry"], true);
    assert_eq!(dups[2]["key"], "balsam fir");
    assert_eq!(dups[2]["entry"], 2);
    assert_eq!(dups[2]["first_entry"], 0);
    assert_eq!(dups[2]["same_entry"], false);
    assert_eq!(dups[2]["kept_entry"], 0);
    assert_eq!(sha_file(&fx.f0), sha0, "the base was modified");

    // G1 and the record on disk.
    let g1 = json(&["genome-verify", s(&fx.f0), s(&f1)]);
    assert_eq!(g1["pass"], true, "{g1}");
    let m = CmfModel::open(&f1).unwrap();
    let rec = &m.header.skills[0];
    assert_eq!(rec.id, SKILL_ID);
    assert_eq!(rec.kind.as_deref(), Some("lookup"));
    assert_eq!(rec.status.as_deref(), Some("quarantine"));
    assert!(rec.selection.is_none());
    let info = rec.lookup.as_ref().unwrap();
    assert_eq!((info.entries, info.keys), (3, 8));
    assert_eq!(info.key_norm, "cmf-key-v2");
    assert_eq!(rec.origin.as_ref().unwrap()["trigger"], "user_corpus");
    assert_eq!(rec.origin.as_ref().unwrap()["duplicates_dropped"], 3);
    assert_eq!(rec.origin.as_ref().unwrap()["duplicates_cross_entry"], 1);
    assert_eq!(rec.origin.as_ref().unwrap()["unreachable_keys_dropped"], 1);
    assert_eq!(rec.origin.as_ref().unwrap()["on_duplicate"], "first");
    assert_eq!(rec.bound.as_ref().unwrap().genome_id, "synth-genome");
    assert_eq!(m.header.lineage.last().unwrap().event, "skill_committed");
    assert_eq!(m.header.lineage.last().unwrap().detail["kind"], "lookup");
    for (leaf, dtype) in [
        (lookup_leaf::KEYS_HASH, TensorDtype::U64),
        (lookup_leaf::KEYS_ENTRY, TensorDtype::U32),
        (lookup_leaf::ENTRIES_OFF, TensorDtype::U64),
        (lookup_leaf::TEXT, TensorDtype::U8),
    ] {
        let t = m.tensor(&lookup_tensor_name(SKILL_ID, leaf)).unwrap();
        assert_eq!(t.dtype, dtype, "{leaf}");
    }
    // Every surviving key is found (any case / punctuation), duplicates
    // resolve to the first entry, the slot is the card.
    let (e, slots) = lookup(&m, "ABIES  balsamea").unwrap();
    assert_eq!(e, 0);
    let ru: serde_json::Value = serde_json::from_str(&slots[0]).unwrap();
    assert_eq!(ru["fields"]["family"], "Сосновые (Pinaceae)");
    let en: serde_json::Value = serde_json::from_str(&slots[1]).unwrap();
    assert_eq!(en["card"], "Balsam fir, a conifer of the family Pinaceae.");
    assert_eq!(lookup(&m, "balsam fir").unwrap().0, 0, "first entry wins");
    assert_eq!(lookup(&m, "Ромашка аптечная").unwrap().0, 1);
    assert_eq!(lookup(&m, "Рома́шка апте́чная").unwrap().0, 1, "stress marks fold");
    assert_eq!(lookup(&m, "chamomile").unwrap().0, 1);
    let (e, slots) = lookup(&m, "зверобой, продырявленный").unwrap();
    assert_eq!(e, 2);
    let en: serde_json::Value = serde_json::from_str(&slots[1]).unwrap();
    assert_eq!(en["card"], "", "a missing language is an empty card");
    assert!(lookup(&m, "hypericum").is_none(), "\"id\" is not a key");
    assert!(lookup(&m, "мята перечная").is_none());
    assert!(
        lookup(&m, "Matricaria chamomilla syn. Chamomilla recutita (L.) Rauschert").is_none(),
        "a key longer than the n-gram window is not stored"
    );
    drop(m);

    // ── route-fit ──
    let (sk_train, sk_fresh) = skill_prompts();
    let (gen_train, gen_fresh) = general_prompts();
    let (skp, genp) = (fx.dir.join("herbs-q.jsonl"), fx.dir.join("general-calib.jsonl"));
    let (ske, gene) = (fx.dir.join("herbs-eval.jsonl"), fx.dir.join("general-eval.jsonl"));
    write_prompts(&skp, &sk_train, "ru", "herbs-train");
    write_prompts(&genp, &gen_train, "en", "general-calib");
    write_prompts(&ske, &sk_fresh, "ru", "herbs-dev");
    write_prompts(&gene, &gen_fresh, "en", "general-eval");
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
    assert_eq!(r["phi"]["layer"], 0, "{r}");
    assert_eq!(r["phi"]["prefix_ids"][0], 256, "<|im_start|> opens the frame");
    assert_eq!(r["phi"]["suffix_ids"][0], 257, "<|im_end|> closes the user text");
    assert_eq!(r["skill_prompts"]["descriptor"]["n"], 50);
    assert_eq!(r["skill_prompts"]["descriptor"]["holdout"], 10);
    assert_eq!(r["skill_prompts"]["descriptor"]["rank"], 2);
    assert_eq!(r["general_prompts"]["descriptor"]["holdout"], 10);
    assert_eq!(r["n_in"], 10);
    assert_eq!(r["n_general"], 10);
    assert_eq!(r["in_scope_recall"], 1.0, "{r}");
    assert_eq!(r["false_accept"], 0.0, "{r}");
    assert_eq!(r["classes"], serde_json::json!([SKILL_ID]));
    assert_eq!(r["auto_routable"], false, "quarantine until skill-gate");
    assert_eq!(r["stale_regate"], serde_json::json!([]), "nothing was active");
    assert_eq!(r["status"], "quarantine");
    assert_eq!(r["measured"]["general_sha256"], sha_file(&genp));
    assert_eq!(r["measured"]["in_sha256"], sha_file(&skp));
    assert_eq!(r["measured"]["phi_backend"], "cpu", "{}", r["measured"]);
    assert_eq!(r["measured"]["cmf_gpu"], "0");
    assert_eq!(r["phi_backend"], "cpu");
    // The runtime replayed 8 held-out prompts of each class before the
    // write: same φ, same decision, same E.
    let rc = &r["runtime_check"];
    assert_eq!(rc["prompts"], 16, "{rc}");
    assert_eq!(rc["decisions_equal"], 16);
    assert!(rc["max_unit_phi_delta"].as_f64().unwrap() <= 1e-6, "{rc}");
    assert!(rc["max_e_delta"].as_f64().unwrap() <= 1e-6, "{rc}");
    assert_eq!(rc["phi_backend"], "cpu");
    assert!(rc["to_skill"].as_u64().unwrap() >= 4, "{rc}");
    assert!(r["novelty_theta"].as_f64().unwrap() > 0.0);
    assert!(r["temperature"].as_f64().unwrap() > 0.0);
    // Header-only append: bytes [128, old_len) unchanged, G1 holds.
    let old_len = r["old_len"].as_u64().unwrap() as usize;
    let after = std::fs::read(&f1).unwrap();
    assert_eq!(old_len, before.len());
    assert!(after.len() > old_len);
    assert_eq!(&after[128..old_len], &before[128..old_len]);
    assert_eq!(json(&["genome-verify", s(&fx.f0), s(&f1)])["pass"], true);
    let m = CmfModel::open(&f1).unwrap();
    let router = m.header.router.as_ref().expect("header.router");
    assert_eq!(router.policy, "backbone_gated");
    assert_eq!(router.phi.layer, 0);
    // `<|im_start|>` + the 5 bytes of "user\n" on the byte-level vocab.
    assert_eq!(router.phi.prefix_ids[0], 256);
    assert_eq!(router.phi.prefix_ids.len(), 6, "{:?}", router.phi.prefix_ids);
    assert_eq!(router.phi.suffix_ids[0], 257);
    assert_eq!(router.base.metric, "mse_unit");
    assert_eq!(router.base.holdout_n, Some(10));
    assert_eq!(
        router.skills_hash,
        format!("{:016x}", cortiq_engine::router::skills_hash(&m.header))
    );
    assert_eq!(router.measured.as_ref().unwrap()["fitted_skill"], SKILL_ID);
    let sel = m.header.skills[0].selection.as_ref().expect("selection");
    assert_eq!(sel.metric, "mse_unit");
    assert_eq!(sel.phi_layer, 0);
    assert_eq!(sel.rank, 2);
    assert_eq!(sel.holdout_n, Some(10));
    assert!(sel.err_mean.is_some() && sel.err_std.is_some());
    assert!(m.header.routing.is_some(), "header.routing");
    let ev = m.header.lineage.last().unwrap();
    assert_eq!(ev.event, "router_fitted");
    assert_eq!(ev.detail["id"], SKILL_ID);
    assert_eq!(ev.detail["skills_hash"], router.skills_hash);
    assert_eq!(m.header.skills[0].kind.as_deref(), Some("lookup"));
    assert_eq!(ev.detail["stale_regate"], serde_json::json!([]));
    assert_eq!(ev.detail["phi_backend"], "cpu");
    drop(m);

    // ── route-eval on disjoint sets (quarantine → the debug flag) ──
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
    assert_eq!(g["n"], 8, "{g}");
    assert_eq!(g["accepted"], 8, "a general prompt reached the lookup skill: {g}");
    assert_eq!(g["status"], "measured", "the summary is a gate file: {g}");
    assert_eq!(g["phi_backend"], "cpu");
    assert_eq!(g["phi_backend_fitted"], serde_json::Value::Null, "same backend: no warning");
    let k = json(&[
        "route-eval",
        s(&f1),
        "--prompts-jsonl",
        s(&ske),
        "--expect",
        SKILL_ID,
        "--include-quarantine",
        "--json",
    ]);
    let recall = k["in_scope_recall"].as_f64().unwrap();
    eprintln!(
        "route-eval fresh: general accepted {}/{}, in-scope recall {recall} ({})",
        g["accepted"], g["n"], k["route_counts"]
    );
    assert!(recall >= 0.5, "in-scope recall {recall} on fresh prompts: {k}");
    // The calibration set itself is refused as a gate set.
    let se = fails(&[
        "route-eval",
        s(&f1),
        "--prompts-jsonl",
        s(&genp),
        "--expect",
        "backbone",
        "--include-quarantine",
        "--json",
    ]);
    assert!(se.contains("calibration set"), "{se}");

    // ── skill-gate: a vacuous route-eval (no --include-quarantine on a
    // quarantined record) is a gate file with status "vacuous" — it can
    // never activate the record; a gate without "measured" is refused
    // for --status active unless --allow-unmeasured ──
    let out = cortiq(&["route-eval", s(&f1), "--prompts-jsonl", s(&gene), "--expect", "backbone", "--json"]);
    assert!(!out.status.success(), "a vacuous route-eval exited 0");
    let vac: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(vac["status"], "vacuous", "{vac}");
    assert_eq!(vac["vacuous"], true);
    let gate_vac = fx.dir.join("gate-vacuous.json");
    std::fs::write(&gate_vac, serde_json::to_string(&vac).unwrap()).unwrap();
    let len_before = std::fs::metadata(&f1).unwrap().len();
    let se = fails(&["skill-gate", s(&f1), "--id", SKILL_ID, "--gate", s(&gate_vac), "--status", "active"]);
    assert!(se.contains("gate.status is \"vacuous\"") && se.contains("--allow-unmeasured"), "{se}");
    assert_eq!(std::fs::metadata(&f1).unwrap().len(), len_before, "a refused skill-gate touched the file");
    let gate_none = fx.dir.join("gate-none.json");
    std::fs::write(&gate_none, r#"{"note": "no status"}"#).unwrap();
    let se = fails(&["skill-gate", s(&f1), "--id", SKILL_ID, "--gate", s(&gate_none), "--status", "active"]);
    assert!(se.contains("gate.status is absent"), "{se}");
    assert_eq!(std::fs::metadata(&f1).unwrap().len(), len_before);
    // --allow-unmeasured commits the state and says it is not routable.
    let sg = json(&["skill-gate", s(&f1), "--id", SKILL_ID, "--gate", s(&gate_none), "--status", "active", "--allow-unmeasured"]);
    assert_eq!(sg["status_to"], "active");
    assert_eq!(sg["gate_measured"], false);
    assert_eq!(sg["auto_routable"], false, "{sg}");
    assert!(!CmfModel::open(&f1).unwrap().header.skills[0].is_auto_routable());

    // ── skill-gate active with the measured route-eval summary as the
    // gate file, as the driver does → auto-routable, calibration kept ──
    let gate = fx.dir.join("gate.json");
    std::fs::write(&gate, serde_json::to_string(&g).unwrap()).unwrap();
    let sg = json(&[
        "skill-gate",
        s(&f1),
        "--id",
        SKILL_ID,
        "--gate",
        s(&gate),
        "--status",
        "active",
    ]);
    assert_eq!(sg["gate_measured"], true, "{sg}");
    assert_eq!(sg["auto_routable"], true, "{sg}");
    assert_eq!(sg["calibration_stale"], false);
    let m = CmfModel::open(&f1).unwrap();
    assert!(m.header.skills[0].is_auto_routable());
    assert_eq!(m.header.skills[0].gate.as_ref().unwrap()["status"], "measured");
    drop(m);
    assert_eq!(json(&["genome-verify", s(&fx.f0), s(&f1)])["pass"], true);
    let g = json(&[
        "route-eval",
        s(&f1),
        "--prompts-jsonl",
        s(&gene),
        "--expect",
        "backbone",
        "--json",
    ]);
    assert_eq!(g["accepted"], 8, "{g}");
    assert_eq!(g["vacuous"], false);
    assert_eq!(g["status"], "measured");

    // ── a second route-fit on the ACTIVE record: new descriptors and
    // calibration, so its measured gate belongs to another decision
    // surface — the record goes to stale_regate and is not auto-routable
    // until re-gated; the event names it ──
    let r2 = json(&[
        "route-fit",
        s(&f1),
        "--id",
        SKILL_ID,
        "--skill-prompts",
        s(&skp),
        "--general-prompts",
        s(&genp),
        "--rank",
        "3",
    ]);
    assert_eq!(r2["stale_regate"], serde_json::json!([SKILL_ID]), "{r2}");
    assert_eq!(r2["status"], "stale_regate");
    assert_eq!(r2["auto_routable"], false);
    assert_eq!(r2["phi"]["layer"], 0, "the file's φ layer is kept without --phi-layer");
    let m = CmfModel::open(&f1).unwrap();
    assert_eq!(m.header.skills[0].status.as_deref(), Some("stale_regate"));
    assert!(!m.header.skills[0].is_auto_routable());
    assert_eq!(m.header.skills[0].selection.as_ref().unwrap().rank, 3);
    let ev = m.header.lineage.last().unwrap();
    assert_eq!(ev.event, "router_fitted");
    assert_eq!(ev.detail["stale_regate"], serde_json::json!([SKILL_ID]));
    assert!(router_ok(&m), "the new calibration binds");
    drop(m);
    // Without the debug flag no prompt can reach it (vacuous); with it the
    // fresh general set is measured, and skill-gate re-activates.
    let out = cortiq(&["route-eval", s(&f1), "--prompts-jsonl", s(&gene), "--expect", "backbone", "--json"]);
    assert!(!out.status.success());
    let g = json(&["route-eval", s(&f1), "--prompts-jsonl", s(&gene), "--expect", "backbone", "--include-quarantine", "--json"]);
    assert_eq!(g["status"], "measured");
    assert_eq!(g["accepted"], 8, "{g}");
    std::fs::write(&gate, serde_json::to_string(&g).unwrap()).unwrap();
    let sg = json(&["skill-gate", s(&f1), "--id", SKILL_ID, "--gate", s(&gate), "--status", "active"]);
    assert_eq!(sg["status_from"], "stale_regate");
    assert_eq!(sg["auto_routable"], true, "{sg}");
    assert_eq!(json(&["genome-verify", s(&fx.f0), s(&f1)])["pass"], true);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

fn router_ok(m: &CmfModel) -> bool {
    m.header.router.as_ref().is_some_and(|r| {
        r.skills_hash == format!("{:016x}", cortiq_engine::router::skills_hash(&m.header))
    })
}

#[test]
fn lookup_build_and_route_fit_refuse_bad_inputs_and_leave_files_alone() {
    let fx = fixture("refuse");
    let sha0 = sha_file(&fx.f0);
    let f1 = fx.dir.join("f1.cmf");

    // --out == base: the same path, a hard link, a symlink — the base's
    // inode would be truncated by the copy; all three are refused with
    // the base byte-identical.
    let se = fails(&build_args(s(&fx.f0), s(&fx.entries), s(&fx.f0), &[]));
    assert!(se.contains("COPY"), "{se}");
    assert_eq!(sha_file(&fx.f0), sha0);
    let link = fx.dir.join("link.cmf");
    std::fs::hard_link(&fx.f0, &link).unwrap();
    let se = fails(&build_args(s(&fx.f0), s(&fx.entries), s(&link), &[]));
    assert!(se.contains("hard link"), "{se}");
    assert_eq!(sha_file(&fx.f0), sha0, "the hard link's inode was truncated");
    assert_eq!(sha_file(&link), sha0);
    std::fs::remove_file(&link).unwrap();
    let sym = fx.dir.join("sym.cmf");
    std::os::unix::fs::symlink(&fx.f0, &sym).unwrap();
    let se = fails(&build_args(s(&fx.f0), s(&fx.entries), s(&sym), &[]));
    assert!(se.contains("COPY"), "{se}");
    assert_eq!(sha_file(&fx.f0), sha0);
    std::fs::remove_file(&sym).unwrap();
    // A read-only (sealed) base: the copy is made writable, the base stays
    // read-only and identical.
    let sealed = fx.dir.join("sealed.cmf");
    std::fs::copy(&fx.f0, &sealed).unwrap();
    let mut perm = std::fs::metadata(&sealed).unwrap().permissions();
    perm.set_readonly(true);
    std::fs::set_permissions(&sealed, perm).unwrap();
    let b = json(&build_args(s(&sealed), s(&fx.entries), s(&f1), &[]));
    assert_eq!(b["keys"], 8, "{b}");
    assert!(std::fs::metadata(&sealed).unwrap().permissions().readonly());
    assert_eq!(sha_file(&sealed), sha0);
    assert!(!std::fs::metadata(&f1).unwrap().permissions().readonly(), "the copy stays writable");
    assert_eq!(CmfModel::open(&f1).unwrap().header.skills[0].id, SKILL_ID);
    // No --policy: the field stays out of the header (= router_and_key).
    assert_eq!(b["policy"], "router_and_key", "{b}");
    assert_eq!(CmfModel::open(&f1).unwrap().header.skills[0].lookup.as_ref().unwrap().policy, None);
    std::fs::remove_file(&f1).unwrap();
    // --policy key_first lands in lookup.policy; an unknown value is
    // refused before anything is written.
    let se = fails(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--policy", "always"]));
    assert!(se.contains("router_and_key | key_first"), "{se}");
    assert!(!f1.exists());
    let b = json(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--policy", "key_first"]));
    assert_eq!(b["policy"], "key_first", "{b}");
    let m = CmfModel::open(&f1).unwrap();
    let li = m.header.skills[0].lookup.as_ref().unwrap();
    assert_eq!((li.policy.as_deref(), li.policy_label()), (Some("key_first"), "key_first"));
    drop(m);
    std::fs::remove_file(&f1).unwrap();
    // A row without keys: no output file is left behind.
    let bad = fx.dir.join("bad.jsonl");
    std::fs::write(&bad, "{\"ru\": {\"card\": \"x\"}}\n").unwrap();
    let se = fails(&[
        "lookup-build",
        s(&fx.f0),
        "--entries",
        s(&bad),
        "--id",
        SKILL_ID,
        "--out",
        s(&f1),
    ]);
    assert!(se.contains("\"keys\""), "{se}");
    assert!(!f1.exists());
    // A language object without a card; a card that is not a string.
    std::fs::write(&bad, "{\"keys\": [\"a\"], \"ru\": {\"fields\": {}}}\n").unwrap();
    assert!(fails(&["lookup-build", s(&fx.f0), "--entries", s(&bad), "--id", SKILL_ID, "--out", s(&f1)]).contains("card"));
    std::fs::write(&bad, "{\"keys\": [\"a\"], \"ru\": {\"card\": 5}}\n").unwrap();
    assert!(fails(&["lookup-build", s(&fx.f0), "--entries", s(&bad), "--id", SKILL_ID, "--out", s(&f1)]).contains("card"));
    // Only empty keys; only unreachable keys.
    std::fs::write(&bad, "{\"keys\": [\"!!!\"], \"ru\": {\"card\": \"x\"}}\n").unwrap();
    assert!(fails(&["lookup-build", s(&fx.f0), "--entries", s(&bad), "--id", SKILL_ID, "--out", s(&f1)]).contains("no key survives"));
    std::fs::write(&bad, "{\"keys\": [\"one two three four five\"], \"ru\": {\"card\": \"x\"}}\n").unwrap();
    let se = fails(&["lookup-build", s(&fx.f0), "--entries", s(&bad), "--id", SKILL_ID, "--out", s(&f1)]);
    assert!(se.contains("no key survives") && se.contains("1 unreachable"), "{se}");
    // A duplicate language in --langs is refused; a bad --on-duplicate too.
    assert!(fails(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--langs", "ru,ru"])).contains("duplicate"));
    assert!(fails(&["lookup-build", s(&fx.f0), "--entries", s(&fx.entries), "--id", SKILL_ID, "--out", s(&f1), "--on-duplicate", "any"]).contains("first | last | error"));
    assert!(!f1.exists());
    // A dotted id is refused by the core writer, the output removed.
    let se = fails(&["lookup-build", s(&fx.f0), "--entries", s(&fx.entries), "--id", "a.b", "--out", s(&f1), "--on-duplicate", "first"]);
    assert!(se.contains("no '.'"), "{se}");
    assert!(!f1.exists(), "a refused append left the output behind");
    // A plain (non-genome) file cannot take a v2 record.
    let plain = fx.dir.join("plain.cmf");
    embryo_synth::write_synth_genome(&plain, &embryo_synth::SynthGeom::tiny_gdn_bounded());
    let se = fails(&build_args(s(&plain), s(&fx.entries), s(&f1), &[]));
    assert!(se.contains("not a sealed genome"), "{se}");
    assert!(!f1.exists());

    // --on-duplicate last: the LAST entry keeps a shared key; the first
    // entry keeps its other keys. The winner is reported.
    let b = json(&["lookup-build", s(&fx.f0), "--entries", s(&fx.entries), "--id", SKILL_ID, "--out", s(&f1), "--on-duplicate", "last"]);
    assert_eq!(b["on_duplicate"], "last");
    assert_eq!(b["keys"], 8, "{b}");
    assert_eq!(b["duplicates_cross_entry"], 1);
    let cross = b["duplicate_keys"].as_array().unwrap().iter().find(|d| d["same_entry"] == false).unwrap();
    assert_eq!(cross["kept_entry"], 2, "{cross}");
    let m = CmfModel::open(&f1).unwrap();
    assert_eq!(lookup(&m, "balsam fir").unwrap().0, 2, "last entry wins");
    assert_eq!(lookup(&m, "Abies balsamea").unwrap().0, 0);
    assert!(b["unreachable_entries"].as_array().unwrap().is_empty());
    drop(m);
    // --drop-keys: a stop list of generic words; the entry that had only
    // stop-listed keys becomes unreachable and is reported.
    let stop = fx.dir.join("stop.txt");
    std::fs::write(&stop, "# generic words\nChamomile\n\n  balsam fir \nЗверобой продырявленный\nHypericum, perforatum!\n").unwrap();
    let b = json(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--drop-keys", s(&stop)]));
    assert_eq!(b["drop_list"], serde_json::json!({"file": s(&stop), "keys": 4}), "{b}");
    // Entries 0 and 1 keep two keys each; entry 2 keeps none.
    assert_eq!(b["keys"], 4, "{b}");
    // 4 listed keys, 5 drops: `balsam fir` is stop-listed in both entries
    // (the list is checked before the duplicate rule).
    assert_eq!(b["dropped_by_list"].as_array().unwrap().len(), 5);
    assert_eq!(b["duplicates_cross_entry"], 0);
    assert_eq!(b["unreachable_entries"], serde_json::json!([2]));
    let m = CmfModel::open(&f1).unwrap();
    assert!(lookup(&m, "chamomile").is_none());
    assert!(lookup(&m, "balsam fir").is_none());
    assert_eq!(lookup(&m, "Abies balsamea").unwrap().0, 0);
    drop(m);
    // --probe-prompts: the built table is probed with a prompt set; a key
    // hit by more than --suspicious-share of the prompts is reported.
    let probe = fx.dir.join("probe.jsonl");
    let mut rows: Vec<String> = (0..6).map(|i| format!("{{\"prompt\": \"What is chamomile good for, take {i}?\"}}")).collect();
    rows.push("{\"prompt\": \"Tell me about balsam fir.\"}".into());
    rows.extend((0..3).map(|i| format!("{{\"prompt\": \"Unrelated question number {i}.\"}}")));
    std::fs::write(&probe, rows.join("\n") + "\n").unwrap();
    let b = json(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--probe-prompts", s(&probe), "--suspicious-share", "0.25"]));
    let p = &b["probe"];
    assert_eq!(p["prompts"], 10, "{p}");
    assert_eq!(p["hits"], 7);
    assert_eq!(b["suspicious_keys"], 1, "{p}");
    assert_eq!(p["suspicious_keys"][0]["norm"], "chamomile");
    assert_eq!(p["suspicious_keys"][0]["hits"], 6);
    assert_eq!(p["suspicious_keys"][0]["entry"], 1);
    let m = CmfModel::open(&f1).unwrap();
    assert_eq!(m.header.skills[0].origin.as_ref().unwrap()["probe"]["hits"], 7);
    drop(m);
    assert!(fails(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--probe-prompts", s(&probe), "--suspicious-share", "1.5"])).contains("--suspicious-share"));

    // Without --langs the languages are sorted by name; the output is
    // overwritten by the next build.
    let b = json(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &[]));
    assert_eq!(b["langs"], serde_json::json!(["en", "ru"]), "{b}");
    // A good build, then --langs restricts the slots and the id is taken.
    let b = json(&build_args(s(&fx.f0), s(&fx.entries), s(&f1), &["--langs", "en"]));
    assert_eq!(b["langs"], serde_json::json!(["en"]), "{b}");
    assert_eq!(b["ignored_slots"], 3, "three ru cards not stored");
    assert_eq!(b["missing_slots"], 1);
    assert_eq!(CmfModel::open(&f1).unwrap().header.skills[0].lookup.as_ref().unwrap().langs, vec!["en"]);
    let se = fails(&build_args(s(&f1), s(&fx.entries), s(&fx.dir.join("f2.cmf")), &[]));
    assert!(se.contains("already exists"), "{se}");
    assert!(!fx.dir.join("f2.cmf").exists());

    // route-fit refusals: unknown skill, φ layer out of range, too few
    // prompts, the same file for both classes — the file is untouched.
    let sha1 = sha_file(&f1);
    let (sk, _) = skill_prompts();
    let (general, _) = general_prompts();
    let (skp, genp) = (fx.dir.join("sk.jsonl"), fx.dir.join("gen.jsonl"));
    write_prompts(&skp, &sk, "ru", "x");
    write_prompts(&genp, &general, "en", "y");
    let base = |id: &str, extra: &[&str]| {
        let mut a = vec![
            "route-fit".to_string(),
            s(&f1).to_string(),
            "--id".into(),
            id.to_string(),
            "--skill-prompts".into(),
            s(&skp).to_string(),
            "--general-prompts".into(),
            s(&genp).to_string(),
        ];
        a.extend(extra.iter().map(|x| x.to_string()));
        a
    };
    let run = |a: Vec<String>| {
        let v: Vec<&str> = a.iter().map(String::as_str).collect();
        fails(&v)
    };
    assert!(run(base("nope", &["--phi-layer", "0"])).contains("not in"));
    assert!(run(base(SKILL_ID, &["--phi-layer", "2"])).contains("has 2 layers"));
    assert!(run(base(SKILL_ID, &[])).contains("--phi-layer is required"));
    assert!(run(base(SKILL_ID, &["--phi-layer", "0", "--max", "3"])).contains("at least 5"));
    let same = base(SKILL_ID, &["--phi-layer", "0"]);
    let mut same_v: Vec<&str> = same.iter().map(String::as_str).collect();
    same_v[7] = same_v[5];
    assert!(fails(&same_v).contains("same file"));
    assert_eq!(sha_file(&f1), sha1, "a refused route-fit touched the file");
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// Common names that are also general phrases (review KF-1): the key
/// `running pop` of *Passiflora foetida* and `scrambled eggs` of
/// *Corydalis aurea* are real names — and `key_first` would answer
/// "Pop!_OS" and omelette questions from the table. A numbered key
/// (`STS-135`) is never strong; a Russian disease name reaches a strong
/// key through the stems.
const GENERAL_PHRASE_ENTRIES: &str = r#"{"keys": ["Corydalis aurea", "scrambled eggs", "golden corydalis"], "en": {"card": "Scrambled eggs, golden corydalis (Corydalis aurea), family Papaveraceae.", "fields": {"family": "Papaveraceae"}}}
{"keys": ["Passiflora foetida", "running pop", "stinking passionflower"], "en": {"card": "Stinking passionflower, running pop (Passiflora foetida).", "fields": {"family": "Passifloraceae"}}}
{"keys": ["Ромашка аптечная", "Matricaria chamomilla", "chamomile"], "ru": {"card": "Ромашка аптечная (Matricaria chamomilla) — семейство Астровые.", "fields": {"family": "Астровые"}}, "en": {"card": "Chamomile (Matricaria chamomilla), family Asteraceae.", "fields": {"family": "Asteraceae"}}}
{"keys": ["STS-135"], "en": {"card": "STS-135, a synthetic cannabinoid.", "fields": {}}}
{"keys": ["жёлтая лихорадка"], "ru": {"card": "Жёлтая лихорадка — вирусная болезнь (растение не описано).", "fields": {}}}
"#;

#[test]
fn general_probe_reports_strong_keys_and_writes_a_stop_list() {
    let fx = fixture("general");
    let entries = fx.dir.join("general-phrases.jsonl");
    std::fs::write(&entries, GENERAL_PHRASE_ENTRIES).unwrap();
    let general = fx.dir.join("general.jsonl");
    write_prompts(
        &general,
        &[
            "How do I make scrambled eggs fluffy?".to_string(),
            "how to trouble shoot USB port on a laptop running Pop!_OS 22.04 LTS?".to_string(),
            "Tell me about the STS-135 mission of Space Shuttle Atlantis".to_string(),
            "What is the capital of France?".to_string(),
            "Scrambled eggs or an omelette for breakfast?".to_string(),
            "Нужна ли прививка от жёлтой лихорадки?".to_string(),
            "Какое семейство у ромашки?".to_string(),
        ],
        "en",
        "general",
    );
    let (f1, f2, stop) = (fx.dir.join("f1.cmf"), fx.dir.join("f2.cmf"), fx.dir.join("stop.txt"));
    // key_first without a general probe: said.
    let (_, se) = ok(&["lookup-build", s(&fx.f0), "--entries", s(&entries), "--id", SKILL_ID, "--out", s(&f1), "--policy", "key_first"]);
    assert!(se.contains("without --general-prompts"), "{se}");
    // --general-stop-out alone is refused before anything is written.
    std::fs::remove_file(&f1).unwrap();
    let se = fails(&["lookup-build", s(&fx.f0), "--entries", s(&entries), "--id", SKILL_ID, "--out", s(&f1), "--general-stop-out", s(&stop)]);
    assert!(se.contains("--general-stop-out needs --general-prompts"), "{se}");
    assert!(!f1.exists() && !stop.exists());

    let (so, se) = ok(&[
        "lookup-build", s(&fx.f0), "--entries", s(&entries), "--id", SKILL_ID, "--out", s(&f1),
        "--policy", "key_first", "--general-prompts", s(&general), "--general-stop-out", s(&stop),
    ]);
    let b: serde_json::Value = serde_json::from_str(&so).unwrap();
    let g = &b["general_probe"];
    assert_eq!((g["prompts"].as_u64(), g["strong_hits"].as_u64(), g["distinct_keys"].as_u64()), (Some(7), Some(4), Some(3)), "{g}");
    assert_eq!(g["sha256"], sha_file(&general));
    let keys = g["keys"].as_array().unwrap();
    assert_eq!((keys[0]["key"].as_str(), keys[0]["hits"].as_u64(), keys[0]["entry"].as_u64()), (Some("scrambled eggs"), Some(2), Some(0)), "{g}");
    let by_key = |k: &str| keys.iter().find(|x| x["key"] == k).unwrap_or_else(|| panic!("{k} not in {g}"));
    assert_eq!(by_key("running pop")["entry"], 1);
    let fever = by_key("жёлт лихорадк");
    assert_eq!((fever["via"].as_str(), fever["entry"].as_u64()), (Some("stem"), Some(4)), "{fever}");
    assert_eq!(fever["stored_keys"], serde_json::json!(["жёлтая лихорадка"]));
    assert!(keys.iter().all(|k| k["key"] != "sts 135"), "a numbered key is never strong: {g}");
    assert!(se.contains("general probe") && se.contains("scrambled eggs"), "{se}");
    let text = std::fs::read_to_string(&stop).unwrap();
    let listed: Vec<&str> = text.lines().filter(|l| !l.starts_with('#')).collect();
    assert_eq!(listed, vec!["scrambled eggs", "running pop", "жёлтая лихорадка"], "{text}");
    assert_eq!(g["stop_out"]["keys"], 3);

    // The stop list removes exactly those keys; the rebuilt table answers
    // no general prompt under key_first, and keeps its Latin names.
    let b2 = json(&[
        "lookup-build", s(&fx.f0), "--entries", s(&entries), "--id", SKILL_ID, "--out", s(&f2),
        "--policy", "key_first", "--drop-keys", s(&stop), "--general-prompts", s(&general),
    ]);
    assert_eq!(b2["general_probe"]["strong_hits"], 0, "{b2}");
    assert_eq!(b2["dropped_by_list"].as_array().unwrap().len(), 3);
    assert_eq!(b2["keys"].as_u64().unwrap(), b["keys"].as_u64().unwrap() - 3);
    let m = CmfModel::open(&f2).unwrap();
    assert!(lookup(&m, "Corydalis aurea").is_some() && lookup(&m, "scrambled eggs").is_none());
    let _ = std::fs::remove_dir_all(&fx.dir);
}
