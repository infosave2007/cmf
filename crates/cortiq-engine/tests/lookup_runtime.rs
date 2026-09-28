//! `lookup` records in the RUNTIME (spec §9.5.2; lookup spec §3): the
//! table read from a mounted record, key extraction on the real
//! [`SKILL_TEXTS`] (Cyrillic n-grams, a Latin binomial in parentheses,
//! misses), field / language selection, the routed outcome under every
//! mode, and the no-forgetting guarantee — with the record mounted, a
//! prompt without a key yields logits bit-identical to F0's. The routing
//! policy `key_first`: a strong key takes a request the φ router sent to
//! the backbone, a one-word key and a pinned decision never do. CPU
//! (`CMF_GPU=0`), synthetic GDN + bounded genome.

#[path = "common/embryo_synth.rs"]
mod embryo_synth;
#[path = "common/knowledge_synth.rs"]
mod knowledge_synth;

use cortiq_core::CmfModel;
use cortiq_engine::lookup::{
    self, CONTEXT_HEADER, DecidedBy, KeyFirstGate, KeySource, LookupMode, LookupOutcome,
    LookupPolicy, LookupTable, LookupTables, MatchVia,
};
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::router::{self, RouteDecision, RouteOptions, RouteTarget};
use cortiq_engine::sampler::SamplerConfig;
use embryo_synth::SynthGeom;
use knowledge_synth::{GENERAL_TEXTS, SKILL_ID, SKILL_TEXTS, lookup_entries};
use std::path::PathBuf;
use std::sync::Arc;

/// A Cyrillic question the router may send to the table, with no key in
/// it: the backbone must run it unchanged.
const NO_KEY_TEXT: &str = "Какие лечебные свойства у мяты перечной?";

fn setup(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    // SAFETY: set before any pipeline of this binary exists; every test
    // sets the same value.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let dir = std::env::temp_dir().join(format!(
        "cmf-lookup-runtime-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let files = knowledge_synth::write_lookup_pair(&dir, &SynthGeom::tiny_gdn_bounded(), "active");
    (dir, files.f0, files.f1)
}

fn field<'a>(v: &'a serde_json::Value, name: &str) -> &'a str {
    v["fields"][name].as_str().unwrap()
}

#[test]
fn table_build_and_answers() {
    let (dir, f0, f1) = setup("table");
    let entries = lookup_entries();
    let model = Arc::new(CmfModel::open(&f1).unwrap());
    assert!(LookupTable::is_lookup(&model, SKILL_ID));
    assert_eq!(LookupTable::lookup_ids(&model), vec![SKILL_ID.to_string()]);
    let table = LookupTable::open(&model, SKILL_ID).unwrap();
    let n_keys: usize = entries.iter().map(|(k, _, _)| k.len()).sum();
    assert_eq!(table.keys(), n_keys);
    assert_eq!(table.entries(), entries.len());
    assert_eq!(table.langs(), &["ru".to_string(), "en".to_string()]);
    assert_eq!(table.fields(), &["family", "uses", "safety"]);
    assert!(!table.blob().is_empty());
    let slot: serde_json::Value = serde_json::from_str(table.slot_text(4, 0).unwrap()).unwrap();
    assert_eq!(slot, entries[4].1);
    assert!(table.slot_text(6, 0).is_none(), "entry beyond E");
    assert!(table.slot_text(0, 2).is_none(), "lang beyond L");

    // Every skill question names its own entry (Cyrillic n-grams in the
    // question's declension).
    for (i, t) in SKILL_TEXTS.iter().enumerate() {
        let hit = table.find_key(t).unwrap_or_else(|| panic!("no key in {t:?}"));
        assert_eq!(hit.entry as usize, i, "{t:?} → {hit:?}");
        assert_eq!((hit.source, hit.via, hit.turn), (KeySource::Ngram, MatchVia::Exact, 0));
        assert_eq!(hit.hash, cortiq_core::key_hash(&hit.key));
    }
    // The stem index: the key texts the cards spell (every key but the
    // two declined spellings `ромашки аптечной` and `календулы`, which no
    // card writes), stemmed; no stem names two entries.
    assert_eq!(table.recovered_keys(), n_keys - 2);
    assert_eq!((table.stems().keys_in, table.stems().ambiguous), (n_keys - 2, 0));
    assert_eq!(table.stems().len(), n_keys - 2);
    assert_eq!(table.find_stem_hash(cortiq_core::key_hash("шалф лекарственн")), Some(2));
    assert_eq!(table.find_stem_hash(cortiq_core::key_hash("календул")), Some(4));
    assert_eq!(table.find_stem_hash(cortiq_core::key_hash("st john s wort")), Some(1));
    // An inflected name no key spells answers through the stems; the
    // exact index is still primary (`календулы` is an exact key).
    let hit = table.find_key("Чем полезен отвар шалфея лекарственного?").unwrap();
    assert_eq!((hit.entry, hit.via, hit.words), (2, MatchVia::Stem, 2));
    assert_eq!((hit.key.as_str(), hit.stem.as_deref()), ("шалфея лекарственного", Some("шалф лекарственн")));
    let hit = table.find_key("Корень валерианы лекарственной — как принимать?").unwrap();
    assert_eq!((hit.entry, hit.via), (3, MatchVia::Stem));
    let hit = table.find_key("Настой календулы").unwrap();
    assert_eq!((hit.entry, hit.via), (4, MatchVia::Exact));
    let hit = table.find_key("Cream with calendulas").unwrap();
    assert_eq!((hit.entry, hit.via, hit.stem.as_deref()), (4, MatchVia::Stem, Some("calendula")));
    // A binomial without parentheses, anywhere in the message.
    let hit = table.find_key("Is Salvia officinalis safe in pregnancy?").unwrap();
    assert_eq!((hit.entry, hit.source, hit.via), (2, KeySource::Binomial, MatchVia::Exact));
    assert!(table.find_key("Is Salvia safe?").is_none(), "a genus alone is no key");
    // Conversation memory: the key from the most recent turn that holds
    // one, the field and the language from the last turn.
    match table.find_answer_turns(&["Какое семейство у этого растения?", "Спасибо.", SKILL_TEXTS[2]]).unwrap() {
        lookup::Found::Answer(a) => {
            assert_eq!((a.key.entry, a.key.turn), (2, 2));
            assert_eq!((a.lang.as_str(), a.field.as_deref()), ("ru", Some("family")));
            assert_eq!(a.text, field(&entries[2].1, "family"));
        }
        other => panic!("{other:?}"),
    }
    match table.find_answer_turns(&["What family is it in?", SKILL_TEXTS[2]]).unwrap() {
        lookup::Found::Answer(a) => {
            assert_eq!((a.key.entry, a.key.turn, a.lang.as_str()), (2, 1, "en"));
            assert_eq!(a.text, field(&entries[2].2, "family"));
        }
        other => panic!("{other:?}"),
    }
    match table.find_answer_turns(&[SKILL_TEXTS[4], SKILL_TEXTS[2]]).unwrap() {
        lookup::Found::Answer(a) => assert_eq!((a.key.entry, a.key.turn), (4, 0), "the latest plant wins"),
        other => panic!("{other:?}"),
    }
    assert_eq!(table.find_answer_turns(&["Какое семейство у этого растения?", NO_KEY_TEXT]).unwrap(), lookup::Found::NoKey);
    assert_eq!(table.find_answer_turns(&[]).unwrap(), lookup::Found::NoKey);
    // Field by keywords, language by script.
    let a = table.answer(SKILL_TEXTS[4]).unwrap().unwrap();
    assert_eq!(a.key.key, "календулы");
    assert_eq!((a.lang.as_str(), a.field.as_deref()), ("ru", Some("family")));
    assert_eq!(a.text, field(&entries[4].1, "family"));
    let card4 = entries[4].1["card"].as_str().unwrap();
    assert_eq!(a.full_card, card4);
    // With a field selected, `context` prepends the card's first
    // sentence and the field — not the whole card.
    assert_eq!(
        a.card,
        format!("{card4}\nСемейство: {}", field(&entries[4].1, "family")),
        "the one-sentence synthetic card is its own first sentence"
    );
    let a = table.answer("What family is calendula in?").unwrap().unwrap();
    assert_eq!((a.lang.as_str(), a.field.as_deref()), ("en", Some("family")));
    assert_eq!(a.text, field(&entries[4].2, "family"));
    let a = table.answer("Is sage safe?").unwrap().unwrap();
    assert_eq!((a.key.entry, a.field.as_deref()), (2, Some("safety")));
    assert_eq!(a.text, field(&entries[2].2, "safety"));
    // No field named → the whole card, in `context` too.
    let a = table.answer(SKILL_TEXTS[0]).unwrap().unwrap();
    assert_eq!((a.key.entry, a.field.as_deref()), (0, None));
    assert_eq!(a.text, entries[0].1["card"].as_str().unwrap());
    assert_eq!(a.card, a.full_card);
    assert_eq!(
        table.find_answer(NO_KEY_TEXT).unwrap(),
        lookup::Found::NoKey
    );
    // A field the card lacks → the whole card too.
    let a = table.answer("Какие части календулы используют?").unwrap().unwrap();
    assert_eq!((a.key.entry, a.field.as_deref()), (4, None));
    // The Latin binomial in parentheses resolves a form no key spells.
    let a = table
        .answer("Что известно о растении ноготки (Calendula officinalis)?")
        .unwrap()
        .unwrap();
    assert_eq!((a.key.entry, a.key.source), (4, KeySource::Parenthesised));
    assert_eq!(a.key.key, "calendula officinalis");
    // Misses.
    assert!(table.find_key(GENERAL_TEXTS[0]).is_none());
    assert!(table.find_key(NO_KEY_TEXT).is_none());
    assert!(table.answer("").unwrap().is_none());

    // The registry: opened once, `None` for anything that is not a lookup.
    let tables = LookupTables::new(model.clone());
    assert!(tables.any());
    assert_eq!(tables.ids(), vec![SKILL_ID.to_string()]);
    let t1 = tables.get(SKILL_ID).unwrap().expect("table");
    let t2 = tables.get(SKILL_ID).unwrap().expect("table");
    assert!(Arc::ptr_eq(&t1, &t2));
    assert!(tables.get("ghost").unwrap().is_none());
    let f0m = Arc::new(CmfModel::open(&f0).unwrap());
    assert!(!LookupTables::new(f0m.clone()).any());
    assert!(!LookupTable::is_lookup(&f0m, SKILL_ID));
    assert!(LookupTable::open(&f0m, SKILL_ID).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn routed_outcome_under_every_mode() {
    let (dir, _f0, f1) = setup("route");
    let entries = lookup_entries();
    let model = Arc::new(CmfModel::open(&f1).unwrap());
    let mut probe = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
    let tables = LookupTables::new(model.clone());
    let opts = RouteOptions::default();

    // Skill questions: the router picks the table, the table answers.
    for (i, t) in SKILL_TEXTS.iter().enumerate() {
        let d = router::route_request_with(&model, &mut probe, t, opts);
        assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()), "{t:?}: {}", d.reason);
        let (d, o) = lookup::resolve_lookup(&tables, d, t, LookupMode::Answer).unwrap();
        assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
        assert!(d.reason.contains("lookup key"), "{}", d.reason);
        let a = match &o {
            LookupOutcome::Answer(a) => a,
            other => panic!("{t:?}: {other:?}"),
        };
        assert_eq!(a.key.entry as usize, i);
        assert!(o.is_hit());
        assert_eq!(o.generation_text(t), *t, "answer mode leaves the text alone");
        let mut s = d.summary_json();
        o.annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["target"], SKILL_ID);
        assert_eq!(s["lookup_hit"], true);
        assert_eq!(s["lookup_entry"], i);
        assert_eq!(s["lookup_mode"], "answer");
    }
    // General questions: the backbone, nothing looked up.
    let d = router::route_request_with(&model, &mut probe, GENERAL_TEXTS[0], opts);
    assert_eq!(d.target, RouteTarget::Backbone, "{}", d.reason);
    let reason = d.reason.clone();
    let (d, o) = lookup::resolve_lookup(&tables, d, GENERAL_TEXTS[0], LookupMode::Answer).unwrap();
    assert_eq!(o, LookupOutcome::NotLookup);
    assert_eq!(d.reason, reason);
    // A decision for the table with no key in the message: the BACKBONE,
    // the reason kept.
    let forced = || RouteDecision::forced(RouteTarget::Skill(SKILL_ID.into()), "forced");
    let (d, o) = lookup::resolve_lookup(&tables, forced(), NO_KEY_TEXT, LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Backbone);
    assert_eq!(o, LookupOutcome::Miss { id: SKILL_ID.into() });
    assert!(d.reason.contains("no key") && d.reason.contains("forced"), "{}", d.reason);
    assert!(o.describe().unwrap().contains("no key"));
    // With memory: the same message answers from the plant named two
    // turns back; the summary says how far.
    let (d, o) = lookup::resolve_lookup_turns(
        &tables,
        forced(),
        &[NO_KEY_TEXT, "Спасибо.", SKILL_TEXTS[2]],
        LookupMode::Answer,
    )
    .unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
    assert!(d.reason.contains("2 turns back"), "{}", d.reason);
    let a = o.hit().unwrap();
    assert_eq!((a.key.entry, a.key.turn, a.field.as_deref()), (2, 2, None), "no field named: the whole card");
    let mut s = d.summary_json();
    o.annotate(&mut s, LookupMode::Answer);
    assert_eq!((s["lookup_turn"].as_u64(), s["lookup_match"].as_str()), (Some(2), Some("exact")));
    assert!(o.describe().unwrap().contains("2 turns back"));
    // Mode off: the backbone even with a key.
    let (d, o) = lookup::resolve_lookup(&tables, forced(), SKILL_TEXTS[4], LookupMode::Off).unwrap();
    assert_eq!(d.target, RouteTarget::Backbone);
    assert_eq!(o, LookupOutcome::Off { id: SKILL_ID.into() });
    assert!(d.reason.contains("lookup mode off"), "{}", d.reason);
    // Mode context: the target stays, the backbone gets the card in front.
    let (d, o) =
        lookup::resolve_lookup(&tables, forced(), SKILL_TEXTS[4], LookupMode::Context).unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
    let LookupOutcome::Context(a) = &o else {
        panic!("{o:?}");
    };
    assert_eq!(a.text, field(&entries[4].1, "family"));
    let g = o.generation_text(SKILL_TEXTS[4]);
    assert!(g.starts_with(CONTEXT_HEADER), "{g}");
    assert!(g.contains(entries[4].1["card"].as_str().unwrap()));
    assert!(g.contains("\nСемейство: Астровые (Asteraceae)\n\n"), "{g}");
    assert!(g.ends_with(SKILL_TEXTS[4]));
    // No field → the whole card in front of the question.
    let (_, o0) =
        lookup::resolve_lookup(&tables, forced(), SKILL_TEXTS[0], LookupMode::Context).unwrap();
    assert_eq!(
        o0.generation_text(SKILL_TEXTS[0]),
        lookup::context_prompt(entries[0].1["card"].as_str().unwrap(), SKILL_TEXTS[0])
    );
    let mut s = d.summary_json();
    o.annotate(&mut s, LookupMode::Context);
    assert_eq!(s["lookup_mode"], "context");
    assert_eq!(s["lookup_key"], "календулы");
    assert_eq!(s["field"], "family");
    assert_eq!(s["lookup_lang"], "ru");
    // A target that is not a lookup record passes through.
    let ghost = RouteDecision::forced(RouteTarget::Skill("ghost".into()), "forced");
    let (d, o) = lookup::resolve_lookup(&tables, ghost, SKILL_TEXTS[4], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Skill("ghost".into()));
    assert_eq!(o, LookupOutcome::NotLookup);
    let _ = std::fs::remove_dir_all(&dir);
}

/// G2 / no-forgetting: the record touches no tensor of the network, so a
/// prompt without a key — general, or in-domain but unknown to the table —
/// yields logits bit-identical to F0's on the F1 backbone pipeline, and
/// on a pipeline "overlaid" with the lookup record (it replaces nothing).
#[test]
fn no_hit_logits_are_bit_identical_to_f0_with_the_record_mounted() {
    let (dir, f0, f1) = setup("g2");
    let m0 = Arc::new(CmfModel::open(&f0).unwrap());
    let m1 = Arc::new(CmfModel::open(&f1).unwrap());
    assert_eq!(m0.trunk_hash(), m1.trunk_hash());
    let mut p0 = Pipeline::from_model(&m0, SamplerConfig::default()).unwrap();
    let mut p1 = Pipeline::from_model(&m1, SamplerConfig::default()).unwrap();
    let mut p1s = Pipeline::from_model_with_skill(&m1, SamplerConfig::default(), Some(SKILL_ID))
        .unwrap();
    let table = LookupTable::open(&m1, SKILL_ID).unwrap();
    for text in [GENERAL_TEXTS[0], GENERAL_TEXTS[3], NO_KEY_TEXT] {
        assert!(table.find_key(text).is_none(), "{text:?} has a key");
        let ids = p0.tokenizer.encode(&router::render_cmf_im_v1(text));
        let mut l0 = p0.forward_ids(&ids, None).unwrap();
        let mut l1 = p1.forward_ids(&ids, None).unwrap();
        let mut l1s = p1s.forward_ids(&ids, None).unwrap();
        assert_eq!(l0.len(), l1.len());
        for step in 0..4 {
            assert!(l0 == l1, "{text:?}: F1 backbone differs from F0 at step {step}");
            assert!(l0 == l1s, "{text:?}: F1 with the record overlaid differs at step {step}");
            let t = argmax(&l0);
            l0 = p0.decode_step_logits(t, ids.len() + step);
            l1 = p1.decode_step_logits(t, ids.len() + step);
            l1s = p1s.decode_step_logits(t, ids.len() + step);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// An entry that carries only one of the table's languages answers in
/// that language whatever the question's script; an entry with no card
/// text in any language is a miss (the backbone runs), not an empty
/// assistant message.
#[test]
fn empty_slot_falls_back_to_another_language_or_misses() {
    use cortiq_core::knowledge::{KEY_NORM, key_hash, lookup_state_effect, lookup_tensors, skill_kind};
    use cortiq_core::{LookupInfo, SkillBound, SkillRecord};
    let (dir, f0, _f1) = setup("empty");
    let f2 = dir.join("f2.cmf");
    std::fs::copy(&f0, &f2).unwrap();
    // Entry 0: ru only. Entry 1: en only. Entry 2: nothing at all (what
    // `--langs` outside every language object of the entry produces).
    let empty = r#"{"card":"","fields":{}}"#;
    let ru0 = r#"{"card":"Ромашка аптечная — однолетник.","fields":{"family":"Астровые"}}"#;
    let en1 = r#"{"card":"Sage is a subshrub.","fields":{"family":"Lamiaceae"}}"#;
    let slots = [ru0, empty, empty, en1, empty, empty];
    let keys = [
        (key_hash("ромашка"), 0u32),
        (key_hash("chamomile"), 0),
        (key_hash("шалфей"), 1),
        (key_hash("sage"), 1),
        (key_hash("мята"), 2),
        (key_hash("mint"), 2),
    ];
    let info = LookupInfo {
        entries: 3,
        keys: keys.len(),
        key_norm: KEY_NORM.into(),
        langs: vec!["ru".into(), "en".into()],
        fields: vec!["family".into()],
        policy: None,
    };
    let tensors = lookup_tensors("half", &info, &keys, &slots).unwrap();
    let m = CmfModel::open(&f2).unwrap();
    let g = m.header.genome.clone().unwrap();
    drop(m);
    let record = SkillRecord {
        id: "half".into(),
        kind: Some(skill_kind::LOOKUP.into()),
        lookup: Some(info),
        bound: Some(SkillBound {
            genome_id: g.id,
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash,
        }),
        state_effect: Some(lookup_state_effect()),
        status: Some("quarantine".into()),
        ..Default::default()
    };
    CmfModel::append_skill(&f2, record, &tensors, None, None, None).unwrap();
    let model = Arc::new(CmfModel::open(&f2).unwrap());
    let table = LookupTable::open(&model, "half").unwrap();
    // English question, ru-only entry: the ru card, its field, `lang` says so.
    let a = table.answer("What family is chamomile in?").unwrap().unwrap();
    assert_eq!((a.key.entry, a.lang.as_str()), (0, "ru"));
    assert_eq!((a.field.as_deref(), a.text.as_str()), (Some("family"), "Астровые"));
    // Russian question, en-only entry: the en card.
    let a = table.answer("Что известно про шалфей?").unwrap().unwrap();
    assert_eq!((a.key.entry, a.lang.as_str()), (1, "en"));
    assert_eq!(a.text, "Sage is a subshrub.");
    // No card anywhere: a miss with the key named, and the backbone runs.
    assert!(table.answer("Что такое мята?").unwrap().is_none());
    match table.find_answer("mint").unwrap() {
        lookup::Found::EmptyCard(k) => assert_eq!((k.entry, k.key.as_str()), (2, "mint")),
        other => panic!("{other:?}"),
    }
    let tables = LookupTables::new(model.clone());
    let forced = RouteDecision::forced(RouteTarget::Skill("half".into()), "forced");
    let (d, o) = lookup::resolve_lookup(&tables, forced, "Что такое мята?", LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Backbone);
    assert_eq!(o, LookupOutcome::Miss { id: "half".into() });
    assert!(d.reason.contains("no card in any language"), "{}", d.reason);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Plant-naming prompts in ENGLISH framing (the synthetic router's
/// backbone class) with a STRONG key — a capitalised binomial, a two-word
/// English name — and the entry they name. The φ router sends each to
/// the backbone (asserted, not assumed).
const STRONG_KEY_GENERAL: &[(&str, u32)] = &[
    ("Write a Rust function that returns the family of Matricaria chamomilla.", 0),
    ("What is the capital of France, and where does pot marigold grow?", 4),
    ("Explain in two sentences why Hypericum perforatum has yellow flowers.", 1),
    ("Say what poison hemlock is made of.", 5),
];
/// A one-word key (`calendula`) in English framing: never strong.
const ONE_WORD_KEY_GENERAL: &str = "Say what calendula is made of.";

/// `src` copied to `dst` with the record `herbs` changed by a header-only
/// update.
fn variant(src: &std::path::Path, dst: &std::path::Path, f: impl FnOnce(&mut cortiq_core::SkillRecord)) {
    std::fs::copy(src, dst).unwrap();
    CmfModel::update_header_append(dst, move |h| {
        f(h.skills.iter_mut().find(|s| s.id == SKILL_ID).unwrap())
    })
    .unwrap();
}

fn key_first(r: &mut cortiq_core::SkillRecord) {
    r.lookup.as_mut().unwrap().policy = Some("key_first".into());
}

/// The routing policy `key_first` (spec §9.5.2): on a backbone decision
/// of the router, a strong key (≥ 2 words, a binomial) of the LAST
/// message sends the request to the table; a one-word key, no key, a
/// pinned decision, mode `off`, a record the router could not pick
/// (quarantine, an unmet prompt contract) and a key named only in an
/// earlier turn leave the router's decision untouched; the default
/// `router_and_key` never overrides.
#[test]
fn key_first_takes_backbone_decisions_on_strong_keys_only() {
    let (dir, _f0, f1) = setup("keyfirst");
    let entries = lookup_entries();
    let fk = dir.join("f1-key-first.cmf");
    variant(&f1, &fk, key_first);
    let m1 = Arc::new(CmfModel::open(&f1).unwrap());
    let mk = Arc::new(CmfModel::open(&fk).unwrap());
    assert_eq!(
        router::skills_hash(&m1.header),
        router::skills_hash(&mk.header),
        "the policy is not part of the calibrated class set"
    );
    let rec = LookupTable::record(&mk, SKILL_ID).unwrap();
    assert_eq!(LookupPolicy::of(rec.lookup.as_ref().unwrap()), LookupPolicy::KeyFirst);
    let t1 = LookupTables::new(m1.clone());
    let tk = LookupTables::new(mk.clone());
    assert_eq!(tk.get(SKILL_ID).unwrap().unwrap().policy(), LookupPolicy::KeyFirst);
    assert_eq!(t1.get(SKILL_ID).unwrap().unwrap().policy(), LookupPolicy::RouterAndKey);
    let mut probe = Pipeline::from_model(&mk, SamplerConfig::default()).unwrap();
    let opts = RouteOptions::default();
    let gate = Some(KeyFirstGate::new(opts, router::PromptFrame::CmfImV1));
    let mut taken = 0usize;
    for &(text, entry) in STRONG_KEY_GENERAL {
        let d = router::route_request_with(&mk, &mut probe, text, opts);
        assert_eq!(d.target, RouteTarget::Backbone, "precondition — {text:?}: {}", d.reason);
        let router_reason = d.reason.clone();
        let e_base = d.e_base();
        // router_and_key (the default): the gate changes nothing.
        let (d1, o1) =
            lookup::resolve_lookup_gated(&t1, d.clone(), gate, &[text], LookupMode::Answer).unwrap();
        assert_eq!((d1.target.clone(), o1.clone()), (RouteTarget::Backbone, LookupOutcome::NotLookup), "{text:?}");
        assert_eq!(d1.reason, router_reason);
        // key_first, but a pinned decision (no gate): taken as given.
        let (dp, op) =
            lookup::resolve_lookup_gated(&tk, d.clone(), None, &[text], LookupMode::Answer).unwrap();
        assert_eq!((dp.target, op), (RouteTarget::Backbone, LookupOutcome::NotLookup), "{text:?}");
        // key_first + the router's decision: the table answers.
        let (dk, ok) =
            lookup::resolve_lookup_gated(&tk, d.clone(), gate, &[text], LookupMode::Answer).unwrap();
        assert_eq!(dk.target, RouteTarget::Skill(SKILL_ID.into()), "{text:?}: {}", dk.reason);
        assert!(dk.reason.starts_with("key_first: lookup 'herbs'"), "{}", dk.reason);
        assert!(dk.reason.contains(&router_reason), "the router's reason is kept: {}", dk.reason);
        assert!(dk.reason.contains("lookup key"), "{}", dk.reason);
        assert_eq!(dk.e_base(), e_base, "the router's scores are kept");
        let LookupOutcome::Answer(a) = &ok else {
            panic!("{text:?}: {ok:?}");
        };
        assert_eq!(a.key.entry, entry, "{text:?}");
        assert!(lookup::is_strong_key(&a.key) && a.key.turn == 0);
        assert_eq!(a.decided_by, DecidedBy::KeyFirst);
        assert_eq!(a.lang, "en", "an English question answers in English");
        assert_eq!(ok.decided_by(), Some(DecidedBy::KeyFirst));
        let mut s = dk.summary_json();
        ok.annotate(&mut s, LookupMode::Answer);
        assert_eq!(s["target"], SKILL_ID);
        assert_eq!(s["decided_target"], SKILL_ID);
        assert_eq!(s["decided_by"], "key_first");
        assert_eq!(s["lookup_hit"], true);
        assert_eq!(s["lookup_entry"], entry);
        assert!(s["lookup_key_words"].as_u64().unwrap() >= 2);
        assert!(ok.describe().unwrap().contains("decided by key_first"));
        // Context mode: the card in front, the backbone lane generates.
        let (dc, oc) =
            lookup::resolve_lookup_gated(&tk, d.clone(), gate, &[text], LookupMode::Context).unwrap();
        assert_eq!(dc.target, RouteTarget::Skill(SKILL_ID.into()));
        let LookupOutcome::Context(ac) = &oc else {
            panic!("{oc:?}");
        };
        assert_eq!(ac.decided_by, DecidedBy::KeyFirst);
        let g = oc.generation_text(text);
        assert!(g.starts_with(CONTEXT_HEADER) && g.ends_with(text), "{g}");
        // Mode off: the router's backbone stands.
        let (doff, ooff) =
            lookup::resolve_lookup_gated(&tk, d.clone(), gate, &[text], LookupMode::Off).unwrap();
        assert_eq!((doff.target, ooff), (RouteTarget::Backbone, LookupOutcome::NotLookup));
        assert_eq!(doff.reason, router_reason);
        taken += 1;
    }
    assert_eq!(taken, STRONG_KEY_GENERAL.len());
    // The binomial + `family` → the English family line.
    let d = router::route_request_with(&mk, &mut probe, STRONG_KEY_GENERAL[0].0, opts);
    let (_, o) = lookup::resolve_lookup_gated(&tk, d, gate, &[STRONG_KEY_GENERAL[0].0], LookupMode::Answer).unwrap();
    let a = o.hit().unwrap();
    assert_eq!((a.key.source, a.key.key.as_str()), (KeySource::Binomial, "matricaria chamomilla"));
    assert_eq!(a.field.as_deref(), Some("family"));
    assert_eq!(a.text, field(&entries[0].2, "family"));

    // A one-word key and no key: the router's decision, untouched.
    for text in [ONE_WORD_KEY_GENERAL, GENERAL_TEXTS[0], GENERAL_TEXTS[3]] {
        let d = router::route_request_with(&mk, &mut probe, text, opts);
        assert_eq!(d.target, RouteTarget::Backbone, "precondition — {text:?}: {}", d.reason);
        let reason = d.reason.clone();
        let (d, o) = lookup::resolve_lookup_gated(&tk, d, gate, &[text], LookupMode::Answer).unwrap();
        assert_eq!((d.target, o), (RouteTarget::Backbone, LookupOutcome::NotLookup), "{text:?}");
        assert_eq!(d.reason, reason, "{text:?}");
    }
    assert!(tk.get(SKILL_ID).unwrap().unwrap().find_key(ONE_WORD_KEY_GENERAL).is_some());
    // A strong key only in an EARLIER turn: not taken (the memory serves
    // the router's decisions for the table, never key_first).
    let d = router::route_request_with(&mk, &mut probe, GENERAL_TEXTS[0], opts);
    let (d, o) = lookup::resolve_lookup_gated(
        &tk,
        d,
        gate,
        &[GENERAL_TEXTS[0], STRONG_KEY_GENERAL[3].0],
        LookupMode::Answer,
    )
    .unwrap();
    assert_eq!((d.target, o), (RouteTarget::Backbone, LookupOutcome::NotLookup));
    // The router picks the table (a one-word key): resolved as always,
    // decided by the router — key_first adds nothing to it.
    let d = router::route_request_with(&mk, &mut probe, SKILL_TEXTS[4], opts);
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()), "{}", d.reason);
    let (d, o) = lookup::resolve_lookup_gated(&tk, d, gate, &[SKILL_TEXTS[4]], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
    assert!(!d.reason.starts_with("key_first"), "{}", d.reason);
    let a = o.hit().unwrap();
    assert_eq!((a.decided_by, a.key.words, a.key.entry), (DecidedBy::Router, 1, 4));
    // A decision for another skill is never overridden.
    let other = RouteDecision::forced(RouteTarget::Skill("ghost".into()), "forced");
    let (d, o) = lookup::resolve_lookup_gated(&tk, other, gate, &[STRONG_KEY_GENERAL[0].0], LookupMode::Answer).unwrap();
    assert_eq!((d.target, o), (RouteTarget::Skill("ghost".into()), LookupOutcome::NotLookup));

    // A quarantined key_first record: the router could not pick it, so
    // key_first does not either — unless the gate measures quarantine.
    let fq = dir.join("f1-key-first-quarantine.cmf");
    variant(&fk, &fq, |r| r.status = Some("quarantine".into()));
    let mq = Arc::new(CmfModel::open(&fq).unwrap());
    let tq = LookupTables::new(mq.clone());
    let (text, entry) = STRONG_KEY_GENERAL[1];
    let backbone = || RouteDecision::forced(RouteTarget::Backbone, "the router picked the backbone");
    let (d, o) = lookup::resolve_lookup_gated(&tq, backbone(), gate, &[text], LookupMode::Answer).unwrap();
    assert_eq!((d.target, o), (RouteTarget::Backbone, LookupOutcome::NotLookup));
    let with_q = Some(KeyFirstGate::new(
        RouteOptions {
            include_quarantine: true,
        },
        router::PromptFrame::CmfImV1,
    ));
    let (d, o) = lookup::resolve_lookup_gated(&tq, backbone(), with_q, &[text], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
    assert_eq!(o.hit().unwrap().key.entry, entry);

    // A prompt contract the frame does not satisfy: not taken; a raw
    // prompt the caller can render: taken, as the router's pick would be.
    let fc = dir.join("f1-key-first-contract.cmf");
    variant(&fk, &fc, |r| r.prompt_contract = Some("cmf-im-v1".into()));
    let tc = LookupTables::new(Arc::new(CmfModel::open(&fc).unwrap()));
    let other_frame = Some(KeyFirstGate::new(opts, router::PromptFrame::Other));
    let (d, o) = lookup::resolve_lookup_gated(&tc, backbone(), other_frame, &[text], LookupMode::Answer).unwrap();
    assert_eq!((d.target, o), (RouteTarget::Backbone, LookupOutcome::NotLookup));
    let raw = KeyFirstGate::new(opts, router::PromptFrame::Raw);
    let (d, _) = lookup::resolve_lookup_gated(&tc, backbone(), Some(raw), &[text], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Backbone, "raw, cannot render");
    let (d, o) = lookup::resolve_lookup_gated(&tc, backbone(), Some(raw.can_render(true)), &[text], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
    assert!(o.is_hit());
    let (d, o) = lookup::resolve_lookup_gated(&tc, backbone(), gate, &[text], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()), "cmf-im-v1 frame");
    assert!(o.is_hit());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Review KF-4: `key_first` never reopens the table behind the router's
/// fail-closed state. A stale `skills_hash` (a record added without a
/// refit), a missing calibration, a file without a router — the router
/// runs the backbone for every request, and so does `key_first`, even on
/// a strong key of an active record. An unknown policy value reads as
/// `router_and_key` (KF-6): the file opens, nothing is taken.
#[test]
fn key_first_stays_closed_while_the_router_is_fail_closed() {
    let (dir, _f0, f1) = setup("keyfirst-closed");
    let fk = dir.join("f1-key-first.cmf");
    variant(&f1, &fk, key_first);
    let (text, entry) = STRONG_KEY_GENERAL[1];
    let gate = Some(KeyFirstGate::new(RouteOptions::default(), router::PromptFrame::CmfImV1));
    let backbone = || RouteDecision::forced(RouteTarget::Backbone, "the router picked the backbone");
    // Precondition: the healthy key_first file takes it.
    let tk = LookupTables::new(Arc::new(CmfModel::open(&fk).unwrap()));
    let (d, o) = lookup::resolve_lookup_gated(&tk, backbone(), gate, &[text], LookupMode::Answer).unwrap();
    assert_eq!(d.target, RouteTarget::Skill(SKILL_ID.into()));
    assert_eq!(o.hit().unwrap().key.entry, entry);
    type Edit = Box<dyn FnOnce(&mut cortiq_core::CmfHeader)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "stale skills_hash",
            Box::new(|h| h.router.as_mut().unwrap().skills_hash = "0000000000000000".into()),
        ),
        ("no calibration", Box::new(|h| h.routing = None)),
        (
            "no router",
            Box::new(|h| {
                h.router = None;
                h.routing = None;
            }),
        ),
        (
            "unknown policy",
            Box::new(|h| {
                h.skills[0].lookup.as_mut().unwrap().policy = Some("key_only".into());
            }),
        ),
    ];
    for (i, (what, edit)) in cases.into_iter().enumerate() {
        let f = dir.join(format!("closed-{i}.cmf"));
        std::fs::copy(&fk, &f).unwrap();
        if what == "unknown policy" {
            // A newer writer's value: this reader's writers refuse it, so it
            // is put in place by rewriting the header JSON directly.
            knowledge_synth::rewrite_header_json(&f, |v| {
                v["skills"][0]["lookup"]["policy"] = serde_json::json!("key_only");
            });
            let _ = edit;
        } else {
            CmfModel::update_header_append(&f, edit).unwrap();
        }
        let m = Arc::new(CmfModel::open(&f).unwrap_or_else(|e| panic!("{what}: {e}")));
        let t = LookupTables::new(m.clone());
        let (d, o) = lookup::resolve_lookup_gated(&t, backbone(), gate, &[text], LookupMode::Answer).unwrap();
        assert_eq!((d.target, o), (RouteTarget::Backbone, LookupOutcome::NotLookup), "{what}");
        if what == "unknown policy" {
            let table = t.get(SKILL_ID).unwrap().unwrap();
            assert_eq!(table.policy(), LookupPolicy::RouterAndKey);
            assert!(!LookupPolicy::is_known(&table.info));
            // Routed by the router, the record still answers as always.
            let pick = RouteDecision::forced(RouteTarget::Skill(SKILL_ID.into()), "router");
            let (_, o) = lookup::resolve_lookup_gated(&t, pick, gate, &[text], LookupMode::Answer).unwrap();
            assert_eq!(o.hit().unwrap().key.entry, entry);
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Review KF-3: one message, one entry — whoever decided. A one-word
/// exact key (`календула`) and a two-word stem key of ANOTHER entry
/// (`зверобоя продырявленного`) in one message: the router's pick of the
/// table, `key_first` on a backbone decision and the conversation memory
/// of a later keyless turn all answer from the strong key's entry.
#[test]
fn one_rule_names_one_entry_on_every_path() {
    let (dir, _f0, f1) = setup("one-rule");
    let fk = dir.join("f1-key-first.cmf");
    variant(&f1, &fk, key_first);
    let tk = LookupTables::new(Arc::new(CmfModel::open(&fk).unwrap()));
    let table = tk.get(SKILL_ID).unwrap().unwrap();
    let m = "Календула или зверобоя продырявленного — что лучше?";
    let weak_only = "Календула — что это?";
    let h = table.find_key(m).unwrap();
    assert_eq!((h.entry, h.via, h.words, h.strong), (1, MatchVia::Stem, 2, true), "{h:?}");
    assert_eq!(table.find_strong_key(m), Some(h.clone()));
    assert_eq!(table.find_key(weak_only).map(|k| (k.entry, k.strong)), Some((4, false)));
    let gate = Some(KeyFirstGate::new(RouteOptions::default(), router::PromptFrame::CmfImV1));
    // The router picked the table.
    let pick = || RouteDecision::forced(RouteTarget::Skill(SKILL_ID.into()), "router");
    let (_, o) = lookup::resolve_lookup_gated(&tk, pick(), gate, &[m], LookupMode::Answer).unwrap();
    let a = o.hit().unwrap();
    assert_eq!((a.key.entry, a.decided_by), (1, DecidedBy::Router));
    // key_first on the router's backbone decision.
    let bb = RouteDecision::forced(RouteTarget::Backbone, "backbone");
    let (_, o) = lookup::resolve_lookup_gated(&tk, bb, gate, &[m], LookupMode::Answer).unwrap();
    let a = o.hit().unwrap();
    assert_eq!((a.key.entry, a.decided_by), (1, DecidedBy::KeyFirst));
    // The memory: a keyless follow-up the router sends to the table.
    let (_, o) = lookup::resolve_lookup_gated(&tk, pick(), gate, &["А противопоказания?", m], LookupMode::Answer).unwrap();
    let a = o.hit().unwrap();
    assert_eq!((a.key.entry, a.key.turn, a.field.as_deref()), (1, 1, Some("safety")));
    let _ = std::fs::remove_dir_all(&dir);
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}
