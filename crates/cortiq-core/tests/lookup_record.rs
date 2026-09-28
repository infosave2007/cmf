//! Format v2 `lookup` records (spec §9.5.2): the explicit key → card table
//! under the resonance router — round trip through the true tail append,
//! every refusal (directory rules and table values; the writer leaves the
//! file alone, `open()` refuses a tampered file), key normalisation
//! `cmf-key-v2` with hash stability, and the raw `u8`/`u32`/`u64` dtypes.

use cortiq_core::format::{TokenizerBundle, features};
use cortiq_core::knowledge::{
    KEY_NORM, KEY_NORM_V1, hex64, lookup_leaf, lookup_policy, read_u32_le, read_u64_le,
    skill_kind,
};
use cortiq_core::quant::expected_nbytes;
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType};
use cortiq_core::{
    CmfHeader, CmfModel, ExpertAppend, GenomeInfo, LineageEvent, LookupInfo, SelectionDescriptor,
    SkillBound, SkillOverride, SkillRecord, TensorDtype, TensorSpec, hash64, key_hash,
    lookup_state_effect, lookup_tensor_name, lookup_tensors, normalize_key, normalized_key_hash,
};
use std::path::{Path, PathBuf};

const H: usize = 64;
const I: usize = 128;
const L: usize = 4;
const V: usize = 10;

// ───────────────────────── fixture ─────────────────────────

fn arch() -> ModelArch {
    ModelArch {
        arch_name: "tiny-genome".into(),
        hidden_size: H,
        intermediate_size: I,
        num_layers: L,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 32,
        vocab_size: V,
        layer_types: vec![LayerType::FullAttention; L],
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: true,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        local_partial_rotary_factor: None,
        mtp: None,
        moe: None,
        qwen4_exp: None,
        deepseek_v41: None,
        anchor_core: None,
        linear_core: None,
        head_clusters: None,
        max_position_embeddings: 256,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_v_norm: false,
        qk_norm_after_rope: false,
        num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        loop_final_norm: false,
        prism_hadamard: None,
        kv_heads_per_layer: None,
        v_head_dim: None,
    }
}

/// Built through serde so the fixture does not list every header field.
fn plain_header() -> CmfHeader {
    let a = arch();
    let mut h: CmfHeader = serde_json::from_value(serde_json::json!({
        "version": 2,
        "arch": serde_json::to_value(&a).unwrap(),
        "quant_type": "F32",
    }))
    .unwrap();
    h.arch = a;
    h.quant_type = QuantType::F32;
    h.tokenizer_config = Some(TokenizerBundle {
        chat_template: None,
        eos_token_ids: vec![2],
        bos_token_id: None,
        pad_token_id: None,
    });
    h
}

fn genome_header() -> CmfHeader {
    let mut h = plain_header();
    h.genome = Some(GenomeInfo::birth("embryo-test", "pre_chat", "f32"));
    h.lineage = vec![LineageEvent::now(0, "birth", serde_json::json!({"step": 0}))];
    h
}

fn f32s(n: usize, seed: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| ((i as f32 * 0.37 + seed).sin()).to_le_bytes())
        .collect()
}

fn spec(name: &str, shape: &[usize], seed: f32) -> TensorSpec {
    TensorSpec {
        name: name.into(),
        dtype: TensorDtype::F32,
        shape: shape.to_vec(),
        data: f32s(shape.iter().product(), seed),
    }
}

fn trunk_specs() -> Vec<TensorSpec> {
    let mut t = vec![spec("model.embed_tokens.weight", &[V, H], 0.1)];
    for l in 0..L {
        t.push(spec(
            &format!("model.layers.{l}.mlp.up_proj.weight"),
            &[I, H],
            1.0 + l as f32,
        ));
        t.push(spec(
            &format!("model.layers.{l}.mlp.down_proj.weight"),
            &[H, I],
            2.0 + l as f32,
        ));
    }
    t.push(spec("model.norm.weight", &[H], 3.0));
    t
}

const VOCAB: &[u8] = br#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cmf-lookup-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// F0: a sealed-at-birth genome with its vocab.
fn write_f0(path: &Path) {
    CmfModel::write(path, &genome_header(), &trunk_specs(), None, Some(VOCAB)).unwrap();
}

fn specs_of(m: &CmfModel) -> Vec<TensorSpec> {
    m.tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: m.entry_bytes(e).to_vec(),
        })
        .collect()
}

fn f16_b64(v: &[f32]) -> String {
    use base64::Engine as _;
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| cortiq_core::quant::f32_to_f16(*x).to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn descriptor(layer: usize) -> SelectionDescriptor {
    let mut mean = vec![0.0f32; H];
    mean[0] = 1.0;
    let mut basis = vec![0.0f32; H];
    basis[1] = 1.0;
    SelectionDescriptor {
        metric: "mse_unit".into(),
        phi_layer: layer,
        mean: f16_b64(&mean),
        basis: f16_b64(&basis),
        rank: 1,
        err_mean: Some(0.01),
        err_std: Some(0.01),
        holdout: None,
        holdout_n: None,
    }
}

fn open_err(p: &Path) -> String {
    match CmfModel::open(p) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{} opened but must be refused", p.display()),
    }
}

/// Publish a MUTATED header without writer-side validation (tests `open()`).
fn raw_header_swap(src: &Path, dst: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let mut bytes = std::fs::read(src).unwrap();
    let m = CmfModel::open(src).unwrap();
    let mut v = serde_json::to_value(&m.header).unwrap();
    drop(m);
    mutate(&mut v);
    let js = serde_json::to_vec(&v).unwrap();
    let off = bytes.len() as u64;
    bytes.extend_from_slice(&js);
    bytes[0x10..0x18].copy_from_slice(&off.to_le_bytes());
    bytes[0x18..0x20].copy_from_slice(&(js.len() as u64).to_le_bytes());
    bytes[0x70..0x78].copy_from_slice(&hash64(&js).to_le_bytes());
    std::fs::write(dst, &bytes).unwrap();
}

// ───────────────────────── the herbs table ─────────────────────────

const LANGS: [&str; 2] = ["ru", "en"];
const FIELDS: [&str; 3] = ["family", "parts", "uses"];

/// Three cards, three keys each (Cyrillic name, Latin binomial, English
/// name), in two languages.
fn herbs() -> (Vec<(&'static str, u32)>, Vec<String>) {
    let keys = vec![
        ("Пихта бальзамическая", 0),
        ("Abies balsamea", 0),
        ("balsam fir", 0),
        ("Зверобой продырявленный", 1),
        ("Hypericum perforatum", 1),
        ("St. John's wort", 1),
        ("Ромашка аптечная", 2),
        ("Matricaria chamomilla", 2),
        ("chamomile", 2),
    ];
    let card = |name: &str, family: &str, parts: &str, uses: &str| {
        serde_json::json!({
            "card": format!("{name} — {family}. Части: {parts}. Применение: {uses}."),
            "fields": {"family": family, "parts": parts, "uses": uses},
        })
        .to_string()
    };
    let slots = vec![
        card("Пихта бальзамическая", "Сосновые (Pinaceae)", "хвоя, смола", "бальзам"),
        card("Balsam fir", "Pinaceae", "needles, resin", "balsam"),
        card("Зверобой продырявленный", "Зверобойные (Hypericaceae)", "трава", "настой"),
        card("St. John's wort", "Hypericaceae", "herb", "infusion"),
        card("Ромашка аптечная", "Астровые (Asteraceae)", "цветки", "чай"),
        card("Chamomile", "Asteraceae", "flowers", "tea"),
    ];
    (keys, slots)
}

fn info() -> LookupInfo {
    LookupInfo {
        entries: 3,
        keys: 9,
        key_norm: KEY_NORM.into(),
        langs: LANGS.iter().map(|s| s.to_string()).collect(),
        fields: FIELDS.iter().map(|s| s.to_string()).collect(),
        policy: None,
    }
}

/// The lookup record `id` over the herbs table, bound to `model`'s genome,
/// with its four tensors built by the core builder.
fn lookup_record(model: &CmfModel, id: &str) -> (SkillRecord, Vec<TensorSpec>) {
    let g = model.header.genome.clone().expect("genome");
    let (keys, slots) = herbs();
    let hashed: Vec<(u64, u32)> = keys.iter().map(|(k, e)| (key_hash(k), *e)).collect();
    let slot_refs: Vec<&str> = slots.iter().map(String::as_str).collect();
    let tensors = lookup_tensors(id, &info(), &hashed, &slot_refs).expect("valid table");
    let record = SkillRecord {
        id: id.into(),
        kind: Some(skill_kind::LOOKUP.into()),
        lookup: Some(info()),
        bound: Some(SkillBound {
            genome_id: g.id,
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash,
        }),
        state_effect: Some(lookup_state_effect()),
        status: Some("quarantine".into()),
        prompt_contract: Some("cmf-im-v1".into()),
        origin: Some(serde_json::json!({
            "trigger": "user_corpus", "entries": 3, "keys": 9, "duplicates_dropped": 0
        })),
        ..Default::default()
    };
    (record, tensors)
}

fn append(path: &Path, rec: SkillRecord, ts: &[TensorSpec]) -> Result<usize, String> {
    CmfModel::append_skill(path, rec, ts, None, None, None)
        .map(|r| r.tensors_added)
        .map_err(|e| e.to_string())
}

// ───────────────────────── round trip ─────────────────────────

#[test]
fn lookup_record_round_trips_through_append_skill() {
    let dir = tempdir("roundtrip");
    let f0 = dir.join("f0.cmf");
    let f1 = dir.join("f1.cmf");
    write_f0(&f0);
    std::fs::copy(&f0, &f1).unwrap();
    let bytes0 = std::fs::read(&f0).unwrap();
    let m0 = CmfModel::open(&f0).unwrap();
    let h0 = m0.trunk_hash();
    let (rec, ts) = lookup_record(&m0, "herbs");
    drop(m0);

    assert_eq!(append(&f1, rec, &ts).unwrap(), 4);
    let bytes1 = std::fs::read(&f1).unwrap();
    assert_eq!(
        &bytes1[128..bytes0.len()],
        &bytes0[128..],
        "a true tail append: the old bytes past the envelope are untouched"
    );

    let m1 = CmfModel::open(&f1).unwrap();
    assert_eq!(m1.trunk_hash(), h0, "the table is not part of the trunk");
    assert_ne!(m1.required_features & features::SKILLS_V2, 0);
    assert_ne!(m1.required_features & features::GENOME, 0);
    assert!(m1.verify().is_empty());
    let rec = &m1.header.skills[0];
    assert_eq!(rec.kind.as_deref(), Some(skill_kind::LOOKUP));
    assert!(skill_kind::IMPLEMENTED.contains(&skill_kind::LOOKUP));
    assert_eq!(rec.lookup.as_ref(), Some(&info()));
    assert!(rec.layers.is_empty() && rec.overrides.is_empty() && rec.experts.is_none());
    assert_eq!(rec.state_effect.as_ref(), Some(&lookup_state_effect()));
    assert_eq!(rec.state_effect.as_ref().unwrap().switch, "transparent");

    // Tensors: dtypes and shapes as the spec fixes them.
    let t = |leaf: &str| m1.tensor(&lookup_tensor_name("herbs", leaf)).unwrap();
    for (leaf, dtype, shape) in [
        (lookup_leaf::KEYS_HASH, TensorDtype::U64, vec![9]),
        (lookup_leaf::KEYS_ENTRY, TensorDtype::U32, vec![9]),
        (lookup_leaf::ENTRIES_OFF, TensorDtype::U64, vec![3 * 2 + 1]),
    ] {
        assert_eq!((t(leaf).dtype, t(leaf).shape.clone()), (dtype, shape), "{leaf}");
    }
    let text_e = t(lookup_leaf::TEXT);
    assert_eq!(text_e.dtype, TensorDtype::U8);
    assert_eq!(text_e.shape, vec![text_e.nbytes as usize]);
    assert_eq!(m1.skill_tensors("herbs").count(), 4);

    // The table decodes: sorted unique hashes, every key found by binary
    // search (also through the runtime's path: normalise the message,
    // hash the n-gram), entry indices right, slots are the cards.
    let hashes = read_u64_le(m1.entry_bytes(t(lookup_leaf::KEYS_HASH)));
    let entries = read_u32_le(m1.entry_bytes(t(lookup_leaf::KEYS_ENTRY)));
    let offs = read_u64_le(m1.entry_bytes(t(lookup_leaf::ENTRIES_OFF)));
    let text = m1.entry_bytes(text_e).to_vec();
    assert!(hashes.windows(2).all(|w| w[0] < w[1]), "sorted, unique");
    assert_eq!(offs[0], 0);
    assert_eq!(*offs.last().unwrap(), text.len() as u64);
    for (k, e) in herbs().0 {
        let i = hashes.binary_search(&key_hash(k)).unwrap_or_else(|_| panic!("{k}"));
        assert_eq!(entries[i], e, "{k}");
        let ngram = normalize_key(&format!("  Что лечит {k}? "));
        let ngram = ngram.trim_start_matches("что лечит ");
        assert_eq!(hashes.binary_search(&normalized_key_hash(ngram)), Ok(i), "{k}");
    }
    let info = rec.lookup.as_ref().unwrap();
    let s = info.slot(1, 1).unwrap();
    assert_eq!(s, 3);
    let card: serde_json::Value =
        serde_json::from_slice(&text[offs[s] as usize..offs[s + 1] as usize]).unwrap();
    assert_eq!(card["fields"]["family"], "Hypericaceae");
    let s = info.slot(2, 0).unwrap();
    let card: serde_json::Value =
        serde_json::from_slice(&text[offs[s] as usize..offs[s + 1] as usize]).unwrap();
    assert_eq!(card["fields"]["parts"], "цветки");
    assert_eq!(info.slot(3, 0), None);
    assert_eq!(info.slot(0, 2), None);

    // Auto-routing: like any v2 record — active + measured gate + selection.
    assert!(!rec.is_auto_routable(), "quarantine");
    drop(m1);
    CmfModel::update_header_append(&f1, |h| {
        let s = &mut h.skills[0];
        s.status = Some("active".into());
        s.gate = Some(serde_json::json!({"status": "measured", "false_accept_upper95": 0.01}));
        s.selection = Some(descriptor(1));
    })
    .unwrap();
    let m2 = CmfModel::open(&f1).unwrap();
    assert!(m2.header.skills[0].is_auto_routable());
    assert_eq!(m2.trunk_hash(), h0);
    assert_eq!(m2.entry_bytes(m2.tensor(&lookup_tensor_name("herbs", lookup_leaf::TEXT)).unwrap()), text);

    // A full rewrite (compaction) carries the record and its bytes.
    let again = dir.join("again.cmf");
    CmfModel::write(&again, &m2.header, &specs_of(&m2), None, m2.vocab.as_deref()).unwrap();
    let m3 = CmfModel::open(&again).unwrap();
    assert_eq!(m3.header.skills[0].lookup, m2.header.skills[0].lookup);
    assert_eq!(m3.trunk_hash(), h0);
    for leaf in lookup_leaf::ALL {
        let name = lookup_tensor_name("herbs", leaf);
        assert_eq!(
            m3.entry_bytes(m3.tensor(&name).unwrap()),
            m2.entry_bytes(m2.tensor(&name).unwrap()),
            "{name}"
        );
    }
}

// ───────────────────────── refusals ─────────────────────────

#[test]
fn every_lookup_refusal_is_precise_and_leaves_the_file_alone() {
    let dir = tempdir("refuse");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let bytes0 = std::fs::read(&f0).unwrap();
    let m = CmfModel::open(&f0).unwrap();
    let (rec, ts) = lookup_record(&m, "herbs");
    drop(m);
    let text_len = ts[3].data.len();
    let first_slot_end = read_u64_le(&ts[2].data)[1] as usize;

    type Mut = Box<dyn Fn(&mut SkillRecord, &mut Vec<TensorSpec>)>;
    let put_u64 = |data: &mut [u8], i: usize, v: u64| data[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    let cases: Vec<(&str, Mut, &str)> = vec![
        (
            "no lookup",
            Box::new(|r, _| r.lookup = None),
            "skill 'herbs': lookup record has no `lookup`",
        ),
        (
            "experts",
            Box::new(|r, _| {
                r.experts = Some(ExpertAppend {
                    count: 1,
                    shell_quantile: 0.5,
                    rank: 0,
                })
            }),
            "`experts` belongs to an expert_append record, not to lookup",
        ),
        (
            "overrides",
            Box::new(|r, _| {
                r.overrides.push(SkillOverride {
                    name: "model.layers.1.mlp.up_proj.weight".into(),
                    base_hash: "0".into(),
                })
            }),
            "a lookup record replaces nothing — `overrides` must be empty",
        ),
        (
            "layers",
            Box::new(|r, _| r.layers = vec![0]),
            "a lookup record touches no layer — `layers` must be empty (is [0])",
        ),
        (
            "key_norm",
            Box::new(|r, _| r.lookup.as_mut().unwrap().key_norm = "cmf-key-v9".into()),
            "lookup.key_norm 'cmf-key-v9' (this reader knows cmf-key-v2)",
        ),
        (
            "key_norm v1 (pre-release)",
            Box::new(|r, _| r.lookup.as_mut().unwrap().key_norm = KEY_NORM_V1.into()),
            "lookup.key_norm 'cmf-key-v1' is the pre-release rule — rebuild the table",
        ),
        (
            "unknown policy",
            Box::new(|r, _| r.lookup.as_mut().unwrap().policy = Some("always".into())),
            "lookup.policy 'always' (expected router_and_key | key_first)",
        ),
        (
            "empty policy",
            Box::new(|r, _| r.lookup.as_mut().unwrap().policy = Some(String::new())),
            "lookup.policy '' (expected router_and_key | key_first)",
        ),
        (
            "entries 0",
            Box::new(|r, _| r.lookup.as_mut().unwrap().entries = 0),
            "lookup.entries must be ≥ 1",
        ),
        (
            "keys 0",
            Box::new(|r, _| r.lookup.as_mut().unwrap().keys = 0),
            "lookup.keys must be ≥ 1",
        ),
        (
            "no langs",
            Box::new(|r, _| r.lookup.as_mut().unwrap().langs.clear()),
            "lookup.langs is empty",
        ),
        (
            "duplicate lang",
            Box::new(|r, _| r.lookup.as_mut().unwrap().langs = vec!["ru".into(), "ru".into()]),
            "duplicate 'ru' in lookup.langs",
        ),
        (
            "empty field",
            Box::new(|r, _| r.lookup.as_mut().unwrap().fields.push(String::new())),
            "lookup.fields has an empty name",
        ),
        (
            "keys disagree with the tensors",
            Box::new(|r, _| r.lookup.as_mut().unwrap().keys = 8),
            "'skill.herbs.lookup.keys.hash' shape [9] != [8] (lookup.keys)",
        ),
        (
            "entries disagree with the offsets",
            Box::new(|r, _| r.lookup.as_mut().unwrap().entries = 4),
            "'skill.herbs.lookup.entries.off' shape [7] != [9] (lookup.entries × langs + 1)",
        ),
        (
            "hash dtype",
            Box::new(|_, t| {
                t[0].dtype = TensorDtype::U32;
                t[0].shape = vec![18];
            }),
            "'skill.herbs.lookup.keys.hash' must be u64 (is u32)",
        ),
        (
            "text 2-D",
            Box::new(move |_, t| t[3].shape = vec![1, text_len]),
            "'skill.herbs.lookup.text' must be 1-D (shape [1,",
        ),
        (
            "missing tensor",
            Box::new(|_, t| {
                t.pop();
            }),
            "skill 'herbs': missing tensor 'skill.herbs.lookup.text'",
        ),
        (
            "extra tensor",
            Box::new(|_, t| t.push(spec("skill.herbs.lookup.extra", &[H], 1.0))),
            "tensor 'skill.herbs.lookup.extra' is not part of a lookup record",
        ),
        (
            "no state effect",
            Box::new(|r, _| r.state_effect = None),
            "v2 record has no `state_effect`",
        ),
        (
            "state switch",
            Box::new(|r, _| r.state_effect.as_mut().unwrap().switch = "sequence_start".into()),
            "a lookup record never touches the network's state",
        ),
        (
            "state layer",
            Box::new(|r, _| r.state_effect.as_mut().unwrap().first_affected_layer = 1),
            "state_effect {first_affected_layer 1, switch 'transparent', state_bytes_added 0}",
        ),
        (
            "state bytes",
            Box::new(|r, _| r.state_effect.as_mut().unwrap().state_bytes_added = 8),
            "a lookup record never touches the network's state",
        ),
        (
            "hashes unsorted",
            Box::new(move |_, t| {
                let h = read_u64_le(&t[0].data);
                put_u64(&mut t[0].data, 0, h[1]);
                put_u64(&mut t[0].data, 1, h[0]);
            }),
            "'lookup.keys.hash' is not sorted ascending at key 1",
        ),
        (
            "duplicate hash",
            Box::new(move |_, t| {
                let h = read_u64_le(&t[0].data);
                put_u64(&mut t[0].data, 1, h[0]);
            }),
            "duplicate key hash",
        ),
        (
            "entry out of range",
            Box::new(|_, t| t[1].data[..4].copy_from_slice(&3u32.to_le_bytes())),
            "key 0 points at entry 3 of 3 (lookup.entries)",
        ),
        (
            "offsets not monotone",
            Box::new(move |_, t| {
                let o = read_u64_le(&t[2].data);
                put_u64(&mut t[2].data, 1, o[2]);
                put_u64(&mut t[2].data, 2, o[1]);
            }),
            "'lookup.entries.off' is not monotone at slot 1",
        ),
        (
            "offset beyond the blob",
            Box::new(move |_, t| put_u64(&mut t[2].data, 6, text_len as u64 + 1)),
            "is beyond the text blob",
        ),
        (
            "slot not a JSON object",
            Box::new(|_, t| t[3].data[0] = b'['),
            "slot 0 (entry 0, lang 'ru') is not a JSON object",
        ),
        (
            "slot a JSON array",
            Box::new(move |_, t| {
                let arr = format!("[{}]", "1".repeat(first_slot_end - 2));
                t[3].data[..first_slot_end].copy_from_slice(arr.as_bytes());
            }),
            "slot 0 (entry 0, lang 'ru') is not a JSON object",
        ),
        (
            "slot not UTF-8",
            Box::new(|_, t| t[3].data[1] = 0xff),
            "slot 0 (entry 0, lang 'ru') is not UTF-8",
        ),
        (
            "slot of the second language not UTF-8",
            Box::new(move |_, t| t[3].data[first_slot_end + 3] = 0xff),
            "slot 1 (entry 0, lang 'en') is not UTF-8",
        ),
        (
            "slot boundary off by one",
            Box::new(move |_, t| {
                // The first slot ends one byte early: its closing `}`
                // moves to the next slot, and the shape check reports it
                // before any JSON is parsed.
                let o = read_u64_le(&t[2].data);
                put_u64(&mut t[2].data, 1, o[1] - 1);
            }),
            "slot 0 (entry 0, lang 'ru') is not a JSON object",
        ),
        (
            "slot boundary inside a character",
            Box::new(move |_, t| {
                // `{"card":"Пихта` — byte 10 is the second byte of `П`.
                let o = read_u64_le(&t[2].data);
                put_u64(&mut t[2].data, 1, o[0] + 10);
            }),
            "slot 0 (entry 0, lang 'ru') does not start and end on a UTF-8 character boundary",
        ),
        (
            "slot is truncated JSON (writers parse every slot)",
            Box::new(move |_, t| {
                // `{"card": ...}` → `{"card": ...` + `}` — object-shaped
                // (first `{`, last `}`) but not JSON: only the deep pass
                // of a writer catches it.
                t[3].data[first_slot_end - 2] = b' ';
                t[3].data[first_slot_end - 3] = b' ';
            }),
            "slot 0 (entry 0, lang 'ru') is not a JSON object:",
        ),
        (
            "slot of the second language",
            Box::new(move |_, t| t[3].data[first_slot_end] = b'x'),
            "slot 1 (entry 0, lang 'en') is not a JSON object",
        ),
    ];
    for (i, (what, mutate, needle)) in cases.iter().enumerate() {
        let f = dir.join(format!("case{i}.cmf"));
        std::fs::copy(&f0, &f).unwrap();
        let (mut r, mut t) = (rec.clone(), ts.clone());
        mutate(&mut r, &mut t);
        let e = append(&f, r, &t).expect_err(what);
        assert!(e.contains(needle), "{what}: refusal was '{e}', expected '{needle}'");
        assert_eq!(std::fs::read(&f).unwrap(), bytes0, "{what}: the file changed");
    }

    // `lookup` on another kind, and on a record without `kind`.
    let m = CmfModel::open(&f0).unwrap();
    let g = m.header.genome.clone().unwrap();
    let base = m.tensor("model.layers.3.mlp.up_proj.weight").unwrap().clone();
    let ffn = SkillRecord {
        id: "ffn".into(),
        layers: vec![3],
        kind: Some(skill_kind::FFN_REPLACE.into()),
        overrides: vec![SkillOverride {
            name: base.name.clone(),
            base_hash: hex64(base.hash),
        }],
        bound: Some(SkillBound {
            genome_id: g.id.clone(),
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash.clone(),
        }),
        state_effect: Some(cortiq_core::ffn_replace_state_effect(&m.header.arch, &[3])),
        status: Some("quarantine".into()),
        lookup: Some(info()),
        ..Default::default()
    };
    let ffn_t = vec![spec(&format!("skill.ffn.{}", base.name), &base.shape, 9.0)];
    drop(m);
    let f = dir.join("ffn.cmf");
    std::fs::copy(&f0, &f).unwrap();
    let e = append(&f, ffn, &ffn_t).unwrap_err();
    assert!(
        e.contains("`lookup` belongs to a lookup record, not to ffn_replace"),
        "{e}"
    );
    assert_eq!(std::fs::read(&f).unwrap(), bytes0);
    let v1 = SkillRecord {
        id: "v1".into(),
        lookup: Some(info()),
        ..Default::default()
    };
    let mut hdr = genome_header();
    hdr.skills.push(v1);
    let e = CmfModel::write(&dir.join("v1.cmf"), &hdr, &trunk_specs(), None, Some(VOCAB))
        .unwrap_err()
        .to_string();
    assert!(e.contains("on a record without `kind`"), "{e}");

    // open() refuses what the writers refuse: a tampered header and a
    // tampered table on disk (the directory still matches, so only the
    // value pass can catch it).
    let f1 = dir.join("f1.cmf");
    std::fs::copy(&f0, &f1).unwrap();
    append(&f1, rec.clone(), &ts).unwrap();
    assert!(CmfModel::open(&f1).is_ok());
    let hdr_tamper = dir.join("hdr.cmf");
    raw_header_swap(&f1, &hdr_tamper, |v| {
        v["skills"][0]["lookup"]["langs"] = serde_json::json!(["ru"]);
    });
    let e = open_err(&hdr_tamper);
    assert!(
        e.contains("'skill.herbs.lookup.entries.off' shape [7] != [4] (lookup.entries × langs + 1)"),
        "{e}"
    );
    raw_header_swap(&f1, &hdr_tamper, |v| {
        v["skills"][0]["state_effect"]["switch"] = serde_json::json!("sequence_start");
    });
    assert!(open_err(&hdr_tamper).contains("never touches the network's state"));

    let mut bytes = std::fs::read(&f1).unwrap();
    let at = bytes
        .windows(ts[0].data.len())
        .position(|w| w == ts[0].data.as_slice())
        .expect("the hash table is in the file");
    let h = read_u64_le(&ts[0].data);
    bytes[at..at + 8].copy_from_slice(&h[1].to_le_bytes());
    bytes[at + 8..at + 16].copy_from_slice(&h[0].to_le_bytes());
    let tampered = dir.join("tampered.cmf");
    std::fs::write(&tampered, &bytes).unwrap();
    let e = open_err(&tampered);
    assert!(e.contains("is not sorted ascending at key 1"), "{e}");
    bytes[at..at + 8].copy_from_slice(&h[0].to_le_bytes());
    bytes[at + 8..at + 16].copy_from_slice(&h[0].to_le_bytes());
    std::fs::write(&tampered, &bytes).unwrap();
    let e = open_err(&tampered);
    assert!(e.contains("duplicate key hash"), "{e}");
}

// ───────────────────────── routing policy ─────────────────────────

#[test]
fn lookup_policy_is_additive_validated_and_switchable_by_a_header_update() {
    let dir = tempdir("policy");
    let f0 = dir.join("f0.cmf");
    let f1 = dir.join("f1.cmf");
    write_f0(&f0);
    std::fs::copy(&f0, &f1).unwrap();
    let m = CmfModel::open(&f0).unwrap();
    let (rec, ts) = lookup_record(&m, "herbs");
    drop(m);
    // Absent = router_and_key, and the header JSON carries no field (the
    // files written before the policy existed read the same).
    assert_eq!(info().policy_label(), lookup_policy::ROUTER_AND_KEY);
    assert_eq!(lookup_policy::DEFAULT, "router_and_key");
    assert_eq!(lookup_policy::ALL, &["router_and_key", "key_first"]);
    let j = serde_json::to_value(info()).unwrap();
    assert!(j.get("policy").is_none(), "{j}");
    let back: LookupInfo = serde_json::from_value(j).unwrap();
    assert_eq!(back.policy, None);
    append(&f1, rec.clone(), &ts).unwrap();
    let m1 = CmfModel::open(&f1).unwrap();
    let h0 = m1.trunk_hash();
    assert_eq!(m1.header.skills[0].lookup.as_ref().unwrap().policy, None);
    drop(m1);

    // A header-only update switches it; the trunk and the table stay.
    CmfModel::update_header_append(&f1, |h| {
        h.skills[0].lookup.as_mut().unwrap().policy = Some(lookup_policy::KEY_FIRST.into());
    })
    .unwrap();
    let m2 = CmfModel::open(&f1).unwrap();
    let li = m2.header.skills[0].lookup.as_ref().unwrap();
    assert_eq!(li.policy.as_deref(), Some("key_first"));
    assert_eq!(li.policy_label(), "key_first");
    assert_eq!(m2.trunk_hash(), h0);
    let len2 = std::fs::metadata(&f1).unwrap().len();
    drop(m2);

    // The explicit default is accepted too.
    CmfModel::update_header_append(&f1, |h| {
        h.skills[0].lookup.as_mut().unwrap().policy = Some(lookup_policy::ROUTER_AND_KEY.into());
    })
    .unwrap();
    let m3 = CmfModel::open(&f1).unwrap();
    assert_eq!(m3.header.skills[0].lookup.as_ref().unwrap().policy_label(), "router_and_key");
    let len3 = std::fs::metadata(&f1).unwrap().len();
    drop(m3);

    // An unknown value is refused by the header update (the file keeps
    // its bytes); open() READS a file that carries one — a newer writer's
    // policy must not make the genome unreadable (review KF-6).
    let e = CmfModel::update_header_append(&f1, |h| {
        h.skills[0].lookup.as_mut().unwrap().policy = Some("KEY_FIRST".into());
    })
    .unwrap_err()
    .to_string();
    assert!(
        e.contains("lookup.policy 'KEY_FIRST' (expected router_and_key | key_first)"),
        "{e}"
    );
    assert_eq!(std::fs::metadata(&f1).unwrap().len(), len3);
    assert!(len3 > len2);
    let tampered = dir.join("tampered.cmf");
    raw_header_swap(&f1, &tampered, |v| {
        v["skills"][0]["lookup"]["policy"] = serde_json::json!("sometimes");
    });
    let mt = CmfModel::open(&tampered).expect("an unknown policy does not refuse the file");
    let li = mt.header.skills[0].lookup.as_ref().unwrap();
    assert_eq!(li.policy_label(), "sometimes", "the value is kept as written");
    assert_eq!(
        cortiq_core::knowledge::unknown_lookup_policies(&mt.header),
        vec![("herbs".to_string(), "sometimes".to_string())]
    );
    assert_eq!(mt.trunk_hash(), h0);
    drop(mt);
    // The writers stay strict: a header update that keeps the unknown
    // value is refused (bytes untouched); one that sets a known policy
    // repairs the file.
    let tlen = std::fs::metadata(&tampered).unwrap().len();
    let e = CmfModel::update_header_append(&tampered, |_h| {})
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(e.contains("lookup.policy 'sometimes' (expected router_and_key | key_first)"), "{e}");
    assert_eq!(std::fs::metadata(&tampered).unwrap().len(), tlen);
    CmfModel::update_header_append(&tampered, |h| {
        h.skills[0].lookup.as_mut().unwrap().policy = Some(lookup_policy::ROUTER_AND_KEY.into());
    })
    .unwrap();
    let mt = CmfModel::open(&tampered).unwrap();
    assert!(cortiq_core::knowledge::unknown_lookup_policies(&mt.header).is_empty());
    drop(mt);
    raw_header_swap(&f1, &tampered, |v| {
        v["skills"][0]["lookup"]["policy"] = serde_json::json!(7);
    });
    assert!(CmfModel::open(&tampered).is_err(), "a non-string policy is not a header");
}

// ───────────────────────── key normalisation ─────────────────────────

#[test]
fn key_normalisation_cmf_key_v2_and_hash_stability() {
    assert_eq!(KEY_NORM, "cmf-key-v2");
    let cases: [(&str, &str); 30] = [
        ("Abies balsamea", "abies balsamea"),
        ("  ABIES   Balsamea  ", "abies balsamea"),
        ("Пихта Бальзамическая", "пихта бальзамическая"),
        (
            "Пихта бальзамическая (Abies balsamea)",
            "пихта бальзамическая abies balsamea",
        ),
        ("Hypericum perforatum (L.)", "hypericum perforatum l"),
        ("St. John's wort", "st john s wort"),
        ("зверобой-продырявленный, трава", "зверобой продырявленный трава"),
        ("Ромашка? №5!", "ромашка 5"),
        ("ABIES balsamea var. balsamea", "abies balsamea var balsamea"),
        // NFC composes a decomposed Cyrillic letter; it stays a letter.
        ("И\u{306}од", "йод"),
        ("Straße", "straße"),
        ("Ёлка/ель", "ёлка ель"),
        ("Ўзбек ў", "ўзбек ў"),
        // A stress mark over a Cyrillic vowel has no precomposed form:
        // it is dropped, the word stays whole (the ru-wiki spelling of
        // every plant name).
        ("Рома́шка апте́чная", "ромашка аптечная"),
        ("Клюква (Oxycóccus)", "клюква oxycoccus"),
        ("Helléborus", "helleborus"),
        // Invisible format characters inside a word do not end it.
        ("ро\u{AD}машка", "ромашка"),
        ("ро\u{200B}маш\u{200D}ка\u{FEFF}", "ромашка"),
        ("\u{FEFF}Abies\u{2060}balsamea", "abiesbalsamea"),
        // A precomposed Latin letter loses its diacritics, whichever
        // way it was spelled.
        ("e\u{301}cole", "ecole"),
        ("École", "ecole"),
        ("Pínus sylvéstris", "pinus sylvestris"),
        ("Taráxacum", "taraxacum"),
        ("Ångström", "angstrom"),
        ("İstanbul", "istanbul"),
        // Letters without a canonical decomposition stay.
        ("Æsculus", "æsculus"),
        ("Łąka", "łaka"),
        ("", ""),
        ("...", ""),
        (" \t\n", ""),
    ];
    for (input, want) in cases {
        assert_eq!(normalize_key(input), want, "{input:?}");
        assert_eq!(normalize_key(want), want, "idempotent on {want:?}");
    }
    // Decomposed and precomposed spellings share one key and one hash;
    // the accented and the plain spelling too.
    assert_eq!(key_hash("И\u{306}од"), key_hash("Йод"));
    assert_eq!(key_hash("e\u{301}cole"), key_hash("École"));
    assert_eq!(key_hash("École"), key_hash("ecole"));
    assert_eq!(key_hash("Рома́шка апте́чная"), key_hash("ромашка аптечная"));
    assert_eq!(key_hash("Pínus"), key_hash("pinus"));
    assert_eq!(key_hash("ро\u{AD}машка"), key_hash("ромашка"));
    assert_ne!(key_hash("йод"), key_hash("иод"), "й is a letter, not и + accent");
    assert_ne!(key_hash("ёлка"), key_hash("елка"));
    assert_eq!(key_hash("St. John's wort"), key_hash("st john s wort"));
    // The hash is hash64 of the normalised UTF-8.
    assert_eq!(key_hash("Abies balsamea"), hash64("abies balsamea".as_bytes()));
    assert_eq!(
        key_hash("  Abies, balsamea!"),
        normalized_key_hash("abies balsamea")
    );
    assert_ne!(key_hash("abies balsamea"), key_hash("abies alba"));
    // Golden values: what an already built table stores — stable across
    // versions (a pure-Python port of hash64 over the normalised key).
    assert_eq!(key_hash("Abies balsamea"), 0x3fa8_94e1_2d37_cea1);
    assert_eq!(key_hash("Пихта бальзамическая"), 0xcf80_8e5d_e438_49d2);
    assert_eq!(key_hash("balsam fir"), 0xb656_fc06_45f5_1cc2);
    assert_eq!(key_hash("Hypericum perforatum (L.)"), 0x9ed7_89f6_b1f6_e6d5);
    assert_eq!(key_hash("St. John's wort"), 0x4ff7_91f1_a00d_b73b);
    assert_eq!(key_hash(""), 0);
    assert_eq!(hash64(b"abc"), 0x2331_a186_8019_3f35);
}

// ───────────────────────── builder ─────────────────────────

#[test]
fn lookup_tensors_builder_sorts_keys_and_refuses_inconsistency() {
    let info = info();
    let (keys, slots) = herbs();
    let hashed: Vec<(u64, u32)> = keys.iter().map(|(k, e)| (key_hash(k), *e)).collect();
    let slot_refs: Vec<&str> = slots.iter().map(String::as_str).collect();
    let ts = lookup_tensors("h", &info, &hashed, &slot_refs).unwrap();
    assert_eq!(
        ts.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        lookup_leaf::ALL
            .iter()
            .map(|l| lookup_tensor_name("h", l))
            .collect::<Vec<_>>()
    );
    let hashes = read_u64_le(&ts[0].data);
    assert!(hashes.windows(2).all(|w| w[0] < w[1]));
    let entries = read_u32_le(&ts[1].data);
    for (k, e) in &keys {
        assert_eq!(entries[hashes.binary_search(&key_hash(k)).unwrap()], *e);
    }
    let offs = read_u64_le(&ts[2].data);
    assert_eq!(offs.len(), 7);
    assert_eq!(
        std::str::from_utf8(&ts[3].data[offs[5] as usize..offs[6] as usize]).unwrap(),
        slots[5]
    );
    let err = |info: &LookupInfo, keys: &[(u64, u32)], slots: &[&str]| {
        lookup_tensors("h", info, keys, slots).unwrap_err().to_string()
    };
    // Two keys that normalise alike.
    let mut dup = hashed.clone();
    dup.push((key_hash("ABIES  balsamea!"), 2));
    let mut info10 = info.clone();
    info10.keys = 10;
    let e = err(&info10, &dup, &slot_refs);
    assert!(e.contains("duplicate key hash") && e.contains("(entries 0 and 2)"), "{e}");
    // Entry out of range, key count, slot count, a slot that is not an object.
    let mut bad = hashed.clone();
    bad[0].1 = 3;
    assert!(err(&info, &bad, &slot_refs).contains("points at entry 3 of 3"));
    assert!(err(&info, &hashed[..8], &slot_refs).contains("8 keys given, lookup.keys says 9"));
    assert!(err(&info, &hashed, &slot_refs[..5]).contains("5 slots given, entries 3 × langs 2 = 6"));
    let mut arr = slot_refs.clone();
    arr[4] = "[1, 2]";
    assert!(err(&info, &hashed, &arr).contains("slot 4 is not a JSON object"));
    arr[4] = "{not json";
    assert!(err(&info, &hashed, &arr).contains("slot 4 is not JSON"));
}

// ───────────────────────── raw dtypes ─────────────────────────

#[test]
fn raw_u8_u32_u64_dtypes_are_fixed_size_and_round_trip_the_directory() {
    for (d, id, name, per) in [
        (TensorDtype::U8, 6u8, "u8", 1usize),
        (TensorDtype::U32, 17, "u32", 4),
        (TensorDtype::U64, 18, "u64", 8),
    ] {
        assert_eq!(d.id(), id);
        assert_eq!(TensorDtype::from_id(id), Some(d));
        assert_eq!(d.name(), name);
        assert!(d.is_raw());
        assert!(!d.is_supported(), "never decoded into f32");
        assert_eq!(expected_nbytes(d, &[5]), Some(5 * per));
        assert_eq!(expected_nbytes(d, &[2, 3]), Some(6 * per));
        assert_eq!(expected_nbytes(d, &[0]), Some(0));
    }
    assert!(!TensorDtype::F32.is_raw());
    assert_eq!(TensorDtype::from_id(19), None, "the next id is unknown");

    // A writer refuses a raw payload that disagrees with its shape …
    let dir = tempdir("raw");
    let p = dir.join("raw.cmf");
    let mut t = trunk_specs();
    t.push(TensorSpec {
        name: "aux.table".into(),
        dtype: TensorDtype::U64,
        shape: vec![3],
        data: vec![0; 16],
    });
    let e = CmfModel::write(&p, &plain_header(), &t, None, Some(VOCAB))
        .unwrap_err()
        .to_string();
    assert!(e.contains("data 16 bytes != expected 24"), "{e}");
    // … and stores the exact bytes of one that agrees, under dtype id 18.
    let payload: Vec<u8> = (0..24).collect();
    t.last_mut().unwrap().data = payload.clone();
    CmfModel::write(&p, &plain_header(), &t, None, Some(VOCAB)).unwrap();
    let m = CmfModel::open(&p).unwrap();
    let e = m.tensor("aux.table").unwrap();
    assert_eq!((e.dtype, e.shape.clone(), e.nbytes), (TensorDtype::U64, vec![3], 24));
    assert_eq!(m.entry_bytes(e), payload.as_slice());
    assert_eq!(read_u64_le(m.entry_bytes(e))[1], u64::from_le_bytes([8, 9, 10, 11, 12, 13, 14, 15]));
    assert!(m.verify().is_empty());
}
