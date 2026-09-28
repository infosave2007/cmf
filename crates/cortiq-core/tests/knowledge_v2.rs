//! Format v2 "knowledge without forgetting" (spec §9.2–§9.5): the frozen
//! genome and its trunk hash, the three derived feature bits, v2 skill
//! records and their open-time validation, the true tail append and its
//! crash ordering, header-only updates, and what an older reader does
//! with the result.

use cortiq_core::format::{CmfStreamWriter, TokenizerBundle, check_required_features, features};
use cortiq_core::knowledge::{hex64, skill_kind, state_switch};
use cortiq_core::mask::{MaskCatalog, MaskPriority, TaskMask};
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType};
use cortiq_core::{
    CmfError, CmfHeader, CmfModel, GenomeInfo, LineageEvent, PhiSpec, RouterPolicy,
    RoutingCalibration, SelectionDescriptor, SkillBound, SkillOverride, SkillRecord, TensorDtype,
    TensorSpec, ffn_replace_state_effect, hash64,
};
use std::path::{Path, PathBuf};

const H: usize = 64;
const I: usize = 128;
const L: usize = 4;
const V: usize = 10;

/// Bits a reader knew before the knowledge bits existed.
const OLD_SUPPORTED: u32 = features::TENSOR_DIR
    | features::BINARY_MASKS
    | features::QUANT_2F
    | features::LOOP_MASKS
    | features::SKILL_FILE
    | features::BOUNDED_STATE;

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
    h.lineage = vec![LineageEvent::now(
        0,
        "birth",
        serde_json::json!({"step": 0}),
    )];
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

fn masks() -> MaskCatalog {
    let a = arch();
    let mut ffn = vec![0u8; a.ffn_mask_bytes()];
    for i in 0..I / 2 {
        ffn[i / 8] |= 1 << (i % 8);
    }
    let head = vec![0b11u8; a.head_mask_bytes()];
    MaskCatalog {
        masks: vec![TaskMask {
            task_id: 0,
            name: "general".into(),
            description: None,
            sparsity: 0.5,
            quality: None,
            ffn_masks: vec![ffn; L],
            head_masks: vec![head; L],
            layer_gates: vec![true; L],
            expert_masks: Vec::new(),
            parent: None,
            has_hot_pack: false,
            priority: MaskPriority::Fallback,
        }],
        default_task: "general".into(),
    }
}

const VOCAB: &[u8] = br#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "cmf-knowledge-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// F0: a sealed-at-birth genome with masks, vocab and sparse index.
fn write_f0(path: &Path) {
    CmfModel::write(
        path,
        &genome_header(),
        &trunk_specs(),
        Some(&masks()),
        Some(VOCAB),
    )
    .unwrap();
}

/// A v2 `ffn_replace` record over `layers` (up+down), bound to `model`'s
/// genome, with its tensors.
fn ffn_skill(model: &CmfModel, id: &str, layers: &[usize]) -> (SkillRecord, Vec<TensorSpec>) {
    let g = model.header.genome.clone().expect("genome");
    let mut overrides = Vec::new();
    let mut tensors = Vec::new();
    for &l in layers {
        for proj in ["up", "down"] {
            let name = format!("model.layers.{l}.mlp.{proj}_proj.weight");
            let base = model.tensor(&name).unwrap();
            tensors.push(spec(
                &format!("skill.{id}.{name}"),
                &base.shape,
                9.0 + l as f32,
            ));
            overrides.push(SkillOverride {
                name,
                base_hash: hex64(base.hash),
            });
        }
    }
    let record = SkillRecord {
        id: id.into(),
        layers: layers.to_vec(),
        kind: Some(skill_kind::FFN_REPLACE.into()),
        overrides,
        bound: Some(SkillBound {
            genome_id: g.id,
            generation: g.generation,
            master_trunk_hash: g.master_trunk_hash,
        }),
        status: Some("quarantine".into()),
        prompt_contract: Some("cmf-im-v1".into()),
        origin: Some(serde_json::json!({"trigger": "user_corpus"})),
        ..Default::default()
    };
    (record, tensors)
}

/// F1 = F0 + skill "herbs" on layer 3, by true append.
fn make_f1(dir: &Path) -> (PathBuf, PathBuf) {
    let f0 = dir.join("f0.cmf");
    let f1 = dir.join("f1.cmf");
    write_f0(&f0);
    std::fs::copy(&f0, &f1).unwrap();
    let m = CmfModel::open(&f1).unwrap();
    let (rec, ts) = ffn_skill(&m, "herbs", &[3]);
    drop(m);
    CmfModel::append_skill(&f1, rec, &ts, None, None, None).unwrap();
    (f0, f1)
}

fn unit(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

fn f16_b64(v: &[f32]) -> String {
    use base64::Engine as _;
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| cortiq_core::quant::f32_to_f16(*x).to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn descriptor(layer: usize, axis: usize) -> SelectionDescriptor {
    let mut mean = vec![0.0f32; H];
    mean[axis] = 1.0;
    unit(&mut mean);
    let mut basis = vec![0.0f32; H];
    basis[(axis + 1) % H] = 1.0;
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

fn policy(layer: usize) -> RouterPolicy {
    RouterPolicy {
        version: 2,
        policy: "backbone_gated".into(),
        granularity: "request".into(),
        phi: PhiSpec {
            layer,
            pool: "span_mean".into(),
            norm: "unit".into(),
            prefix_ids: vec![1, 5],
            suffix_ids: vec![2, 1, 6],
        },
        base: descriptor(layer, 0),
        margin: 0.05,
        skills_hash: "0000000000000000".into(),
        measured: None,
    }
}

fn open_err(p: &Path) -> String {
    match CmfModel::open(p) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("{} opened but must be refused", p.display()),
    }
}

fn bits_of(p: &Path) -> u32 {
    let b = std::fs::read(p).unwrap();
    u32::from_le_bytes(b[12..16].try_into().unwrap())
}

fn set_bits(src: &Path, dst: &Path, bits: u32) {
    let mut b = std::fs::read(src).unwrap();
    b[12..16].copy_from_slice(&bits.to_le_bytes());
    std::fs::write(dst, &b).unwrap();
}

/// Publish a MUTATED header without any writer-side validation: the JSON
/// goes to the end of a copy and the envelope points at it — so what is
/// tested is open()'s refusal, not the writer's.
fn raw_header_swap(
    src: &Path,
    dst: &Path,
    clear_bits: u32,
    mutate: impl FnOnce(&mut serde_json::Value),
) {
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
    let bits = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) & !clear_bits;
    bytes[12..16].copy_from_slice(&bits.to_le_bytes());
    std::fs::write(dst, &bytes).unwrap();
}

fn assert_refused(p: &Path, needle: &str) {
    let e = open_err(p);
    assert!(
        e.contains(needle),
        "refusal of {} was '{e}', expected '{needle}'",
        p.display()
    );
}

// ───────────────────────── trunk hash ─────────────────────────

#[test]
fn trunk_hash_is_stable_and_covers_exactly_the_trunk() {
    let dir = tempdir("trunk");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let m0 = CmfModel::open(&f0).unwrap();
    let h0 = m0.trunk_hash();
    let g = m0.header.genome.clone().unwrap();
    assert_eq!(
        g.trunk_hash,
        hex64(h0),
        "writer fills the trunk hash from content"
    );
    assert_eq!(g.master_trunk_hash, g.trunk_hash, "f32: master == trunk");
    assert_ne!(m0.required_features & features::GENOME, 0);
    assert_eq!(
        m0.required_features & (features::SKILLS_V2 | features::ROUTER_V2),
        0
    );
    assert!(m0.verify().is_empty());

    // Re-write the opened file (stored trunk hash kept): same trunk hash.
    let again = dir.join("again.cmf");
    let specs: Vec<TensorSpec> = m0
        .tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: m0.entry_bytes(e).to_vec(),
        })
        .collect();
    CmfModel::write(
        &again,
        &m0.header,
        &specs,
        Some(&m0.masks),
        m0.vocab.as_deref(),
    )
    .unwrap();
    assert_eq!(CmfModel::open(&again).unwrap().trunk_hash(), h0);

    // Variants on plain files (no genome): what moves the hash and what not.
    let hash_of = |name: &str, header: &CmfHeader, t: &[TensorSpec], vocab: &[u8]| {
        let p = dir.join(name);
        CmfModel::write(&p, header, t, Some(&masks()), Some(vocab)).unwrap();
        CmfModel::open(&p).unwrap().trunk_hash()
    };
    let base = hash_of("plain.cmf", &plain_header(), &trunk_specs(), VOCAB);
    assert_eq!(
        base, h0,
        "the genome record itself is not part of the trunk"
    );

    let mut t = trunk_specs();
    t[3].data[17] ^= 1;
    assert_ne!(
        hash_of("byte.cmf", &plain_header(), &t, VOCAB),
        base,
        "one payload byte"
    );
    let mut hdr = plain_header();
    hdr.arch.rope_theta = 20_000.0;
    assert_ne!(
        hash_of("arch.cmf", &hdr, &trunk_specs(), VOCAB),
        base,
        "arch"
    );
    let mut vocab2 = VOCAB.to_vec();
    vocab2[5] ^= 1;
    assert_ne!(
        hash_of("vocab.cmf", &plain_header(), &trunk_specs(), &vocab2),
        base,
        "vocab"
    );
    let mut hdr = plain_header();
    hdr.tokenizer_config.as_mut().unwrap().eos_token_ids = vec![3];
    assert_ne!(
        hash_of("eos.cmf", &hdr, &trunk_specs(), VOCAB),
        base,
        "tokenizer bundle"
    );
    let mut t = trunk_specs();
    t[0].name = "model.embed.weight".into();
    assert_ne!(
        hash_of("name.cmf", &plain_header(), &t, VOCAB),
        base,
        "tensor name"
    );

    // skill.* tensors, router, routing and lineage are NOT the trunk.
    let mut hdr = plain_header();
    hdr.skills.push(SkillRecord {
        id: "v1".into(),
        layers: vec![1],
        ..Default::default()
    });
    hdr.routing = Some(RoutingCalibration {
        temperature: 0.5,
        novelty_theta: 0.4,
        samples: 10,
        target_fpr: 0.05,
    });
    hdr.router = Some(policy(0));
    hdr.lineage = vec![LineageEvent::now(7, "recalibrate", serde_json::Value::Null)];
    let mut t = trunk_specs();
    t.push(spec(
        "skill.v1.model.layers.1.mlp.up_proj.weight",
        &[I, H],
        5.0,
    ));
    t.push(spec("route.probe.weight", &[H], 6.0));
    assert_eq!(hash_of("skills.cmf", &hdr, &t, VOCAB), base);

    // The writer refuses to carry a genome over a changed trunk.
    let mut t = trunk_specs();
    t[3].data[17] ^= 1;
    let e = CmfModel::write(dir.join("stale.cmf"), &m0.header, &t, None, Some(VOCAB))
        .unwrap_err()
        .to_string();
    assert!(e.contains("trunk changed without a new genome"), "{e}");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn open_refuses_a_directory_level_trunk_change_and_verify_catches_a_payload_change() {
    let dir = tempdir("tamper");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let m = CmfModel::open(&f0).unwrap();
    let idx = m.tensor_index("model.layers.2.mlp.up_proj.weight").unwrap();
    let abs = m.entry_abs_offset(&m.tensors[idx]).unwrap();
    let bytes = std::fs::read(&f0).unwrap();
    let dir_off = u64::from_le_bytes(bytes[0x20..0x28].try_into().unwrap()) as usize;
    drop(m);

    // 1 payload byte under an unchanged directory: open cannot see it (the
    // trunk hash is directory-level), verify does.
    let mut b = bytes.clone();
    b[abs + 5] ^= 0x40;
    let p = dir.join("payload.cmf");
    std::fs::write(&p, &b).unwrap();
    let opened = CmfModel::open(&p).expect("payload change is verify's job");
    let problems = opened.verify();
    assert!(
        problems
            .iter()
            .any(|s| s.contains("model.layers.2.mlp.up_proj.weight")),
        "{problems:?}"
    );

    // The directory entry's hash changed (payload re-hashed, no new
    // genome): refused at open.
    let mut b = bytes.clone();
    let rec = dir_off + 16 + idx * 56;
    b[rec + 48] ^= 0x01;
    let p = dir.join("dir.cmf");
    std::fs::write(&p, &b).unwrap();
    assert_refused(&p, "trunk_hash mismatch");
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── feature bits ─────────────────────────

#[test]
fn knowledge_bits_are_derived_and_forgery_is_refused_both_ways() {
    let dir = tempdir("bits");
    let (f0, f1) = make_f1(&dir);
    let plain = dir.join("plain.cmf");
    CmfModel::write(&plain, &plain_header(), &trunk_specs(), None, None).unwrap();
    assert_eq!(
        bits_of(&plain) & (features::GENOME | features::SKILLS_V2 | features::ROUTER_V2),
        0
    );
    assert_ne!(bits_of(&f0) & features::GENOME, 0);
    assert_eq!(bits_of(&f0) & features::SKILLS_V2, 0);
    let b1 = bits_of(&f1);
    assert_ne!(b1 & features::GENOME, 0);
    assert_ne!(b1 & features::SKILLS_V2, 0);
    assert_eq!(b1 & features::ROUTER_V2, 0, "no router declared yet");

    // Content present, bit cleared → refused.
    let p = dir.join("x.cmf");
    set_bits(&f1, &p, b1 & !features::GENOME);
    assert_refused(&p, "GENOME feature bit (clear) disagree");
    set_bits(&f1, &p, b1 & !features::SKILLS_V2);
    assert_refused(&p, "SKILLS_V2 feature bit (clear) disagree");
    // Bit forged on content that is absent → refused.
    let pb = bits_of(&plain);
    for (bit, name) in [
        (features::GENOME, "GENOME"),
        (features::SKILLS_V2, "SKILLS_V2"),
        (features::ROUTER_V2, "ROUTER_V2"),
    ] {
        set_bits(&plain, &p, pb | bit);
        assert_refused(&p, &format!("{name} feature bit (set) disagree"));
    }

    // Router: bit follows content through a header-only update, and
    // clearing it by hand is refused.
    let f2 = dir.join("f2.cmf");
    std::fs::copy(&f1, &f2).unwrap();
    CmfModel::update_header_append(&f2, |h| h.router = Some(policy(1))).unwrap();
    let b2 = bits_of(&f2);
    assert_ne!(b2 & features::ROUTER_V2, 0);
    set_bits(&f2, &p, b2 & !features::ROUTER_V2);
    assert_refused(&p, "ROUTER_V2 feature bit (clear) disagree");

    // The streaming writer derives the same bits.
    let sp = dir.join("stream.cmf");
    let mut w = CmfStreamWriter::new(&sp, CmfStreamWriter::head_reserve_for(16, 40)).unwrap();
    for t in trunk_specs() {
        w.push(&t.name, t.dtype, &t.shape, &t.data).unwrap();
    }
    w.finish(&genome_header(), Some(&masks()), Some(VOCAB))
        .unwrap();
    let ms = CmfModel::open(&sp).unwrap();
    assert_ne!(ms.required_features & features::GENOME, 0);
    assert_eq!(ms.trunk_hash(), CmfModel::open(&f0).unwrap().trunk_hash());
    std::fs::remove_dir_all(dir).ok();
}

/// G6: the reader as it was before the knowledge bits (its SUPPORTED mask)
/// refuses every new file, and still opens the old kind.
#[test]
fn a_reader_without_the_new_bits_refuses_new_files() {
    let dir = tempdir("old-reader");
    let (f0, f1) = make_f1(&dir);
    let plain = dir.join("plain.cmf");
    CmfModel::write(
        &plain,
        &plain_header(),
        &trunk_specs(),
        Some(&masks()),
        Some(VOCAB),
    )
    .unwrap();
    assert!(check_required_features(bits_of(&plain), OLD_SUPPORTED).is_ok());
    for (p, expect) in [
        (&f0, features::GENOME),
        (&f1, features::GENOME | features::SKILLS_V2),
    ] {
        match check_required_features(bits_of(p), OLD_SUPPORTED) {
            Err(CmfError::UnsupportedFeature(u)) => assert_eq!(u, expect),
            other => panic!("old reader accepted {}: {other:?}", p.display()),
        }
        // …and this reader accepts it.
        assert!(check_required_features(bits_of(p), features::SUPPORTED).is_ok());
    }
    std::fs::remove_dir_all(dir).ok();
}

/// A pre-0.8.1 embryo-o1 build wrote GENOME on bit 8 (today's
/// PRISM_AFFINE). Such a file is refused with a pointer at the migration,
/// is recognised by content (not by bit number), and the in-place
/// migration changes ONLY envelope bytes 12..16 — the appended record
/// survives and the trunk hash is unchanged.
#[test]
fn legacy_embryo_bits_are_migrated_in_place_and_appends_survive() {
    use cortiq_core::format::{
        LegacyBitsMigration, header_json_is_genome, legacy_embryo_bits, peek_header,
    };
    let dir = tempdir("legacy-bits");
    let (_f0, f1) = make_f1(&dir);
    let b1 = bits_of(&f1);
    assert_ne!(b1 & features::GENOME, 0);
    assert_eq!(b1 & legacy_embryo_bits::BOTH, 0);
    let legacy_bits = (b1 & !features::GENOME) | legacy_embryo_bits::GENOME;
    let leg = dir.join("legacy.cmf");
    set_bits(&f1, &leg, legacy_bits);
    let e = open_err(&leg);
    assert!(e.contains("PRISM_AFFINE") && e.contains("migrate-embryo-bits"), "{e}");
    let (req, header) = peek_header(&leg).unwrap().unwrap();
    assert_eq!(req, legacy_bits);
    assert!(header_json_is_genome(&header), "content check must see the genome");

    // Dry run reports the move and writes nothing.
    assert_eq!(
        CmfModel::migrate_legacy_embryo_bits(&leg, true).unwrap(),
        LegacyBitsMigration::Migrated { from: legacy_bits, to: b1 }
    );
    assert_eq!(bits_of(&leg), legacy_bits);
    assert_eq!(
        CmfModel::migrate_legacy_embryo_bits(&leg, false).unwrap(),
        LegacyBitsMigration::Migrated { from: legacy_bits, to: b1 }
    );
    let (a, b) = (std::fs::read(&leg).unwrap(), std::fs::read(&f1).unwrap());
    assert_eq!(a, b, "the migration must restore the file byte for byte");
    let m = CmfModel::open(&leg).unwrap();
    assert!(m.header.skills.iter().any(|s| s.id == "herbs"));
    assert_eq!(m.trunk_hash(), CmfModel::open(&f1).unwrap().trunk_hash());
    drop(m);
    assert!(matches!(
        CmfModel::migrate_legacy_embryo_bits(&leg, false).unwrap(),
        LegacyBitsMigration::NothingToDo { .. }
    ));

    // A plain file has nothing to move; a stray bit 8 without any Embryo
    // or Prism record is refused, not guessed; a non-CMF file is reported.
    let plain = dir.join("plain.cmf");
    CmfModel::write(&plain, &plain_header(), &trunk_specs(), None, None).unwrap();
    assert!(!header_json_is_genome(&peek_header(&plain).unwrap().unwrap().1));
    assert!(matches!(
        CmfModel::migrate_legacy_embryo_bits(&plain, false).unwrap(),
        LegacyBitsMigration::NothingToDo { .. }
    ));
    let stray = dir.join("stray.cmf");
    set_bits(&plain, &stray, bits_of(&plain) | legacy_embryo_bits::GENOME);
    let e = CmfModel::migrate_legacy_embryo_bits(&stray, false)
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a pre-0.8.1 Embryo file"), "{e}");
    let txt = dir.join("notes.txt");
    std::fs::write(&txt, vec![b'x'; 200]).unwrap();
    assert_eq!(
        CmfModel::migrate_legacy_embryo_bits(&txt, true).unwrap(),
        LegacyBitsMigration::NotCmf
    );
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── skill records v2 ─────────────────────────

#[test]
fn ffn_replace_state_effect_follows_the_layer_schedule() {
    // fam-a: anchors at 3 and 7, GDN elsewhere.
    let mut a = arch();
    a.num_layers = 8;
    a.layer_types = (0..8)
        .map(|i| {
            if i == 3 || i == 7 {
                LayerType::BoundedAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    let e = ffn_replace_state_effect(&a, &[7]);
    assert_eq!(
        (e.first_affected_layer, e.switch.as_str()),
        (7, state_switch::TRANSPARENT)
    );
    assert_eq!(e.state_bytes_added, 0);
    let e = ffn_replace_state_effect(&a, &[6, 7]);
    assert_eq!(
        (e.first_affected_layer, e.switch.as_str()),
        (6, state_switch::SEQUENCE_START)
    );
    let e = ffn_replace_state_effect(&a, &[4, 5, 6, 7]);
    assert_eq!(
        (e.first_affected_layer, e.switch.as_str()),
        (4, state_switch::SEQUENCE_START)
    );
    a.num_loops = 2;
    assert_eq!(
        ffn_replace_state_effect(&a, &[7]).switch,
        state_switch::SEQUENCE_START
    );
}

#[test]
fn skills_v2_validation_refuses_each_violation_precisely() {
    let dir = tempdir("v2");
    let (_f0, f1) = make_f1(&dir);
    let m1 = CmfModel::open(&f1).unwrap();
    let rec = &m1.header.skills[0];
    assert_eq!(rec.kind.as_deref(), Some("ffn_replace"));
    let se = rec
        .state_effect
        .clone()
        .expect("writer computed the state effect");
    assert_eq!(
        (se.first_affected_layer, se.switch.as_str()),
        (3, "transparent")
    );
    drop(m1);

    let p = dir.join("bad.cmf");
    type Mutation = Box<dyn Fn(&mut serde_json::Value)>;
    let cases: Vec<(&str, Mutation, &str)> = vec![
        (
            "missing bound",
            Box::new(|v| {
                v["skills"][0].as_object_mut().unwrap().remove("bound");
            }),
            "v2 record has no `bound`",
        ),
        (
            "wrong master hash",
            Box::new(|v| v["skills"][0]["bound"]["master_trunk_hash"] = "0123456789abcdef".into()),
            "bound.master_trunk_hash 0123456789abcdef != genome.master_trunk_hash",
        ),
        (
            "wrong genome id",
            Box::new(|v| v["skills"][0]["bound"]["genome_id"] = "other".into()),
            "bound.genome_id 'other' != genome.id 'embryo-test'",
        ),
        (
            "unknown kind",
            Box::new(|v| v["skills"][0]["kind"] = "lora_delta".into()),
            "unknown kind 'lora_delta'",
        ),
        (
            "reserved kind",
            Box::new(|v| v["skills"][0]["kind"] = "anchor_sinks".into()),
            "kind 'anchor_sinks' is reserved and not implemented",
        ),
        (
            "expert_append parameters on an ffn_replace record",
            Box::new(|v| {
                v["skills"][0]["experts"] =
                    serde_json::json!({"count": 1, "shell_quantile": 0.99, "rank": 0});
            }),
            "`experts` belongs to an expert_append record, not to ffn_replace",
        ),
        (
            "override without tensor",
            Box::new(|v| {
                let o = serde_json::json!({"name": "model.layers.2.mlp.up_proj.weight",
                                           "base_hash": "0000000000000001"});
                v["skills"][0]["overrides"].as_array_mut().unwrap().push(o);
            }),
            "override 'model.layers.2.mlp.up_proj.weight' has no tensor \
             'skill.herbs.model.layers.2.mlp.up_proj.weight'",
        ),
        (
            "tensor without override",
            Box::new(|v| {
                v["skills"][0]["overrides"]
                    .as_array_mut()
                    .unwrap()
                    .remove(1);
            }),
            "tensor 'skill.herbs.model.layers.3.mlp.down_proj.weight' has no override entry",
        ),
        (
            "base hash of other bytes",
            Box::new(|v| v["skills"][0]["overrides"][0]["base_hash"] = "00000000000000aa".into()),
            "base_hash 00000000000000aa != trunk entry hash",
        ),
        (
            "layers disagree with tensors",
            Box::new(|v| v["skills"][0]["layers"] = serde_json::json!([2, 3])),
            "layers [2, 3] != layers named by its tensors {3}",
        ),
        (
            "missing state effect",
            Box::new(|v| {
                v["skills"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("state_effect");
            }),
            "v2 record has no `state_effect`",
        ),
        (
            "first affected layer",
            Box::new(|v| v["skills"][0]["state_effect"]["first_affected_layer"] = 1.into()),
            "state_effect.first_affected_layer 1 != min(layers) 3",
        ),
        (
            "unknown status",
            Box::new(|v| v["skills"][0]["status"] = "shipping".into()),
            "status 'shipping' is not one of",
        ),
        (
            "v2 fields on a v1 record",
            Box::new(|v| {
                let legacy = serde_json::json!({"id": "legacy", "layers": [1], "status": "active"});
                v["skills"].as_array_mut().unwrap().push(legacy);
            }),
            "v2 fields (overrides/bound/state_effect/status/gate/prompt_contract/origin/experts/\
             lookup) on a record without `kind`",
        ),
    ];
    for (name, mutate, needle) in cases {
        raw_header_swap(&f1, &p, 0, mutate);
        let e = open_err(&p);
        assert!(
            e.contains(needle),
            "case '{name}': refusal was '{e}', expected '{needle}'"
        );
    }

    // Shape mismatch: an equal-size transposed skill tensor. Refused by the
    // append before a byte is written…
    let m1 = CmfModel::open(&f1).unwrap();
    let (rec, mut ts) = ffn_skill(&m1, "bent", &[2]);
    drop(m1);
    ts[0].shape = vec![H, I];
    let len_before = std::fs::metadata(&f1).unwrap().len();
    let e = CmfModel::append_skill(&f1, rec, &ts, None, None, None)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("'skill.bent.model.layers.2.mlp.up_proj.weight' shape [64, 128] != trunk"),
        "{e}"
    );
    assert_eq!(
        std::fs::metadata(&f1).unwrap().len(),
        len_before,
        "nothing written"
    );
    // …and by open() when the directory says so (dims patched in place).
    let m1 = CmfModel::open(&f1).unwrap();
    let idx = m1
        .tensor_index("skill.herbs.model.layers.3.mlp.up_proj.weight")
        .unwrap();
    drop(m1);
    let mut b = std::fs::read(&f1).unwrap();
    let dir_off = u64::from_le_bytes(b[0x20..0x28].try_into().unwrap()) as usize;
    let rec_at = dir_off + 16 + idx * 56;
    b[rec_at + 8..rec_at + 12].copy_from_slice(&(H as u32).to_le_bytes());
    b[rec_at + 12..rec_at + 16].copy_from_slice(&(I as u32).to_le_bytes());
    std::fs::write(&p, &b).unwrap();
    assert_refused(
        &p,
        "shape [64, 128] != trunk 'model.layers.3.mlp.up_proj.weight' shape [128, 64]",
    );

    // A state effect more permissive than the schedule allows: layer 2's
    // FFN feeds layer 3's attention.
    let m1 = CmfModel::open(&f1).unwrap();
    let (mut rec, ts) = ffn_skill(&m1, "early", &[2]);
    drop(m1);
    rec.state_effect = Some(cortiq_core::StateEffect {
        first_affected_layer: 2,
        switch: "transparent".into(),
        state_bytes_added: 0,
    });
    let e = CmfModel::append_skill(&f1, rec, &ts, None, None, None)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("more permissive than the layer schedule allows ('sequence_start')"),
        "{e}"
    );

    // A skill tensor that belongs to no record.
    raw_header_swap(&f1, &p, features::SKILLS_V2, |v| {
        v["skills"] = serde_json::json!([]);
    });
    assert_refused(
        &p,
        "tensor 'skill.herbs.model.layers.3.mlp.up_proj.weight' belongs to no skill record",
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn resolve_tensor_never_falls_back_for_a_v2_override() {
    let dir = tempdir("resolve");
    let (f0, f1) = make_f1(&dir);
    let m0 = CmfModel::open(&f0).unwrap();
    let mut m1 = CmfModel::open(&f1).unwrap();
    let up = "model.layers.3.mlp.up_proj.weight";
    let s = m1.resolve_tensor(up, Some("herbs")).unwrap();
    assert_eq!(s.name, format!("skill.herbs.{up}"));
    assert_eq!(m1.entry_bytes(s), &f32s(I * H, 12.0)[..]);
    let b = m1.resolve_tensor(up, None).unwrap();
    assert_eq!(
        m1.entry_bytes(b),
        m0.tensor_bytes(up).unwrap(),
        "backbone bytes untouched"
    );
    // Not overridden → the backbone tensor, as for v1 records.
    let other = "model.layers.1.mlp.up_proj.weight";
    assert_eq!(m1.resolve_tensor(other, Some("herbs")).unwrap().name, other);

    // An override whose replacement is missing (open() refuses such a file;
    // simulate a header edited in memory): an error, never the backbone.
    m1.header.skills[0].overrides.push(SkillOverride {
        name: other.into(),
        base_hash: "0".into(),
    });
    assert!(m1.resolve_tensor(other, Some("herbs")).is_none());
    assert!(matches!(
        m1.try_resolve_tensor(other, Some("herbs")),
        Err(CmfError::MissingTensor(_))
    ));
    std::fs::remove_dir_all(dir).ok();
}

// ───────────────────────── true append ─────────────────────────

#[test]
fn append_skill_is_a_true_tail_append() {
    let dir = tempdir("append");
    let (f0, f1) = make_f1(&dir);
    let b0 = std::fs::read(&f0).unwrap();
    let b1 = std::fs::read(&f1).unwrap();
    assert!(b1.len() > b0.len());
    assert_eq!(
        &b1[128..b0.len()],
        &b0[128..],
        "prefix [128, len(F0)) is byte-identical"
    );

    let m0 = CmfModel::open(&f0).unwrap();
    let m1 = CmfModel::open(&f1).unwrap();
    let n0 = m0.tensors.len();
    assert_eq!(
        &m1.tensors[..n0],
        &m0.tensors[..],
        "old entries: same off/hash/shape/dtype"
    );
    assert_eq!(m1.tensors.len(), n0 + 2);
    for t in &m1.tensors[n0..] {
        assert!(t.name.starts_with("skill.herbs.model.layers.3.mlp."));
        assert_eq!(t.off % 4096, 0, "large tail tensors stay page-aligned");
    }
    assert_eq!(m1.trunk_hash(), m0.trunk_hash(), "G1: trunk hash unchanged");
    assert_eq!(m1.header.genome, m0.header.genome);
    assert_eq!(m1.vocab, m0.vocab);
    assert_eq!(m1.masks.masks.len(), 1);
    assert_eq!(m1.sparse_index, m0.sparse_index);
    assert!(m1.verify().is_empty(), "{:?}", m1.verify());

    // masks/vocab/index now lie INSIDE the extended data range — legal.
    let u64at = |b: &[u8], o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let (data_off, data_len1) = (u64at(&b1, 0x30), u64at(&b1, 0x38));
    let vocab_off = u64at(&b1, 0x50);
    assert_eq!(u64at(&b0, 0x30), data_off, "data_off never moves");
    assert!(vocab_off > data_off && vocab_off < data_off + data_len1);
    assert!(
        u64at(&b1, 0x10) >= b0.len() as u64,
        "new header lives in the tail"
    );

    // Segments + lineage.
    let seg = &m1.header.segments;
    assert_eq!(seg.len(), 2);
    assert_eq!(
        (seg[0].kind.as_str(), seg[0].id.as_str()),
        ("genome", "embryo-test")
    );
    assert_eq!((seg[0].data_start, seg[0].data_end), (0, u64at(&b0, 0x38)));
    assert_eq!(
        (seg[1].kind.as_str(), seg[1].id.as_str()),
        ("skill", "herbs")
    );
    assert_eq!(seg[1].data_end, data_len1);
    let last = m1.header.lineage.last().unwrap();
    assert_eq!((last.seq, last.event.as_str()), (1, "skill_committed"));
    assert_eq!(last.detail["skill"], "herbs");
    drop((m0, m1));

    // A second append works on top: F1's bytes are F2's prefix.
    let f2 = dir.join("f2.cmf");
    std::fs::copy(&f1, &f2).unwrap();
    let m = CmfModel::open(&f2).unwrap();
    let (rec, ts) = ffn_skill(&m, "herbs2", &[2]);
    drop(m);
    let rep = CmfModel::append_skill(&f2, rec, &ts, None, None, None).unwrap();
    assert_eq!(rep.tensors_added, 2);
    assert_eq!(rep.old_len, b1.len() as u64);
    let b2 = std::fs::read(&f2).unwrap();
    assert_eq!(&b2[128..b1.len()], &b1[128..]);
    let m1 = CmfModel::open(&f1).unwrap();
    let m2 = CmfModel::open(&f2).unwrap();
    assert_eq!(&m2.tensors[..m1.tensors.len()], &m1.tensors[..]);
    assert_eq!(m2.header.skills.len(), 2);
    assert_eq!(
        m2.header.skills[1].state_effect.as_ref().unwrap().switch,
        "sequence_start",
        "layer 2 feeds layer 3's attention"
    );
    assert_eq!(m2.header.segments.len(), 3);
    assert_eq!(m2.header.lineage.last().unwrap().seq, 2);
    assert!(m2.verify().is_empty());
    assert_eq!(m2.trunk_hash(), m1.trunk_hash());
    drop(m2);

    // Refusals: a duplicate id, a tensor outside the namespace.
    let m = CmfModel::open(&f2).unwrap();
    let (rec, ts) = ffn_skill(&m, "herbs", &[1]);
    let (rec3, mut ts3) = ffn_skill(&m, "third", &[1]);
    drop(m);
    let e = CmfModel::append_skill(&f2, rec, &ts, None, None, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("skill 'herbs' already exists"), "{e}");
    ts3[0].name = "model.layers.1.mlp.up_proj.weight".into();
    let e = CmfModel::append_skill(&f2, rec3, &ts3, None, None, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("outside the skill's namespace"), "{e}");
    assert_eq!(
        std::fs::read(&f2).unwrap(),
        b2,
        "refused appends leave the file alone"
    );

    // A GENOME file refuses in-place recoding.
    let e = CmfModel::recode_entries_in_place(f2.to_str().unwrap(), &[])
        .unwrap_err()
        .to_string();
    assert!(e.contains("carries a frozen genome"), "{e}");
    std::fs::remove_dir_all(dir).ok();
}

/// Killed between the tail and the envelope: the file is F0, exactly.
#[test]
fn a_crash_before_the_envelope_leaves_the_old_file() {
    let dir = tempdir("crash");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let b0 = std::fs::read(&f0).unwrap();
    let p = dir.join("f.cmf");
    std::fs::copy(&f0, &p).unwrap();
    let m = CmfModel::open(&p).unwrap();
    let (rec, ts) = ffn_skill(&m, "herbs", &[3]);
    drop(m);

    let pending = CmfModel::prepare_append_skill(&p, rec.clone(), &ts, None, None, None).unwrap();
    assert!(pending.report().new_len > b0.len() as u64);
    drop(pending); // process dies here: payload + header + dir on disk, envelope not

    let bc = std::fs::read(&p).unwrap();
    assert!(bc.len() > b0.len(), "the tail was written");
    assert_eq!(&bc[..b0.len()], &b0[..], "envelope and prefix untouched");
    let m0 = CmfModel::open(&f0).unwrap();
    let mc = CmfModel::open(&p).unwrap();
    assert_eq!(
        serde_json::to_value(&mc.header).unwrap(),
        serde_json::to_value(&m0.header).unwrap()
    );
    assert_eq!(mc.tensors, m0.tensors);
    assert_eq!(mc.required_features, m0.required_features);
    assert!(mc.verify().is_empty());
    assert!(mc.header.skills.is_empty());
    drop(mc);

    // The next append simply starts after the abandoned tail.
    CmfModel::append_skill(&p, rec, &ts, None, None, None).unwrap();
    let m = CmfModel::open(&p).unwrap();
    assert_eq!(m.header.skills.len(), 1);
    assert!(m.verify().is_empty());
    assert_eq!(m.trunk_hash(), m0.trunk_hash());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn update_header_append_rewrites_only_the_header() {
    let dir = tempdir("hdr");
    let (_f0, f1) = make_f1(&dir);
    let b1 = std::fs::read(&f1).unwrap();
    let m1 = CmfModel::open(&f1).unwrap();
    let tensors1 = m1.tensors.clone();
    drop(m1);

    let p = dir.join("f.cmf");
    std::fs::copy(&f1, &p).unwrap();
    let rep = CmfModel::update_header_append(&p, |h| {
        let s = &mut h.skills[0];
        s.status = Some("active".into());
        s.gate = Some(serde_json::json!({"status": "measured", "false_accept_upper95": 0.01}));
        s.selection = Some(descriptor(1, 5));
        h.router = Some(policy(1));
        let seq = h.lineage.last().unwrap().seq + 1;
        h.lineage
            .push(LineageEvent::now(seq, "recalibrate", serde_json::json!({})));
        h.segments.clear(); // layout facts are kept whatever the closure does
    })
    .unwrap();
    assert_eq!(rep.tensors_added, 0);
    let b = std::fs::read(&p).unwrap();
    assert_eq!(&b[128..b1.len()], &b1[128..]);
    assert_eq!(b[0x20..0x30], b1[0x20..0x30], "directory not moved");
    let m = CmfModel::open(&p).unwrap();
    assert_eq!(m.tensors, tensors1);
    assert!(m.header.skills[0].is_auto_routable());
    assert_eq!(m.header.segments.len(), 2);
    assert_ne!(m.required_features & features::ROUTER_V2, 0);
    assert!(m.verify().is_empty());
    drop(m);

    // Router rules are checked before anything is written.
    let len = std::fs::metadata(&p).unwrap().len();
    let e = CmfModel::update_header_append(&p, |h| {
        h.router.as_mut().unwrap().base.metric = "mse".into()
    })
    .unwrap_err()
    .to_string();
    assert!(e.contains("router.base: selection.metric 'mse'"), "{e}");
    let e = CmfModel::update_header_append(&p, |h| {
        h.router = Some(policy(3));
        h.skills[0].selection = Some(descriptor(3, 5));
    })
    .unwrap_err()
    .to_string();
    assert!(
        e.contains("router.phi.layer 3 must be < min(layers) 3"),
        "{e}"
    );
    // The genome identity is frozen against header updates; its status is not.
    let e = CmfModel::update_header_append(&p, |h| h.genome = None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("identity"), "{e}");
    let e = CmfModel::update_header_append(&p, |h| h.genome.as_mut().unwrap().generation = 1)
        .unwrap_err()
        .to_string();
    assert!(e.contains("identity"), "{e}");
    // A trunk change through the header is refused (arch is trunk).
    let e = CmfModel::update_header_append(&p, |h| h.arch.rope_theta = 1.0)
        .unwrap_err()
        .to_string();
    assert!(e.contains("trunk changed without a new genome"), "{e}");
    assert_eq!(
        std::fs::metadata(&p).unwrap().len(),
        len,
        "refusals write nothing"
    );
    CmfModel::update_header_append(&p, |h| h.genome.as_mut().unwrap().status = "sealed".into())
        .unwrap();
    let m = CmfModel::open(&p).unwrap();
    assert_eq!(m.header.genome.as_ref().unwrap().status, "sealed");
    assert!(m.verify().is_empty());
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn a_full_rewrite_is_a_compaction() {
    let dir = tempdir("compact");
    let (_f0, f1) = make_f1(&dir);
    let m1 = CmfModel::open(&f1).unwrap();
    let specs: Vec<TensorSpec> = m1
        .tensors
        .iter()
        .map(|e| TensorSpec {
            name: e.name.clone(),
            dtype: e.dtype,
            shape: e.shape.clone(),
            data: m1.entry_bytes(e).to_vec(),
        })
        .collect();
    let c = dir.join("c.cmf");
    CmfModel::write(&c, &m1.header, &specs, Some(&m1.masks), m1.vocab.as_deref()).unwrap();
    let mc = CmfModel::open(&c).unwrap();
    assert!(mc.header.segments.is_empty(), "stale segment table dropped");
    assert_eq!(mc.trunk_hash(), m1.trunk_hash());
    assert_eq!(mc.required_features, m1.required_features);
    assert!(std::fs::metadata(&c).unwrap().len() < std::fs::metadata(&f1).unwrap().len());
    std::fs::remove_dir_all(dir).ok();
}

/// The trunk hash is over the header JSON as written: canonical (sorted
/// keys, any map order), f32 fields hashed in their written form, and an
/// arch field this reader's struct does not know is covered, not dropped.
#[test]
fn trunk_hash_covers_the_arch_as_written() {
    use cortiq_core::knowledge::{canonical_json, trunk_hash_json};
    let a: serde_json::Value =
        serde_json::from_str(r#"{"b":1,"a":{"d":[1,{"z":0,"y":2}],"c":0.1}}"#).unwrap();
    let b: serde_json::Value =
        serde_json::from_str(r#"{"a":{"c":0.1,"d":[1,{"y":2,"z":0}]},"b":1}"#).unwrap();
    assert_eq!(canonical_json(&a), canonical_json(&b));
    assert_eq!(
        canonical_json(&a),
        br#"{"a":{"c":0.1,"d":[1,{"y":2,"z":0}]},"b":1}"#.to_vec()
    );

    let dir = tempdir("as-written");
    // f32 fields with no exact decimal form, a free-form map in the arch.
    let mut h = genome_header();
    h.arch.embed_multiplier = 0.1;
    h.arch.partial_rotary_factor = 0.3;
    h.arch.deepseek_v41 = Some(serde_json::json!({"z": 1, "a": {"k": 0.7}}));
    let f = dir.join("f.cmf");
    CmfModel::write(&f, &h, &trunk_specs(), None, Some(VOCAB)).unwrap();
    let m = CmfModel::open(&f).expect("writer and reader agree on the written form");
    assert_eq!(
        m.header.genome.as_ref().unwrap().trunk_hash,
        hex64(m.trunk_hash())
    );
    let tensors = m.tensors.clone();
    drop(m);

    // A newer writer's additive arch field: covered by the hash.
    let p = dir.join("future.cmf");
    raw_header_swap(&f, &p, 0, |v| v["arch"]["future_field"] = 7.into());
    assert_refused(&p, "trunk_hash mismatch");
    let mut raw = serde_json::Value::Null;
    raw_header_swap(&f, &p, 0, |v| {
        v["arch"]["future_field"] = 7.into();
        let th = trunk_hash_json(&v["arch"], &v["tokenizer_config"], &tensors, Some(VOCAB));
        v["genome"]["trunk_hash"] = hex64(th).into();
        v["genome"]["master_trunk_hash"] = hex64(th).into();
        raw = v.clone();
    });
    let m = CmfModel::open(&p).expect("a genome sealed over the field opens");
    let th = trunk_hash_json(
        &raw["arch"],
        &raw["tokenizer_config"],
        &tensors,
        Some(VOCAB),
    );
    assert_eq!(m.trunk_hash(), th);
    std::fs::remove_dir_all(dir).ok();
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

/// NF-1: the mask catalog / sparse index run with the trunk (`run` and
/// `serve` apply the catalog's fallback mask to every request), so the
/// genome's hash covers them. A genome born without masks keeps the
/// pre-NF-1 hash; adding a catalog to a sealed genome by a full rewrite
/// (`cortiq skill add --sparse`) is refused by the writer, and a mask byte
/// changed on disk is refused by the reader.
#[test]
fn genome_trunk_hash_covers_the_mask_catalog() {
    use cortiq_core::knowledge::ExecHashes;
    let dir = tempdir("exec");
    let bare = dir.join("bare.cmf");
    CmfModel::write(&bare, &genome_header(), &trunk_specs(), None, Some(VOCAB)).unwrap();
    let mb = CmfModel::open(&bare).unwrap();
    assert_eq!(
        mb.trunk_hash(),
        cortiq_core::trunk_hash(&mb.header, &mb.tensors, Some(VOCAB)),
        "no mask sections: the hash is the pre-NF-1 one"
    );
    assert_eq!(mb.exec_hashes(), ExecHashes::default());

    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let m0 = CmfModel::open(&f0).unwrap();
    assert!(m0.exec_hashes().masks.is_some() && m0.exec_hashes().index.is_some());
    assert_ne!(
        m0.trunk_hash(),
        mb.trunk_hash(),
        "the same trunk with a mask catalog is another trunk"
    );
    assert_eq!(
        m0.trunk_hash(),
        cortiq_core::trunk_hash_exec(&m0.header, &m0.tensors, Some(VOCAB), m0.exec_hashes())
    );

    // Full rewrite of the sealed bare genome WITH a catalog: refused.
    let masked = dir.join("masked.cmf");
    let e = CmfModel::write(&masked, &mb.header, &specs_of(&mb), Some(&masks()), Some(VOCAB))
        .unwrap_err()
        .to_string();
    assert!(e.contains("trunk changed without a new genome"), "{e}");
    assert!(!masked.exists() || CmfModel::open(&masked).is_err());

    // A byte of the sparse index / mask section changed on disk: refused
    // at open (the directory, arch and vocab are untouched).
    let bytes = std::fs::read(&f0).unwrap();
    let u64at = |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap());
    for (what, at) in [("masks", 0x40usize), ("index", 0x60usize)] {
        let (off, len) = (u64at(&bytes, at) as usize, u64at(&bytes, at + 8) as usize);
        assert!(len > 0, "{what} section present");
        let mut t = bytes.clone();
        // the last byte: a bitmap / index payload byte, not a length field
        t[off + len - 1] ^= 0x01;
        let p = dir.join(format!("tamper-{what}.cmf"));
        std::fs::write(&p, &t).unwrap();
        let e = open_err(&p);
        assert!(
            e.contains("trunk_hash mismatch"),
            "{what}: {e}"
        );
    }
    std::fs::remove_dir_all(dir).ok();
}

/// NF-5: a genome binds its tokenizer — no genome without an embedded
/// VOCAB section (writer and reader).
#[test]
fn a_genome_must_embed_its_vocab() {
    let dir = tempdir("novocab");
    let p = dir.join("g.cmf");
    let e = CmfModel::write(&p, &genome_header(), &trunk_specs(), None, None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("must embed its tokenizer"), "{e}");
    // On disk: an F0 whose envelope drops the vocab section.
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let mut b = std::fs::read(&f0).unwrap();
    b[0x58..0x60].copy_from_slice(&0u64.to_le_bytes());
    let t = dir.join("t.cmf");
    std::fs::write(&t, &b).unwrap();
    assert_refused(&t, "must embed its tokenizer");
    // A plain (non-genome) file without vocab keeps working.
    let plain = dir.join("plain.cmf");
    CmfModel::write(&plain, &plain_header(), &trunk_specs(), None, None).unwrap();
    CmfModel::open(&plain).unwrap();
    std::fs::remove_dir_all(dir).ok();
}

/// NF-6: the trunk's attention operator is the arch's; a provenance
/// `o1_attn` hint on a genome would switch it without a new genome.
#[test]
fn a_genome_refuses_a_provenance_o1_hint() {
    let dir = tempdir("o1hint");
    let mut h = genome_header();
    h.provenance = Some(serde_json::json!({"o1_attn": {"layers": "all", "m": 8}}));
    let e = CmfModel::write(&dir.join("w.cmf"), &h, &trunk_specs(), None, Some(VOCAB))
        .unwrap_err()
        .to_string();
    assert!(e.contains("o1_attn"), "{e}");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    let e = CmfModel::update_header_append(&f0, |h| {
        h.provenance = Some(serde_json::json!({"o1_attn": {"layers": "all"}}))
    })
    .unwrap_err()
    .to_string();
    assert!(e.contains("o1_attn"), "header-only update: {e}");
    let p = dir.join("swap.cmf");
    raw_header_swap(&f0, &p, 0, |v| {
        v["provenance"] = serde_json::json!({"o1_attn": {"layers": "all"}})
    });
    assert_refused(&p, "o1_attn");
    // Other provenance keys stay free.
    CmfModel::update_header_append(&f0, |h| {
        h.provenance = Some(serde_json::json!({"producer": "test"}))
    })
    .unwrap();
    CmfModel::open(&f0).unwrap();
    std::fs::remove_dir_all(dir).ok();
}

/// R8: `backbone`, `none`, `auto` name the backbone / the automatic
/// decision in the tools — a v2 skill may not take them.
#[test]
fn v2_skill_ids_backbone_none_auto_are_reserved() {
    let dir = tempdir("reserved");
    let f0 = dir.join("f0.cmf");
    write_f0(&f0);
    for id in cortiq_core::knowledge::RESERVED_SKILL_IDS {
        let m = CmfModel::open(&f0).unwrap();
        let (rec, ts) = ffn_skill(&m, id, &[3]);
        drop(m);
        let e = CmfModel::append_skill(&f0, rec, &ts, None, None, None)
            .unwrap_err()
            .to_string();
        assert!(e.contains("reserved"), "{id}: {e}");
    }
    let m = CmfModel::open(&f0).unwrap();
    assert!(m.header.skills.is_empty(), "nothing was appended");
    std::fs::remove_dir_all(dir).ok();
}
