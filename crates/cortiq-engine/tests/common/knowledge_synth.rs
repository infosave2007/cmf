//! Format-v2 knowledge files over a synthetic Embryo genome, for the
//! router-v2 runtime tests (engine, cli, server):
//!
//! * **F0** — a frozen genome (`header.genome`, bit GENOME) with no skill
//!   and no router: the reference the backbone must stay bit-identical to;
//! * **F1** — F0 + one `ffn_replace` skill record (FFN of the last layer,
//!   tensors appended at the tail by `CmfModel::append_skill`) + a
//!   `backbone_gated` router whose two descriptors (backbone, skill) are
//!   fitted on φ of [`GENERAL_TEXTS`] / [`SKILL_TEXTS`] computed by the
//!   runtime itself ([`Pipeline::probe_phi_span`]), then calibrated with
//!   `router::calibrate_v2` and published with a matching `skills_hash`;
//! * **legacy** — the genome without a genome record + a v1 skill (no
//!   `kind`, uncalibrated `mse_unit` descriptor): the pre-v2 routing path.
//!
//! Needs the including test crate to declare `embryo_synth` at its root.
//! The caller selects the backend (the tests run with `CMF_GPU=0`).
#![allow(dead_code)]

use super::embryo_synth::{SynthGeom, write_synth_genome, write_synth_genome_with};
use cortiq_core::format::{RoutingCalibration, SelectionDescriptor, TensorSpec};
use cortiq_core::knowledge::{hex64, skill_kind};
use cortiq_core::{
    CmfModel, GenomeInfo, PhiSpec, RouterPolicy, SkillBound, SkillOverride, SkillRecord,
    ffn_replace_state_effect,
};
use cortiq_engine::pipeline::Pipeline;
use cortiq_engine::router;
use cortiq_engine::sampler::SamplerConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// In-scope user texts of the skill (Cyrillic: disjoint bytes from the
/// general set under the byte-level fallback tokenizer).
pub const SKILL_TEXTS: &[&str] = &[
    "Какие лечебные свойства у ромашки аптечной?",
    "Чем полезен зверобой продырявленный?",
    "Как заваривать шалфей лекарственный?",
    "Где растёт валериана лекарственная?",
    "Какое семейство у календулы?",
    "Чем опасен болиголов пятнистый?",
];

/// General user texts (the backbone class).
pub const GENERAL_TEXTS: &[&str] = &[
    "What is the capital of France?",
    "Write a Rust function that returns the maximum element.",
    "Explain why Earth has seasons in two sentences.",
    "Compute exactly: 17 * 19 + 23.",
    "Say what water is made of.",
    "Why does a hash table offer constant-time lookup?",
];

pub const SKILL_ID: &str = "herbs";
pub const GENOME_ID: &str = "synth-genome";

/// The layer whose FFN the skill replaces (the last layer of the geometry).
pub fn skill_layer(g: &SynthGeom) -> usize {
    g.layers - 1
}

/// φ at the hidden after layer 0 (< the skill layer), the cmf-im-v1 shape
/// with arbitrary synthetic template ids.
pub fn phi_spec() -> PhiSpec {
    PhiSpec {
        layer: 0,
        pool: "span_mean".into(),
        norm: "unit".into(),
        prefix_ids: vec![1, 2, 3],
        suffix_ids: vec![4, 5, 6],
    }
}

pub struct KnowledgeFiles {
    pub f0: PathBuf,
    pub f1: PathBuf,
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 (with padding) — no dependency for the including crate.
pub fn b64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            B64[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            B64[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

pub fn f16_b64(v: &[f32]) -> String {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| cortiq_core::quant::f32_to_f16(*x).to_le_bytes())
        .collect();
    b64(&bytes)
}

fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter()
        .map(|x| cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(*x)))
        .collect()
}

pub fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Canonical router-v2 φ of `text` on a backbone pipeline (unit-normalized).
pub fn phi_of(p: &mut Pipeline, spec: &PhiSpec, text: &str) -> Vec<f32> {
    let q = p.tokenizer.encode_plain(text);
    let (ids, span) = router::phi_span_ids(spec, &q);
    unit(&p.probe_phi_span(&ids, spec.layer, span))
}

/// A rank-1 `mse_unit` descriptor fitted on unit φ samples: mean = their
/// centroid, basis = the direction of the largest centroid residual,
/// error statistics over the samples, holdout = the samples themselves.
pub fn descriptor(samples: &[Vec<f32>], layer: usize) -> SelectionDescriptor {
    let h = samples[0].len();
    let mut mean = vec![0.0f32; h];
    for s in samples {
        for (m, x) in mean.iter_mut().zip(s) {
            *m += x / samples.len() as f32;
        }
    }
    let far = samples
        .iter()
        .max_by(|a, b| {
            let d = |s: &Vec<f32>| {
                s.iter()
                    .zip(&mean)
                    .map(|(x, m)| (x - m).powi(2))
                    .sum::<f32>()
            };
            d(a).total_cmp(&d(b))
        })
        .unwrap();
    let dir: Vec<f32> = far.iter().zip(&mean).map(|(x, m)| x - m).collect();
    let basis = unit(&dir);
    let (mq, bq) = (f16_round(&mean), f16_round(&basis));
    let errs: Vec<f32> = samples
        .iter()
        .map(|s| router::recon_error(s, &mq, &bq, 1))
        .collect();
    let em = errs.iter().sum::<f32>() / errs.len() as f32;
    let es = (errs.iter().map(|e| (e - em).powi(2)).sum::<f32>() / errs.len() as f32)
        .sqrt()
        .max(0.02);
    SelectionDescriptor {
        metric: "mse_unit".into(),
        phi_layer: layer,
        mean: f16_b64(&mean),
        basis: f16_b64(&basis),
        rank: 1,
        err_mean: Some(em),
        err_std: Some(es),
        holdout: Some(f16_b64(&samples.concat())),
        holdout_n: Some(samples.len()),
    }
}

/// Replacement for a trunk FFN tensor: every value negated and scaled —
/// a skill that visibly changes the layer's output.
fn replaced(model: &CmfModel, name: &str) -> TensorSpec {
    let e = model
        .tensor(name)
        .unwrap_or_else(|| panic!("trunk tensor {name}"));
    let data: Vec<u8> = model
        .tensor_bytes(name)
        .unwrap()
        .chunks_exact(4)
        .flat_map(|b| (-1.25 * f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
        .collect();
    TensorSpec {
        name: format!("skill.{SKILL_ID}.{name}"),
        dtype: e.dtype,
        shape: e.shape.clone(),
        data,
    }
}

/// The FFN tensors of `layer` the synthetic skill replaces.
pub fn skill_tensor_names(g: &SynthGeom, layer: usize) -> Vec<String> {
    let pf = format!("model.layers.{layer}.mlp.");
    if g.experts == 0 {
        return ["gate_proj", "up_proj", "down_proj"]
            .iter()
            .map(|m| format!("{pf}{m}.weight"))
            .collect();
    }
    let mut v: Vec<String> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|m| format!("{pf}shared_expert.{m}.weight"))
        .collect();
    for e in 0..g.experts {
        v.push(format!("{pf}experts.{e}.down_proj.weight"));
    }
    v
}

/// φ samples of both text sets on F0's backbone.
pub fn phi_sets(f0: &Path, spec: &PhiSpec) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let model = Arc::new(CmfModel::open(f0).expect("open F0"));
    let mut p = Pipeline::from_model(&model, SamplerConfig::default()).expect("F0 pipeline");
    let skill = SKILL_TEXTS
        .iter()
        .map(|t| phi_of(&mut p, spec, t))
        .collect();
    let general = GENERAL_TEXTS
        .iter()
        .map(|t| phi_of(&mut p, spec, t))
        .collect();
    (skill, general)
}

/// Write F0 and F1 into `dir`. `status` is the skill record's status
/// (`active` makes it auto-routable: its gate is `measured`).
pub fn write_knowledge_pair(dir: &Path, g: &SynthGeom, status: &str) -> KnowledgeFiles {
    std::fs::create_dir_all(dir).unwrap();
    let f0 = dir.join("f0.cmf");
    let f1 = dir.join("f1.cmf");
    write_synth_genome_with(
        &f0,
        g,
        Some(GenomeInfo::birth(GENOME_ID, "pre_chat", "f32")),
    );
    std::fs::copy(&f0, &f1).expect("copy F0 → F1");

    let spec = phi_spec();
    let (skill_phi, general_phi) = phi_sets(&f0, &spec);
    let base_desc = descriptor(&general_phi, spec.layer);
    let skill_desc = descriptor(&skill_phi, spec.layer);

    let model = CmfModel::open(&f1).expect("open F1 before append");
    let genome = model.header.genome.clone().expect("genome record");
    let layer = skill_layer(g);
    let names = skill_tensor_names(g, layer);
    let tensors: Vec<TensorSpec> = names.iter().map(|n| replaced(&model, n)).collect();
    let overrides = names
        .iter()
        .map(|n| SkillOverride {
            name: n.clone(),
            base_hash: hex64(model.tensor(n).unwrap().hash),
        })
        .collect();
    let record = SkillRecord {
        id: SKILL_ID.into(),
        name: Some("synthetic herbs".into()),
        layers: vec![layer],
        selection: Some(skill_desc),
        kind: Some(skill_kind::FFN_REPLACE.into()),
        overrides,
        bound: Some(SkillBound {
            genome_id: genome.id.clone(),
            generation: genome.generation,
            master_trunk_hash: genome.master_trunk_hash.clone(),
        }),
        state_effect: Some(ffn_replace_state_effect(model.arch(), &[layer])),
        status: Some(status.into()),
        gate: Some(serde_json::json!({"status": "measured", "synthetic": true})),
        prompt_contract: Some("cmf-im-v1".into()),
        origin: Some(serde_json::json!({"trigger": "test"})),
        ..Default::default()
    };
    let policy = RouterPolicy {
        version: 2,
        policy: "backbone_gated".into(),
        granularity: "request".into(),
        phi: spec,
        base: base_desc,
        margin: 0.05,
        skills_hash: "0000000000000000".into(),
        measured: None,
    };
    drop(model);
    CmfModel::append_skill(&f1, record, &tensors, Some(policy), None, None).expect("append skill");

    // Calibrate over backbone + skill holdouts and publish with the
    // matching skills hash. θ is set generously so the in-scope texts
    // are never novel (the truth table exercises novelty explicitly).
    let m = CmfModel::open(&f1).expect("open F1 after append");
    let (mut cal, measured) = router::calibrate_v2(&m.header, 0.05).expect("calibrate_v2");
    cal.novelty_theta = cal.novelty_theta.max(0.9);
    let hash = router::skills_hash(&m.header);
    drop(m);
    CmfModel::update_header_append(&f1, move |h| {
        h.routing = Some(cal);
        let r = h.router.as_mut().unwrap();
        r.skills_hash = hex64(hash);
        r.measured = Some(measured);
    })
    .expect("publish calibration");
    KnowledgeFiles { f0, f1 }
}

/// A legacy (pre-v2) file: the genome without a genome record plus a v1
/// skill (no `kind`) whose uncalibrated `mse_unit` descriptor sits on φ
/// of [`SKILL_TEXTS`] as the legacy router computes it (`probe_phi` over
/// the whole encoded prompt at layer 0).
pub fn write_legacy_skill_file(path: &Path, g: &SynthGeom) {
    let plain = path.with_extension("plain.cmf");
    write_synth_genome(&plain, g);
    let model = Arc::new(CmfModel::open(&plain).expect("open plain genome"));
    let mut p = Pipeline::from_model(&model, SamplerConfig::default()).expect("pipeline");
    let samples: Vec<Vec<f32>> = SKILL_TEXTS
        .iter()
        .map(|t| {
            let ids = p.tokenizer.encode(t);
            unit(&p.probe_phi(&ids, 0))
        })
        .collect();
    let mut desc = descriptor(&samples, 0);
    desc.holdout = None;
    desc.holdout_n = None;
    let layer = skill_layer(g);
    let mut specs: Vec<TensorSpec> = model
        .tensors
        .iter()
        .map(|t| TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model.tensor_bytes(&t.name).unwrap().to_vec(),
        })
        .collect();
    specs.extend(
        skill_tensor_names(g, layer)
            .iter()
            .map(|n| replaced(&model, n)),
    );
    let mut header = model.header.clone();
    header.skills = vec![SkillRecord {
        id: SKILL_ID.into(),
        layers: vec![layer],
        selection: Some(desc),
        ..Default::default()
    }];
    header.routing = None::<RoutingCalibration>;
    drop(p);
    CmfModel::write(path, &header, &specs, None, None).expect("write legacy skill file");
    let _ = std::fs::remove_file(&plain);
}

// ───────────────────────── lookup record ─────────────────────────

/// The synthetic reference table: `(keys, ru card, en card)` per entry.
/// Entry `i` is named by [`SKILL_TEXTS`]`[i]` (a Cyrillic key in the
/// question's own form), its Latin binomial and an English name.
pub fn lookup_entries() -> Vec<(Vec<&'static str>, serde_json::Value, serde_json::Value)> {
    let card = |text: &str, family: &str, uses: &str, safety: &str| {
        serde_json::json!({
            "card": text,
            "fields": {"family": family, "uses": uses, "safety": safety},
        })
    };
    vec![
        (
            vec!["ромашка аптечная", "ромашки аптечной", "Matricaria chamomilla", "chamomile"],
            card(
                "Ромашка аптечная (Matricaria chamomilla) — однолетник семейства Астровые.",
                "Астровые (Asteraceae)",
                "Противовоспалительное и спазмолитическое средство.",
                "Возможна аллергия.",
            ),
            card(
                "Chamomile (Matricaria chamomilla) is an annual of the daisy family.",
                "Asteraceae",
                "Anti-inflammatory, antispasmodic.",
                "Possible allergy.",
            ),
        ),
        (
            vec!["зверобой продырявленный", "Hypericum perforatum", "St. John's wort"],
            card(
                "Зверобой продырявленный (Hypericum perforatum) — многолетник семейства Зверобойные.",
                "Зверобойные (Hypericaceae)",
                "Лёгкие депрессивные состояния, наружно при ранах.",
                "Фотосенсибилизация; взаимодействия с лекарствами.",
            ),
            card(
                "St. John's wort (Hypericum perforatum) is a perennial of the family Hypericaceae.",
                "Hypericaceae",
                "Mild depression; topical for wounds.",
                "Photosensitivity; drug interactions.",
            ),
        ),
        (
            vec!["шалфей лекарственный", "Salvia officinalis", "sage"],
            card(
                "Шалфей лекарственный (Salvia officinalis) — полукустарник семейства Яснотковые.",
                "Яснотковые (Lamiaceae)",
                "Полоскания при воспалении горла.",
                "Не при беременности.",
            ),
            card(
                "Sage (Salvia officinalis) is a subshrub of the mint family.",
                "Lamiaceae",
                "Gargles for a sore throat.",
                "Not in pregnancy.",
            ),
        ),
        (
            vec!["валериана лекарственная", "Valeriana officinalis", "valerian"],
            card(
                "Валериана лекарственная (Valeriana officinalis) — многолетник семейства Жимолостные.",
                "Жимолостные (Caprifoliaceae)",
                "Седативное средство.",
                "Сонливость.",
            ),
            card(
                "Valerian (Valeriana officinalis) is a perennial of the honeysuckle family.",
                "Caprifoliaceae",
                "Sedative.",
                "Drowsiness.",
            ),
        ),
        (
            vec!["календула", "календулы", "Calendula officinalis", "calendula", "pot marigold"],
            card(
                "Календула лекарственная (Calendula officinalis) — однолетник семейства Астровые.",
                "Астровые (Asteraceae)",
                "Наружно при ранах и воспалении.",
                "Редко аллергия.",
            ),
            card(
                "Pot marigold (Calendula officinalis) is an annual of the daisy family.",
                "Asteraceae",
                "Topical for wounds and inflammation.",
                "Rare allergy.",
            ),
        ),
        (
            vec!["болиголов пятнистый", "Conium maculatum", "poison hemlock"],
            card(
                "Болиголов пятнистый (Conium maculatum) — ядовитый двулетник семейства Зонтичные.",
                "Зонтичные (Apiaceae)",
                "В медицине не применяется.",
                "Смертельно ядовит.",
            ),
            card(
                "Poison hemlock (Conium maculatum) is a poisonous biennial of the carrot family.",
                "Apiaceae",
                "Not used in medicine.",
                "Deadly poisonous.",
            ),
        ),
    ]
}

/// Slot languages of the synthetic table, in slot order.
pub const LOOKUP_LANGS: [&str; 2] = ["ru", "en"];

/// The `LookupInfo` + four tensors of the synthetic table under `id`.
pub fn lookup_record_tensors(
    id: &str,
) -> (cortiq_core::LookupInfo, Vec<TensorSpec>) {
    let entries = lookup_entries();
    let mut keys: Vec<(u64, u32)> = Vec::new();
    let mut slots: Vec<String> = Vec::new();
    for (e, (ks, ru, en)) in entries.iter().enumerate() {
        for k in ks {
            keys.push((cortiq_core::key_hash(k), e as u32));
        }
        slots.push(ru.to_string());
        slots.push(en.to_string());
    }
    let info = cortiq_core::LookupInfo {
        entries: entries.len(),
        keys: keys.len(),
        key_norm: cortiq_core::knowledge::KEY_NORM.into(),
        langs: LOOKUP_LANGS.iter().map(|s| s.to_string()).collect(),
        fields: vec!["family".into(), "uses".into(), "safety".into()],
        policy: None,
    };
    let slot_refs: Vec<&str> = slots.iter().map(String::as_str).collect();
    let tensors = cortiq_core::lookup_tensors(id, &info, &keys, &slot_refs).expect("lookup tensors");
    (info, tensors)
}

/// F0 and F1 where F1 = F0 + ONE `lookup` record ([`SKILL_ID`], the table
/// of [`lookup_entries`]) with the same selection descriptor and the same
/// calibrated `backbone_gated` router as [`write_knowledge_pair`]: the
/// router sends [`SKILL_TEXTS`] to the table and [`GENERAL_TEXTS`] to the
/// backbone. The record touches no tensor of the network.
pub fn write_lookup_pair(dir: &Path, g: &SynthGeom, status: &str) -> KnowledgeFiles {
    std::fs::create_dir_all(dir).unwrap();
    let f0 = dir.join("f0.cmf");
    let f1 = dir.join("f1.cmf");
    write_synth_genome_with(
        &f0,
        g,
        Some(GenomeInfo::birth(GENOME_ID, "pre_chat", "f32")),
    );
    std::fs::copy(&f0, &f1).expect("copy F0 → F1");

    let spec = phi_spec();
    let (skill_phi, general_phi) = phi_sets(&f0, &spec);
    let base_desc = descriptor(&general_phi, spec.layer);
    let skill_desc = descriptor(&skill_phi, spec.layer);

    let model = CmfModel::open(&f1).expect("open F1 before append");
    let genome = model.header.genome.clone().expect("genome record");
    let (info, tensors) = lookup_record_tensors(SKILL_ID);
    let record = SkillRecord {
        id: SKILL_ID.into(),
        name: Some("synthetic herbs table".into()),
        layers: Vec::new(),
        selection: Some(skill_desc),
        kind: Some(skill_kind::LOOKUP.into()),
        lookup: Some(info),
        overrides: Vec::new(),
        bound: Some(SkillBound {
            genome_id: genome.id.clone(),
            generation: genome.generation,
            master_trunk_hash: genome.master_trunk_hash.clone(),
        }),
        state_effect: Some(cortiq_core::lookup_state_effect()),
        status: Some(status.into()),
        gate: Some(serde_json::json!({"status": "measured", "synthetic": true})),
        prompt_contract: None,
        origin: Some(serde_json::json!({"trigger": "test", "entries": 6})),
        ..Default::default()
    };
    let policy = RouterPolicy {
        version: 2,
        policy: "backbone_gated".into(),
        granularity: "request".into(),
        phi: spec,
        base: base_desc,
        margin: 0.05,
        skills_hash: "0000000000000000".into(),
        measured: None,
    };
    drop(model);
    CmfModel::append_skill(&f1, record, &tensors, Some(policy), None, None)
        .expect("append lookup record");

    let m = CmfModel::open(&f1).expect("open F1 after append");
    let (mut cal, measured) = router::calibrate_v2(&m.header, 0.05).expect("calibrate_v2");
    cal.novelty_theta = cal.novelty_theta.max(0.9);
    let hash = router::skills_hash(&m.header);
    drop(m);
    CmfModel::update_header_append(&f1, move |h| {
        h.routing = Some(cal);
        let r = h.router.as_mut().unwrap();
        r.skills_hash = hex64(hash);
        r.measured = Some(measured);
    })
    .expect("publish calibration");
    KnowledgeFiles { f0, f1 }
}

/// Publish a MUTATED header JSON at the tail of `path` WITHOUT the
/// writers' validation (the envelope's header offset, length and hash
/// follow): a header a newer writer — or a hand edit — could have left,
/// for the tests of what `open()` and the runtime do with it.
pub fn rewrite_header_json(path: &Path, mutate: impl FnOnce(&mut serde_json::Value)) {
    let mut bytes = std::fs::read(path).expect("read the file");
    let m = CmfModel::open(path).expect("open before the header rewrite");
    let mut v = serde_json::to_value(&m.header).expect("header JSON");
    drop(m);
    mutate(&mut v);
    let js = serde_json::to_vec(&v).expect("header JSON bytes");
    let off = bytes.len() as u64;
    bytes.extend_from_slice(&js);
    bytes[0x10..0x18].copy_from_slice(&off.to_le_bytes());
    bytes[0x18..0x20].copy_from_slice(&(js.len() as u64).to_le_bytes());
    bytes[0x70..0x78].copy_from_slice(&cortiq_core::hash64(&js).to_le_bytes());
    std::fs::write(path, &bytes).expect("write the file");
}
