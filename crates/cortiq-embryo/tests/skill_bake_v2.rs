//! Skill bake v2 (format v2 "knowledge without forgetting") on a tiny
//! GDN + bounded-anchor + experts genome (the fam-a family at test size):
//!  - export stamps the frozen genome (trunk / master hashes, lineage birth);
//!  - the canonical span-mean φ probe (one prompt per row, right padding,
//!    dropless) equals a host recomputation and the runtime's own forward;
//!  - the φ frame is exactly the single-turn prompt of the SFT records;
//!  - bake v2 end to end: the output opens with GENOME|SKILLS_V2|ROUTER_V2,
//!    its prefix bytes / trunk / directory equal the base's, the backbone
//!    logits are bit-identical to the base file's, the skill overlay loads;
//!  - refusals: out == base, ckpt ≠ base, phi_layer ≥ min(layers), no genome,
//!    τ outside (0, σ(3)), dev records seen in train, a manifest that is not
//!    the shards' group split;
//!  - regressions of the review findings: whole-shard dev evaluation on the
//!    dropless instance (T1/T2), the f16 genome probed and polished on its
//!    served trunk (T4/PHI-4), private temp + no-overwrite publish (T5/NF-8),
//!    the base router's descriptor/margin kept and active skills re-gated on
//!    a second bake (NF-3/PHI-8), legacy tools refusing genome files (NF-4),
//!    φ parity with the runtime's own `probe_phi_span` + tokenizer at 1e-5
//!    over GDN / vmf_phase / phase-delta and non-anchor layers (PHI-5/6),
//!    and the post-append runtime replay of the holdouts (PHI-7).
//!
//! Runs on the native Metal backend on macOS and on the Vulkan backend in
//! the Linux `--features vulkan` build (skipped when no device exists).
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_core::format::{CmfModel, features};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, Mixer, init_params};
use cortiq_embryo::sft::{IGNORE, SftShard};
use cortiq_embryo::skill::{
    BakeV2Args, BakeV2Inputs, PhiCase, bake_gpu, bake_v2, create_bake_tmp, eval_answer_nll,
    eval_records, phi_spec, probe_prompts, publish_new_file, runtime_route_check,
    runtime_tokenizer, sample_prompts, user_text_ids,
};
use cortiq_embryo::tokenizer::Bpe;
use cortiq_embryo::train::{Checkpoint, Shard, save_checkpoint};
use std::path::{Path, PathBuf};

/// fam-a geometry at test size: GDN mixer, bounded anchors (window 16, 2
/// sinks) at layers 1 and 3, a shared expert + 4 routed experts.
fn fam_a_tiny() -> EmbryoCfg {
    let mut cfg = EmbryoCfg::tiny();
    cfg.layers = 4;
    cfg.anchor_layers = Some(vec![1, 3]);
    cfg.mixer = Mixer::Gdn;
    cfg.gdn_heads = 2;
    cfg.gdn_dk = 32;
    cfg.gdn_dv = 32;
    cfg.anchor_window = 16;
    cfg.anchor_sink = 2;
    cfg.experts = 4;
    cfg
}

fn scratch(tag: &str) -> PathBuf {
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "embryo_bake_v2_{tag}_{}_{}",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

const PLANTS: &[&str] = &[
    "Achillea millefolium",
    "Arnica montana",
    "Calendula officinalis",
    "Hypericum perforatum",
    "Matricaria chamomilla",
    "Mentha piperita",
    "Salvia officinalis",
    "Thymus vulgaris",
    "Urtica dioica",
    "Valeriana officinalis",
    "Plantago major",
    "Taraxacum officinale",
    "Echinacea purpurea",
    "Melissa officinalis",
    "Rosmarinus officinalis",
    "Lavandula angustifolia",
    "Tilia cordata",
    "Sambucus nigra",
    "Crataegus monogyna",
    "Artemisia absinthium",
    "Leonurus cardiaca",
    "Origanum vulgare",
    "Tanacetum vulgare",
    "Bidens tripartita",
    "Equisetum arvense",
    "Viola tricolor",
    "Primula veris",
    "Filipendula ulmaria",
    "Glycyrrhiza glabra",
    "Inula helenium",
    "Rhodiola rosea",
    "Silybum marianum",
    "Hippophae rhamnoides",
    "Rosa canina",
    "Vaccinium myrtillus",
    "Betula pendula",
    "Quercus robur",
    "Tussilago farfara",
    "Chelidonium majus",
    "Polygonum aviculare",
];
const FAMILIES: &[&str] = &[
    "Asteraceae",
    "Lamiaceae",
    "Rosaceae",
    "Apiaceae",
    "Urticaceae",
];

/// A genome with seeded expert descriptors (one lr-0 step, as a birth
/// leaves them), its tokenizer, and the files every test needs.
struct Fixture {
    dir: PathBuf,
    ck: Checkpoint,
    tok_json: String,
    ckpt: PathBuf,
    base: PathBuf,
    sft: [PathBuf; 3],
    manifest: PathBuf,
    lm: [PathBuf; 2],
    phi_prompts: PathBuf,
    general_prompts: PathBuf,
}

fn general_questions() -> Vec<String> {
    let mut v = Vec::new();
    for (i, c) in [
        "France", "Japan", "Brazil", "Canada", "Egypt", "Norway", "India", "Chile",
    ]
    .iter()
    .enumerate()
    {
        v.push(format!("What is the capital of {c}?"));
        v.push(format!(
            "How many people live in {c} today, roughly {i} or more?"
        ));
    }
    for l in ["Rust", "Python", "C", "Go", "Java", "Haskell", "Zig", "Lua"] {
        v.push(format!("How do I sort a vector of integers in {l}?"));
        v.push(format!("Write a {l} function that reverses a string."));
    }
    for n in 2..10 {
        v.push(format!("Solve {n}x + 3 = {} for x.", n * 5));
    }
    v
}

fn skill_questions() -> Vec<String> {
    let mut v = Vec::new();
    for p in PLANTS {
        v.push(format!("Which family does {p} belong to?"));
        v.push(format!("What is {p} used for in herbal medicine?"));
    }
    v
}

fn write_jsonl_prompts(path: &Path, qs: &[String]) {
    let mut s = String::new();
    for q in qs {
        s.push_str(&serde_json::json!({ "prompt": q }).to_string());
        s.push('\n');
    }
    std::fs::write(path, s).unwrap();
}

/// The fixture corpus: plant facts + the general questions.
fn corpus_text() -> String {
    let mut corpus = String::new();
    for p in PLANTS {
        corpus.push_str(&format!(
            "{p} belongs to the family {} and is used for tea. ",
            FAMILIES[p.len() % 5]
        ));
    }
    for q in general_questions() {
        corpus.push_str(&q);
        corpus.push(' ');
    }
    corpus
}

/// Byte-level BPE over the fixture corpus (ids < vocab; the chat specials
/// at the end).
fn tokenizer_json(vocab: usize) -> String {
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    cortiq_embryo::tokenizer::count_words(&corpus_text(), &re, &mut counts);
    cortiq_embryo::tokenizer::train(&counts, vocab, false).to_hf_json()
}

/// A genome checkpoint: seeded params, expert descriptors seeded with one
/// lr-0 step (as a birth leaves them).
fn seeded_ck(cfg: &EmbryoCfg, bpe: &Bpe) -> Checkpoint {
    let lay = Layout::new(cfg);
    let params = init_params(cfg, &lay, 11);
    let mut gpu = EmbryoGpu::new(cfg.clone(), 4, 64, &params).unwrap();
    let mut ids = Vec::new();
    bpe.encode(
        &corpus_text(),
        &mut std::collections::HashMap::new(),
        &mut ids,
    );
    let tk: Vec<u32> = ids.iter().cycle().take(256).copied().collect();
    let tg: Vec<u32> = ids.iter().cycle().skip(1).take(256).copied().collect();
    let _ = gpu.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    let extras: Vec<(String, Vec<f32>)> = gpu
        .desc_host()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    drop(gpu);
    Checkpoint {
        cfg: cfg.clone(),
        step: 10_000,
        params,
        m: None,
        v: None,
        extras,
    }
}

fn fixture(tag: &str) -> Option<Fixture> {
    cortiq_embryo::metal::ctx()?;
    // the CPU pipeline is the runtime reference in these tests
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let cfg = fam_a_tiny();
    let dir = scratch(tag);
    let corpus = corpus_text();
    let tok_json = tokenizer_json(cfg.vocab);
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, &tok_json).unwrap();
    let bpe = Bpe::load(&tok_path).unwrap();
    let ck = seeded_ck(&cfg, &bpe);
    let ckpt = dir.join("genome.ckpt");
    let ex: Vec<(&str, &[f32])> = ck
        .extras
        .iter()
        .map(|(n, x)| (n.as_str(), x.as_slice()))
        .collect();
    save_checkpoint(&ckpt, &ck.cfg, ck.step, &ck.params, None, None, &ex).unwrap();
    // the base: the genome exported with its frozen-genome block
    let base = dir.join("base.cmf");
    cortiq_embryo::export::export_genome(
        &ck,
        tok_json.as_bytes(),
        &base,
        cortiq_core::TensorDtype::F32,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny".into(),
            status: "pre_chat".into(),
        }),
    )
    .unwrap();
    // response-only SFT shards, group (plant) disjoint
    let mut msgs = String::new();
    for (k, p) in PLANTS.iter().enumerate() {
        let fam = FAMILIES[k % FAMILIES.len()];
        for (q, a) in [
            (
                format!("Which family does {p} belong to?"),
                format!("{p} belongs to the family {fam}."),
            ),
            (
                format!("What is {p} used for in herbal medicine?"),
                format!("{p} is used as a mild tea."),
            ),
            (
                format!("Describe {p}."),
                format!("{p} is a perennial plant of the family {fam}."),
            ),
        ] {
            msgs.push_str(
                &serde_json::json!({
                    "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    "plant": p, "lang": "en",
                })
                .to_string(),
            );
            msgs.push('\n');
        }
    }
    let msgs_path = dir.join("herbs.jsonl");
    std::fs::write(&msgs_path, msgs).unwrap();
    let sft = [
        dir.join("train.sft"),
        dir.join("dev.sft"),
        dir.join("final.sft"),
    ];
    cortiq_embryo::sft::prepare_messages(
        &msgs_path,
        &tok_path,
        64,
        &sft[0],
        &sft[1],
        &sft[2],
        &dir.join("manifest.json"),
        None,
    )
    .expect("prepare SFT shards");
    // raw-LM shards (train / dev disjoint halves of a text)
    let mut text = String::new();
    for f in [
        "../../README.md",
        "../../docs/SKILLS.md",
        "../../docs/COMPARISON.md",
    ] {
        if let Ok(s) = std::fs::read_to_string(f) {
            text.push_str(&s);
        }
    }
    text.push_str(&corpus.repeat(8));
    let mut lm_ids = Vec::new();
    bpe.encode(&text, &mut std::collections::HashMap::new(), &mut lm_ids);
    let half = lm_ids.len() / 2;
    let lm = [dir.join("lm-train.u16"), dir.join("lm-dev.u16")];
    for (path, part) in lm.iter().zip([&lm_ids[..half], &lm_ids[half..]]) {
        Shard {
            tokens: part.iter().map(|&x| x as u16).collect(),
        }
        .save(path)
        .unwrap();
    }
    let phi_prompts = dir.join("phi.jsonl");
    let general_prompts = dir.join("general.jsonl");
    write_jsonl_prompts(&phi_prompts, &skill_questions());
    write_jsonl_prompts(&general_prompts, &general_questions());
    Some(Fixture {
        manifest: dir.join("manifest.json"),
        dir,
        ck,
        tok_json,
        ckpt,
        base,
        sft,
        lm,
        phi_prompts,
        general_prompts,
    })
}

fn inputs(fx: &Fixture, out: PathBuf) -> BakeV2Inputs {
    BakeV2Inputs {
        ckpt: fx.ckpt.clone(),
        base: fx.base.clone(),
        out,
        tokenizer: None,
        sft_train: fx.sft[0].clone(),
        sft_dev: fx.sft[1].clone(),
        sft_final: Some(fx.sft[2].clone()),
        sft_manifest: Some(fx.manifest.clone()),
        lm_train: Some(fx.lm[0].clone()),
        lm_dev: Some(fx.lm[1].clone()),
        phi_prompts: fx.phi_prompts.clone(),
        general_prompts: Some(fx.general_prompts.clone()),
    }
}

fn args(id: &str) -> BakeV2Args {
    BakeV2Args {
        id: id.into(),
        layers: vec![2, 3],
        steps_a: 8,
        steps_b: 6,
        lr_a: 5e-2,
        lr_b: 1e-3,
        l1: 2e-3,
        tau: 0.5,
        eval_every: 3,
        batch: 4,
        dev_batches: 0,
        lm_frac: 0.5,
        phi_layer: 1,
        phi_max: 4000,
        phi_batch: 4,
        phi_max_len: 512,
        rank: 4,
        route_margin: None,
        refit_base: false,
        target_fpr: 0.02,
        seed: 7,
    }
}

#[test]
fn export_stamps_the_frozen_genome() {
    let Some(fx) = fixture("export") else {
        return;
    };
    let m = CmfModel::open(&fx.base).unwrap();
    let g = m.header.genome.as_ref().expect("genome block");
    assert_eq!(g.id, "embryo-o1-fam-a-tiny");
    assert_eq!(
        (g.generation, g.status.as_str(), g.encoding.as_str()),
        (0, "pre_chat", "f32")
    );
    assert!(g.parent.is_none() && g.reference.is_none());
    assert_eq!(g.trunk_hash, format!("{:016x}", m.trunk_hash()));
    assert_eq!(g.master_trunk_hash, g.trunk_hash, "f32: master == trunk");
    assert_eq!(m.header.lineage.len(), 1);
    assert_eq!(m.header.lineage[0].event, "birth");
    assert_ne!(m.required_features & features::GENOME, 0);
    let prov = m.header.provenance.as_ref().unwrap();
    assert_eq!(
        prov["genome"], "embryo-o1-fam-a-tiny",
        "provenance carries the id"
    );
    // f16: the master hash is the f32 trunk's, the file's own trunk differs
    let f16 = fx.dir.join("base-f16.cmf");
    cortiq_embryo::export::export_genome(
        &fx.ck,
        fx.tok_json.as_bytes(),
        &f16,
        cortiq_core::TensorDtype::F16,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny".into(),
            status: "candidate".into(),
        }),
    )
    .unwrap();
    let m16 = CmfModel::open(&f16).unwrap();
    let g16 = m16.header.genome.as_ref().unwrap();
    assert_eq!(g16.encoding, "f16");
    assert_eq!(
        g16.master_trunk_hash, g.trunk_hash,
        "f16 master = f32 trunk"
    );
    assert_ne!(g16.trunk_hash, g16.master_trunk_hash);
    // without --genome-id: a legacy file (no bits, no genome in provenance)
    let plain = fx.dir.join("plain.cmf");
    cortiq_embryo::export::export(&fx.ck, fx.tok_json.as_bytes(), &plain).unwrap();
    let mp = CmfModel::open(&plain).unwrap();
    assert!(mp.header.genome.is_none() && mp.header.lineage.is_empty());
    assert_eq!(mp.required_features & features::GENOME, 0);
    assert!(
        mp.header
            .provenance
            .as_ref()
            .unwrap()
            .get("genome")
            .is_none()
    );
    // a bad status is refused before any file is written
    let bad = fx.dir.join("bad.cmf");
    assert!(
        cortiq_embryo::export::export_genome(
            &fx.ck,
            fx.tok_json.as_bytes(),
            &bad,
            cortiq_core::TensorDtype::F32,
            Some(&cortiq_embryo::export::ExportGenome {
                id: "x".into(),
                status: "rejected".into(),
            }),
        )
        .is_err()
    );
    assert!(!bad.exists());
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn phi_frame_is_the_single_turn_prompt_of_the_sft_records() {
    let Some(fx) = fixture("frame") else {
        return;
    };
    let bpe = Bpe::from_json(fx.tok_json.as_bytes()).unwrap();
    let spec = phi_spec(&bpe, 1).unwrap();
    assert_eq!(spec.pool, "span_mean");
    assert_eq!(spec.norm, "unit");
    let im_start = bpe.special_id("<|im_start|>").unwrap();
    let im_end = bpe.special_id("<|im_end|>").unwrap();
    assert_eq!(spec.prefix_ids[0], im_start);
    assert_eq!(spec.suffix_ids[0], im_end);
    assert!(spec.suffix_ids.contains(&im_start));
    // every train record starts with prefix ++ encode(q) ++ suffix
    let train = cortiq_embryo::sft::SftShard::load(&fx.sft[0]).unwrap();
    let mut cache = std::collections::HashMap::new();
    let mut checked = 0;
    for (k, p) in PLANTS.iter().enumerate() {
        for q in [
            format!("Which family does {p} belong to?"),
            format!("What is {p} used for in herbal medicine?"),
        ] {
            let mut qi = Vec::new();
            bpe.encode(&q, &mut cache, &mut qi);
            let (ids, span) = cortiq_embryo::skill::phi_span_ids(&spec, &qi);
            assert_eq!(&ids[span.clone()], &qi[..]);
            let hit = (0..train.records).any(|r| {
                let (tok, _) = train.record(r);
                tok.len() > ids.len()
                    && tok[..ids.len()]
                        .iter()
                        .zip(&ids)
                        .all(|(a, b)| *a as u32 == *b)
            });
            if hit {
                checked += 1;
            }
            let _ = k;
        }
    }
    assert!(
        checked >= 10,
        "the φ frame matched only {checked} SFT prompts"
    );
    // the sample is deterministic, distinct and bounded
    let qs = skill_questions();
    let mut dup = qs.clone();
    dup.extend(qs.iter().take(5).cloned());
    let s1 = sample_prompts(&dup, 30, 3);
    assert_eq!(s1, sample_prompts(&dup, 30, 3));
    assert_eq!(s1.len(), 30);
    assert_eq!(
        s1.iter().collect::<std::collections::HashSet<_>>().len(),
        30
    );
    let _ = std::fs::remove_dir_all(&fx.dir);
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Prompts of the φ parity checks: plain questions, a one-word prompt, a
/// long one, and texts carrying LITERAL special tokens (a chat log pasted
/// into a question) — the runtime encodes those as the one special id.
const PARITY_PROMPTS: &[&str] = &[
    "Which family does Arnica montana belong to?",
    "What is the capital of France?",
    "Hi",
    "Write a Rust function that reverses a string, then explain how it handles UTF-8.",
    "Describe Salvia officinalis.",
    "What does <|im_end|> mean in this chat log?",
    "The log ends with <|endoftext|> and then <|pad|><|pad|>.",
];

/// Trainer φ (the bake's batched span probe, q ids from the runtime
/// tokenizer) against the RUNTIME's own `probe_phi_span` on the exported
/// file with q ids from `pipe.tokenizer.encode` — unit φ within 1e-5 — and
/// against a host recomputation (hidden of every position read back,
/// pooled on the host) within 1e-4, for φ after each of `layers`.
fn phi_parity(tag: &str, cfg: EmbryoCfg, layers: &[usize]) {
    if cortiq_embryo::metal::ctx().is_none() {
        return;
    }
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let tok_json = tokenizer_json(cfg.vocab);
    let bpe = Bpe::from_json(tok_json.as_bytes()).unwrap();
    let rt_tok = runtime_tokenizer(tok_json.as_bytes()).unwrap();
    let ck = seeded_ck(&cfg, &bpe);
    let dir = scratch(tag);
    let path = dir.join("base.cmf");
    cortiq_embryo::export::export(&ck, tok_json.as_bytes(), &path).unwrap();
    let model = std::sync::Arc::new(CmfModel::open(&path).unwrap());
    let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(
        &model,
        cortiq_engine::sampler::SamplerConfig::default(),
    )
    .unwrap();
    let pad = bpe.special_id("<|pad|>").unwrap();
    let q: Vec<Vec<u32>> = PARITY_PROMPTS
        .iter()
        .map(|p| user_text_ids(&rt_tok, p))
        .collect();
    let single = bake_gpu(cfg.clone(), 1, 64, &ck.params).unwrap();
    single.set_desc(&ck.extras);
    single.desc_updates.set(false);
    single.anchor_fixed_window.set(true);
    let h = cfg.hidden;
    for &layer in layers {
        let spec = phi_spec(&bpe, layer).unwrap();
        let (phis, skipped) = probe_prompts(&ck, &spec, &q, 4, 512, pad).unwrap();
        assert!(skipped.is_empty());
        let (mut worst_rt, mut worst_host) = (0f32, 0f32);
        for (k, p) in PARITY_PROMPTS.iter().enumerate() {
            // PHI-5: the runtime's tokenization of the user text is the trainer's
            let q_rt = pipe.tokenizer.encode(p);
            assert_eq!(
                q_rt, q[k],
                "{tag}: runtime and trainer encode {p:?} differently"
            );
            let (ids, span) = cortiq_embryo::skill::phi_span_ids(&spec, &q_rt);
            let rt = pipe.probe_phi_span(&ids, layer, span.clone());
            worst_rt = worst_rt.max(max_abs_diff(&unit(&phis[k]), &unit(&rt)));
            if ids.len() <= 64 {
                let mut toks = vec![pad; 64];
                toks[..ids.len()].copy_from_slice(&ids);
                let hid = single.hidden_after_layer(&toks, layer);
                let mut host = vec![0f32; h];
                for t in span.clone() {
                    for j in 0..h {
                        host[j] += hid[t * h + j];
                    }
                }
                host.iter_mut().for_each(|x| *x /= span.len() as f32);
                worst_host = worst_host.max(max_abs_diff(&phis[k], &host));
            }
        }
        eprintln!(
            "{tag} φ after layer {layer}: unit φ trainer vs runtime probe_phi_span {worst_rt:.2e}, \
             batched vs host {worst_host:.2e}"
        );
        assert!(
            worst_rt <= 1e-5,
            "{tag} layer {layer}: trainer φ vs runtime φ {worst_rt}"
        );
        assert!(
            worst_host <= 1e-4,
            "{tag} layer {layer}: span probe vs host {worst_host}"
        );
    }
    drop((pipe, model));
    let _ = std::fs::remove_dir_all(&dir);
}

/// fam-a geometry: φ after the bounded anchor (layer 1) and after a GDN
/// layer (2); the special-literal prompts encode as the runtime serves them.
#[test]
fn phi_parity_gdn_anchor_and_gdn_layers() {
    if cortiq_embryo::metal::ctx().is_none() {
        return;
    }
    // PHI-5: the old trainer path (Bpe::encode, no added-token split) breaks
    // a literal special into pieces; the runtime's encode keeps the one id
    let tok_json = tokenizer_json(fam_a_tiny().vocab);
    let bpe = Bpe::from_json(tok_json.as_bytes()).unwrap();
    let rt_tok = runtime_tokenizer(tok_json.as_bytes()).unwrap();
    let im_end = bpe.special_id("<|im_end|>").unwrap();
    let p = PARITY_PROMPTS[5];
    let ids = user_text_ids(&rt_tok, p);
    assert!(ids.contains(&im_end), "{ids:?}");
    let mut old = Vec::new();
    bpe.encode(p, &mut std::collections::HashMap::new(), &mut old);
    assert!(!old.contains(&im_end) && old != ids);
    phi_parity("parity_gdn", fam_a_tiny(), &[1, 2]);
}

/// The legacy vmf_phase mixer (full-causal anchor at layer 3): φ after two
/// non-anchor layers.
#[test]
fn phi_parity_vmf_phase() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.layers = 4;
    cfg.anchor_every = 4;
    cfg.experts = 4;
    phi_parity("parity_vmf", cfg, &[1, 2]);
}

/// Mixed Phase-Delta (layer 1 runs the Phase-Delta operator, layer 2 the
/// legacy vmf_phase): φ after each.
#[test]
fn phi_parity_mixed_phase_delta() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.layers = 4;
    cfg.anchor_every = 4;
    cfg.experts = 4;
    cfg.phase_delta_layer = Some(1);
    phi_parity("parity_phase_delta", cfg, &[1, 2]);
}

#[test]
fn bake_v2_appends_a_bound_routed_record_and_leaves_the_genome_untouched() {
    let Some(fx) = fixture("bake") else {
        return;
    };
    let out = fx.dir.join("skilled.cmf");
    let mut a = args("herbs");
    a.route_margin = Some(0.05);
    let summary = bake_v2(&inputs(&fx, out.clone()), &a, &|| false).expect("bake v2");
    eprintln!("{summary}");
    for k in ["base_dev_nll", "bestA", "bestB"] {
        assert!(summary[k].as_f64().unwrap().is_finite(), "{k}");
    }
    let (b0, ba, bb) = (
        summary["base_dev_nll"].as_f64().unwrap(),
        summary["bestA"].as_f64().unwrap(),
        summary["bestB"].as_f64().unwrap(),
    );
    assert!(
        bb <= ba + 1e-9 && ba <= b0 + 1e-9,
        "selection is monotone: {b0} {ba} {bb}"
    );
    assert_eq!(summary["layers"], serde_json::json!([2, 3]));
    assert_eq!(summary["phi_layer"], 1);
    assert_eq!(summary["kept"].as_array().unwrap().len(), 2);
    // T1: selection ran on EVERY dev record; T2: on the dropless operator
    let dev = SftShard::load(&fx.sft[1]).unwrap();
    assert_eq!(summary["dev_records"]["used"], dev.records);
    assert_eq!(summary["dev_records"]["total"], dev.records);
    assert_eq!(summary["dev_records"]["selection"], "all");
    assert_eq!(summary["moe_capacity"], "dropless");
    assert_eq!(summary["trunk_params"], "f32");
    // PHI-7: the runtime replayed the holdouts before the file was published
    let rc = &summary["runtime_check"];
    assert!(rc["prompts"].as_u64().unwrap() > 0, "{rc}");
    assert_eq!(rc["decisions_equal"], rc["prompts"]);
    assert!(rc["max_unit_phi_delta"].as_f64().unwrap() <= 1e-5, "{rc}");
    assert_eq!(summary["router"]["base_descriptor"], "fitted");
    assert!((summary["router"]["margin"].as_f64().unwrap() - 0.05).abs() < 1e-6);
    // T5: no temp file is left next to --out
    for e in std::fs::read_dir(&fx.dir).unwrap().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        assert!(!n.contains(".bake-"), "leftover temp {n}");
    }
    assert!(summary["calib"]["temperature"].as_f64().unwrap() > 0.0);
    assert!(summary["lm_dev_nll"]["base"].as_f64().is_some());
    assert!(summary["final_nll"]["skill"].as_f64().is_some());

    // ---- the file ----
    let m0 = CmfModel::open(&fx.base).unwrap();
    let m1 = CmfModel::open(&out).unwrap();
    let bits = features::GENOME | features::SKILLS_V2 | features::ROUTER_V2;
    assert_eq!(m1.required_features & bits, bits);
    assert_eq!(
        m0.required_features & (features::SKILLS_V2 | features::ROUTER_V2),
        0
    );
    assert_eq!(m0.trunk_hash(), m1.trunk_hash(), "G1: trunk hash");
    for t in &m0.tensors {
        assert_eq!(
            m1.tensor(&t.name),
            Some(t),
            "G1: directory entry {}",
            t.name
        );
    }
    let (f0, f1) = (
        std::fs::read(&fx.base).unwrap(),
        std::fs::read(&out).unwrap(),
    );
    assert!(f1.len() > f0.len());
    assert!(
        f0[128..] == f1[128..f0.len()],
        "G1: prefix bytes [128, len(F0))"
    );
    // the record
    let g = m1.header.genome.as_ref().unwrap();
    let s = &m1.header.skills[0];
    assert_eq!(s.id, "herbs");
    assert_eq!(s.kind.as_deref(), Some("ffn_replace"));
    assert_eq!(s.status.as_deref(), Some("quarantine"));
    assert!(s.gate.is_none());
    assert_eq!(s.prompt_contract.as_deref(), Some("cmf-im-v1"));
    let b = s.bound.as_ref().unwrap();
    assert_eq!(
        (b.genome_id.as_str(), b.generation),
        (g.id.as_str(), g.generation)
    );
    assert_eq!(b.master_trunk_hash, g.master_trunk_hash);
    let se = s.state_effect.as_ref().unwrap();
    assert_eq!(se.first_affected_layer, 2);
    assert_eq!(
        se.switch, "sequence_start",
        "the anchor of layer 3 follows FFN 2"
    );
    assert_eq!(s.overrides.len(), 6);
    for o in &s.overrides {
        let base_entry = m0.tensor(&o.name).expect("override names a trunk tensor");
        assert_eq!(o.base_hash, format!("{:016x}", base_entry.hash));
        assert!(m1.tensor(&format!("skill.herbs.{}", o.name)).is_some());
    }
    let origin = s.origin.as_ref().unwrap();
    assert_eq!(origin["trigger"], "user_corpus");
    assert_eq!(
        origin["recipe"],
        "DTG-MA mask (phase A) + polish under the hard mask (phase B)"
    );
    // sft train/dev/final + manifest + lm train/dev + phi + general
    assert_eq!(origin["dataset_sha256"].as_array().unwrap().len(), 8);
    assert!(origin["dataset_sha256"][0].as_str().unwrap().len() == 64);
    assert_eq!(origin["mask"]["3"]["neurons"], fx.ck.cfg.inter);
    assert_eq!(origin["moe_capacity"], "dropless");
    assert_eq!(origin["dev_records"]["used"], dev.records);
    // PHI-7: the calibration holdouts are named (lines + sha256) for route-eval
    for class in ["skill", "general"] {
        let hold = &origin["phi"][class]["holdout"];
        assert!(!hold["lines"].as_array().unwrap().is_empty(), "{class}");
        assert_eq!(hold["sha256"].as_str().unwrap().len(), 64);
    }
    // T6: quality states what was verified about the held-out set
    let q = s.quality.as_ref().unwrap();
    assert_eq!(q["set"], "sft_dev (response-only)");
    assert!(
        q["held_out"]["records_disjoint"]
            .as_str()
            .unwrap()
            .starts_with("verified")
    );
    assert!(
        q["held_out"]["group_disjoint"]["status"]
            .as_str()
            .unwrap()
            .starts_with("verified by --sft-manifest"),
        "{q}"
    );
    // router v2
    let r = m1.header.router.as_ref().unwrap();
    assert_eq!(
        (r.version, r.policy.as_str(), r.granularity.as_str()),
        (2, "backbone_gated", "request")
    );
    assert_eq!(r.phi.layer, 1);
    assert_eq!(r.base.metric, "mse_unit");
    assert_eq!(s.selection.as_ref().unwrap().phi_layer, 1);
    assert_eq!(
        r.skills_hash,
        format!("{:016x}", cortiq_engine::router::skills_hash(&m1.header))
    );
    let measured = r.measured.as_ref().unwrap();
    assert!(measured["in_sha256"].as_str().unwrap().len() == 64);
    assert!(measured["false_accept_upper95"].as_f64().is_some());
    let cal = m1.header.routing.as_ref().unwrap();
    assert!((cal.target_fpr - 0.02).abs() < 1e-6);
    assert_eq!(m1.header.lineage.last().unwrap().event, "skill_committed");
    assert_eq!(m1.header.segments.len(), 2);
    // a quarantine skill is never auto-routed
    assert!(!s.is_auto_routable());

    // ---- runtime: the backbone is the base bit for bit, the skill is live ----
    let m0 = std::sync::Arc::new(m0);
    let m1 = std::sync::Arc::new(m1);
    let sc = cortiq_engine::sampler::SamplerConfig::default();
    let mut p0 = cortiq_engine::pipeline::Pipeline::from_model(&m0, sc.clone()).unwrap();
    let mut p1 = cortiq_engine::pipeline::Pipeline::from_model(&m1, sc.clone()).unwrap();
    let mut ps =
        cortiq_engine::pipeline::Pipeline::from_model_with_skill(&m1, sc, Some("herbs")).unwrap();
    let bpe = Bpe::from_json(fx.tok_json.as_bytes()).unwrap();
    let mut ids = Vec::new();
    bpe.encode_with_specials(
        "<|im_start|>user\nWhich family does Arnica montana belong to?<|im_end|>\n<|im_start|>assistant\n",
        &mut std::collections::HashMap::new(),
        &mut ids,
    );
    let l0 = p0.prefill_next_logits(&ids, None);
    let l1 = p1.prefill_next_logits(&ids, None);
    let lsk = ps.prefill_next_logits(&ids, None);
    assert!(
        l0.iter().zip(&l1).all(|(a, b)| a.to_bits() == b.to_bits()),
        "G2 (in-process, CPU): backbone logits of F1 must be bit-identical to F0"
    );
    let d = l0
        .iter()
        .zip(&lsk)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("runtime: max|Δ logits| backbone vs skill = {d:.3e}");
    assert!(
        d > 0.0,
        "the skill overlay must change the runtime's output"
    );
    drop((p0, p1, ps));

    // ---- PHI-7: the runtime replay refuses a φ or an encoding that differs ----
    let rt_tok = runtime_tokenizer(fx.tok_json.as_bytes()).unwrap();
    let spec = m1.header.router.as_ref().unwrap().phi.clone();
    let text = "Which family does Arnica montana belong to?";
    let q = user_text_ids(&rt_tok, text);
    let pad = bpe.special_id("<|pad|>").unwrap();
    let (phis, _) = probe_prompts(&fx.ck, &spec, std::slice::from_ref(&q), 1, 512, pad).unwrap();
    let case = |q: &[u32], phi: &[f32]| -> Result<serde_json::Value, String> {
        runtime_route_check(
            &out,
            &[PhiCase {
                text,
                q_ids: q,
                phi,
            }],
            1e-5,
        )
        .map_err(|e| format!("{e:#}"))
    };
    let ok = case(&q, &phis[0]).expect("the trainer's own φ passes");
    assert_eq!(ok["decisions_equal"], 1);
    let mut off = phis[0].clone();
    off[0] += 1e-2 * off.iter().map(|x| x * x).sum::<f32>().sqrt();
    let e = case(&q, &off).expect_err("a drifted φ must refuse");
    assert!(e.contains("unit φ"), "{e}");
    let e = case(&q[..q.len() - 1], &phis[0]).expect_err("another encoding must refuse");
    assert!(e.contains("encodes"), "{e}");
    drop((m0, m1));

    // ---- NF-3 / PHI-8: herbs goes live (measured gate) under router R1 ----
    let r1 = CmfModel::open(&out).unwrap().header.router.clone().unwrap();
    CmfModel::update_header_append(&out, |h| {
        let s = h.skills.iter_mut().find(|s| s.id == "herbs").unwrap();
        s.status = Some("active".into());
        s.gate = Some(serde_json::json!({"status": "measured", "set": "test"}));
    })
    .unwrap();
    assert!(
        CmfModel::open(&out).unwrap().header.skills[0].is_auto_routable(),
        "herbs is live in F1"
    );
    // ---- a second bake over F1 appends a second record, recalibrates ----
    // (another general set is passed and must NOT replace F1's backbone
    // descriptor; no --route-margin keeps F1's margin)
    let general2 = fx.dir.join("general2.jsonl");
    write_jsonl_prompts(&general2, &general_questions()[..20]);
    let out2 = fx.dir.join("skilled2.cmf");
    let mut in2 = inputs(&fx, out2.clone());
    in2.base = out.clone();
    in2.general_prompts = Some(general2);
    let mut a2 = args("herbs2");
    a2.layers = vec![3];
    a2.steps_a = 3;
    a2.steps_b = 0;
    let s2 = bake_v2(&in2, &a2, &|| false).expect("second bake");
    let m2 = CmfModel::open(&out2).unwrap();
    assert_eq!(m2.header.skills.len(), 2);
    let r2 = m2.header.router.as_ref().unwrap();
    assert_eq!(
        serde_json::to_value(&r2.base).unwrap(),
        serde_json::to_value(&r1.base).unwrap(),
        "PHI-8: the backbone descriptor of F1 is kept"
    );
    assert!((r2.margin - 0.05).abs() < 1e-6, "NF-3: F1's margin is kept");
    assert_eq!(
        r2.measured.as_ref().unwrap()["general_sha256"],
        r1.measured.as_ref().unwrap()["general_sha256"]
    );
    assert_ne!(r2.skills_hash, r1.skills_hash);
    let herbs = &m2.header.skills[0];
    assert_eq!(
        herbs.status.as_deref(),
        Some("stale_regate"),
        "NF-3: a skill gated under R1 must be re-gated under R2"
    );
    assert!(!herbs.is_auto_routable());
    let ev: Vec<&str> = m2.header.lineage.iter().map(|e| e.event.as_str()).collect();
    assert_eq!(&ev[ev.len() - 2..], &["skill_committed", "recalibrate"]);
    let rec = &m2.header.lineage.last().unwrap().detail;
    assert_eq!(rec["stale_regate"], serde_json::json!(["herbs"]));
    assert_eq!(rec["base_descriptor"], "kept");
    assert_eq!(s2["stale_regate"], serde_json::json!(["herbs"]));
    assert_eq!(s2["router"]["base_descriptor"], "kept from --base");
    let o2 = m2.header.skills[1].origin.as_ref().unwrap();
    assert_eq!(
        o2["dataset_sha256"].as_array().unwrap().len(),
        7,
        "the unused general set is not an input of the record"
    );
    assert_eq!(
        m2.header.skills[1].state_effect.as_ref().unwrap().switch,
        "transparent",
        "FFN of the last layer: nothing stateful follows"
    );
    assert_eq!(m2.header.segments.len(), 3);
    assert!(
        s2["calib"]["samples"].as_u64().unwrap() > summary["calib"]["samples"].as_u64().unwrap()
    );
    let f2 = std::fs::read(&out2).unwrap();
    assert!(
        f1[128..] == f2[128..f1.len()],
        "F2 extends F1 by a tail only"
    );
    let _ = std::fs::remove_dir_all(&fx.dir);
}

#[test]
fn bake_v2_refuses_unsafe_or_unbound_inputs() {
    let Some(fx) = fixture("refuse") else {
        return;
    };
    let err = |inp: BakeV2Inputs, a: BakeV2Args| -> String {
        format!(
            "{:#}",
            bake_v2(&inp, &a, &|| false).expect_err("must refuse")
        )
    };
    // out == base (also through a different spelling of the same path)
    let same = fx.dir.join(".").join("base.cmf");
    let e = err(inputs(&fx, same), args("x"));
    assert!(e.contains("--out") && e.contains("--base"), "{e}");
    // out exists
    let exists = fx.dir.join("exists.cmf");
    std::fs::write(&exists, b"x").unwrap();
    let e = err(inputs(&fx, exists), args("x"));
    assert!(e.contains("overwrite"), "{e}");
    // phi_layer ≥ min(layers)
    let mut a = args("x");
    a.phi_layer = 2;
    let e = err(inputs(&fx, fx.dir.join("o1.cmf")), a);
    assert!(
        e.contains("--phi-layer") && e.contains("min(--layers)"),
        "{e}"
    );
    // ckpt ≠ base: one parameter of an FFN moved
    let mut params = fx.ck.params.clone();
    let lay = Layout::new(&fx.ck.cfg);
    params[lay.embed + 5] += 1e-3;
    let other = fx.dir.join("other.ckpt");
    let ex: Vec<(&str, &[f32])> = fx
        .ck
        .extras
        .iter()
        .map(|(n, x)| (n.as_str(), x.as_slice()))
        .collect();
    save_checkpoint(&other, &fx.ck.cfg, fx.ck.step, &params, None, None, &ex).unwrap();
    let mut inp = inputs(&fx, fx.dir.join("o2.cmf"));
    inp.ckpt = other;
    let e = err(inp, args("x"));
    assert!(
        e.contains("is not the checkpoint") && e.contains("embed_tokens"),
        "{e}"
    );
    // a base without a genome
    let plain = fx.dir.join("plain.cmf");
    cortiq_embryo::export::export(&fx.ck, fx.tok_json.as_bytes(), &plain).unwrap();
    let mut inp = inputs(&fx, fx.dir.join("o3.cmf"));
    inp.base = plain;
    let e = err(inp, args("x"));
    assert!(e.contains("GENOME"), "{e}");
    // a tokenizer that is not the base's vocab
    let tok2 = fx.dir.join("tok2.json");
    std::fs::write(&tok2, fx.tok_json.replace("\"1.0\"", "\"1.1\"")).unwrap();
    let mut inp = inputs(&fx, fx.dir.join("o4.cmf"));
    inp.tokenizer = Some(tok2);
    let e = err(inp, args("x"));
    assert!(e.contains("--tokenizer"), "{e}");
    // a bad id
    let e = err(inputs(&fx, fx.dir.join("o5.cmf")), args("a.b"));
    assert!(e.contains("'.'"), "{e}");
    // T3: τ ≥ σ(3) makes the "all-on" base a model with every FFN off; τ ≤ 0
    // never drops a neuron; a negative L1 is nonsense
    for tau in [0.96f32, 0.9527, 0.0, -0.5] {
        let mut a = args("x");
        a.tau = tau;
        let e = err(inputs(&fx, fx.dir.join("o6.cmf")), a);
        assert!(e.contains("--tau"), "τ {tau}: {e}");
    }
    let mut a = args("x");
    a.l1 = -1e-3;
    let e = err(inputs(&fx, fx.dir.join("o6.cmf")), a);
    assert!(e.contains("--l1"), "{e}");
    // T6: a dev shard that is (a copy of) train under another name
    let dev_copy = fx.dir.join("dev-copy.sft");
    std::fs::copy(&fx.sft[0], &dev_copy).unwrap();
    let mut inp = inputs(&fx, fx.dir.join("o7.cmf"));
    inp.sft_dev = dev_copy;
    inp.sft_manifest = None;
    let e = err(inp, args("x"));
    assert!(e.contains("also occur in --sft-train"), "{e}");
    // T6: a manifest whose group map breaks the group rule
    let mut man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fx.manifest).unwrap()).unwrap();
    let groups = man["groups"].as_object_mut().unwrap();
    let (g, sp) = groups
        .iter()
        .next()
        .map(|(g, s)| (g.clone(), s.clone()))
        .unwrap();
    let moved = if sp == "train" { "dev" } else { "train" };
    groups.insert(g, serde_json::json!(moved));
    let bad_man = fx.dir.join("manifest-bad.json");
    std::fs::write(&bad_man, serde_json::to_vec(&man).unwrap()).unwrap();
    let mut inp = inputs(&fx, fx.dir.join("o8.cmf"));
    inp.sft_manifest = Some(bad_man);
    let e = err(inp, args("x"));
    assert!(e.contains("group rule"), "{e}");
    // T6: a manifest of another prepare run (split sizes differ)
    let mut man: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fx.manifest).unwrap()).unwrap();
    man["splits"]["dev"]["examples"] = serde_json::json!(999_999);
    let other_man = fx.dir.join("manifest-other.json");
    std::fs::write(&other_man, serde_json::to_vec(&man).unwrap()).unwrap();
    let mut inp = inputs(&fx, fx.dir.join("o9.cmf"));
    inp.sft_manifest = Some(other_man);
    let e = err(inp, args("x"));
    assert!(e.contains("split dev declares"), "{e}");
    // the first record needs the backbone class
    let mut inp = inputs(&fx, fx.dir.join("o10.cmf"));
    inp.general_prompts = None;
    let e = err(inp, args("x"));
    assert!(e.contains("--general-prompts"), "{e}");
    // nothing was written by any refusal (no output, no temp)
    for k in 1..=10 {
        assert!(!fx.dir.join(format!("o{k}.cmf")).exists());
    }
    for e in std::fs::read_dir(&fx.dir).unwrap().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        assert!(!n.contains(".bake-"), "leftover temp {n}");
    }
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// T1 + T2: a dev evaluation covers every record of the shard exactly once
/// (an even stride when capped — never the file's prefix, never repeated
/// rows), token-weighted, and on the bake's instance a record's NLL does not
/// depend on the other rows of its batch: capacity for every token
/// (dropless, the runtime's operator). The capacity-2 training instance
/// has room for 2·M/E tokens per expert only.
#[test]
fn dev_eval_covers_every_record_once_on_the_dropless_instance() {
    assert_eq!(eval_records(10, 0), (0..10).collect::<Vec<_>>());
    assert_eq!(eval_records(3, 8), vec![0, 1, 2]);
    let s = eval_records(100, 8);
    assert_eq!(s.len(), 8);
    assert!(s.windows(2).all(|w| w[0] < w[1]));
    assert!(
        s[0] == 0 && *s.last().unwrap() >= 80,
        "spread over the whole shard: {s:?}"
    );
    let Some(fx) = fixture("deveval") else {
        return;
    };
    let dev = SftShard::load(&fx.sft[1]).unwrap();
    let (cfg, t) = (fx.ck.cfg.clone(), dev.seq);
    let mk = |b: usize| {
        let g = bake_gpu(cfg.clone(), b, t, &fx.ck.params).unwrap();
        g.set_desc(&fx.ck.extras);
        g.desc_updates.set(false);
        g.anchor_fixed_window.set(true);
        g
    };
    let g5 = mk(5);
    assert!(
        g5.moe_cap >= 5 * t,
        "the bake instance holds every token in one expert (cap {})",
        g5.moe_cap
    );
    let cap2 = EmbryoGpu::new(cfg.clone(), 5, t, &fx.ck.params).unwrap();
    assert!(cap2.moe_cap < 5 * t);
    // reference: every record alone, token-weighted
    let g1 = mk(1);
    let (mut sum, mut n) = (0f64, 0usize);
    for r in 0..dev.records {
        let v = dev.record(r).1.iter().filter(|&&x| x != IGNORE).count();
        if v > 0 {
            sum += eval_answer_nll(&g1, &dev, 1, &[r]) as f64 * v as f64;
            n += v;
        }
    }
    let want = sum / n as f64;
    let all = eval_records(dev.records, 0);
    let rev: Vec<usize> = all.iter().rev().copied().collect();
    let big = dev.records + 3; // more rows than records: padding rows, no repeats
    let gbig = mk(big);
    let got = [
        eval_answer_nll(&g5, &dev, 5, &all),
        eval_answer_nll(&g5, &dev, 5, &rev),
        eval_answer_nll(&gbig, &dev, big, &all),
    ];
    eprintln!(
        "dev NLL over {} records: per-record {want:.6}, batched {got:?}",
        dev.records
    );
    for g in got {
        assert!(
            (g as f64 - want).abs() < 1e-4 * want.abs().max(1.0),
            "batched dev NLL {g} vs per-record {want}"
        );
    }
    // the old default: capacity 2·M/E — the right-padded rows compete for
    // expert slots, and the answer NLL depends on the batch (report only)
    cap2.set_desc(&fx.ck.extras);
    cap2.desc_updates.set(false);
    cap2.anchor_fixed_window.set(true);
    let c2 = eval_answer_nll(&cap2, &dev, 5, &all);
    let drops: u32 = cap2
        .routing_counts()
        .iter()
        .flatten()
        .map(|&n| n.saturating_sub(cap2.moe_cap as u32))
        .sum();
    eprintln!(
        "capacity-2 instance (cap {} of {} tokens): dev NLL {c2:.6} (Δ {:.2e}), {drops} \
         tokens over capacity in the last batch",
        cap2.moe_cap,
        5 * t,
        c2 as f64 - want
    );
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// T4 / PHI-4 (+ NF-8): an f16 genome is probed and polished on the trunk
/// the runtime serves — `served_params` rounds exactly the tensors the f16
/// export stores in half precision — so φ matches the f16 runtime at 1e-5
/// (the f32 master does not) and the bake's runtime replay passes. The base
/// here carries the name the old fixed temp of `--out` had; the bake must
/// leave it untouched.
#[test]
fn f16_genome_is_probed_and_baked_on_the_served_trunk() {
    let Some(fx) = fixture("f16") else {
        return;
    };
    let tok = fx.tok_json.as_bytes();
    let f16 = fx.dir.join(".f16.cmf.bake-tmp");
    cortiq_embryo::export::export_genome(
        &fx.ck,
        tok,
        &f16,
        cortiq_core::TensorDtype::F16,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny".into(),
            status: "candidate".into(),
        }),
    )
    .unwrap();
    let served =
        cortiq_embryo::export::served_params(&fx.ck, tok, cortiq_core::TensorDtype::F16).unwrap();
    assert_ne!(served, fx.ck.params);
    let ck_s = Checkpoint {
        cfg: fx.ck.cfg.clone(),
        step: fx.ck.step,
        params: served,
        m: None,
        v: None,
        extras: fx.ck.extras.clone(),
    };
    // the rounding is the export's rule: the f32 export of the served
    // parameters is the f16 file decoded, tensor by tensor
    let m16 = CmfModel::open(&f16).unwrap();
    let (_, specs32) =
        cortiq_embryo::export::build_export(&ck_s, tok, cortiq_core::TensorDtype::F32, None)
            .unwrap();
    let mut halves = 0;
    for t in &specs32 {
        let e = m16.tensor(&t.name).unwrap();
        let bytes = m16.tensor_bytes(&t.name).unwrap();
        let decoded: Vec<u8> = if e.dtype == cortiq_core::TensorDtype::F16 {
            halves += 1;
            bytes
                .chunks_exact(2)
                .flat_map(|c| {
                    cortiq_core::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()
                })
                .collect()
        } else {
            bytes.to_vec()
        };
        assert!(decoded == t.data, "{} is not the served tensor", t.name);
    }
    assert!(halves > 0);
    // φ: served trunk = f16 runtime (1e-5); the f32 master is off
    let bpe = Bpe::from_json(tok).unwrap();
    let rt_tok = runtime_tokenizer(tok).unwrap();
    let spec = phi_spec(&bpe, 1).unwrap();
    let pad = bpe.special_id("<|pad|>").unwrap();
    let q: Vec<Vec<u32>> = PARITY_PROMPTS[..5]
        .iter()
        .map(|p| user_text_ids(&rt_tok, p))
        .collect();
    let (phi_served, _) = probe_prompts(&ck_s, &spec, &q, 4, 512, pad).unwrap();
    let (phi_master, _) = probe_prompts(&fx.ck, &spec, &q, 4, 512, pad).unwrap();
    let model = std::sync::Arc::new(m16);
    let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(
        &model,
        cortiq_engine::sampler::SamplerConfig::default(),
    )
    .unwrap();
    let (mut d_served, mut d_master) = (0f32, 0f32);
    for (k, qk) in q.iter().enumerate() {
        let (ids, span) = cortiq_embryo::skill::phi_span_ids(&spec, qk);
        let rt = unit(&pipe.probe_phi_span(&ids, 1, span));
        d_served = d_served.max(max_abs_diff(&unit(&phi_served[k]), &rt));
        d_master = d_master.max(max_abs_diff(&unit(&phi_master[k]), &rt));
    }
    eprintln!(
        "f16 genome: unit φ vs f16 runtime — served {d_served:.2e}, f32 master {d_master:.2e}"
    );
    assert!(d_served <= 1e-5, "served trunk φ {d_served}");
    assert!(d_master > d_served, "the f32 master is not what is served");
    drop((pipe, model));
    // the bake on the f16 genome: its runtime replay (1e-5) passes
    let before = std::fs::read(&f16).unwrap();
    let out = fx.dir.join("f16.cmf");
    let mut inp = inputs(&fx, out.clone());
    inp.base = f16.clone();
    let mut a = args("herbs16");
    a.steps_a = 2;
    a.steps_b = 1;
    a.eval_every = 1;
    let s = bake_v2(&inp, &a, &|| false).expect("bake over the f16 genome");
    assert!(
        s["trunk_params"].as_str().unwrap().starts_with("f16"),
        "{s}"
    );
    assert!(s["runtime_check"]["max_unit_phi_delta"].as_f64().unwrap() <= 1e-5);
    let m = CmfModel::open(&out).unwrap();
    assert!(
        m.header.skills[0].origin.as_ref().unwrap()["trunk_params"]
            .as_str()
            .unwrap()
            .starts_with("f16")
    );
    assert_eq!(
        std::fs::read(&f16).unwrap(),
        before,
        "NF-8: the base (named like the old fixed temp) is untouched"
    );
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// T5 / NF-8: temps are fresh private files (`create_new`, unique names)
/// and publishing never replaces an existing `--out` — the refused result
/// stays at its temp path.
#[test]
fn publish_never_overwrites_and_temps_are_private() {
    let dir = scratch("publish");
    let out = dir.join("x.cmf");
    let (t1, mut f1) = create_bake_tmp(&out).unwrap();
    let (t2, mut f2) = create_bake_tmp(&out).unwrap();
    assert_ne!(t1, t2);
    use std::io::Write;
    f1.write_all(b"first").unwrap();
    f2.write_all(b"second").unwrap();
    drop((f1, f2));
    publish_new_file(&t1, &out).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), b"first");
    assert!(!t1.exists(), "the temp name is dropped after publishing");
    let e = format!("{:#}", publish_new_file(&t2, &out).unwrap_err());
    assert!(
        e.contains("refusing to overwrite") && e.contains(&t2.display().to_string()),
        "{e}"
    );
    assert_eq!(std::fs::read(&out).unwrap(), b"first", "--out untouched");
    assert_eq!(std::fs::read(&t2).unwrap(), b"second", "the result is kept");
    let _ = std::fs::remove_dir_all(&dir);
}

/// NF-4: the legacy full-rewrite tools refuse a genome file and leave it
/// byte-identical — the v1 append, the v1 calibration and the sleep daemon.
#[test]
fn legacy_tools_refuse_genome_files() {
    let Some(fx) = fixture("legacy") else {
        return;
    };
    let before = std::fs::read(&fx.base).unwrap();
    let h = fx.ck.cfg.hidden;
    let phis: Vec<Vec<f32>> = (0..12)
        .map(|k| {
            (0..h)
                .map(|j| ((k * 7 + j * 3) % 11) as f32 - 5.0)
                .collect()
        })
        .collect();
    let sel = cortiq_embryo::skill::fit_selection(&phis, 1, 2);
    let o = fx.dir.join("legacy.cmf");
    let e = cortiq_embryo::skill::append_to_cmf(
        &fx.base,
        &o,
        "night",
        &[3],
        &[],
        sel,
        serde_json::json!({}),
    )
    .expect_err("append_to_cmf on a genome file");
    assert!(format!("{e:#}").contains("GENOME"), "{e:#}");
    assert!(!o.exists());
    let e = cortiq_embryo::skill::calibrate_file(&fx.base, 0.05)
        .expect_err("calibrate_file on a genome file");
    assert!(format!("{e:#}").contains("GENOME"), "{e:#}");
    #[cfg(target_os = "macos")]
    {
        let ood = fx.dir.join("ood");
        let e = cortiq_embryo::sleep::run(cortiq_embryo::sleep::SleepArgs {
            ckpt: fx.ckpt.clone(),
            tokenizer: fx.dir.join("tokenizer.json"),
            cmf: fx.base.clone(),
            ood_dir: ood.clone(),
            idle_min: 0.0,
            min_tokens: 0,
            gate: 0.0,
            requant_gate: 0.0,
            held_out: None,
            cortiq_bin: "cortiq".into(),
            layers: vec![3],
            steps_a: 1,
            steps_b: 1,
            batch: 1,
            seq: 64,
            once: true,
            force: true,
            poll_secs: 1,
            grow_after: 1,
        })
        .expect_err("the sleep daemon on a genome file");
        assert!(format!("{e:#}").contains("sleep daemon"), "{e:#}");
        assert!(!ood.exists(), "refused before the first journal line");
    }
    assert_eq!(std::fs::read(&fx.base).unwrap(), before);
    let _ = std::fs::remove_dir_all(&fx.dir);
}

/// Phase B's trainable set at fam-a widths (hidden 384, FFN 768) over two
/// layers needs 6 × 1152 grad-norm partials — more than the 4096-entry
/// partial buffer. The reported norm (which drives the clip and the
/// non-finite guard) must still cover every range.
#[test]
fn phase_b_grad_norm_covers_every_ffn_range() {
    let Some(c) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let mut cfg = EmbryoCfg::tiny();
    cfg.hidden = 384;
    cfg.inter = 768;
    let lay = Layout::new(&cfg);
    let p0 = init_params(&cfg, &lay, 3);
    let (b, t) = (2usize, 64usize);
    let mut gpu = EmbryoGpu::new(cfg.clone(), b, t, &p0).unwrap();
    gpu.desc_updates.set(false);
    gpu.skill = Some(cortiq_embryo::model::SkillState::new(
        c,
        vec![0, 1],
        cfg.inter,
        3.0,
        0.5,
    ));
    gpu.skill.as_ref().unwrap().hard.set(true);
    let tokens: Vec<u32> = (0..b * t).map(|k| (k * 97 % cfg.vocab) as u32).collect();
    let targets: Vec<u32> = (0..b * t)
        .map(|k| ((k * 97 + 5) % cfg.vocab) as u32)
        .collect();
    let (_, gn) = gpu.train_step_skill(&tokens, &targets, 0.0, 0.0, 1e9, true);
    let g = gpu.grads_host();
    let ranges = cortiq_embryo::skill::ffn_ranges(&cfg, &lay, &[0, 1]);
    let groups: usize = ranges.iter().map(|&(_, n)| n.div_ceil(256)).sum();
    assert!(
        groups > 4096,
        "the fixture must overflow one partial buffer ({groups})"
    );
    let want = ranges
        .iter()
        .map(|&(o, n)| g[o..o + n].iter().map(|x| (*x as f64).powi(2)).sum::<f64>())
        .sum::<f64>()
        .sqrt();
    let rel = (gn as f64 - want).abs() / want.max(1e-30);
    eprintln!("phase-B grad norm {gn:.6e} vs host {want:.6e} (rel {rel:.2e}, {groups} partials)");
    assert!(rel < 1e-4, "grad norm misses ranges: {gn} vs {want}");
    assert_eq!(gpu.params_host(), p0, "lr 0 leaves the arena unchanged");
}
