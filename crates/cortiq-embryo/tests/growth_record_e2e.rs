//! Growth records END TO END (SPEC_GROWTH_RECORDS §1–§3 + addendum §5):
//! a tiny fam-a genome (GDN mixer, bounded anchors at layers 1 and 3, 4
//! resonance experts, hidden 64) is TRAINED a few steps on synthetic
//! corpus B (capitals / code / arithmetic as cmf-im-v1 dialogues — the
//! frame `growth-eval` renders prompts with), exported with a genome block
//! (`moe_experts = 4`, status `sealed`) = F0, then grown K = 1 on corpus A
//! (herbal facts in Latin — a token-disjoint distribution) with `--held`,
//! once over every layer and once over the top half (`--layers 2,3`),
//! each `--record-out` = F1. For each F1:
//!
//!  * it opens with the GENOME | SKILLS_V2 bits; `trunk_hash` equals F0's
//!    and the bytes `[128, len(F0))` are F0's (a true tail append);
//!  * the loader mounts 5 experts in the grown layers and 4 elsewhere
//!    (`CMF_GROWTH=all` — the record is quarantined); `CMF_GROWTH=off`
//!    gives 4 everywhere;
//!  * on corpus-B prompts with zero grown wins the per-op logits of F1
//!    (growth mounted, shell on) are bit-identical to F0's over the prompt
//!    + 16 greedy tokens (G2 by construction), and such prompts exist;
//!  * the trainer's witness (`trace_routes` + `coverage` with the runtime
//!    formula) covers more corpus-A than corpus-B tokens;
//!  * `cortiq growth-eval F1 --growth all --shell both` runs (per-op) and
//!    its hit rate on B prompts is below the one on A prompts, shell on
//!    and off.
//!
//! `novel_source_mode_grows_a_novel_expert_not_a_general_one`: on the
//! same genome, a growth corpus M that is MOSTLY the genome's own world
//! (corpus B's sentences) with a Latin herbal minority (corpus A) — the
//! shape of the real herbal corpus — grown with `--source-mode novel`
//! (descriptor = the cluster of the M tokens novel for the trunk, τ from
//! corpus B, frozen) against `hottest` (today's default, whose copy of
//! the expert hottest on M becomes a second general expert): far less of
//! corpus B routes to the novel expert without a shell, most of the novel
//! M tokens do, the frozen descriptor is exactly the initialisation,
//! `reshell` reproduces the record (sources, shells, desc mode inferred),
//! K = 2 clusters, the record validates and mounts (fewer grown wins on B
//! prompts than the hottest record's); the refusals (novel without
//! `--general`, too few novel tokens, `--desc-mode frozen` on an adapted
//! checkpoint).
//!
//! Metal on macOS, Vulkan with `--features vulkan` on Linux (skipped
//! without a device). CPU runtime (`CMF_GPU=0`); the tests hold
//! [`ENV_LOCK`] for their whole run, so the `CMF_GROWTH*` environment is
//! read sequentially.
//!
//! Run (stand, via srv.sh; the `cortiq` binary is built first into the
//! same target dir, else the test builds it itself with `cargo build`):
//!   srv.sh build -p cortiq-cli -p cortiq-embryo --features cortiq-embryo/vulkan --bin cortiq
//!   SRV_ENV=CMF_GPU=0 srv.sh --no-sync test -p cortiq-embryo --features vulkan \
//!       --test growth_record_e2e -- --nocapture
//! `CORTIQ_BIN=/path/to/cortiq` overrides the binary lookup.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_core::format::{CmfModel, features};
use cortiq_core::knowledge::skill_kind;
use cortiq_embryo::cli::{BiasMode, DescMode, GrowCli, ReshellCli, ShellMode, SourceMode, grow, reshell};
use cortiq_embryo::growth::{GrowthDesc, NOVEL_MIN_PER_EXPERT, coverage, trace_routes};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, Mixer, init_params};
use cortiq_embryo::tokenizer::Bpe;
use cortiq_embryo::train::{Checkpoint, Shard, load_checkpoint, save_checkpoint};
use cortiq_engine::pipeline::{FfnKind, MoeFfn, set_growth_shell};
use cortiq_engine::{Pipeline, SamplerConfig};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

const E0: usize = 4;
const SEQ: usize = 64;
const BATCH: usize = 4;
const GREEDY: usize = 16;

/// The tests of this file set `CMF_GROWTH*` for the pipelines they build:
/// each holds this for its whole run.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
    cfg.experts = E0;
    cfg
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("embryo_growth_e2e_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

// ───────────────────────── the two corpora ─────────────────────────

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
];
const FAMILIES: &[&str] = &["Asteraceae", "Lamiaceae", "Rosaceae", "Apiaceae", "Urticaceae"];

const LATIN: &[&str] = &["hortus", "silva", "pratum"];
/// The plants of corpus A (a small vocabulary keeps the growth cluster
/// compact: the shells then exclude the genome's tokens).
const A_PLANTS: usize = 6;

/// Corpus A: herbal facts in Latin only (the knowledge to grow) — no
/// English function words, digits, newlines or B's punctuation, so the
/// two corpora share no token (the growth's shells then say something
/// about the distributions, not about `the` and `.`).
fn corpus_a_text(reps: usize) -> String {
    let mut s = String::new();
    for _ in 0..reps {
        for (k, p) in PLANTS.iter().enumerate().take(A_PLANTS) {
            let fam = FAMILIES[k % FAMILIES.len()];
            let loc = LATIN[k % LATIN.len()];
            s.push_str(&format!(
                "{p} familia {fam} herba perennis ~ {p} folia sicca infusum lene {loc} ~ "
            ));
        }
    }
    s
}

const COUNTRIES: &[&str] = &["France", "Japan", "Brazil", "Canada", "Egypt", "Norway", "India", "Chile"];

const IM_USER: &str = "<|im_start|>user\n";
const IM_ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n";
const IM_END: &str = "<|im_end|>\n";

/// One cmf-im-v1 dialogue turn (the frame `growth-eval` renders prompts
/// with — a chat genome has seen it).
fn dialogue(user: &str, assistant: &str) -> String {
    format!("{IM_USER}{user}{IM_ASSISTANT}{assistant}{IM_END}")
}

/// The user turns of corpus B (the `growth-eval` B prompts are these).
fn b_questions() -> Vec<(String, String)> {
    let mut v = Vec::new();
    for (i, c) in COUNTRIES.iter().enumerate() {
        v.push((
            format!("What is the capital of {c}?"),
            format!("Roughly {i} million people live there."),
        ));
    }
    for l in ["Rust", "Python", "C", "Go"] {
        v.push((
            format!("How do I sort a vector of integers in {l}?"),
            format!("Write a {l} function that reverses a string."),
        ));
    }
    for n in 2..10 {
        v.push((format!("Solve {n}x + 3 = {} for x.", n * 5), "The answer is 5.".to_string()));
    }
    v
}

/// Corpus B: the genome's world (capitals, code, arithmetic) as cmf-im-v1
/// dialogues.
fn corpus_b_text(reps: usize) -> String {
    let mut s = String::new();
    for _ in 0..reps {
        for (q, a) in b_questions() {
            s.push_str(&dialogue(&q, &a));
        }
    }
    s
}

fn strip_specials(text: &str) -> String {
    let mut t = text.to_string();
    for sp in cortiq_embryo::tokenizer::SPECIALS {
        t = t.replace(sp, " ");
    }
    t
}

fn tokenizer_json(vocab: usize) -> String {
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    let text = format!("{}{}", corpus_a_text(1), strip_specials(&corpus_b_text(1)));
    cortiq_embryo::tokenizer::count_words(&text, &re, &mut counts);
    cortiq_embryo::tokenizer::train(&counts, vocab, false).to_hf_json()
}

fn encode(bpe: &Bpe, text: &str) -> Shard {
    let mut ids = Vec::new();
    bpe.encode_with_specials(text, &mut std::collections::HashMap::new(), &mut ids);
    Shard {
        tokens: ids.iter().map(|&x| x as u16).collect(),
    }
}

// ───────────────────────── the genome ─────────────────────────

/// A genome checkpoint trained `steps` steps on corpus B (descriptor EMA
/// during training, the P1 subspace step at the end — the routing
/// resonates with B's tokens).
fn genome_ck(cfg: &EmbryoCfg, corpus_b: &Shard, steps: usize) -> Checkpoint {
    let lay = Layout::new(cfg);
    let params = init_params(cfg, &lay, 11);
    let mut gpu = EmbryoGpu::new(cfg.clone(), BATCH, SEQ, &params).unwrap();
    let toks = &corpus_b.tokens;
    let n_w = (toks.len() - 1) / SEQ;
    assert!(n_w >= BATCH, "corpus B has {n_w} windows");
    let mut cov = Vec::new();
    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for s in 0..steps {
        let mut tk = Vec::with_capacity(BATCH * SEQ);
        let mut tg = Vec::with_capacity(BATCH * SEQ);
        for r in 0..BATCH {
            let w = (s * BATCH + r) % n_w;
            tk.extend(toks[w * SEQ..w * SEQ + SEQ].iter().map(|&x| x as u32));
            tg.extend(toks[w * SEQ + 1..w * SEQ + SEQ + 1].iter().map(|&x| x as u32));
        }
        let (loss, _, _) = gpu.train_step(&tk, &tg, 2e-3, 0.0, 1.0);
        if s == 0 {
            first = loss;
        }
        last = loss;
        if s % 10 == 9 {
            gpu.update_subspaces(&mut cov, 0.9);
        }
    }
    gpu.update_subspaces(&mut cov, 0.9);
    eprintln!("genome: {steps} steps on corpus B, loss {first:.4} → {last:.4}");
    assert!(last.is_finite() && last < first, "the genome did not learn corpus B");
    let trained = gpu.params_host();
    assert_eq!(trained.len(), params.len());
    let extras: Vec<(String, Vec<f32>)> = gpu
        .desc_host()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    drop(gpu);
    Checkpoint {
        cfg: cfg.clone(),
        step: steps as u32,
        params: trained,
        m: None,
        v: None,
        extras,
    }
}

// ───────────────────────── the runtime walk ─────────────────────────

fn moe(p: &Pipeline, layer: usize) -> &MoeFfn {
    match &p.weights.layers[layer].ffn {
        FfnKind::Moe(m) => m,
        _ => panic!("layer {layer} is not an MoE layer"),
    }
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

/// Grown wins per MoE layer since the counters were last read.
fn grown_wins(p: &Pipeline) -> Vec<u64> {
    (0..p.weights.layers.len())
        .filter_map(|l| match &p.weights.layers[l].ffn {
            FfnKind::Moe(m) => {
                let mut st = m.stats.borrow().clone();
                st.resize(m.experts.len(), 0);
                Some(st[m.experts.len() - m.grown.len()..].iter().sum())
            }
            _ => None,
        })
        .collect()
}

/// The prefill of `ids` + `GREEDY` greedy tokens, per-op: per-position
/// logits and the grown wins over the whole record.
fn walk(p: &mut Pipeline, ids: &[u32]) -> (Vec<Vec<f32>>, u64) {
    p.reset_session();
    let before = grown_wins(p);
    let mut out = vec![p.forward_ids(ids, None).expect("forward_ids")];
    for s in 0..GREEDY {
        let t = argmax(out.last().unwrap());
        out.push(p.decode_step_logits(t, ids.len() + s));
    }
    let after = grown_wins(p);
    let wins = after.iter().zip(&before).map(|(a, b)| a - b).sum();
    (out, wins)
}

fn bits_equal(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y).all(|(u, v)| u.to_bits() == v.to_bits())
        })
}

fn set_growth(mode: Option<&str>) {
    // SAFETY: before the pipeline that reads it is built; one test thread.
    unsafe {
        match mode {
            Some(m) => std::env::set_var("CMF_GROWTH", m),
            None => std::env::remove_var("CMF_GROWTH"),
        }
    }
}

fn pipeline(m: &Arc<CmfModel>) -> Pipeline {
    let p = Pipeline::from_model(m, SamplerConfig::default()).expect("pipeline");
    // the routing counters are the host route's: per-op only
    p.mark_graph_refused();
    p
}

/// Prompt windows (`n` × `len` tokens) cut from a token stream.
fn windows(shard: &Shard, n: usize, len: usize) -> Vec<Vec<u32>> {
    let stride = (shard.tokens.len() - len) / n;
    (0..n)
        .map(|i| shard.tokens[i * stride..i * stride + len].iter().map(|&x| x as u32).collect())
        .collect()
}

// ───────────────────────── the cortiq binary ─────────────────────────

/// `cortiq` next to this test's profile dir (`CORTIQ_BIN` overrides);
/// built with `cargo build -p cortiq-cli --bin cortiq` into the same
/// target dir when absent.
fn cortiq_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CORTIQ_BIN") {
        return PathBuf::from(p);
    }
    let exe = std::env::current_exe().unwrap();
    let profile_dir = exe.parent().unwrap().parent().unwrap().to_path_buf();
    let cand = profile_dir.join(format!("cortiq{}", std::env::consts::EXE_SUFFIX));
    if cand.exists() {
        return cand;
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut c = Command::new(cargo);
    c.args(["build", "-p", "cortiq-cli", "--bin", "cortiq"]);
    if profile_dir.file_name().and_then(|n| n.to_str()) == Some("release") {
        c.arg("--release");
    }
    eprintln!("building {} …", cand.display());
    let st = c.status().expect("spawn cargo build");
    assert!(st.success() && cand.exists(), "cargo build -p cortiq-cli --bin cortiq failed");
    cand
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

// ───────────────────────── one grown file ─────────────────────────

struct Grown {
    tag: &'static str,
    layers: Vec<usize>,
    out: PathBuf,
    out_ckpt: PathBuf,
    shells: Vec<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
fn check_grown(
    g: &Grown,
    base: &Path,
    f0_bytes: &[u8],
    n_layers: usize,
    prompts_b: &[Vec<u32>],
    prompts_a: &[Vec<u32>],
    shard_a_held: &Shard,
    shard_b: &Shard,
    cortiq: &Path,
    prompts_jsonl: &Path,
) {
    let tag = g.tag;
    // ---- format: bits, trunk hash, prefix bytes ----
    let f1_bytes = std::fs::read(&g.out).unwrap();
    assert!(f1_bytes.len() > f0_bytes.len(), "{tag}: F1 is not longer than F0");
    assert!(
        f1_bytes[128..f0_bytes.len()] == f0_bytes[128..],
        "{tag}: bytes [128, len(F0)) of F1 differ from F0"
    );
    let m0 = CmfModel::open(base).unwrap();
    let m1 = CmfModel::open(&g.out).unwrap();
    assert_ne!(m1.required_features & features::GENOME, 0, "{tag}: GENOME bit");
    assert_ne!(m1.required_features & features::SKILLS_V2, 0, "{tag}: SKILLS_V2 bit");
    assert_eq!(m0.trunk_hash(), m1.trunk_hash(), "{tag}: trunk hash");
    assert_eq!(
        m1.header.genome.as_ref().unwrap().trunk_hash,
        m0.header.genome.as_ref().unwrap().trunk_hash
    );
    cortiq_embryo::skill::verify_append(base, &g.out).unwrap();
    assert_eq!(m1.header.skills.len(), 1);
    let rec = &m1.header.skills[0];
    assert_eq!(rec.kind.as_deref(), Some(skill_kind::EXPERT_APPEND));
    assert_eq!(rec.layers, g.layers, "{tag}: record layers");
    assert_eq!(rec.status.as_deref(), Some("quarantine"));
    assert_eq!(rec.experts.as_ref().unwrap().count, 1);
    let m1 = Arc::new(m1);
    let m0 = Arc::new(m0);
    // ---- loader: 5 experts in the grown layers, 4 elsewhere; off → 4 everywhere ----
    set_growth(Some("all"));
    let mut p1 = pipeline(&m1);
    for l in 0..n_layers {
        let m = moe(&p1, l);
        let want = if g.layers.contains(&l) { E0 + 1 } else { E0 };
        assert_eq!(m.experts.len(), want, "{tag}: layer {l} experts under CMF_GROWTH=all");
        assert_eq!(m.grown.len(), want - E0, "{tag}: layer {l} grown");
        if want > E0 {
            assert_eq!(m.grown[0].expert, E0);
            assert_eq!(m.grown[0].record, "herbs");
        }
    }
    set_growth(Some("off"));
    let p_off = pipeline(&m1);
    for l in 0..n_layers {
        assert_eq!(moe(&p_off, l).experts.len(), E0, "{tag}: layer {l} experts under CMF_GROWTH=off");
        assert!(moe(&p_off, l).grown.is_empty());
    }
    drop(p_off);
    set_growth(None);
    // ---- the trainer's witness: coverage on A > coverage on B ----
    let ck1 = load_checkpoint(&g.out_ckpt).unwrap();
    assert_eq!(ck1.cfg.experts, E0 + 1);
    let desc = GrowthDesc::from_checkpoint(&ck1, E0).unwrap();
    let gpu = EmbryoGpu::new_eval_dropless(ck1.cfg.clone(), BATCH, SEQ, &ck1.params).unwrap();
    gpu.set_desc(&ck1.extras);
    gpu.desc_updates.set(false);
    let tr_a = trace_routes(&gpu, shard_a_held, &g.layers, &desc).unwrap();
    let tr_b = trace_routes(&gpu, shard_b, &g.layers, &desc).unwrap();
    let cov_a = coverage(&tr_a, &g.shells);
    let cov_b = coverage(&tr_b, &g.shells);
    eprintln!(
        "{tag}: coverage A (held) {:.4} on {} tokens (per layer {:?}) vs B {:.4} on {} tokens (per layer {:?})",
        cov_a.overall, cov_a.tokens, cov_a.per_layer, cov_b.overall, cov_b.tokens, cov_b.per_layer
    );
    assert!(
        cov_a.overall > cov_b.overall,
        "{tag}: coverage on corpus A ({}) must exceed coverage on corpus B ({})",
        cov_a.overall,
        cov_b.overall
    );
    drop(gpu);
    // ---- G2: no-hit corpus-B prompts are bit-identical to F0 (shell on) ----
    set_growth_shell(None);
    let mut p0 = pipeline(&m0);
    for l in 0..n_layers {
        assert_eq!(moe(&p0, l).experts.len(), E0);
    }
    let (mut no_hit, mut hit_b, mut hit_a) = (0usize, 0usize, 0usize);
    for ids in prompts_b {
        let (l1, wins) = walk(&mut p1, ids);
        if wins > 0 {
            hit_b += 1;
            continue;
        }
        no_hit += 1;
        let (l0, w0) = walk(&mut p0, ids);
        assert_eq!(w0, 0);
        assert_eq!(l0.len(), GREEDY + 1);
        assert!(
            bits_equal(&l0, &l1),
            "{tag}: a corpus-B prompt with zero grown wins gave logits that differ from F0's"
        );
    }
    for ids in prompts_a {
        let (_, wins) = walk(&mut p1, ids);
        hit_a += usize::from(wins > 0);
    }
    eprintln!(
        "{tag}: in-process per-op walk (prompt + {GREEDY} greedy): B prompts {}/{} hit, {no_hit} no-hit prompts \
         bit-identical to F0; A prompts {hit_a}/{} hit",
        hit_b,
        prompts_b.len(),
        prompts_a.len()
    );
    assert!(no_hit > 0, "{tag}: every corpus-B prompt hit a grown expert — G2 has no witness");
    drop((p0, p1));
    // ---- growth-eval CLI: shell on and off, hit rate B < A ----
    let out = Command::new(cortiq)
        .args([
            "growth-eval",
            s(&g.out),
            "--prompts-jsonl",
            s(prompts_jsonl),
            "--max-tokens",
            "16",
            "--json",
            "--growth",
            "all",
            "--shell",
            "both",
        ])
        .env("CMF_GPU", "0")
        .env("RUST_LOG", "warn")
        .env_remove("CMF_EMBRYO_RESIDENT")
        .env_remove("CMF_GROWTH")
        .env_remove("CMF_GROWTH_SHELL")
        .output()
        .expect("spawn cortiq");
    let so = String::from_utf8_lossy(&out.stdout).into_owned();
    let se = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(out.status.success(), "{tag}: growth-eval failed\nstdout:\n{so}\nstderr:\n{se}");
    let j: serde_json::Value =
        serde_json::from_str(&so).unwrap_or_else(|e| panic!("{tag}: growth-eval output is not JSON ({e}):\n{so}"));
    assert_eq!(j["vacuous"], false, "{tag}: {j}");
    assert_eq!(j["path"], "per-op");
    assert_eq!(j["growth_mode"], "all");
    assert_eq!(j["mounted"].as_array().unwrap().len(), 1);
    let modes = j["modes"].as_array().unwrap();
    assert_eq!(modes.len(), 2);
    for m in modes {
        let shell = m["shell"].as_str().unwrap();
        assert_eq!(m["n"], (A_PLANTS + COUNTRIES.len()) as u64, "{tag}: growth-eval prompts");
        for l in 0..n_layers {
            let pl = &m["per_layer"][l.to_string()];
            assert_eq!(pl["trunk_experts"], E0 as u64, "{tag}: shell {shell} layer {l}");
            assert_eq!(
                pl["grown_experts"],
                u64::from(g.layers.contains(&l)),
                "{tag}: shell {shell} layer {l} grown_experts"
            );
        }
        let rate_a = m["per_src"]["herbs"]["rate"].as_f64().unwrap();
        let rate_b = m["per_src"]["general"]["rate"].as_f64().unwrap();
        eprintln!(
            "{tag}: growth-eval shell {shell}: hit rate A (herbs) {rate_a:.4} ({}/{}), B (general) {rate_b:.4} ({}/{}), \
             overall {:.4} CP95 {:.4}, per_record {}",
            m["per_src"]["herbs"]["hits"],
            m["per_src"]["herbs"]["n"],
            m["per_src"]["general"]["hits"],
            m["per_src"]["general"]["n"],
            m["hit_rate"].as_f64().unwrap(),
            m["hit_upper95"].as_f64().unwrap(),
            m["per_record"]
        );
        assert!(
            rate_b < rate_a,
            "{tag}: growth-eval (shell {shell}): hit rate on B prompts {rate_b} must be below A's {rate_a}"
        );
    }
}

// ───────────────────────── the test ─────────────────────────

#[test]
fn growth_record_end_to_end_on_a_fam_a_genome() {
    if cortiq_embryo::metal::ctx().is_none() {
        eprintln!("no GPU device: skipped");
        return;
    }
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: before any pipeline of this process exists; the CPU pipeline
    // is the runtime reference.
    unsafe {
        std::env::set_var("CMF_GPU", "0");
        std::env::remove_var("CMF_EMBRYO_RESIDENT");
        std::env::remove_var("CMF_GROWTH_SHELL");
    }
    set_growth(None);
    set_growth_shell(None);
    let cortiq = cortiq_bin();
    let cfg = fam_a_tiny();
    let n_layers = cfg.layers;
    let dir = scratch("fam_a");
    let tok_json = tokenizer_json(cfg.vocab);
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, &tok_json).unwrap();
    let bpe = Bpe::load(&tok_path).unwrap();
    // corpora: B = the genome's world, A = the knowledge to grow
    let shard_b = encode(&bpe, &corpus_b_text(8));
    let shard_a = encode(&bpe, &corpus_a_text(30));
    let shard_a_held = encode(&bpe, &corpus_a_text(8).replace("lene", "forte"));
    eprintln!(
        "corpus B {} tokens, corpus A {} tokens, A held-out {} tokens",
        shard_b.tokens.len(),
        shard_a.tokens.len(),
        shard_a_held.tokens.len()
    );
    // the two distributions are clearly different: their token sets barely overlap
    {
        let set = |s: &Shard| s.tokens.iter().copied().collect::<std::collections::BTreeSet<u16>>();
        let (sa, sb) = (set(&shard_a), set(&shard_b));
        let inter = sa.intersection(&sb).count();
        eprintln!("token sets: A {} B {} shared {inter}", sa.len(), sb.len());
        assert!(inter * 2 < sa.len().min(sb.len()), "corpus A and B share most of their tokens");
    }
    // ---- the genome: trained on B, exported sealed with a genome block ----
    let ck = genome_ck(&cfg, &shard_b, 40);
    let ckpt = dir.join("genome.ckpt");
    let ex: Vec<(&str, &[f32])> = ck.extras.iter().map(|(n, x)| (n.as_str(), x.as_slice())).collect();
    save_checkpoint(&ckpt, &ck.cfg, ck.step, &ck.params, None, None, &ex).unwrap();
    let base = dir.join("F0.cmf");
    cortiq_embryo::export::export_genome(
        &ck,
        tok_json.as_bytes(),
        &base,
        cortiq_core::TensorDtype::F32,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny".into(),
            status: "sealed".into(),
        }),
    )
    .unwrap();
    let f0_bytes = std::fs::read(&base).unwrap();
    {
        let m0 = CmfModel::open(&base).unwrap();
        let gi = m0.header.genome.as_ref().expect("genome block");
        assert_eq!(gi.status, "sealed");
        assert_eq!(gi.moe_experts, Some(E0));
        assert_eq!(cortiq_core::knowledge::genome_moe_experts(&m0.header).unwrap(), E0);
        assert_ne!(m0.required_features & features::GENOME, 0);
        assert_eq!(m0.required_features & features::SKILLS_V2, 0);
        assert!(m0.header.skills.is_empty());
        assert_eq!(cortiq_core::knowledge::moe_layers(&m0.tensors), (0..n_layers).collect::<Vec<_>>());
    }
    // ---- growth inputs: corpus A (train + held-out files), B as --general ----
    let corpus_a = dir.join("A.txt");
    std::fs::write(&corpus_a, corpus_a_text(30)).unwrap();
    let held_a = dir.join("A-held.txt");
    std::fs::write(&held_a, corpus_a_text(8).replace("lene", "forte")).unwrap();
    let general_b = dir.join("B.u16");
    shard_b.save(&general_b).unwrap();
    // ---- grow K = 1: once over all layers, once over the top half ----
    let mut grown = Vec::new();
    for (tag, layers) in [("all-layers", None), ("top-half", Some(vec![2usize, 3]))] {
        let out = dir.join(format!("F1-{tag}.cmf"));
        let out_ckpt = dir.join(format!("grown-{tag}.ckpt"));
        let a = GrowCli {
            ckpt: ckpt.clone(),
            tokenizer: tok_path.clone(),
            corpus: vec![corpus_a.clone()],
            held: vec![held_a.clone()],
            general: Some(general_b.clone()),
            trace_tokens: 0,
            trace_docs: 0,
            experts: 1,
            layers: layers.clone(),
            shell_mode: ShellMode::WonQuantile,
            shell_quantile: 0.95,
            shell_target_shift: 0.005,
            bias_mode: BiasMode::Zero,
            source_mode: SourceMode::Hottest,
            novel_quantile: 0.995,
            desc_mode: None,
            record_out: Some(out.clone()),
            base: Some(base.clone()),
            id: Some("herbs".into()),
            out_ckpt: Some(out_ckpt.clone()),
            export: None,
            steps: 16,
            lr: 1e-3,
            batch: BATCH,
            seq: SEQ,
            gate: -1.0,
            held_batches: 4,
            noise: 1e-3,
            shift: 0.1,
            seed: 7,
        };
        let sj = grow(&a).unwrap();
        eprintln!("grow {tag}: {}", serde_json::to_string(&sj).unwrap());
        assert_eq!(std::fs::read(&base).unwrap(), f0_bytes, "{tag}: F0 rewritten by grow");
        let want_layers: Vec<usize> = layers.clone().unwrap_or_else(|| (0..n_layers).collect());
        assert_eq!(sj["layers"], serde_json::json!(want_layers));
        assert_eq!(sj["K"], 1);
        assert_eq!(sj["record"]["record_index"], 0);
        for &l in &want_layers {
            assert_eq!(sj["record"]["experts"][l.to_string()], serde_json::json!([E0]), "{tag}: layer {l}");
        }
        let shells: Vec<Vec<f32>> = serde_json::from_value(sj["shells"].clone()).unwrap();
        assert_eq!(shells.len(), want_layers.len());
        assert!(shells.iter().all(|v| v.len() == 1 && v[0].is_finite()));
        let cov = sj["coverage"].as_f64().unwrap();
        let rs_no = sj["routing_shift_noshell"].as_f64().unwrap();
        let rs_sh = sj["routing_shift_shell"].as_f64().unwrap();
        eprintln!(
            "grow {tag}: held-out genome {:.4} → after {:.4}; shells {shells:?}; coverage (A held, framed) {cov:.4}; \
             routing shift on B: no shell {rs_no:.4} / shell {rs_sh:.4}",
            sj["held_genome"].as_f64().unwrap(),
            sj["held_after"].as_f64().unwrap()
        );
        assert!(rs_sh <= rs_no);
        grown.push(Grown {
            tag,
            layers: want_layers,
            out,
            out_ckpt,
            shells,
        });
    }
    // ---- prompts: token windows for the in-process walk, text for growth-eval ----
    let prompts_b = windows(&shard_b, 10, 24);
    let prompts_a = windows(&shard_a_held, 10, 24);
    let prompts_jsonl = dir.join("prompts.jsonl");
    {
        let mut text = String::new();
        for (k, p) in PLANTS.iter().enumerate().take(A_PLANTS) {
            let fam = FAMILIES[k % FAMILIES.len()];
            let q = format!("{p} familia {fam} herba perennis ~ {p} folia sicca");
            text += &(serde_json::json!({"prompt": q, "lang": "en", "src": "herbs"}).to_string() + "\n");
        }
        for (q, _) in b_questions().into_iter().take(COUNTRIES.len()) {
            text += &(serde_json::json!({"prompt": q, "lang": "en", "src": "general"}).to_string() + "\n");
        }
        std::fs::write(&prompts_jsonl, text).unwrap();
    }
    for g in &grown {
        check_grown(
            g,
            &base,
            &f0_bytes,
            n_layers,
            &prompts_b,
            &prompts_a,
            &shard_a_held,
            &shard_b,
            &cortiq,
            &prompts_jsonl,
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── source-mode novel ─────────────────────────

/// The `skill.{id}.*` tensors of a record, by name.
fn record_tensors(path: &Path, id: &str) -> std::collections::BTreeMap<String, Vec<f32>> {
    let m = CmfModel::open(path).unwrap();
    let prefix = format!("skill.{id}.");
    let mut out = std::collections::BTreeMap::new();
    for t in m.tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
        let b = m.tensor_bytes(&t.name).unwrap();
        let v: Vec<f32> = b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.insert(t.name.clone(), v);
    }
    out
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// The numbers of a JSON array as f32 (the header's JSON round-trips f32
/// values, not their f64 rendering).
fn f32s(v: &serde_json::Value) -> Vec<f32> {
    f64s(v).into_iter().map(|x| x as f32).collect()
}

/// The grown `desc.mu` / `desc.u` of every layer of a grown checkpoint,
/// concatenated (`E0 + k`, k in 0..K).
fn grown_desc(ck: &Checkpoint) -> Vec<f32> {
    let (l_n, e1, h) = (ck.cfg.layers, ck.cfg.experts, ck.cfg.hidden);
    let k = cortiq_embryo::model::MOE_K;
    let ex = |name: &str| &ck.extras.iter().find(|(n, _)| n == name).unwrap().1;
    let (mu, u) = (ex("desc.mu"), ex("desc.u"));
    let mut out = Vec::new();
    for l in 0..l_n {
        for e in E0..e1 {
            out.extend_from_slice(&mu[(l * e1 + e) * h..(l * e1 + e + 1) * h]);
            out.extend_from_slice(&u[(l * e1 + e) * k * h..(l * e1 + e + 1) * k * h]);
        }
    }
    out
}

fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
fn novel_source_mode_grows_a_novel_expert_not_a_general_one() {
    if cortiq_embryo::metal::ctx().is_none() {
        eprintln!("no GPU device: skipped");
        return;
    }
    let _env = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: before any pipeline of this process exists (the lock is
    // held); the CPU pipeline is the runtime reference.
    unsafe {
        std::env::set_var("CMF_GPU", "0");
        std::env::remove_var("CMF_EMBRYO_RESIDENT");
        std::env::remove_var("CMF_GROWTH_SHELL");
    }
    set_growth(None);
    set_growth_shell(None);
    let cfg = fam_a_tiny();
    let n_layers = cfg.layers;
    let dir = scratch("novel");
    let tok_json = tokenizer_json(cfg.vocab);
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, &tok_json).unwrap();
    let bpe = Bpe::load(&tok_path).unwrap();
    let shard_b = encode(&bpe, &corpus_b_text(8));
    let shard_a_held = encode(&bpe, &corpus_a_text(8).replace("lene", "forte"));
    let ck = genome_ck(&cfg, &shard_b, 40);
    let ckpt = dir.join("genome.ckpt");
    let ex: Vec<(&str, &[f32])> = ck.extras.iter().map(|(n, x)| (n.as_str(), x.as_slice())).collect();
    save_checkpoint(&ckpt, &ck.cfg, ck.step, &ck.params, None, None, &ex).unwrap();
    let base = dir.join("F0.cmf");
    cortiq_embryo::export::export_genome(
        &ck,
        tok_json.as_bytes(),
        &base,
        cortiq_core::TensorDtype::F32,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny".into(),
            status: "sealed".into(),
        }),
    )
    .unwrap();
    let f0_bytes = std::fs::read(&base).unwrap();
    // the growth corpus M: mostly the genome's own world (B's sentences,
    // frame stripped) with a Latin herbal minority — the real corpus' shape
    let mixed = (0..24)
        .map(|_| format!("{}{}\n", corpus_a_text(2), strip_specials(&corpus_b_text(1))))
        .collect::<String>();
    let corpus_m = dir.join("M.txt");
    std::fs::write(&corpus_m, &mixed).unwrap();
    let held_a = dir.join("A-held.txt");
    std::fs::write(&held_a, corpus_a_text(8).replace("lene", "forte")).unwrap();
    let general_b = dir.join("B.u16");
    shard_b.save(&general_b).unwrap();
    // the grown layers: the top half, where the Latin tokens are novel for
    // this young trunk (its descriptors are loose in the first layers: the
    // Latin tokens there sit inside the wide tail of B's own errors)
    let grown_layers: Vec<usize> = vec![2, 3];
    const NOVEL_Q: f32 = 0.995;
    let cli = |tag: &str, source_mode: SourceMode, desc_mode: Option<DescMode>, steps: usize| GrowCli {
        ckpt: ckpt.clone(),
        tokenizer: tok_path.clone(),
        corpus: vec![corpus_m.clone()],
        held: vec![held_a.clone()],
        general: Some(general_b.clone()),
        trace_tokens: 0,
        trace_docs: 0,
        experts: 1,
        layers: Some(grown_layers.clone()),
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.95,
        shell_target_shift: 0.005,
        bias_mode: BiasMode::Zero,
        source_mode,
        novel_quantile: NOVEL_Q,
        desc_mode,
        record_out: Some(dir.join(format!("F1-{tag}.cmf"))),
        base: Some(base.clone()),
        id: Some("herbs".into()),
        out_ckpt: Some(dir.join(format!("grown-{tag}.ckpt"))),
        export: None,
        steps,
        lr: 1e-3,
        batch: BATCH,
        seq: SEQ,
        gate: -1.0,
        held_batches: 4,
        noise: 1e-3,
        shift: 0.1,
        seed: 7,
    };
    // ---- the two arms: hottest (adapt) and novel (frozen), the defaults ----
    let a_hot = cli("hot", SourceMode::Hottest, None, 16);
    let s_hot = grow(&a_hot).unwrap();
    eprintln!("grow hottest: {}", serde_json::to_string(&s_hot).unwrap());
    let a_nov = cli("novel", SourceMode::Novel, None, 16);
    let s_nov = grow(&a_nov).unwrap();
    eprintln!("grow novel: {}", serde_json::to_string(&s_nov).unwrap());
    assert_eq!(std::fs::read(&base).unwrap(), f0_bytes, "F0 rewritten by grow");
    assert_eq!(s_hot["source_mode"], "hottest");
    assert_eq!(s_hot["desc_mode"], "adapt");
    assert_eq!(s_nov["source_mode"], "novel");
    assert_eq!(s_nov["desc_mode"], "frozen");
    assert_eq!(s_nov["novel_quantile"].as_f64().unwrap() as f32, NOVEL_Q);
    assert_eq!(s_nov["layers"], serde_json::json!(grown_layers));
    let n_grown = grown_layers.len();
    // the novelty witness: τ per layer from B, the novel share of M, the
    // same numbers in the hottest arm (both see --general)
    let tau = f64s(&s_nov["novel_tau"]);
    assert_eq!(tau.len(), n_grown);
    assert!(tau.iter().all(|t| t.is_finite() && *t > 0.0), "{tau:?}");
    assert_eq!(s_hot["novel_tau"], s_nov["novel_tau"]);
    assert_eq!(s_hot["novel_share_corpus"], s_nov["novel_share_corpus"]);
    let share = f64s(&s_nov["novel_share_corpus"]);
    let novel_tokens: Vec<u64> = s_nov["novel_tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
    eprintln!("novelty: τ {tau:?}; novel share of M {share:?} ({novel_tokens:?} tokens)");
    assert!(novel_tokens.iter().all(|&n| n as usize >= NOVEL_MIN_PER_EXPERT), "{novel_tokens:?}");
    let nw = &s_nov["novel_witness"];
    assert_eq!(nw["cluster_rows"].as_array().unwrap().len(), n_grown);
    assert!(s_hot["novel_witness"]["cluster_rows"].is_null());
    assert_eq!(nw["general_tokens"].as_u64().unwrap() as usize, shard_b.tokens.len() / SEQ * SEQ);
    // the sources: a trunk expert per layer, and the hottest arm's are the
    // corpus-wins order (unchanged behaviour)
    for li in 0..n_grown {
        assert!(s_nov["sources"][li][0].as_u64().unwrap() < E0 as u64);
        let wins: Vec<u64> = s_hot["trunk_corpus_wins"][li].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        let src = s_hot["sources"][li][0].as_u64().unwrap() as usize;
        assert_eq!(wins[src], *wins.iter().max().unwrap(), "layer {li}: the hottest source wins the most corpus tokens");
    }
    // ---- THE claim: the novel expert captures far less of B (no shell) ----
    let rs_hot = s_hot["routing_shift_noshell"].as_f64().unwrap();
    let rs_nov = s_nov["routing_shift_noshell"].as_f64().unwrap();
    let per_hot = f64s(&s_hot["routing_shift"]["per_layer_noshell"]);
    let per_nov = f64s(&s_nov["routing_shift"]["per_layer_noshell"]);
    let nc_nov = s_nov["novel_coverage"].as_f64().unwrap();
    let nc_nov_no = s_nov["novel_coverage_noshell"].as_f64().unwrap();
    let nc_hot = s_hot["novel_coverage"].as_f64().unwrap();
    eprintln!(
        "routing shift on B (no shell): hottest {rs_hot:.4} (per layer {per_hot:?}) vs novel {rs_nov:.4} (per layer \
         {per_nov:?}); novel coverage of M (shell on): novel {nc_nov:.4} (no shell {nc_nov_no:.4}) vs hottest {nc_hot:.4}; \
         held-out after: hottest {:.4}, novel {:.4} (genome {:.4})",
        s_hot["held_after"].as_f64().unwrap(),
        s_nov["held_after"].as_f64().unwrap(),
        s_nov["held_genome"].as_f64().unwrap()
    );
    assert!(rs_nov < rs_hot, "novel {rs_nov} must route less of B than hottest {rs_hot}");
    assert!(rs_nov <= 0.5 * rs_hot, "novel {rs_nov} is not far below hottest {rs_hot}");
    // (per layer the picture varies on a tiny genome: the claim is the
    // overall share of B a grown expert takes)
    assert!(per_nov.iter().zip(&per_hot).any(|(n, h)| n < h), "per layer: novel {per_nov:?} vs hottest {per_hot:?}");
    assert!(nc_nov >= 0.5, "novel coverage of the novel A tokens {nc_nov} (shell on) is low");
    assert!(nc_nov <= nc_nov_no + 1e-6);
    let per_nc = f64s(&s_nov["novel_coverage_detail"]["per_layer_shell"]);
    assert_eq!(per_nc.len(), n_grown);
    // ---- frozen: the grown descriptor is exactly the initialisation ----
    let ck_nov = load_checkpoint(&dir.join("grown-novel.ckpt")).unwrap();
    let a_n0 = cli("novel0", SourceMode::Novel, None, 0);
    let s_n0 = grow(&a_n0).unwrap();
    assert_eq!(s_n0["sources"], s_nov["sources"]);
    let ck_n0 = load_checkpoint(&dir.join("grown-novel0.ckpt")).unwrap();
    assert!(bits_eq(&grown_desc(&ck_nov), &grown_desc(&ck_n0)), "frozen: the trained grown μ / U moved");
    assert_ne!(ck_nov.params, ck_n0.params, "the novel expert's weights trained");
    let a_na = cli("novel-adapt", SourceMode::Novel, Some(DescMode::Adapt), 16);
    let s_na = grow(&a_na).unwrap();
    assert_eq!(s_na["desc_mode"], "adapt");
    assert_eq!(s_na["sources"], s_nov["sources"]);
    let ck_na = load_checkpoint(&dir.join("grown-novel-adapt.ckpt")).unwrap();
    assert!(!bits_eq(&grown_desc(&ck_na), &grown_desc(&ck_n0)), "adapt: the grown μ did not move");
    // the record carries the frozen descriptor and the bias 0
    let rec_nov = record_tensors(&dir.join("F1-novel.cmf"), "herbs");
    let (e1, h) = (ck_nov.cfg.experts, ck_nov.cfg.hidden);
    for &l in &grown_layers {
        let pre = format!("skill.herbs.model.layers.{l}.mlp.experts.{E0}.");
        let mu0 = &ck_n0.extras.iter().find(|(n, _)| n == "desc.mu").unwrap().1;
        assert!(bits_eq(&rec_nov[&format!("{pre}desc.mu")], &mu0[(l * e1 + E0) * h..(l * e1 + E0 + 1) * h]));
        assert_eq!(rec_nov[&format!("{pre}desc.bias")], vec![0.0]);
        assert!(rec_nov[&format!("{pre}desc.shell")][0].is_finite());
    }
    // ---- reshell reproduces the novel record; the desc mode is inferred ----
    let rcli = |ckpt: &Path, tag: &str, source_mode: SourceMode, desc_mode: Option<DescMode>, general: bool| ReshellCli {
        ckpt: ckpt.to_path_buf(),
        genome_ckpt: None,
        tokenizer: tok_path.clone(),
        corpus: vec![corpus_m.clone()],
        held: vec![held_a.clone()],
        general: general.then(|| general_b.clone()),
        trace_tokens: 0,
        trace_docs: 0,
        layers: None,
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.95,
        shell_target_shift: 0.005,
        bias_mode: None,
        source_mode,
        novel_quantile: NOVEL_Q,
        desc_mode,
        seed: 7,
        record_out: Some(dir.join(format!("F1-{tag}.cmf"))),
        base: base.clone(),
        id: Some("herbs".into()),
        batch: BATCH,
        seq: SEQ,
    };
    let s_re = reshell(&rcli(&dir.join("grown-novel.ckpt"), "novel-reshell", SourceMode::Novel, None, true)).unwrap();
    eprintln!("reshell novel: {}", serde_json::to_string(&s_re).unwrap());
    assert_eq!(s_re["reshell"], true);
    assert_eq!(s_re["source_mode"], "novel");
    assert_eq!(s_re["desc_mode"], "frozen", "inferred from the checkpoint");
    assert_eq!(s_re["sources"], s_nov["sources"]);
    assert_eq!(s_re["shells"], s_nov["shells"]);
    assert_eq!(s_re["wins"], s_nov["wins"]);
    assert_eq!(s_re["novel_tau"], s_nov["novel_tau"]);
    assert_eq!(s_re["novel_share_corpus"], s_nov["novel_share_corpus"]);
    assert_eq!(s_re["novel_coverage"], s_nov["novel_coverage"]);
    assert_eq!(s_re["routing_shift"], s_nov["routing_shift"]);
    assert_eq!(s_re["coverage"], s_nov["coverage"]);
    let rec_re = record_tensors(&dir.join("F1-novel-reshell.cmf"), "herbs");
    assert_eq!(rec_re.keys().collect::<Vec<_>>(), rec_nov.keys().collect::<Vec<_>>());
    for (name, v) in &rec_nov {
        let w = &rec_re[name];
        if name.ends_with(".desc.shell") {
            assert!((v[0] - w[0]).abs() <= 1e-6, "{name}: {} vs {}", v[0], w[0]);
        } else {
            assert!(bits_eq(v, w), "{name}: reshell's record tensor differs from grow's");
        }
    }
    let m_re = CmfModel::open(&dir.join("F1-novel-reshell.cmf")).unwrap();
    let origin = m_re.header.skills[0].origin.clone().unwrap();
    assert_eq!(origin["source_mode"], "novel");
    assert_eq!(origin["desc_mode"], "frozen");
    assert_eq!(origin["novel_quantile"].as_f64().unwrap() as f32, NOVEL_Q);
    assert_eq!(f32s(&origin["novel_tau"]), f32s(&s_nov["novel_tau"]));
    assert_eq!(f32s(&origin["novel_share_corpus"]), f32s(&s_nov["novel_share_corpus"]));
    assert_eq!(origin["novel_coverage"].as_f64().unwrap() as f32, s_nov["novel_coverage"].as_f64().unwrap() as f32);
    assert_eq!(origin["seed"], 7);
    let q = m_re.header.skills[0].quality.clone().unwrap();
    assert_eq!(q["source_mode"], "novel");
    assert_eq!(q["desc_mode"], "frozen");
    assert_eq!(q["novel_coverage"].as_f64().unwrap() as f32, s_nov["novel_coverage"].as_f64().unwrap() as f32);
    drop(m_re);
    // the adapted checkpoint: inferred adapt; `frozen` given → refused
    let s_ra = reshell(&rcli(&dir.join("grown-novel-adapt.ckpt"), "novel-adapt-reshell", SourceMode::Novel, None, true)).unwrap();
    assert_eq!(s_ra["desc_mode"], "adapt");
    let e = reshell(&rcli(&dir.join("grown-novel-adapt.ckpt"), "bad1", SourceMode::Novel, Some(DescMode::Frozen), true))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("--desc-mode frozen"), "{e}");
    // ---- refusals: novel without --general (grow and reshell); too few novel tokens ----
    let mut bad = cli("bad2", SourceMode::Novel, None, 16);
    bad.general = None;
    let e = grow(&bad).err().unwrap().to_string();
    assert!(e.contains("--general"), "{e}");
    let e = reshell(&rcli(&dir.join("grown-novel.ckpt"), "bad3", SourceMode::Novel, None, false)).err().unwrap().to_string();
    assert!(e.contains("--general"), "{e}");
    let corpus_bb = dir.join("B-as-corpus.txt");
    std::fs::write(&corpus_bb, strip_specials(&corpus_b_text(8))).unwrap();
    let mut bad = cli("bad4", SourceMode::Novel, None, 16);
    bad.corpus = vec![corpus_bb];
    bad.layers = None;
    bad.novel_quantile = 0.999;
    let e = grow(&bad).err().unwrap().to_string();
    assert!(e.contains("novel for the trunk"), "{e}");
    for t in ["bad1", "bad2", "bad3", "bad4"] {
        assert!(!dir.join(format!("F1-{t}.cmf")).exists(), "{t}: a refusal left a file behind");
    }
    // ---- K = 2: two novel clusters per grown layer (K-means) ----
    let mut a_k2 = cli("novel-k2", SourceMode::Novel, None, 16);
    a_k2.experts = 2;
    let s_k2 = grow(&a_k2).unwrap();
    eprintln!("grow novel K=2: {}", serde_json::to_string(&s_k2).unwrap());
    assert_eq!(s_k2["K"], 2);
    assert_eq!(s_k2["layers"], serde_json::json!([2, 3]));
    let cr = s_k2["novel_witness"]["cluster_rows"].as_array().unwrap();
    assert_eq!(cr.len(), 2);
    for (li, l) in [2usize, 3].iter().enumerate() {
        let rows: Vec<u64> = cr[li].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|&r| r >= 1), "layer {l}: cluster rows {rows:?}");
        assert_eq!(rows.iter().sum::<u64>(), s_k2["novel_witness"]["reservoir_rows"][li].as_u64().unwrap());
        assert_eq!(s_k2["sources"][li].as_array().unwrap().len(), 2);
        assert_eq!(s_k2["record"]["experts"][l.to_string()], serde_json::json!([E0, E0 + 1]));
    }
    let rec_k2 = record_tensors(&dir.join("F1-novel-k2.cmf"), "herbs");
    for l in [2usize, 3] {
        let a = &rec_k2[&format!("skill.herbs.model.layers.{l}.mlp.experts.{E0}.desc.mu")];
        let b = &rec_k2[&format!("skill.herbs.model.layers.{l}.mlp.experts.{}.desc.mu", E0 + 1)];
        assert!(!bits_eq(a, b), "layer {l}: the two novel clusters have the same mean");
    }
    // ---- the novel record validates and mounts; A prompts hit it at runtime ----
    let out_nov = dir.join("F1-novel.cmf");
    let f1_bytes = std::fs::read(&out_nov).unwrap();
    assert!(f1_bytes[128..f0_bytes.len()] == f0_bytes[128..], "F1 is not a tail append of F0");
    cortiq_embryo::skill::verify_append(&base, &out_nov).unwrap();
    let m0 = Arc::new(CmfModel::open(&base).unwrap());
    let m1 = Arc::new(CmfModel::open(&out_nov).unwrap());
    assert_eq!(m0.trunk_hash(), m1.trunk_hash());
    assert_ne!(m1.required_features & features::SKILLS_V2, 0);
    assert_eq!(m1.header.skills[0].kind.as_deref(), Some(skill_kind::EXPERT_APPEND));
    assert_eq!(m1.header.skills[0].status.as_deref(), Some("quarantine"));
    set_growth(Some("all"));
    let mut p1 = pipeline(&m1);
    for l in 0..n_layers {
        let m = moe(&p1, l);
        let want = if grown_layers.contains(&l) { E0 + 1 } else { E0 };
        assert_eq!(m.experts.len(), want, "layer {l} experts under CMF_GROWTH=all");
        assert_eq!(m.grown.len(), want - E0);
        if want > E0 {
            assert_eq!(m.grown[0].record, "herbs");
        }
    }
    set_growth(Some("off"));
    let p_off = pipeline(&m1);
    for l in 0..n_layers {
        assert_eq!(moe(&p_off, l).experts.len(), E0, "layer {l} experts under CMF_GROWTH=off");
    }
    drop(p_off);
    set_growth(Some("all"));
    let m_hot = Arc::new(CmfModel::open(&dir.join("F1-hot.cmf")).unwrap());
    let mut p_hot = pipeline(&m_hot);
    set_growth(None);
    let prompts_a = windows(&shard_a_held, 10, 24);
    let prompts_b = windows(&shard_b, 10, 24);
    let tally = |p: &mut Pipeline, prompts: &[Vec<u32>]| -> (usize, u64) {
        let mut hit = 0usize;
        let mut wins = 0u64;
        for ids in prompts {
            let (_, w) = walk(p, ids);
            hit += usize::from(w > 0);
            wins += w;
        }
        (hit, wins)
    };
    let (hit_a, wins_a) = tally(&mut p1, &prompts_a);
    let (hit_b, wins_b) = tally(&mut p1, &prompts_b);
    let (hit_a_hot, wins_a_hot) = tally(&mut p_hot, &prompts_a);
    let (hit_b_hot, wins_b_hot) = tally(&mut p_hot, &prompts_b);
    eprintln!(
        "runtime walk (shell on, prompt + {GREEDY} greedy): novel record — A prompts {hit_a}/{} hit ({wins_a} grown \
         wins), B prompts {hit_b}/{} hit ({wins_b}); hottest record — A {hit_a_hot}/{} ({wins_a_hot}), B {hit_b_hot}/{} \
         ({wins_b_hot})",
        prompts_a.len(),
        prompts_b.len(),
        prompts_a.len(),
        prompts_b.len()
    );
    assert!(hit_a > 0 && wins_a > 0, "no corpus-A prompt reaches the novel expert at runtime");
    assert!(wins_b < wins_b_hot, "the novel record wins {wins_b} B tokens, the hottest record {wins_b_hot}");
    drop((p1, p_hot));
    let _ = std::fs::remove_dir_all(&dir);
}
