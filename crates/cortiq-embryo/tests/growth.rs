//! Growth as records (SPEC_GROWTH_RECORDS §3 + addendum), on a tiny fam-a
//! genome (GDN mixer, bounded anchors, 4 routed experts):
//!  - the host route probe is EXACTLY the runtime's `Resonance::scores`
//!    (bit-for-bit, with and without shells) and its top-1 tie rule;
//!  - K = 2 experts grown in a layer SUBSET from the K hottest trunk
//!    experts: old tensors / descriptors byte-identical, undeclared layers
//!    carry inert slots, the grown bias stays pinned at 0 through training
//!    (Metal / WGSL `bias_frozen_from`), only the grown layers train;
//!  - shells by quantile, coverage and routing shift from the witness;
//!  - the `expert_append` record over a COPY of F0: trunk hash and prefix
//!    bytes equal, the format's layout plan satisfied (E0+k, then E0+2+k
//!    for a second record — the per-layer chain), status quarantine;
//!  - `CMF_GROWTH=off` mounts nothing: F1's logits == F0's bit-for-bit;
//!    `CMF_GROWTH=all` mounts the record and runs (shell on and off);
//!  - the writer's refusals leave no output behind;
//!  - the CLI driver (`cli::grow`) end to end with --held / --general /
//!    --record-out and its one JSON summary;
//!  - the legacy full-genome growth (sleep daemon) still works;
//!  - the descriptor of a grown expert differs from its source's on a
//!    genome with a real U (corpus init; legacy shift outside span(U));
//!  - an f16 genome grows on the served (rounded) trunk;
//!  - held-out: consecutive windows, the genome's loss as the gate
//!    reference, the dropless loss of the trained checkpoint;
//!  - `--export` / `export` never truncate a genome file; `E0 + K ≤ 8`;
//!    no growth on a base whose records already occupy the grown layers.
//!
//! Metal on macOS, Vulkan with `--features vulkan` on Linux (skipped
//! without a device). CMF_GPU=0: the CPU pipeline is the runtime reference.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_core::format::CmfModel;
use cortiq_core::knowledge::{expert_append_layout, expert_leaf, skill_kind, state_switch};
use cortiq_embryo::cli::{GrowCli, ReshellCli, SourceMode, grow, reshell};
use cortiq_embryo::growth::{
    BiasMode, DEAD_BIAS, GrowArgs, GrowSpec, GrowTrain, GrowthDesc, MAX_RUNTIME_EXPERTS, RecordArgs,
    ShellMode, check_base_growth_records, check_layers, cluster_inits, coverage, eval_windows, frame_docs,
    grow_experts, grow_experts_init, grow_experts_k, grown_layers_of, held_windows, quantile_sorted,
    resonance_scores, resonance_winner, routing_shift, shells_from_traces, shells_general_target,
    shells_won_quantile, shrink_experts, sources_by_corpus_wins, split_docs_at_eot, trace_routes,
    trace_routes_rows, train_grown_experts, train_new_experts, write_growth_record,
};
use cortiq_embryo::model::{EmbryoCfg, EmbryoGpu, Layout, MOE_K, Mixer, gauss_vec, init_params};
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
        "embryo_growth_{tag}_{}_{}",
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
];
const FAMILIES: &[&str] = &["Asteraceae", "Lamiaceae", "Rosaceae", "Apiaceae", "Urticaceae"];

/// The growth corpus: plant facts (the knowledge to grow).
fn herbs_text(reps: usize) -> String {
    let mut s = String::new();
    for _ in 0..reps {
        for (k, p) in PLANTS.iter().enumerate() {
            let fam = FAMILIES[k % FAMILIES.len()];
            s.push_str(&format!(
                "{p} belongs to the family {fam} and is used for a mild tea. {p} is a perennial \
                 herb of {fam}; dried leaves of {p} are brewed for {} minutes.\n",
                3 + k % 7
            ));
        }
    }
    s
}

/// General text (the backbone's world: capitals, code, arithmetic).
fn general_text(reps: usize) -> String {
    let mut s = String::new();
    for _ in 0..reps {
        for (i, c) in ["France", "Japan", "Brazil", "Canada", "Egypt", "Norway", "India", "Chile"]
            .iter()
            .enumerate()
        {
            s.push_str(&format!(
                "What is the capital of {c}? Roughly {i} million people live there.\n"
            ));
        }
        for l in ["Rust", "Python", "C", "Go"] {
            s.push_str(&format!(
                "How do I sort a vector of integers in {l}? Write a {l} function that reverses a string.\n"
            ));
        }
        for n in 2..10 {
            s.push_str(&format!("Solve {n}x + 3 = {} for x. The answer is 5.\n", n * 5));
        }
    }
    s
}

fn tokenizer_json(vocab: usize) -> String {
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    let text = format!("{}{}", herbs_text(1), general_text(1));
    cortiq_embryo::tokenizer::count_words(&text, &re, &mut counts);
    cortiq_embryo::tokenizer::train(&counts, vocab, false).to_hf_json()
}

fn encode(bpe: &Bpe, text: &str) -> Shard {
    let mut ids = Vec::new();
    bpe.encode(text, &mut std::collections::HashMap::new(), &mut ids);
    Shard {
        tokens: ids.iter().map(|&x| x as u16).collect(),
    }
}

/// A genome checkpoint: seeded params, expert descriptors seeded with one
/// lr-0 step (as a birth leaves them).
fn seeded_ck(cfg: &EmbryoCfg, bpe: &Bpe) -> Checkpoint {
    let lay = Layout::new(cfg);
    let params = init_params(cfg, &lay, 11);
    let mut gpu = EmbryoGpu::new(cfg.clone(), 4, 64, &params).unwrap();
    let ids = encode(bpe, &general_text(2)).tokens;
    let tk: Vec<u32> = ids.iter().cycle().take(256).map(|&x| x as u32).collect();
    let tg: Vec<u32> = ids.iter().cycle().skip(1).take(256).map(|&x| x as u32).collect();
    let _ = gpu.train_step(&tk, &tg, 0.0, 0.0, 1e9);
    // the birth's P1 step: descriptor subspaces from the routed inputs
    // (orthonormal rows) — a genome routes on a real U, not zeros
    let mut cov = Vec::new();
    gpu.update_subspaces(&mut cov, 0.9);
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

struct Fixture {
    dir: PathBuf,
    ck: Checkpoint,
    tok_json: String,
    tok_path: PathBuf,
    ckpt: PathBuf,
    base: PathBuf,
    bpe: Bpe,
}

fn fixture(tag: &str) -> Option<Fixture> {
    cortiq_embryo::metal::ctx()?;
    // the CPU pipeline is the runtime reference in these tests
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let cfg = fam_a_tiny();
    let dir = scratch(tag);
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
    Some(Fixture {
        dir,
        ck,
        tok_json,
        tok_path,
        ckpt,
        base,
        bpe,
    })
}

fn by_name(lay: &Layout) -> std::collections::HashMap<&str, (usize, usize)> {
    lay.names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect()
}

/// Every tensor of `a` that `b`'s layout also names is byte-identical.
fn assert_old_tensors_identical(a: &Checkpoint, b: &Checkpoint, what: &str) {
    let la = Layout::new(&a.cfg);
    let lb = Layout::new(&b.cfg);
    let ba = by_name(&la);
    for (name, ob, lb_n) in &lb.names {
        if let Some((oa, la_n)) = ba.get(name.as_str()) {
            assert_eq!(la_n, lb_n, "{name}: length");
            assert!(
                a.params[*oa..*oa + la_n] == b.params[*ob..*ob + lb_n],
                "{name} changed by {what}"
            );
        }
    }
}

fn desc<'a>(ck: &'a Checkpoint, name: &str) -> &'a [f32] {
    &ck.extras.iter().find(|(n, _)| n == name).unwrap().1
}

/// The fixed inputs of a record, so refusal cases can vary one field each.
struct RecCtx<'a> {
    ck0: &'a Checkpoint,
    trained: &'a Checkpoint,
    e0: usize,
    layers: &'a [usize],
    shells: &'a [Vec<f32>],
    grown_bias: &'a [Vec<f32>],
    origin: serde_json::Value,
    quality: serde_json::Value,
}

impl<'a> RecCtx<'a> {
    fn args<'b>(&'b self, base: &'b Path, out: &'b Path, id: &'b str) -> RecordArgs<'b>
    where
        'a: 'b,
    {
        RecordArgs {
            base,
            out,
            id,
            ck0: self.ck0,
            trained: self.trained,
            e0: self.e0,
            layers: self.layers,
            shell_quantile: 0.99,
            shells: self.shells,
            bias_mode: BiasMode::Zero,
            grown_bias: self.grown_bias,
            origin: self.origin.clone(),
            quality: self.quality.clone(),
        }
    }
}

fn prefill(path: &Path, ids: &[u32]) -> Vec<f32> {
    let m = std::sync::Arc::new(CmfModel::open(path).unwrap());
    let mut p = cortiq_engine::pipeline::Pipeline::from_model(
        &m,
        cortiq_engine::sampler::SamplerConfig::default(),
    )
    .unwrap();
    p.prefill_next_logits(ids, None)
}

// ───────────────────────── the runtime formula ─────────────────────────

#[test]
fn host_probe_is_the_runtime_formula_bit_for_bit() {
    let (h, e, k) = (8usize, 5usize, 3usize);
    let mu = gauss_vec(1, e * h);
    let u = gauss_vec(2, e * k * h);
    let bias: Vec<f32> = gauss_vec(3, e).iter().map(|x| x * 0.1).collect();
    let shell = vec![f32::INFINITY, f32::INFINITY, f32::INFINITY, 4.0, 6.0];
    let mut out_r = vec![0.0f32; e];
    let mut out_h = vec![0.0f32; e];
    let mut err = vec![0.0f32; e];
    let mut shell_fired = 0;
    for s in 0..64u64 {
        let x = gauss_vec(100 + s, h);
        // no shell (legacy constructor: empty shell vector)
        let r = cortiq_engine::pipeline::Resonance {
            mu: mu.clone(),
            u: u.clone(),
            k,
            bias: bias.clone(),
            shell: Vec::new(),
        };
        r.scores(&x, &mut out_r);
        resonance_scores(&x, &mu, &u, k, &bias, None, &mut out_h, &mut err);
        assert!(
            out_r.iter().zip(&out_h).all(|(a, b)| a.to_bits() == b.to_bits()),
            "no-shell scores differ: {out_r:?} vs {out_h:?}"
        );
        for i in 0..e {
            assert_eq!((bias[i] - err[i]).to_bits(), out_r[i].to_bits());
        }
        // shell on (CMF_GROWTH_SHELL unset → on)
        let r = cortiq_engine::pipeline::Resonance {
            mu: mu.clone(),
            u: u.clone(),
            k,
            bias: bias.clone(),
            shell: shell.clone(),
        };
        r.scores(&x, &mut out_r);
        resonance_scores(&x, &mu, &u, k, &bias, Some(&shell), &mut out_h, &mut err);
        assert!(
            out_r.iter().zip(&out_h).all(|(a, b)| a.to_bits() == b.to_bits()),
            "shell scores differ: {out_r:?} vs {out_h:?}"
        );
        shell_fired += out_h.iter().filter(|v| **v == f32::NEG_INFINITY).count();
    }
    assert!(shell_fired > 0, "the fixture shells never fired");
    // top-1: the first index of the maximum (moe_route: lower index wins ties)
    assert_eq!(resonance_winner(&[1.0, 3.0, 3.0, 2.0]), 1);
    assert_eq!(resonance_winner(&[f32::NEG_INFINITY, -2.0, -1.0, -1.0]), 2);
    assert_eq!(resonance_winner(&[f32::NEG_INFINITY; 3]), 0);
    // nearest-rank quantile
    let s = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(quantile_sorted(&s, 0.99), 5.0);
    assert_eq!(quantile_sorted(&s, 0.5), 3.0);
    assert_eq!(quantile_sorted(&s, 0.0), 1.0);
    assert_eq!(quantile_sorted(&s, 1.0), 5.0);
    // record growth: sources = trunk experts hottest ON THE GROWTH CORPUS
    // (most wins; ties → the hotter balancing bias, then the lower index;
    // `k mod E0` past E0) — never the bias order alone
    {
        let bias = vec![0.0, 0.0, 0.0, 0.0, -1.0, -2.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0];
        let wins = vec![vec![10usize, 0, 30, 30], vec![5, 5, 5, 5]];
        let src = sources_by_corpus_wins(&wins, &bias, 4, &[1, 2], 5).unwrap();
        assert_eq!(src[0], vec![2, 3, 0, 1, 2], "layer 1: 30/30 tie with equal bias → index, then 10, then 0 wins, then k mod E0");
        assert_eq!(src[1], vec![0, 1, 2, 3, 0], "layer 2: all tied → index order");
        assert!(sources_by_corpus_wins(&wins, &bias, 4, &[1], 1).is_err(), "one win list per grown layer");
        assert!(sources_by_corpus_wins(&[vec![1, 2]], &bias, 4, &[1], 1).is_err(), "E0 wins per layer");
    }
    // layer checks
    assert_eq!(check_layers(4, &[3, 1]).unwrap(), vec![1, 3]);
    assert!(check_layers(4, &[]).is_err());
    assert!(check_layers(4, &[1, 1]).is_err());
    assert!(check_layers(4, &[4]).is_err());
}

// ───────────────────────── growth + record ─────────────────────────

#[test]
fn growth_record_over_fam_a_genome() {
    let Some(f) = fixture("record") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let (e0, kn) = (4usize, 2usize);
    let layers = vec![1usize, 3];
    let train = encode(&f.bpe, &herbs_text(12));
    let held = encode(&f.bpe, &herbs_text(3).replace("mild", "strong"));
    let general = encode(&f.bpe, &general_text(6));
    assert!(train.tokens.len() > 20 * 66, "train corpus {} tokens", train.tokens.len());
    // ---- surgery: K = 2 in layers 1 and 3 from the 2 hottest trunk experts ----
    let spec = GrowSpec {
        experts: kn,
        layers: layers.clone(),
        noise: 1e-3,
        shift: 0.1,
        seed: 5,
        zero_bias: true,
    };
    let (grown, sources) = grow_experts_k(&f.ck, &spec).unwrap();
    assert_eq!(grown.cfg.experts, e0 + kn);
    assert_eq!(sources.len(), 4);
    assert!(sources[0].is_empty() && sources[2].is_empty());
    for &l in &layers {
        assert_eq!(sources[l].len(), kn);
        assert_ne!(sources[l][0], sources[l][1], "two hottest trunk experts differ");
        assert!(sources[l].iter().all(|&s| s < e0), "sources are trunk experts");
        let b0 = &desc(&f.ck, "desc.bias")[l * e0..(l + 1) * e0];
        let hottest = (0..e0).min_by(|&a, &b| b0[a].partial_cmp(&b0[b]).unwrap()).unwrap();
        assert_eq!(sources[l][0], hottest, "layer {l}: the first copy is the hottest");
    }
    assert_old_tensors_identical(&f.ck, &grown, "growth");
    let e1 = grown.cfg.experts;
    let b1 = desc(&grown, "desc.bias");
    for l in 0..4 {
        for kk in 0..kn {
            let b = b1[l * e1 + e0 + kk];
            if layers.contains(&l) {
                assert_eq!(b, 0.0, "layer {l}: grown bias starts at 0");
            } else {
                assert_eq!(b, DEAD_BIAS, "layer {l}: undeclared slot is inert");
            }
        }
        // trunk descriptors copied verbatim
        let h = grown.cfg.hidden;
        assert_eq!(&desc(&grown, "desc.mu")[l * e1 * h..l * e1 * h + e0 * h], &desc(&f.ck, "desc.mu")[l * e0 * h..(l + 1) * e0 * h]);
        assert_eq!(&b1[l * e1..l * e1 + e0], &desc(&f.ck, "desc.bias")[l * e0..(l + 1) * e0]);
    }
    // ---- train only the grown experts of the grown layers, bias pinned ----
    let a = GrowArgs {
        steps: 12,
        lr: 1e-3,
        batch: 4,
        seq: 64,
        eval_every: 6,
        seed: 3,
        held_batches: 2,
    };
    let gt = GrowTrain {
        e0,
        layers: &layers,
        train: &train,
        held: &held,
        freeze_bias: true,
        freeze_desc: false,
        genome: Some(&f.ck),
    };
    let (trained, ho) = train_grown_experts(&grown, &gt, &a, &|| false).unwrap();
    let (l0, l1) = (ho.genome.unwrap(), ho.after);
    eprintln!("growth: held-out genome {l0:.4}, untrained {:.4} → {l1:.4}; sources {sources:?}", ho.untrained);
    assert!(l0.is_finite() && l1.is_finite() && ho.untrained.is_finite());
    // T4: the held-out is read in consecutive non-overlapping windows, at
    // most `held_batches` batches of `batch` rows
    assert_eq!(ho.batches, 2);
    assert_eq!(ho.windows, 8);
    let wins_held = held_windows(&held, 4, 64, 2).unwrap();
    assert_eq!(wins_held.len(), 2);
    for (bi, (tk, tg)) in wins_held.iter().enumerate() {
        for r in 0..4 {
            let w = (bi * 4 + r) % ((held.tokens.len() - 1) / 64);
            assert_eq!(tk[r * 64], held.tokens[w * 64] as u32, "window {w} starts at w·seq");
            assert_eq!(tg[r * 64 + 63], held.tokens[w * 64 + 64] as u32, "targets shifted by one");
        }
    }
    // T3: `genome` is the loss of the PRE-growth genome (E0 experts) on
    // exactly those windows — not the untrained grown model's
    {
        let g0 = EmbryoGpu::new_eval_dropless(f.ck.cfg.clone(), 4, 64, &f.ck.params).unwrap();
        g0.set_desc(&f.ck.extras);
        let again = eval_windows(&g0, &wins_held);
        assert!((again - l0).abs() <= 1e-4 * l0.abs().max(1.0), "genome loss {l0} vs recomputed {again}");
        assert_eq!(ho.improvement(), (l0 - l1) / l0.abs().max(1e-6));
    }
    // F3 / T2: the gate loss is the DROPLESS loss of the trained checkpoint
    // (the runtime's routing) — a fresh dropless instance reproduces it
    {
        let g1 = EmbryoGpu::new_eval_dropless(trained.cfg.clone(), 4, 64, &trained.params).unwrap();
        g1.set_desc(&trained.extras);
        let again = eval_windows(&g1, &wins_held);
        assert!((again - l1).abs() <= 1e-4 * l1.abs().max(1.0), "after {l1} vs dropless recomputation {again}");
    }
    assert_old_tensors_identical(&f.ck, &trained, "training the grown experts");
    let bt = desc(&trained, "desc.bias");
    let h = trained.cfg.hidden;
    for l in 0..4 {
        // trunk descriptors never move (desc_frozen_below = E0)
        assert_eq!(&bt[l * e1..l * e1 + e0], &desc(&f.ck, "desc.bias")[l * e0..(l + 1) * e0], "layer {l}: trunk bias moved");
        assert_eq!(&desc(&trained, "desc.mu")[l * e1 * h..l * e1 * h + e0 * h], &desc(&f.ck, "desc.mu")[l * e0 * h..(l + 1) * e0 * h], "layer {l}: trunk μ moved");
        for kk in 0..kn {
            let b = bt[l * e1 + e0 + kk];
            if layers.contains(&l) {
                assert_eq!(b, 0.0, "layer {l}: the grown bias must stay 0 (bias_frozen_from)");
            } else {
                assert_eq!(b, DEAD_BIAS);
            }
        }
    }
    // undeclared layers' slots are not in the trainable ranges
    let lay1 = Layout::new(&trained.cfg);
    let ew = 3 * h * trained.cfg.inter;
    for l in [0usize, 2] {
        let ffn = match &lay1.layers[l] {
            cortiq_embryo::model::LayerOffs::Mixer { ffn, .. }
            | cortiq_embryo::model::LayerOffs::Anchor { ffn, .. }
            | cortiq_embryo::model::LayerOffs::Gdn { ffn, .. } => ffn,
        };
        let r = ffn.experts + e0 * ew..ffn.experts + (e0 + kn) * ew;
        assert!(grown.params[r.clone()] == trained.params[r], "layer {l}: inert slots trained");
    }
    // ---- the witness: shells, coverage, routing shift ----
    let desc_t = GrowthDesc::from_checkpoint(&trained, e0).unwrap();
    assert_eq!(desc_t.k(), kn);
    let gpu = EmbryoGpu::new_eval_dropless(trained.cfg.clone(), 4, 64, &trained.params).unwrap();
    gpu.set_desc(&trained.extras);
    let tr_train = trace_routes(&gpu, &train, &layers, &desc_t).unwrap();
    assert_eq!(tr_train.len(), 2);
    assert_eq!(tr_train[0].layer, 1);
    assert_eq!(tr_train[1].layer, 3);
    let n_w = train.tokens.len() / 64;
    for tr in &tr_train {
        assert_eq!(tr.tokens, n_w * 64);
        assert_eq!(tr.grown_err.len(), tr.tokens * kn);
        assert!(tr.grown_err.iter().all(|e| e.is_finite()));
        assert!(tr.trunk_best.iter().all(|&b| (b as usize) < e0));
        // the witness winner equals the runtime's top-1 over the full score vector
        for t in (0..tr.tokens).step_by(97) {
            let mut full: Vec<f32> = vec![0.0; e1];
            full[tr.trunk_best[t] as usize] = tr.trunk_best_score[t];
            // trunk experts other than the best: anything lower
            for e in 0..e0 {
                if e != tr.trunk_best[t] as usize {
                    full[e] = tr.trunk_best_score[t] - 1.0;
                }
            }
            full[e0..].copy_from_slice(&tr.grown_score[t * kn..(t + 1) * kn]);
            assert_eq!(tr.winner(t, None), resonance_winner(&full));
        }
    }
    let (shells, wins) = shells_from_traces(&tr_train, 0.99);
    assert_eq!(shells.len(), 2);
    for (sh, wn) in shells.iter().zip(&wins) {
        assert_eq!(sh.len(), kn);
        assert!(sh.iter().all(|s| s.is_finite() && *s >= 0.0), "shells {sh:?}");
        for (s, w) in sh.iter().zip(wn) {
            if *w == 0 {
                assert_eq!(*s, 0.0, "an expert without wins has a closed shell");
            }
        }
    }
    eprintln!("shells {shells:?} wins {wins:?}");
    let tr_held = trace_routes(&gpu, &held, &layers, &desc_t).unwrap();
    let cov = coverage(&tr_held, &shells);
    assert_eq!(cov.tokens, (held.tokens.len() / 64) * 64);
    assert!((0.0..=1.0).contains(&cov.overall));
    assert!(cov.per_layer.iter().all(|c| (0.0..=1.0).contains(c)));
    assert!(cov.per_layer.iter().all(|c| *c <= cov.overall + 1e-6));
    let tr_gen = trace_routes(&gpu, &general, &layers, &desc_t).unwrap();
    let shift = routing_shift(&tr_gen, &shells);
    assert_eq!(shift.layers, layers);
    for i in 0..2 {
        assert!((0.0..=1.0).contains(&shift.per_layer_noshell[i]));
        assert!(shift.per_layer_shell[i] <= shift.per_layer_noshell[i], "the shell only removes grown wins");
    }
    assert!(shift.overall_shell <= shift.overall_noshell);
    eprintln!("coverage {cov:?}; shift {shift:?}");
    // with every shell closed nothing is covered and no grown expert wins
    let closed = vec![vec![-1.0f32; kn]; 2];
    assert_eq!(coverage(&tr_held, &closed).overall, 0.0);
    assert_eq!(routing_shift(&tr_gen, &closed).overall_shell, 0.0);
    // with open shells the shell path equals the no-shell path
    let open = vec![vec![f32::MAX; kn]; 2];
    let s_open = routing_shift(&tr_gen, &open);
    assert_eq!(s_open.overall_shell, s_open.overall_noshell);
    assert_eq!(coverage(&tr_held, &open).overall, 1.0);
    drop(gpu);
    // ---- the record over a copy of F0 ----
    let out = f.dir.join("grown.cmf");
    let origin = serde_json::json!({"trigger": "user_corpus", "test": "growth_record_over_fam_a_genome"});
    let quality = serde_json::json!({"held_out": {"before": l0, "after": l1}, "coverage": cov.overall});
    let zero_bias: Vec<Vec<f32>> = layers.iter().map(|_| vec![0.0; e1 - e0]).collect();
    let ctx = RecCtx {
        ck0: &f.ck,
        trained: &trained,
        e0,
        layers: &layers,
        shells: &shells,
        grown_bias: &zero_bias,
        origin,
        quality,
    };
    let summary = write_growth_record(&ctx.args(&f.base, &out, "herbs")).unwrap();
    eprintln!("record: {summary}");
    assert_eq!(summary["record_index"], 0);
    assert_eq!(summary["experts"]["1"], serde_json::json!([4, 5]));
    assert_eq!(summary["experts"]["3"], serde_json::json!([4, 5]));
    assert_eq!(summary["rank"], MOE_K);
    let m0 = CmfModel::open(&f.base).unwrap();
    let m1 = CmfModel::open(&out).unwrap();
    assert_eq!(m0.trunk_hash(), m1.trunk_hash(), "growth changed the trunk hash");
    cortiq_embryo::skill::verify_append(&f.base, &out).unwrap();
    assert_eq!(m1.header.skills.len(), 1);
    let rec = &m1.header.skills[0];
    assert_eq!(rec.id, "herbs");
    assert_eq!(rec.kind.as_deref(), Some(skill_kind::EXPERT_APPEND));
    assert_eq!(rec.layers, layers);
    assert_eq!(rec.status.as_deref(), Some("quarantine"));
    let ea = rec.experts.as_ref().unwrap();
    assert_eq!((ea.count, ea.rank), (kn, MOE_K));
    assert_eq!(ea.shell_quantile, 0.99);
    let se = rec.state_effect.as_ref().unwrap();
    assert_eq!(se.first_affected_layer, 1);
    assert_eq!(se.switch, state_switch::SEQUENCE_START);
    assert_eq!(se.state_bytes_added, 0);
    assert!(rec.selection.is_none() && rec.overrides.is_empty());
    assert_eq!(rec.bound.as_ref().unwrap().genome_id, "embryo-o1-fam-a-tiny");
    let plan = expert_append_layout(&m1.header, &m1.tensors, 0, rec).unwrap();
    assert_eq!(plan.len(), 2 * kn * expert_leaf::ALL.len());
    for p in &plan {
        let t = m1.tensor(&p.name).unwrap_or_else(|| panic!("{} missing", p.name));
        assert_eq!(t.shape, p.shape, "{}", p.name);
        assert!(p.expert >= e0 && p.expert < e0 + kn);
        let li = layers.iter().position(|&l| l == p.layer).unwrap();
        let bytes = m1.tensor_bytes(&p.name).unwrap();
        match p.leaf {
            expert_leaf::BIAS => assert_eq!(cortiq_core::knowledge::f32_le_head(bytes), Some(0.0)),
            expert_leaf::SHELL => assert_eq!(
                cortiq_core::knowledge::f32_le_head(bytes),
                Some(shells[li][p.expert - e0])
            ),
            _ => {}
        }
    }
    assert!(
        m1.tensors.iter().all(|t| !t.name.starts_with("skill.herbs.model.layers.0.") && !t.name.starts_with("skill.herbs.model.layers.2.")),
        "undeclared layers carry no record tensors"
    );
    assert_eq!(m1.tensors.len(), m0.tensors.len() + plan.len());
    drop((m0, m1));
    // ---- runtime: CMF_GROWTH=off mounts nothing → F1 == F0 bit-for-bit ----
    let ids: Vec<u32> = train.tokens[100..132].iter().map(|&x| x as u32).collect();
    unsafe { std::env::set_var("CMF_GROWTH", "off") };
    let lg0 = prefill(&f.base, &ids);
    let lg1 = prefill(&out, &ids);
    assert!(
        lg0.iter().zip(&lg1).all(|(a, b)| a.to_bits() == b.to_bits()),
        "CMF_GROWTH=off: F1 logits must equal F0's bit-for-bit"
    );
    // CMF_GROWTH=all mounts the quarantine record: it loads and runs, shell on and off
    unsafe { std::env::set_var("CMF_GROWTH", "all") };
    let lg_on = prefill(&out, &ids);
    assert!(lg_on.iter().all(|v| v.is_finite()));
    cortiq_engine::pipeline::set_growth_shell(Some(false));
    let lg_off = prefill(&out, &ids);
    assert!(lg_off.iter().all(|v| v.is_finite()));
    cortiq_engine::pipeline::set_growth_shell(None);
    unsafe { std::env::remove_var("CMF_GROWTH") };
    // the default (active only) mounts nothing for a quarantine record
    let lg_def = prefill(&out, &ids);
    assert!(lg_def.iter().zip(&lg0).all(|(a, b)| a.to_bits() == b.to_bits()));
    // ---- a second record on F1: the per-layer chain continues at E0+2 ----
    let out2 = f.dir.join("grown2.cmf");
    let summary2 = write_growth_record(&ctx.args(&out, &out2, "herbs2")).unwrap();
    assert_eq!(summary2["record_index"], 1);
    assert_eq!(summary2["experts"]["1"], serde_json::json!([6, 7]));
    assert_eq!(summary2["experts"]["3"], serde_json::json!([6, 7]));
    let m2 = CmfModel::open(&out2).unwrap();
    assert_eq!(m2.header.skills.len(), 2);
    assert_eq!(m2.trunk_hash(), CmfModel::open(&f.base).unwrap().trunk_hash());
    cortiq_embryo::skill::verify_append(&out, &out2).unwrap();
    let plan2 = expert_append_layout(&m2.header, &m2.tensors, 1, &m2.header.skills[1]).unwrap();
    assert!(plan2.iter().all(|p| p.expert >= 6 && p.expert < 8 && m2.tensor(&p.name).is_some()));
    drop(m2);
    unsafe { std::env::set_var("CMF_GROWTH", "all") };
    assert!(prefill(&out2, &ids).iter().all(|v| v.is_finite()));
    unsafe { std::env::remove_var("CMF_GROWTH") };
    // ---- refusals leave nothing behind ----
    fn refuse(args: RecordArgs<'_>, what: &str) -> String {
        let out_p = args.out.to_path_buf();
        let existed = out_p.exists();
        let e = write_growth_record(&args).err().unwrap_or_else(|| panic!("{what}: accepted"));
        eprintln!("refused ({what}): {e:#}");
        assert_eq!(out_p.exists(), existed, "{what}: output state changed");
        format!("{e:#}")
    }
    let (p_r3, p_r4, p_r5, p_r6, p_r9, p_r10) = (
        f.dir.join("r3.cmf"),
        f.dir.join("r4.cmf"),
        f.dir.join("r5.cmf"),
        f.dir.join("r6.cmf"),
        f.dir.join("r9.cmf"),
        f.dir.join("r10.cmf"),
    );
    refuse(ctx.args(&f.base, &f.base, "x"), "out == base");
    refuse(ctx.args(&f.base, &out, "x"), "out exists");
    let mut a3 = ctx.args(&f.base, &p_r3, "x");
    a3.e0 = 3;
    refuse(a3, "wrong E0");
    let mut a4 = ctx.args(&f.base, &p_r4, "x");
    let desc_layers = vec![3usize, 1];
    a4.layers = &desc_layers;
    refuse(a4, "layers not ascending");
    let short = vec![vec![0.0f32; kn]];
    let mut a5 = ctx.args(&f.base, &p_r5, "x");
    a5.shells = &short;
    refuse(a5, "shells for one layer only");
    let bad_shell = vec![vec![0.5f32, f32::INFINITY], vec![0.5, 0.5]];
    let mut a6 = ctx.args(&f.base, &p_r6, "x");
    a6.shells = &bad_shell;
    refuse(a6, "infinite shell");
    refuse(ctx.args(&f.base, &f.dir.join("r7.cmf"), "a.b"), "id with a dot");
    // a base without a genome
    let plain = f.dir.join("plain.cmf");
    cortiq_embryo::export::export(&f.ck, f.tok_json.as_bytes(), &plain).unwrap();
    let e = refuse(ctx.args(&plain, &f.dir.join("r8.cmf"), "x"), "no genome");
    assert!(e.contains("GENOME"), "{e}");
    // a checkpoint that is not the base's
    let mut other = Checkpoint {
        cfg: f.ck.cfg.clone(),
        step: f.ck.step,
        params: f.ck.params.clone(),
        m: None,
        v: None,
        extras: f.ck.extras.clone(),
    };
    other.params[0] += 1.0;
    let mut a9 = ctx.args(&f.base, &p_r9, "x");
    a9.ck0 = &other;
    let e = refuse(a9, "foreign checkpoint");
    assert!(e.contains("not the checkpoint"), "{e}");
    // a grown checkpoint whose bias moved (bias trained, not pinned)
    let mut moved = Checkpoint {
        cfg: trained.cfg.clone(),
        step: trained.step,
        params: trained.params.clone(),
        m: None,
        v: None,
        extras: trained.extras.clone(),
    };
    for (n, x) in moved.extras.iter_mut() {
        if n == "desc.bias" {
            x[1 * e1 + e0] = -0.01;
        }
    }
    let mut a10 = ctx.args(&f.base, &p_r10, "x");
    a10.trained = &moved;
    let e = refuse(a10, "bias moved");
    assert!(e.contains("carries bias") && e.contains("--bias-mode zero"), "{e}");
    // surgery refusals
    assert!(grow_experts_k(&f.ck, &GrowSpec { experts: 0, ..spec.clone() }).is_err());
    assert!(grow_experts_k(&f.ck, &GrowSpec { layers: vec![7], ..spec.clone() }).is_err());
    assert!(grow_experts_k(&f.ck, &GrowSpec { layers: vec![1, 1], ..spec.clone() }).is_err());
    assert!(grow_experts_k(&f.ck, &GrowSpec { layers: vec![], ..spec.clone() }).is_err());
    let _ = std::fs::remove_dir_all(&f.dir);
}

// ───────────────────────── the CLI driver ─────────────────────────

#[test]
fn grow_cli_end_to_end() {
    let Some(f) = fixture("cli") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let corpus = f.dir.join("herbs.txt");
    std::fs::write(&corpus, herbs_text(10)).unwrap();
    let held = f.dir.join("herbs-held.txt");
    std::fs::write(&held, herbs_text(3).replace("tea", "infusion")).unwrap();
    let general = f.dir.join("general.u16");
    encode(&f.bpe, &general_text(6)).save(&general).unwrap();
    let out = f.dir.join("F1.cmf");
    let out_ckpt = f.dir.join("grown.ckpt");
    let export = f.dir.join("full.cmf");
    let a = cortiq_embryo::cli::GrowCli {
        ckpt: f.ckpt.clone(),
        tokenizer: f.tok_path.clone(),
        corpus: vec![corpus.clone()],
        held: vec![held.clone()],
        general: Some(general.clone()),
        trace_tokens: 0,
        trace_docs: 0,
        experts: 2,
        layers: Some(vec![2, 3]),
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.95,
        shell_target_shift: 0.005,
        bias_mode: BiasMode::Zero,
        source_mode: SourceMode::Hottest,
        novel_quantile: 0.995,
        desc_mode: None,
        record_out: Some(out.clone()),
        base: Some(f.base.clone()),
        id: Some("herbs".into()),
        out_ckpt: Some(out_ckpt.clone()),
        export: Some(export.clone()),
        steps: 8,
        lr: 1e-3,
        batch: 4,
        seq: 64,
        gate: -1.0,
        held_batches: 0,
        noise: 1e-3,
        shift: 0.1,
        seed: 7,
    };
    let base_bytes = std::fs::read(&f.base).unwrap();
    let s = cortiq_embryo::cli::grow(&a).unwrap();
    eprintln!("{}", serde_json::to_string(&s).unwrap());
    assert_eq!(s["id"], "herbs");
    assert_eq!(s["K"], 2);
    assert_eq!(s["layers"], serde_json::json!([2, 3]));
    assert_eq!(s["shells"].as_array().unwrap().len(), 2);
    assert!(s["shells"][0].as_array().unwrap().iter().all(|v| v.as_f64().unwrap().is_finite()));
    assert!(s["held_before"].as_f64().unwrap().is_finite());
    assert!(s["held_after"].as_f64().unwrap().is_finite());
    // the gate reference is the genome's loss; the untrained grown loss is reported too
    assert_eq!(s["held_genome"], s["held_before"]);
    assert!(s["held_untrained"].as_f64().unwrap().is_finite());
    assert!(s["held_windows"].as_u64().unwrap() >= 4);
    assert_eq!(s["served"]["encoding"], "f32");
    assert_eq!(s["served"]["rounded_values"], 0);
    // the descriptors of the new experts came from the corpus witness
    let init_tok = s["descriptor_init_tokens"].as_array().unwrap();
    assert_eq!(init_tok.len(), 2);
    assert!(init_tok.iter().any(|l| l.as_array().unwrap().iter().any(|n| n.as_u64().unwrap() > 0)));
    // the held-out witness: documents as framed rows
    assert!(s["held_witness"]["frame"].is_string());
    assert!(s["held_witness"]["rows"].as_u64().unwrap() >= 1);
    assert_eq!(s["held_witness"]["docs"], 1);
    assert_eq!(std::fs::read(&f.base).unwrap(), base_bytes, "F0 rewritten");
    let cov = s["coverage"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&cov));
    let rs_no = s["routing_shift_noshell"].as_f64().unwrap();
    let rs_sh = s["routing_shift_shell"].as_f64().unwrap();
    assert!(rs_sh <= rs_no && (0.0..=1.0).contains(&rs_no));
    assert_eq!(s["record"]["experts"]["2"], serde_json::json!([4, 5]));
    assert_eq!(s["record"]["experts"]["3"], serde_json::json!([4, 5]));
    assert!(out_ckpt.exists() && export.exists());
    // the record file
    let m0 = CmfModel::open(&f.base).unwrap();
    let m1 = CmfModel::open(&out).unwrap();
    assert_eq!(m0.trunk_hash(), m1.trunk_hash());
    cortiq_embryo::skill::verify_append(&f.base, &out).unwrap();
    let rec = &m1.header.skills[0];
    assert_eq!(rec.kind.as_deref(), Some(skill_kind::EXPERT_APPEND));
    assert_eq!(rec.layers, vec![2, 3]);
    let origin = rec.origin.as_ref().unwrap();
    assert_eq!(origin["trigger"], "user_corpus");
    assert_eq!(origin["dataset_sha256"].as_array().unwrap().len(), 3, "corpus + held + general");
    assert_eq!(origin["held_out"]["source"], "--held files (group held-out)");
    assert_eq!(origin["held_out"]["instance"], "dropless");
    assert!(origin["held_out"]["genome"].is_number() && origin["held_out"]["grown_untrained"].is_number());
    assert!(origin["routing_shift"]["overall_shell"].is_number());
    assert!(origin["coverage"]["overall"].is_number());
    assert_eq!(origin["served"]["encoding"], "f32");
    // the full export is NOT a record: it changes the arch
    let full = CmfModel::open(&export).unwrap();
    assert_eq!(full.header.arch.moe.as_ref().unwrap().num_experts, 6);
    assert!(full.header.genome.is_none());
    drop((m0, m1, full));
    // the grown checkpoint reloads with E0+K experts in every layer
    let ck1 = cortiq_embryo::train::load_checkpoint(&out_ckpt).unwrap();
    assert_eq!(ck1.cfg.experts, 6);
    // refusals of the driver
    let mut b = cortiq_embryo::cli::GrowCli { ..a };
    b.export = None; // full.cmf exists now; the export refusals come below
    b.record_out = Some(f.base.clone());
    let e = cortiq_embryo::cli::grow(&b).err().unwrap().to_string();
    assert!(e.contains("--base"), "{e}");
    b.record_out = Some(f.dir.join("F2.cmf"));
    b.base = None;
    assert!(cortiq_embryo::cli::grow(&b).is_err());
    b.base = Some(f.base.clone());
    b.experts = 0;
    assert!(cortiq_embryo::cli::grow(&b).is_err());
    // T6: E0 + K ≤ 8 (the runtime graph's limit)
    b.experts = MAX_RUNTIME_EXPERTS - 4 + 1;
    let e = cortiq_embryo::cli::grow(&b).err().unwrap().to_string();
    assert!(e.contains("> 8"), "{e}");
    b.experts = 1;
    b.shell_quantile = 1.5;
    assert!(cortiq_embryo::cli::grow(&b).is_err());
    b.shell_quantile = 0.9;
    b.layers = Some(vec![4]);
    assert!(cortiq_embryo::cli::grow(&b).is_err());
    b.layers = Some(vec![2, 3]);
    // NF-4: `--export` never truncates the genome file (nor an existing file)
    b.export = Some(f.base.clone());
    let e = cortiq_embryo::cli::grow(&b).err().unwrap().to_string();
    assert!(e.contains("GENOME") || e.contains("--base"), "{e}");
    assert_eq!(std::fs::read(&f.base).unwrap(), base_bytes, "F0 truncated by --export");
    b.export = Some(export.clone());
    let e = cortiq_embryo::cli::grow(&b).err().unwrap().to_string();
    assert!(e.contains("--export"), "{e}");
    b.export = Some(f.dir.join("F2.cmf"));
    let e = cortiq_embryo::cli::grow(&b).err().unwrap().to_string();
    assert!(e.contains("--record-out"), "{e}");
    b.export = None;
    // T5: growing on F1, whose record already occupies layers 2 and 3
    b.base = Some(out.clone());
    b.record_out = Some(f.dir.join("F2.cmf"));
    let e = cortiq_embryo::cli::grow(&b).err().unwrap().to_string();
    assert!(e.contains("expert_append") && e.contains("herbs"), "{e}");
    let m1 = CmfModel::open(&out).unwrap();
    assert!(check_base_growth_records(&m1.header, &[2, 3]).is_err());
    assert!(check_base_growth_records(&m1.header, &[0, 1]).is_ok());
    assert!(check_base_growth_records(&CmfModel::open(&f.base).unwrap().header, &[2, 3]).is_ok());
    drop(m1);
    assert!(!f.dir.join("F2.cmf").exists());
    let _ = std::fs::remove_dir_all(&f.dir);
}

// ───────────────────────── descriptor init (F1) ─────────────────────────

/// The runtime score `bias − ‖(x−μ)⊥U‖²` cannot tell a copy whose μ was
/// shifted INSIDE span(U) from its source: the copy scored every token
/// like the source and, at bias 0 against a negative balancing bias, won
/// a superset of its tokens on any input. The record growth initialises
/// the copy from the corpus (μ = cluster mean, U = its principal
/// subspace); the legacy copy shifts OUTSIDE span(U). Both move the
/// reconstruction error off the source's on a genome with a real U.
#[test]
fn grown_descriptor_differs_from_its_source_on_a_real_subspace() {
    let Some(f) = fixture("init") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let e0 = 4usize;
    let h = f.ck.cfg.hidden;
    let layers = vec![1usize, 3];
    let train = encode(&f.bpe, &herbs_text(12));
    // the genome's U is a real orthonormal basis (the fixture ran the P1
    // step; an expert that saw fewer than two tokens keeps a zero U)
    let u0 = desc(&f.ck, "desc.u");
    let u_norm = |l: usize, e: usize, i: usize| -> f32 {
        let r = &u0[((l * e0 + e) * MOE_K + i) * h..((l * e0 + e) * MOE_K + i + 1) * h];
        r.iter().map(|x| x * x).sum::<f32>().sqrt()
    };
    for l in 0..4 {
        for e in 0..e0 {
            for i in 0..MOE_K {
                let n = u_norm(l, e, i);
                assert!((n - 1.0).abs() < 1e-3 || n < 1e-6, "layer {l} expert {e} row {i}: |u| = {n}");
            }
        }
    }
    let spec = GrowSpec {
        experts: 1,
        layers: layers.clone(),
        noise: 0.0,
        shift: 0.1,
        seed: 5,
        zero_bias: true,
    };
    let gpu0 = EmbryoGpu::new_eval_dropless(f.ck.cfg.clone(), 4, 64, &f.ck.params).unwrap();
    gpu0.set_desc(&f.ck.extras);
    let desc0 = GrowthDesc::from_checkpoint(&f.ck, e0).unwrap();
    assert_eq!(desc0.k(), 0);
    let sources = cortiq_embryo::growth::grown_sources(&f.ck, &spec).unwrap();
    assert_eq!(sources.len(), 2);
    let inits = cluster_inits(&gpu0, &train, &layers, &desc0, &sources, 5).unwrap();
    assert_eq!(inits.len(), 2);
    assert!(inits.iter().all(|v| v.len() == 1));
    let (grown_c, src_c) = grow_experts_init(&f.ck, &spec, Some(&inits)).unwrap();
    let (grown_l, src_l) = grow_experts_k(&f.ck, &spec).unwrap();
    assert_eq!(src_c, src_l);
    assert_old_tensors_identical(&f.ck, &grown_c, "corpus init");
    let e1 = e0 + 1;
    // one window of the growth corpus: the MoE input of every grown layer
    let tokens: Vec<u32> = train.tokens[..256].iter().map(|&x| x as u32).collect();
    let _ = gpu0.forward_hidden(&tokens);
    for (li, &l) in layers.iter().enumerate() {
        let src = sources[li][0];
        let init = &inits[li][0];
        assert!(init.rows > 0, "layer {l}: the hottest source wins no token of the corpus");
        for i in 0..MOE_K {
            assert!((u_norm(l, src, i) - 1.0).abs() < 1e-3, "layer {l}: the source's U row {i} is not unit");
        }
        assert_eq!(init.mu.len(), h);
        assert_eq!(init.u.len(), MOE_K * h);
        // U_new: orthonormal rows
        for i in 0..MOE_K {
            for j in 0..MOE_K {
                let d: f32 = (0..h).map(|t| init.u[i * h + t] * init.u[j * h + t]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-3, "layer {l}: u{i}·u{j} = {d}");
            }
        }
        // the grown checkpoint carries exactly the init
        let mu_c = desc(&grown_c, "desc.mu");
        let u_c = desc(&grown_c, "desc.u");
        assert_eq!(&mu_c[(l * e1 + e0) * h..(l * e1 + e0 + 1) * h], &init.mu[..]);
        assert_eq!(&u_c[(l * e1 + e0) * MOE_K * h..(l * e1 + e0 + 1) * MOE_K * h], &init.u[..]);
        let x2 = gpu0.ffn_input_host(l);
        let mut scores = vec![0.0f32; e1];
        let mut errs = vec![0.0f32; e1];
        for (what, g) in [("corpus init", &grown_c), ("legacy shift outside span(U)", &grown_l)] {
            let mu = &desc(g, "desc.mu")[l * e1 * h..(l + 1) * e1 * h];
            let u = &desc(g, "desc.u")[l * e1 * MOE_K * h..(l + 1) * e1 * MOE_K * h];
            let bias = &desc(g, "desc.bias")[l * e1..(l + 1) * e1];
            assert_eq!(bias[e0], 0.0);
            let mut differ = 0usize;
            for row in 0..256 {
                let x = &x2[row * h..(row + 1) * h];
                resonance_scores(x, mu, u, MOE_K, bias, None, &mut scores, &mut errs);
                if errs[e0].to_bits() != errs[src].to_bits() {
                    differ += 1;
                }
            }
            eprintln!("layer {l} ({what}): err_new != err_src on {differ}/256 tokens");
            assert!(
                differ > 128,
                "layer {l} ({what}): the copy's reconstruction error equals the source's on {} of 256 tokens — \
                 the descriptor was shifted inside span(U)",
                256 - differ
            );
        }
        // the legacy shift left span(U_src): its direction has a component orthogonal to every row
        let mu_l = &desc(&grown_l, "desc.mu")[(l * e1 + e0) * h..(l * e1 + e0 + 1) * h];
        let mu_s = &desc(&f.ck, "desc.mu")[(l * e0 + src) * h..(l * e0 + src + 1) * h];
        let d: Vec<f32> = (0..h).map(|j| mu_l[j] - mu_s[j]).collect();
        let dn: f32 = d.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(dn > 0.0);
        let mut inside = 0.0f32;
        for i in 0..MOE_K {
            let r = &u0[((l * e0 + src) * MOE_K + i) * h..((l * e0 + src) * MOE_K + i + 1) * h];
            let p: f32 = d.iter().zip(r).map(|(a, b)| a * b).sum();
            inside += p * p;
        }
        assert!(inside.sqrt() < 1e-2 * dn, "layer {l}: the legacy shift lies inside span(U) ({} of {dn})", inside.sqrt());
    }
    // K > E0: copies of the same source split its cluster
    let spec5 = GrowSpec { experts: 4, layers: vec![1], ..spec.clone() };
    let s5 = cortiq_embryo::growth::grown_sources(&f.ck, &spec5).unwrap();
    assert_eq!(s5[0].len(), 4);
    let spec6 = GrowSpec { experts: 5, ..spec5.clone() };
    assert!(grow_experts_k(&f.ck, &spec6).is_err(), "E0 4 + K 5 > 8 must be refused");
    let e = grow_experts_k(&f.ck, &spec6).err().unwrap().to_string();
    assert!(e.contains("> 8"), "{e}");
    // documents as framed rows: only the document positions are witnessed
    let docs = split_docs_at_eot(&[1u16, 2, 3, 0, 4, 5, 0, 0, 6], 0);
    assert_eq!(docs, vec![vec![1u16, 2, 3], vec![4, 5], vec![6]]);
    let (rows, spans) = frame_docs(&docs, &[10, 11], &[12], 8, 99).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0], vec![10, 11, 1, 2, 3, 12, 99, 99]);
    assert_eq!(spans[0], 2..5);
    assert_eq!(rows[2], vec![10, 11, 6, 12, 99, 99, 99, 99]);
    assert_eq!(spans[2], 2..3);
    let long = vec![(1..=12).collect::<Vec<u16>>()];
    let (rows, spans) = frame_docs(&long, &[10, 11], &[12], 8, 99).unwrap();
    assert_eq!(rows.len(), 3, "12 tokens in chunks of 5 (8 − frame 3)");
    assert_eq!(spans[2], 2..4);
    assert!(frame_docs(&docs, &[1; 8], &[], 8, 0).is_err());
    let desc_c = GrowthDesc::from_checkpoint(&grown_c, e0).unwrap();
    let gpu1 = EmbryoGpu::new_eval_dropless(grown_c.cfg.clone(), 4, 64, &grown_c.params).unwrap();
    gpu1.set_desc(&grown_c.extras);
    let held_docs = split_docs_at_eot(&train.tokens[..500], train.tokens[7]);
    let eot = f.bpe.special_id(cortiq_embryo::tokenizer::EOT).unwrap();
    let (rows, spans) = frame_docs(&held_docs, &[eot], &[], 64, eot).unwrap();
    let tr = trace_routes_rows(&gpu1, &rows, &spans, &layers, &desc_c).unwrap();
    let want: usize = spans.iter().map(|s| s.len()).sum();
    assert_eq!(tr[0].tokens, want, "only the documents' own positions are witnessed");
    assert_eq!(tr[1].tokens, want);
    let _ = std::fs::remove_dir_all(&f.dir);
}

// ───────────────────────── the served trunk (F2) ─────────────────────────

/// An f16 genome executes on the f16-rounded trunk: the copies, their
/// training and every witness must run on that arena, not on the f32
/// checkpoint. With no training step and no noise the record's grown
/// weights are exactly the rounded source expert.
#[test]
fn f16_genome_grows_on_the_served_trunk() {
    let Some(f) = fixture("f16") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let base16 = f.dir.join("base16.cmf");
    cortiq_embryo::export::export_genome(
        &f.ck,
        f.tok_json.as_bytes(),
        &base16,
        cortiq_core::TensorDtype::F16,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny-f16".into(),
            status: "pre_chat".into(),
        }),
    )
    .unwrap();
    let m16 = CmfModel::open(&base16).unwrap();
    assert_eq!(m16.header.genome.as_ref().unwrap().encoding, "f16");
    let (served, enc, changed) = cortiq_embryo::cli::served_checkpoint(&f.ck, &m16).unwrap();
    assert_eq!(enc, "f16");
    assert!(changed > 0, "an f16 genome rounds the arena");
    let (_, enc32, changed32) = cortiq_embryo::cli::served_checkpoint(&f.ck, &CmfModel::open(&f.base).unwrap()).unwrap();
    assert_eq!((enc32.as_str(), changed32), ("f32", 0));
    let round = |x: f32| cortiq_core::quant::f16_to_f32(cortiq_core::quant::f32_to_f16(x));
    assert!(served.params.iter().zip(&f.ck.params).all(|(s, o)| *s == *o || *s == round(*o)));
    drop(m16);
    let corpus = f.dir.join("herbs.txt");
    std::fs::write(&corpus, herbs_text(10)).unwrap();
    let out = f.dir.join("F1-16.cmf");
    let a = cortiq_embryo::cli::GrowCli {
        ckpt: f.ckpt.clone(),
        tokenizer: f.tok_path.clone(),
        corpus: vec![corpus.clone()],
        held: vec![],
        general: None,
        trace_tokens: 0,
        trace_docs: 0,
        experts: 1,
        layers: Some(vec![1, 3]),
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.99,
        shell_target_shift: 0.005,
        bias_mode: BiasMode::Zero,
        source_mode: SourceMode::Hottest,
        novel_quantile: 0.995,
        desc_mode: None,
        record_out: Some(out.clone()),
        base: Some(base16.clone()),
        id: Some("herbs16".into()),
        out_ckpt: Some(f.dir.join("grown16.ckpt")),
        export: None,
        steps: 0,
        lr: 1e-3,
        batch: 4,
        seq: 64,
        gate: -1.0,
        held_batches: 0,
        noise: 0.0,
        shift: 0.1,
        seed: 7,
    };
    let s = cortiq_embryo::cli::grow(&a).unwrap();
    assert_eq!(s["served"]["encoding"], "f16");
    assert!(s["served"]["rounded_values"].as_u64().unwrap() > 0);
    let m1 = CmfModel::open(&out).unwrap();
    assert_eq!(m1.trunk_hash(), CmfModel::open(&base16).unwrap().trunk_hash());
    let e0 = 4usize;
    let lay = Layout::new(&f.ck.cfg);
    let (h, i) = (f.ck.cfg.hidden, f.ck.cfg.inter);
    let ew = 3 * h * i;
    let mut checked = 0usize;
    for (li, &l) in [1usize, 3].iter().enumerate() {
        let src = s["sources"][li][0].as_u64().unwrap() as usize;
        let ffn = match &lay.layers[l] {
            cortiq_embryo::model::LayerOffs::Mixer { ffn, .. }
            | cortiq_embryo::model::LayerOffs::Anchor { ffn, .. }
            | cortiq_embryo::model::LayerOffs::Gdn { ffn, .. } => ffn,
        };
        let src_gate = &f.ck.params[ffn.experts + src * ew..ffn.experts + src * ew + i * h];
        assert!(src_gate.iter().any(|&x| round(x) != x), "the f32 source must not be f16-exact already");
        let name = format!("skill.herbs16.model.layers.{l}.mlp.experts.{e0}.gate_proj.weight");
        let bytes = m1.tensor_bytes(&name).unwrap_or_else(|e| panic!("{name} missing: {e}"));
        let got: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        assert_eq!(got.len(), i * h);
        for (g, o) in got.iter().zip(src_gate) {
            assert_eq!(*g, round(*o), "{name}: the copy was taken from the f32 checkpoint, not the served f16 trunk");
            checked += 1;
        }
    }
    assert!(checked > 0);
    drop(m1);
    // reshell on an f16 genome: the grown checkpoint's trunk is the served
    // (rounded) arena, so the master is only reachable through --genome-ckpt
    let mut r = ReshellCli {
        ckpt: f.dir.join("grown16.ckpt"),
        genome_ckpt: None,
        tokenizer: f.tok_path.clone(),
        corpus: vec![corpus.clone()],
        held: vec![],
        general: None,
        trace_tokens: 0,
        trace_docs: 0,
        layers: None,
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.99,
        shell_target_shift: 0.005,
        bias_mode: None,
        source_mode: SourceMode::Hottest,
        novel_quantile: 0.995,
        desc_mode: None,
        seed: 7,
        record_out: Some(f.dir.join("F1-16-reshell.cmf")),
        base: base16.clone(),
        id: Some("herbs16".into()),
        batch: 4,
        seq: 64,
    };
    let e = reshell(&r).err().unwrap().to_string();
    assert!(e.contains("--genome-ckpt"), "{e}");
    assert!(!f.dir.join("F1-16-reshell.cmf").exists());
    r.genome_ckpt = Some(f.ckpt.clone());
    let s2 = reshell(&r).unwrap();
    assert_eq!(s2["served"]["encoding"], "f16");
    assert_eq!(s2["served"]["rounded_values"], s["served"]["rounded_values"]);
    assert_eq!(s2["layers"], serde_json::json!([1, 3]));
    assert_eq!(s2["sources"], s["sources"]);
    assert_eq!(s2["shells"], s["shells"]);
    let a16 = record_tensors(&out, "herbs16");
    let b16 = record_tensors(&f.dir.join("F1-16-reshell.cmf"), "herbs16");
    assert_eq!(a16.len(), b16.len());
    for (name, v) in &a16 {
        assert!(v.iter().zip(&b16[name]).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}");
    }
    let _ = std::fs::remove_dir_all(&f.dir);
}

// ───────────────────────── a genome file is never rewritten (NF-4) ─────────────────────────

#[test]
fn export_never_overwrites_a_genome_file() {
    let Some(f) = fixture("nf4") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let before = std::fs::read(&f.base).unwrap();
    let e = cortiq_embryo::export::export(&f.ck, f.tok_json.as_bytes(), &f.base).err().unwrap().to_string();
    assert!(e.contains("GENOME"), "{e}");
    assert_eq!(std::fs::read(&f.base).unwrap(), before, "export truncated the genome file");
    let e = cortiq_embryo::export::export_genome(
        &f.ck,
        f.tok_json.as_bytes(),
        &f.base,
        cortiq_core::TensorDtype::F16,
        Some(&cortiq_embryo::export::ExportGenome { id: "x".into(), status: "pre_chat".into() }),
    )
    .err()
    .unwrap()
    .to_string();
    assert!(e.contains("GENOME"), "{e}");
    assert_eq!(std::fs::read(&f.base).unwrap(), before);
    assert!(cortiq_embryo::export::refuse_genome_overwrite(&f.base).is_err());
    // a plain export (no genome block) may be rewritten; a missing file passes
    let plain = f.dir.join("plain.cmf");
    cortiq_embryo::export::export(&f.ck, f.tok_json.as_bytes(), &plain).unwrap();
    assert!(cortiq_embryo::export::refuse_genome_overwrite(&plain).is_ok());
    cortiq_embryo::export::export(&f.ck, f.tok_json.as_bytes(), &plain).unwrap();
    assert!(cortiq_embryo::export::refuse_genome_overwrite(&f.dir.join("absent.cmf")).is_ok());
    let txt = f.dir.join("notes.txt");
    std::fs::write(&txt, "not a cmf").unwrap();
    assert!(cortiq_embryo::export::refuse_genome_overwrite(&txt).is_ok());
    // a pre-0.8.1 genome file carries GENOME on bit 8: still refused (by content)
    let legacy = f.dir.join("legacy-bits.cmf");
    let mut bytes = std::fs::read(&f.base).unwrap();
    let bits = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let old_bits = (bits & !cortiq_core::format::features::GENOME)
        | cortiq_core::format::legacy_embryo_bits::GENOME;
    bytes[12..16].copy_from_slice(&old_bits.to_le_bytes());
    std::fs::write(&legacy, &bytes).unwrap();
    let e = cortiq_embryo::export::refuse_genome_overwrite(&legacy)
        .unwrap_err()
        .to_string();
    assert!(e.contains("migrate-embryo-bits"), "{e}");
    let _ = std::fs::remove_dir_all(&f.dir);
}

// ───────────────────────── legacy full-genome growth ─────────────────────────

#[test]
fn growth_appends_experts_without_touching_old_records() {
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    unsafe { std::env::set_var("CMF_GPU", "0") };
    let mut cfg = EmbryoCfg::tiny();
    cfg.experts = 2;
    let lay = Layout::new(&cfg);
    let params = init_params(&cfg, &lay, 21);
    let mut text = Vec::new();
    for f in ["../../docs/CMF_V2_SPEC.md", "../../docs/SKILLS.md", "../../docs/COMPARISON.md"] {
        if let Ok(b) = std::fs::read(f) {
            text.extend_from_slice(&b);
        }
    }
    while text.len() < 40_000 {
        text.extend_from_slice(format!("{}{}", herbs_text(2), general_text(2)).as_bytes());
    }
    let corpus = Shard::from_bytes(&text);
    // a few training steps to seed descriptors and give the routing shape
    let mut gpu = EmbryoGpu::new(cfg.clone(), 4, 64, &params).unwrap();
    for s in 0..3 {
        let tk: Vec<u32> = corpus.tokens[s * 256..s * 256 + 256]
            .iter()
            .map(|&x| x as u32)
            .collect();
        let tg: Vec<u32> = corpus.tokens[s * 256 + 1..s * 256 + 257]
            .iter()
            .map(|&x| x as u32)
            .collect();
        gpu.train_step(&tk, &tg, 1e-3, 0.0, 1.0);
    }
    let params = gpu.params_host();
    let extras: Vec<(String, Vec<f32>)> = gpu
        .desc_host()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    drop(gpu);
    let ck = Checkpoint {
        cfg: cfg.clone(),
        step: 3,
        params,
        m: None,
        v: None,
        extras,
    };
    let (grown, sources) = grow_experts(&ck, 1e-3, 0.1, 5);
    assert_eq!(grown.cfg.experts, 3);
    assert_eq!(sources.len(), cfg.layers);
    assert_old_tensors_identical(&ck, &grown, "growth");
    let a = GrowArgs {
        steps: 20,
        lr: 1e-3,
        batch: 4,
        seq: 64,
        eval_every: 10,
        seed: 3,
        held_batches: 0,
    };
    let (trained, l0, l1) = train_new_experts(&grown, &corpus, &a, &|| false).unwrap();
    eprintln!("legacy growth: held-out {l0:.4} → {l1:.4}; sources {sources:?}");
    assert_old_tensors_identical(&ck, &trained, "training the new experts");
    let re = fancy_regex::Regex::new(cortiq_embryo::tokenizer::SPLIT).unwrap();
    let mut counts = std::collections::HashMap::new();
    cortiq_embryo::tokenizer::count_words("hello world", &re, &mut counts);
    let tok_json = cortiq_embryo::tokenizer::train(&counts, cfg.vocab, false).to_hf_json();
    let dir = scratch("legacy");
    let p0 = dir.join("before.cmf");
    let p1 = dir.join("after.cmf");
    cortiq_embryo::export::export(&ck, tok_json.as_bytes(), &p0).unwrap();
    cortiq_embryo::export::export(&trained, tok_json.as_bytes(), &p1).unwrap();
    let m0 = CmfModel::open(&p0).unwrap();
    let m1 = CmfModel::open(&p1).unwrap();
    for t in &m0.tensors {
        assert!(m1.tensor(&t.name).is_some(), "{}: old tensor kept", t.name);
        assert_eq!(m0.tensor_bytes(&t.name).unwrap(), m1.tensor_bytes(&t.name).unwrap(), "{}: bytes changed", t.name);
    }
    assert!(m1.tensor("model.layers.0.mlp.experts.2.gate_proj.weight").is_some());
    assert!(m1.tensor("model.layers.0.mlp.experts.2.desc.mu").is_some());
    assert_eq!(m1.header.arch.moe.as_ref().unwrap().num_experts, 3);
    drop((m0, m1));
    let ids: Vec<u32> = corpus.tokens[100..132].iter().map(|&x| x as u32).collect();
    let lg = prefill(&p1, &ids);
    assert!(lg.iter().all(|v| v.is_finite()));
    let _ = std::fs::remove_dir_all(&dir);
}

// ───────────────────────── shell modes, reshell, bias mode ─────────────────────────

/// A synthetic one-layer witness (`K = 1`, `E0 = 4`): the grown expert wins
/// token `t` iff `err[t] < 1` (trunk best score `−1`, grown bias 0).
fn synthetic_trace(errs: &[f32]) -> cortiq_embryo::growth::LayerTrace {
    cortiq_embryo::growth::LayerTrace {
        layer: 2,
        e0: 4,
        k: 1,
        tokens: errs.len(),
        trunk_best: vec![0; errs.len()],
        trunk_best_score: vec![-1.0; errs.len()],
        grown_score: errs.iter().map(|e| -e).collect(),
        grown_err: errs.to_vec(),
    }
}

/// `general-target`: an expert capturing more than the target of the
/// layer's general tokens gets the `(target / share)`-quantile of those
/// captured errors — the shell admits EXACTLY the target here (no ties);
/// under the target it keeps the won-quantile shell; the won-quantile
/// shell admits far more.
#[test]
fn general_target_shell_admits_at_most_the_target_of_the_general_tokens() {
    // train: 200 tokens all won, errors 0.002·t → q 0.99 shell = err[197]
    let train = synthetic_trace(&(0..200).map(|t| 0.002 * t as f32).collect::<Vec<_>>());
    // general: 1000 tokens, the first 400 captured (errors 0.001·t), the rest far outside
    let general_errs: Vec<f32> = (0..1000).map(|t| if t < 400 { 0.001 * t as f32 } else { 5.0 }).collect();
    let general = synthetic_trace(&general_errs);
    let won = shells_won_quantile(std::slice::from_ref(&train), 0.99);
    assert_eq!(won.mode, ShellMode::WonQuantile);
    // the nearest-rank 0.99 quantile of the 200 won errors (f32 0.99 is a
    // hair above 0.99: rank ceil(198.000002) = 199 → err[198])
    let won_expect = quantile_sorted(&train.grown_err, 0.99);
    assert_eq!(won.shells, vec![vec![won_expect]]);
    assert!((won_expect - 0.396).abs() < 1e-6, "{won_expect}");
    assert_eq!(won.wins, vec![vec![200]]);
    assert_eq!(won.rule, vec![vec!["won-quantile"]]);
    let s_won = routing_shift(std::slice::from_ref(&general), &won.shells);
    assert_eq!(s_won.per_layer_noshell, vec![0.4]);
    // the won shell admits every general token with err ≤ 0.396: t ≤ 396 → 397 of 1000
    assert!((s_won.per_layer_shell[0] - 0.397).abs() < 1e-6, "{:?}", s_won.per_layer_shell);
    // target 0.05 < share 0.4 → the (0.05 / 0.4)-quantile of the 400 captured
    // errors: rank ceil(50.00000x) tokens admitted = the target + at most
    // 1 / N from the rank rounding
    let gt = shells_general_target(std::slice::from_ref(&train), std::slice::from_ref(&general), 0.99, 0.05).unwrap();
    assert_eq!(gt.mode, ShellMode::GeneralTarget);
    assert_eq!(gt.target_shift, Some(0.05));
    assert_eq!(gt.wins, won.wins);
    assert_eq!(gt.general_captured, Some(vec![vec![400]]));
    assert_eq!(gt.general_share, Some(vec![vec![0.4]]));
    assert_eq!(gt.rule, vec![vec!["general-target"]]);
    let qq = 0.05f32 / 0.4f32;
    assert!((gt.applied_quantile[0][0] - qq).abs() < 1e-7);
    let captured: Vec<f32> = general_errs[..400].to_vec();
    assert_eq!(gt.shells, vec![vec![quantile_sorted(&captured, qq)]]);
    assert!((gt.shells[0][0] - 0.049).abs() < 0.0015, "{:?}", gt.shells);
    let s_gt = routing_shift(std::slice::from_ref(&general), &gt.shells);
    assert_eq!(s_gt.per_layer_noshell, vec![0.4]);
    let admitted = s_gt.per_layer_shell[0];
    assert!(admitted <= 0.05 + 1.0 / 1000.0 + 1e-6 && admitted >= 0.05 - 1e-6, "{:?}", s_gt.per_layer_shell);
    assert!(admitted < s_won.per_layer_shell[0]);
    // target above the share → the won-quantile shell, rule says so
    let under = shells_general_target(std::slice::from_ref(&train), std::slice::from_ref(&general), 0.99, 0.5).unwrap();
    assert_eq!(under.shells, won.shells);
    assert_eq!(under.rule, vec![vec!["won-quantile"]]);
    assert_eq!(under.applied_quantile, vec![vec![0.99]]);
    // target 0 closes the shell of a capturing expert
    let closed = shells_general_target(std::slice::from_ref(&train), std::slice::from_ref(&general), 0.99, 0.0).unwrap();
    assert_eq!(closed.shells, vec![vec![0.0]]);
    // an expert capturing nothing keeps the won shell
    let none = synthetic_trace(&vec![5.0; 100]);
    let g0 = shells_general_target(std::slice::from_ref(&train), std::slice::from_ref(&none), 0.99, 0.05).unwrap();
    assert_eq!(g0.shells, won.shells);
    assert_eq!(g0.general_captured, Some(vec![vec![0]]));
    // mismatched witnesses refused
    let other = cortiq_embryo::growth::LayerTrace { layer: 3, ..synthetic_trace(&[0.1]) };
    assert!(shells_general_target(std::slice::from_ref(&train), std::slice::from_ref(&other), 0.99, 0.05).is_err());
    assert!(ShellMode::parse("nope").is_err());
    assert_eq!(ShellMode::parse("general-target").unwrap(), ShellMode::GeneralTarget);
    assert_eq!(BiasMode::parse("source").unwrap(), BiasMode::Source);
    assert!(BiasMode::parse("one").is_err());
}

/// The f32 payloads of every tensor of a record (`skill.{id}.…`), by name.
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

fn grow_cli(f: &Fixture, corpus: &Path, held: &Path, general: Option<&Path>) -> GrowCli {
    GrowCli {
        ckpt: f.ckpt.clone(),
        tokenizer: f.tok_path.clone(),
        corpus: vec![corpus.to_path_buf()],
        held: vec![held.to_path_buf()],
        general: general.map(|p| p.to_path_buf()),
        trace_tokens: 0,
        trace_docs: 0,
        experts: 1,
        layers: Some(vec![2, 3]),
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.99,
        shell_target_shift: 0.005,
        bias_mode: BiasMode::Zero,
        source_mode: SourceMode::Hottest,
        novel_quantile: 0.995,
        desc_mode: None,
        record_out: None,
        base: Some(f.base.clone()),
        id: Some("herbs".into()),
        out_ckpt: None,
        export: None,
        steps: 8,
        lr: 1e-3,
        batch: 4,
        seq: 64,
        gate: -1.0,
        held_batches: 0,
        noise: 1e-3,
        shift: 0.1,
        seed: 7,
    }
}

fn reshell_cli(f: &Fixture, ckpt: &Path, corpus: &Path, held: &Path, general: Option<&Path>) -> ReshellCli {
    ReshellCli {
        ckpt: ckpt.to_path_buf(),
        genome_ckpt: None,
        tokenizer: f.tok_path.clone(),
        corpus: vec![corpus.to_path_buf()],
        held: vec![held.to_path_buf()],
        general: general.map(|p| p.to_path_buf()),
        trace_tokens: 0,
        trace_docs: 0,
        layers: None,
        shell_mode: ShellMode::WonQuantile,
        shell_quantile: 0.99,
        shell_target_shift: 0.005,
        bias_mode: None,
        source_mode: SourceMode::Hottest,
        novel_quantile: 0.995,
        desc_mode: None,
        seed: 7,
        record_out: None,
        base: f.base.clone(),
        id: Some("herbs".into()),
        batch: 4,
        seq: 64,
    }
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// `grow --out-ckpt` saves the TRAINED grown checkpoint; `reshell` from it
/// redoes only the tail: the same sources, the same shells (won-quantile),
/// the same coverage / routing shift, a record whose expert tensors are
/// BIT-identical to `grow`'s (shells within 1e-6); with
/// `--shell-mode general-target` the shell of every grown expert admits at
/// most `--shell-target-shift` of the layer's general tokens
/// (`routing_shift_shell ≤ target + 1/N`), less than the won-quantile
/// shell admits; the refusals (general-target without `--general`, a
/// `--layers` mismatch, a foreign checkpoint).
#[test]
fn general_target_shells_and_reshell_from_the_trained_checkpoint() {
    let Some(f) = fixture("reshell") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    let corpus = f.dir.join("herbs.txt");
    std::fs::write(&corpus, herbs_text(10)).unwrap();
    let held = f.dir.join("herbs-held.txt");
    std::fs::write(&held, herbs_text(3).replace("tea", "infusion")).unwrap();
    let general = f.dir.join("general.u16");
    encode(&f.bpe, &general_text(6)).save(&general).unwrap();
    let out_w = f.dir.join("F1-won.cmf");
    let out_ckpt = f.dir.join("grown.ckpt");
    // ---- 1. grow, won-quantile, the trained checkpoint saved ----
    let mut a = grow_cli(&f, &corpus, &held, Some(&general));
    a.record_out = Some(out_w.clone());
    a.out_ckpt = Some(out_ckpt.clone());
    let s1 = grow(&a).unwrap();
    eprintln!("grow: {}", serde_json::to_string(&s1).unwrap());
    assert_eq!(s1["shell_mode"], "won-quantile");
    assert_eq!(s1["bias_mode"], "zero");
    assert_eq!(s1["shell_target_shift"].as_f64().unwrap() as f32, 0.005f32);
    assert_eq!(s1["grown_bias"], serde_json::json!([[0.0], [0.0]]));
    assert_eq!(s1["shell_witness"]["rule"], serde_json::json!([["won-quantile"], ["won-quantile"]]));
    assert!(s1["reshell"].is_null());
    let e0 = 4usize;
    let ck1 = cortiq_embryo::train::load_checkpoint(&out_ckpt).unwrap();
    assert_eq!(ck1.cfg.experts, e0 + 1);
    assert_eq!(grown_layers_of(&ck1, e0).unwrap(), vec![2, 3]);
    // the saved checkpoint is the TRAINED one: the record's expert tensors
    // are its slices, and the grown experts differ from their untrained copies
    let rec_w = record_tensors(&out_w, "herbs");
    let lay1 = Layout::new(&ck1.cfg);
    let bn = by_name(&lay1);
    let (h, i) = (ck1.cfg.hidden, ck1.cfg.inter);
    for &l in &[2usize, 3] {
        let (og, _) = bn[format!("layers.{l}.experts.{e0}.gate").as_str()];
        let (ou, _) = bn[format!("layers.{l}.experts.{e0}.up").as_str()];
        let (od, _) = bn[format!("layers.{l}.experts.{e0}.down").as_str()];
        let pre = format!("skill.herbs.model.layers.{l}.mlp.experts.{e0}.");
        assert_eq!(rec_w[&format!("{pre}gate_proj.weight")], ck1.params[og..og + i * h]);
        assert_eq!(rec_w[&format!("{pre}up_proj.weight")], ck1.params[ou..ou + i * h]);
        assert_eq!(rec_w[&format!("{pre}down_proj.weight")], ck1.params[od..od + h * i]);
        let e1 = ck1.cfg.experts;
        assert_eq!(rec_w[&format!("{pre}desc.mu")], desc(&ck1, "desc.mu")[(l * e1 + e0) * h..(l * e1 + e0 + 1) * h]);
        assert_eq!(rec_w[&format!("{pre}desc.bias")], vec![0.0]);
        let src = s1["sources"][[2usize, 3].iter().position(|&x| x == l).unwrap()][0].as_u64().unwrap() as usize;
        let (sg, _) = bn[format!("layers.{l}.experts.{src}.gate").as_str()];
        assert_ne!(ck1.params[og..og + i * h], ck1.params[sg..sg + i * h], "layer {l}: the grown expert trained");
    }
    // the E0 shrink of the trained checkpoint is the genome checkpoint (f32 genome)
    let back = shrink_experts(&ck1, e0).unwrap();
    assert_eq!(back.cfg.experts, e0);
    assert_eq!(back.params, f.ck.params);
    for (n, x) in &back.extras {
        assert_eq!(x, desc(&f.ck, n), "{n} of the shrunk checkpoint");
    }
    // ---- 2. reshell, won-quantile: the same record ----
    let out_r = f.dir.join("F1-reshell.cmf");
    let mut r = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    r.record_out = Some(out_r.clone());
    let s2 = reshell(&r).unwrap();
    eprintln!("reshell: {}", serde_json::to_string(&s2).unwrap());
    assert_eq!(s2["reshell"], true);
    assert_eq!(s2["K"], 1);
    assert_eq!(s2["layers"], serde_json::json!([2, 3]));
    assert_eq!(s2["sources"], s1["sources"]);
    assert_eq!(s2["trunk_corpus_wins"], s1["trunk_corpus_wins"]);
    assert_eq!(s2["bias_mode"], "zero");
    assert_eq!(s2["shell_mode"], "won-quantile");
    assert_eq!(s2["shells"], s1["shells"]);
    assert_eq!(s2["wins"], s1["wins"]);
    assert_eq!(s2["coverage"], s1["coverage"]);
    assert_eq!(s2["routing_shift"], s1["routing_shift"]);
    assert_eq!(s2["served"], s1["served"]);
    assert_eq!(s2["record"]["experts"], s1["record"]["experts"]);
    // (appended_bytes differ: the origin JSON of a reshell names the
    // checkpoint instead of the training)
    let rec_r = record_tensors(&out_r, "herbs");
    assert_eq!(rec_r.keys().collect::<Vec<_>>(), rec_w.keys().collect::<Vec<_>>());
    for (name, v) in &rec_w {
        let w = &rec_r[name];
        if name.ends_with(".desc.shell") {
            assert!((v[0] - w[0]).abs() <= 1e-6, "{name}: {} vs {}", v[0], w[0]);
        } else {
            assert!(
                v.len() == w.len() && v.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{name}: reshell's record tensor differs from grow's"
            );
        }
    }
    let m_r = CmfModel::open(&out_r).unwrap();
    assert_eq!(m_r.trunk_hash(), CmfModel::open(&f.base).unwrap().trunk_hash());
    let origin = m_r.header.skills[0].origin.clone().unwrap();
    assert_eq!(origin["reshell"], true);
    assert_eq!(origin["shell_mode"], "won-quantile");
    assert_eq!(origin["bias_mode"], "zero");
    assert_eq!(origin["shell_target_shift"].as_f64().unwrap() as f32, 0.005f32);
    assert_eq!(origin["sources"], s1["sources"]);
    drop(m_r);
    // ---- 3. reshell, general-target: the shell admits at most the target ----
    let per_no = f64s(&s1["routing_shift"]["per_layer_noshell"]);
    let per_won = f64s(&s1["routing_shift"]["per_layer_shell"]);
    let n_gen = s1["routing_shift"]["tokens"].as_u64().unwrap() as f64;
    let min_no = per_no.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        min_no > 0.0,
        "the grown copy wins no general token in some layer ({per_no:?}): the fixture cannot exercise general-target"
    );
    let target = (min_no / 2.0) as f32;
    let out_g = f.dir.join("F1-gt.cmf");
    let mut rg = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    rg.shell_mode = ShellMode::GeneralTarget;
    rg.shell_target_shift = target;
    rg.record_out = Some(out_g.clone());
    let s3 = reshell(&rg).unwrap();
    eprintln!("reshell general-target {target}: {}", serde_json::to_string(&s3).unwrap());
    assert_eq!(s3["shell_mode"], "general-target");
    assert_eq!(s3["shell_target_shift"].as_f64().unwrap() as f32, target);
    assert_eq!(s3["shell_witness"]["rule"], serde_json::json!([["general-target"], ["general-target"]]));
    assert_eq!(f64s(&s3["routing_shift"]["per_layer_noshell"]), per_no, "the no-shell shift never changes");
    let per_gt = f64s(&s3["routing_shift"]["per_layer_shell"]);
    let tol = 2.0 / n_gen + 1e-6;
    for (li, &sh) in per_gt.iter().enumerate() {
        assert!(
            sh <= target as f64 + tol,
            "layer {}: general-target shell shift {sh} > target {target} + {tol}",
            [2, 3][li]
        );
        assert!(sh <= per_won[li], "layer {}: general-target admits more than the won-quantile shell", [2, 3][li]);
        assert!(sh < per_no[li]);
    }
    assert!(s3["routing_shift_shell"].as_f64().unwrap() <= 2.0 * target as f64 + 2.0 * tol);
    let share = f64s(&s3["shell_witness"]["general_share"][0]);
    assert!((share[0] - per_no[0]).abs() < 1e-6, "share = the no-shell per-layer shift");
    let shells_gt: Vec<Vec<f32>> = serde_json::from_value(s3["shells"].clone()).unwrap();
    let shells_w: Vec<Vec<f32>> = serde_json::from_value(s1["shells"].clone()).unwrap();
    assert!(shells_gt.iter().zip(&shells_w).all(|(g, w)| g[0] <= w[0]), "{shells_gt:?} vs {shells_w:?}");
    let rec_g = record_tensors(&out_g, "herbs");
    for (name, v) in &rec_w {
        if !name.ends_with(".desc.shell") {
            assert!(v.iter().zip(&rec_g[name]).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}");
        }
    }
    let m_g = CmfModel::open(&out_g).unwrap();
    let origin = m_g.header.skills[0].origin.clone().unwrap();
    assert_eq!(origin["shell_mode"], "general-target");
    assert_eq!(m_g.header.skills[0].quality.as_ref().unwrap()["shell_mode"], "general-target");
    drop(m_g);
    // ---- 4. grow itself in general-target mode (the default of the CLI) ----
    let out_gg = f.dir.join("F1-grow-gt.cmf");
    let mut ag = grow_cli(&f, &corpus, &held, Some(&general));
    ag.shell_mode = ShellMode::GeneralTarget;
    ag.shell_target_shift = target;
    ag.record_out = Some(out_gg.clone());
    let s4 = grow(&ag).unwrap();
    assert_eq!(s4["shell_mode"], "general-target");
    assert_eq!(s4["shell_witness"]["rule"], serde_json::json!([["general-target"], ["general-target"]]));
    for &sh in &f64s(&s4["routing_shift"]["per_layer_shell"]) {
        assert!(sh <= target as f64 + tol, "grow general-target: shell shift {sh} > target {target} + {tol}");
    }
    // ---- refusals ----
    let mut bad = grow_cli(&f, &corpus, &held, None);
    bad.shell_mode = ShellMode::GeneralTarget;
    let e = grow(&bad).err().unwrap().to_string();
    assert!(e.contains("--general"), "{e}");
    let mut bad = reshell_cli(&f, &out_ckpt, &corpus, &held, None);
    bad.shell_mode = ShellMode::GeneralTarget;
    let e = reshell(&bad).err().unwrap().to_string();
    assert!(e.contains("--general"), "{e}");
    let mut bad = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    bad.layers = Some(vec![1, 3]);
    let e = reshell(&bad).err().unwrap().to_string();
    assert!(e.contains("live grown slots"), "{e}");
    let mut bad = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    bad.shell_target_shift = 1.5;
    assert!(reshell(&bad).is_err());
    let mut bad = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    bad.bias_mode = Some(BiasMode::Source);
    let e = reshell(&bad).err().unwrap().to_string();
    assert!(e.contains("--bias-mode source"), "{e}");
    // the genome checkpoint itself is not a grown one
    let mut bad = reshell_cli(&f, &f.ckpt, &corpus, &held, Some(&general));
    bad.record_out = Some(f.dir.join("F-none.cmf"));
    let e = reshell(&bad).err().unwrap().to_string();
    assert!(e.contains("not a grown checkpoint"), "{e}");
    // a grown checkpoint of another genome (trunk perturbed)
    let mut foreign = Checkpoint {
        cfg: ck1.cfg.clone(),
        step: ck1.step,
        params: ck1.params.clone(),
        m: None,
        v: None,
        extras: ck1.extras.clone(),
    };
    foreign.params[0] += 1.0;
    let fp = f.dir.join("foreign.ckpt");
    let ex: Vec<(&str, &[f32])> = foreign.extras.iter().map(|(n, x)| (n.as_str(), x.as_slice())).collect();
    save_checkpoint(&fp, &foreign.cfg, foreign.step, &foreign.params, None, None, &ex).unwrap();
    let mut bad = reshell_cli(&f, &fp, &corpus, &held, Some(&general));
    bad.record_out = Some(f.dir.join("F-foreign.cmf"));
    let e = reshell(&bad).err().unwrap().to_string();
    assert!(e.contains("not the checkpoint") || e.contains("not a grown checkpoint"), "{e}");
    assert!(!f.dir.join("F-none.cmf").exists() && !f.dir.join("F-foreign.cmf").exists());
    let _ = std::fs::remove_dir_all(&f.dir);
}

/// `--bias-mode source`: the grown expert starts at its SOURCE trunk
/// expert's balancing bias, keeps it through training (frozen) and the
/// record stores that value (the core accepts a finite bias); the runtime
/// mounts it as loaded; `reshell` infers the mode from the checkpoint.
#[test]
fn bias_mode_source_writes_the_source_bias() {
    let Some(f0) = fixture("bias") else {
        eprintln!("no GPU device: skipped");
        return;
    };
    // a genome whose trunk experts carry DISTINCT negative balancing biases
    let mut ck = Checkpoint {
        cfg: f0.ck.cfg.clone(),
        step: f0.ck.step,
        params: f0.ck.params.clone(),
        m: None,
        v: None,
        extras: f0.ck.extras.clone(),
    };
    let e0 = ck.cfg.experts;
    for (n, x) in ck.extras.iter_mut() {
        if n == "desc.bias" {
            for (j, b) in x.iter_mut().enumerate() {
                *b = -0.05 * ((j % e0) as f32 + 1.0) - 0.01 * (j / e0) as f32;
            }
        }
    }
    let ckpt = f0.dir.join("genome-b.ckpt");
    let ex: Vec<(&str, &[f32])> = ck.extras.iter().map(|(n, x)| (n.as_str(), x.as_slice())).collect();
    save_checkpoint(&ckpt, &ck.cfg, ck.step, &ck.params, None, None, &ex).unwrap();
    let base = f0.dir.join("base-b.cmf");
    cortiq_embryo::export::export_genome(
        &ck,
        f0.tok_json.as_bytes(),
        &base,
        cortiq_core::TensorDtype::F32,
        Some(&cortiq_embryo::export::ExportGenome {
            id: "embryo-o1-fam-a-tiny-b".into(),
            status: "pre_chat".into(),
        }),
    )
    .unwrap();
    let f = Fixture {
        dir: f0.dir.clone(),
        ck,
        tok_json: f0.tok_json.clone(),
        tok_path: f0.tok_path.clone(),
        ckpt,
        base,
        bpe: Bpe::load(&f0.tok_path).unwrap(),
    };
    let corpus = f.dir.join("herbs.txt");
    std::fs::write(&corpus, herbs_text(10)).unwrap();
    let held = f.dir.join("herbs-held.txt");
    std::fs::write(&held, herbs_text(3).replace("tea", "infusion")).unwrap();
    let general = f.dir.join("general.u16");
    encode(&f.bpe, &general_text(6)).save(&general).unwrap();
    let out = f.dir.join("F1-src.cmf");
    let out_ckpt = f.dir.join("grown-src.ckpt");
    let mut a = grow_cli(&f, &corpus, &held, Some(&general));
    a.layers = Some(vec![1, 3]);
    a.bias_mode = BiasMode::Source;
    a.record_out = Some(out.clone());
    a.out_ckpt = Some(out_ckpt.clone());
    let s = grow(&a).unwrap();
    eprintln!("grow (bias source): {}", serde_json::to_string(&s).unwrap());
    assert_eq!(s["bias_mode"], "source");
    let layers = [1usize, 3];
    let b0 = desc(&f.ck, "desc.bias");
    let want: Vec<f32> = layers
        .iter()
        .enumerate()
        .map(|(li, &l)| b0[l * e0 + s["sources"][li][0].as_u64().unwrap() as usize])
        .collect();
    assert!(want.iter().all(|b| *b != 0.0 && b.is_finite()), "{want:?}");
    let got: Vec<Vec<f32>> = serde_json::from_value(s["grown_bias"].clone()).unwrap();
    assert_eq!(got, want.iter().map(|b| vec![*b]).collect::<Vec<_>>());
    // frozen through training: the checkpoint carries it
    let ck1 = cortiq_embryo::train::load_checkpoint(&out_ckpt).unwrap();
    let e1 = ck1.cfg.experts;
    let b1 = desc(&ck1, "desc.bias");
    for (li, &l) in layers.iter().enumerate() {
        assert_eq!(b1[l * e1 + e0].to_bits(), want[li].to_bits(), "layer {l}: the grown bias moved");
        assert_eq!(&b1[l * e1..l * e1 + e0], &b0[l * e0..(l + 1) * e0], "layer {l}: trunk bias moved");
    }
    // the record stores it and the core accepts it
    let rec = record_tensors(&out, "herbs");
    for (li, &l) in layers.iter().enumerate() {
        assert_eq!(rec[&format!("skill.herbs.model.layers.{l}.mlp.experts.{e0}.desc.bias")], vec![want[li]]);
    }
    let m1 = CmfModel::open(&out).unwrap();
    assert_eq!(m1.trunk_hash(), CmfModel::open(&f.base).unwrap().trunk_hash());
    cortiq_embryo::skill::verify_append(&f.base, &out).unwrap();
    let origin = m1.header.skills[0].origin.clone().unwrap();
    assert_eq!(origin["bias_mode"], "source");
    assert_eq!(origin["grown_bias"], s["grown_bias"]);
    assert_eq!(m1.header.skills[0].quality.as_ref().unwrap()["bias_mode"], "source");
    drop(m1);
    // the runtime mounts the bias as loaded (per-op resonance and the graph read the same vector)
    // SAFETY: before the pipeline that reads it is built; one test thread
    unsafe { std::env::set_var("CMF_GROWTH", "all") };
    let m = std::sync::Arc::new(CmfModel::open(&out).unwrap());
    let p = cortiq_engine::pipeline::Pipeline::from_model(&m, cortiq_engine::sampler::SamplerConfig::default()).unwrap();
    for (li, &l) in layers.iter().enumerate() {
        let r = match &p.weights.layers[l].ffn {
            cortiq_engine::pipeline::FfnKind::Moe(m) => m.resonance.as_ref().unwrap(),
            _ => panic!("layer {l} is not MoE"),
        };
        assert_eq!(r.bias.len(), e0 + 1);
        assert_eq!(r.bias[e0].to_bits(), want[li].to_bits(), "layer {l}: mounted bias");
        assert!(r.shell[e0].is_finite() && r.shell[..e0].iter().all(|s| s.is_infinite()));
    }
    drop(p);
    unsafe { std::env::remove_var("CMF_GROWTH") };
    // reshell infers `source` from the checkpoint and writes the same bias
    let out_r = f.dir.join("F1-src-reshell.cmf");
    let mut r = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    r.record_out = Some(out_r.clone());
    let s2 = reshell(&r).unwrap();
    assert_eq!(s2["bias_mode"], "source");
    assert_eq!(s2["grown_bias"], s["grown_bias"]);
    assert_eq!(s2["layers"], serde_json::json!([1, 3]));
    let rec_r = record_tensors(&out_r, "herbs");
    for (name, v) in &rec {
        if !name.ends_with(".desc.shell") {
            assert!(v.iter().zip(&rec_r[name]).all(|(a, b)| a.to_bits() == b.to_bits()), "{name}");
        }
    }
    // and refuses the other mode
    let mut bad = reshell_cli(&f, &out_ckpt, &corpus, &held, Some(&general));
    bad.bias_mode = Some(BiasMode::Zero);
    let e = reshell(&bad).err().unwrap().to_string();
    assert!(e.contains("--bias-mode zero"), "{e}");
    // zero mode on this genome: the grown bias is 0 while the sources' are not
    let out_z = f.dir.join("F1-zero.cmf");
    let mut az = grow_cli(&f, &corpus, &held, Some(&general));
    az.layers = Some(vec![1, 3]);
    az.record_out = Some(out_z.clone());
    let sz = grow(&az).unwrap();
    assert_eq!(sz["bias_mode"], "zero");
    assert_eq!(sz["grown_bias"], serde_json::json!([[0.0], [0.0]]));
    let _ = std::fs::remove_dir_all(&f.dir);
}
