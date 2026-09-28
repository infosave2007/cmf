//! Skill bake for the Embryo (DTG-MA / P2 + P15 record): from a genome
//! checkpoint and a task corpus, train a neuron mask over the shared FFN of
//! selected layers to its denoising bottom (phase A, L1 pressure, held-out
//! gate), polish those FFNs under the hard mask (phase B), fold the mask
//! into the tensors, fit the recon-argmin routing descriptor (P1) and
//! append the record `skill.{id}.*` to the .cmf — every pre-existing tensor
//! byte-for-byte unchanged (the directory hashes prove it).
//!
//! Two paths:
//! - v2 ([`bake_v2`], format v2 "knowledge without forgetting"): a
//!   `ffn_replace` record bound to the FROZEN genome of `--base` (bit
//!   GENOME; the checkpoint is re-exported in memory and must hash to the
//!   base's trunk; an f16 genome is probed and polished on its served,
//!   f16-rounded trunk), trained response-only on SFT shards (optional
//!   raw-LM rows mixed per batch) on the dropless MoE operator the runtime
//!   executes, selected on the answer NLL of the WHOLE dev shard (verified
//!   record-disjoint from train, group split checked against the
//!   `sft-prepare` manifest), routed by the backbone-gated router v2
//!   (canonical span-mean φ over the user text encoded by the runtime's
//!   tokenizer, a backbone descriptor fitted once, calibration with the
//!   backbone as a class; earlier active records go to `stale_regate`),
//!   replayed through the runtime before publishing, and written by a true
//!   tail append onto a private copy of the base, published without
//!   overwriting;
//! - legacy ([`bake`] + [`append_to_cmf`]): flat LM corpus, v1 record,
//!   full rewrite — only for files without a genome / router v2 / v2
//!   records ([`refuse_knowledge_file`]).

use crate::model::{EmbryoCfg, EmbryoGpu, LayerOffs, Layout, SkillState};
use crate::sft::{IGNORE, SftShard};
use crate::tokenizer::Bpe;
use crate::train::{Checkpoint, Sampler, Shard};
use cortiq_core::format::{CmfModel, SelectionDescriptor, SkillRecord, TensorSpec};
use cortiq_core::knowledge::{
    BACKBONE_CLASS_ID, LineageEvent, PhiSpec, RouterPolicy, SkillBound, SkillOverride,
    ffn_replace_state_effect, hex64, is_trunk_tensor, skill_kind, trunk_hash,
};
use cortiq_core::types::TensorDtype;
use cortiq_engine::tokenizer::Tokenizer as RuntimeTokenizer;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Initial logit of every mask neuron: σ(3) ≈ 0.953. The all-on hard mask
/// (neuron kept iff σ(m) > τ) is the exact genome only for 0 < τ < σ(3).
pub const SKILL_INIT_LOGIT: f32 = 3.0;

/// σ([`SKILL_INIT_LOGIT`]) — the exclusive upper bound of `--tau`.
pub fn tau_ceiling() -> f32 {
    1.0 / (1.0 + (-SKILL_INIT_LOGIT).exp())
}

/// The legacy (v1, full-rewrite) tools — [`append_to_cmf`],
/// [`calibrate_file`], the sleep daemon and its growth — refuse a file with
/// a frozen genome, a router v2 or v2 records: they rewrite the whole file
/// (append-only and G1 gone), overwrite `header.routing` with a v1-only
/// calibration the v2 decision would then read, or replace the trunk. Such
/// a file changes only through `skill-bake --sft-train …` (tail append);
/// growth is a new genome (a new `genome.id`).
pub fn refuse_knowledge_file(model: &CmfModel, path: &Path, tool: &str) -> anyhow::Result<()> {
    let h = &model.header;
    let what = if let Some(g) = &h.genome {
        format!("the frozen GENOME '{}'", g.id)
    } else if h.router.is_some() {
        "a router v2 (ROUTER_V2)".to_string()
    } else if let Some(s) = h.skills.iter().find(|s| s.is_v2()) {
        format!("the v2 skill record '{}'", s.id)
    } else {
        return Ok(());
    };
    anyhow::bail!(
        "refusing: {} carries {what} — {tool} is a legacy (v1) path that rewrites the whole \
         file and its routing calibration; records over a frozen genome are added only by \
         `skill-bake --sft-train …` (true tail append), and growth is a new genome",
        path.display()
    )
}

/// [`refuse_knowledge_file`] on a path.
pub fn refuse_knowledge_path(path: &Path, tool: &str) -> anyhow::Result<()> {
    let model = CmfModel::open(path)?;
    refuse_knowledge_file(&model, path, tool)
}

pub struct BakeArgs {
    pub id: String,
    pub layers: Vec<usize>,
    pub steps_a: usize,
    pub steps_b: usize,
    pub lr_a: f32,
    pub lr_b: f32,
    pub l1: f32,
    pub tau: f32,
    pub eval_every: usize,
    pub batch: usize,
    pub seq: usize,
    pub phi_layer: usize,
    /// positions averaged for φ (prompt-length statistics, not whole windows)
    pub phi_len: usize,
    pub rank: usize,
    pub seed: u64,
}

fn f16_bits(x: f32) -> u16 {
    // round-to-nearest-even f32 → f16 (finite range; the descriptors are O(1))
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e);
        let round = (m >> 13) as u16
            + if (m & 0x1fff) > 0x1000 || ((m & 0x1fff) == 0x1000 && (m & 0x2000) != 0) {
                1
            } else {
                0
            };
        return sign | round;
    }
    let mut half = sign | ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half += 1;
    }
    half
}

fn f16_base64(x: &[f32]) -> String {
    use base64::Engine;
    let mut bytes = Vec::with_capacity(x.len() * 2);
    for v in x {
        bytes.extend_from_slice(&f16_bits(*v).to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Fit the P1 selection descriptor from φ samples (cortiq-router recipe):
/// mean + rank principal directions (orthonormal rows) on the TRAIN part,
/// training reconstruction-error statistics (err_mean/err_std) for the
/// novelty z-score, and the HELD-OUT φ samples carried in the record so the
/// container can recalibrate temperature/θ over all its skills later.
pub fn fit_selection(phis_raw: &[Vec<f32>], phi_layer: usize, rank: usize) -> SelectionDescriptor {
    // unit-norm φ (the cortiq-router recipe scores unit embeddings: errors
    // and margins live on an O(1) scale its constants were tuned for)
    let phis: Vec<Vec<f32>> = phis_raw
        .iter()
        .map(|p| {
            let n = p.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            p.iter().map(|x| x / n).collect()
        })
        .collect();
    let n = phis.len();
    let h = phis[0].len();
    // 80/20 split, held-out never enters the mean/basis
    let n_hold = holdout_count(n);
    let (train, hold) = phis.split_at(n - n_hold);
    let nt = train.len();
    let mut mean = vec![0f32; h];
    for p in train {
        for (m, v) in mean.iter_mut().zip(p) {
            *m += v / nt as f32;
        }
    }
    let rank = rank.min(nt.saturating_sub(1)).max(1).min(16);
    let mut cov = vec![0f32; h * h];
    for p in train {
        let c: Vec<f32> = p.iter().zip(&mean).map(|(v, m)| v - m).collect();
        for i in 0..h {
            if c[i] == 0.0 {
                continue;
            }
            for j in 0..h {
                cov[i * h + j] += c[i] * c[j] / nt as f32;
            }
        }
    }
    let basis = crate::model::top_eigenvectors(&cov, h, rank, 120, 99);
    // training error distribution
    let errs: Vec<f32> = train
        .iter()
        .map(|p| cortiq_engine::router::recon_error(p, &mean, &basis, rank))
        .collect();
    let em = errs.iter().sum::<f32>() / nt as f32;
    let es = (errs.iter().map(|e| (e - em).powi(2)).sum::<f32>() / nt as f32)
        .sqrt()
        .max(1e-4);
    let mut hold_flat = Vec::with_capacity(hold.len() * h);
    for p in hold {
        hold_flat.extend_from_slice(p);
    }
    SelectionDescriptor {
        metric: "mse_unit".into(),
        phi_layer,
        mean: f16_base64(&mean),
        basis: f16_base64(&basis),
        rank,
        err_mean: Some(em),
        err_std: Some(es),
        holdout: Some(f16_base64(&hold_flat)),
        holdout_n: Some(hold.len()),
    }
}

/// Size of the held-out tail [`fit_selection`] keeps out of the fit (its
/// last `holdout_count(n)` samples, in input order): 20 %, at least 1.
pub fn holdout_count(n: usize) -> usize {
    (n / 5).clamp(1, n.saturating_sub(2).max(1))
}

/// Recalibrate a container's skill router (temperature + θ) from the
/// held-out φ every skill carries; rewrites the header only (tensor bytes
/// copied verbatim). Returns the calibration written.
pub fn calibrate_file(
    path: &Path,
    target_fpr: f32,
) -> anyhow::Result<Option<cortiq_core::format::RoutingCalibration>> {
    let model = CmfModel::open(path)?;
    refuse_knowledge_file(&model, path, "calibrate_file")?;
    let Some(cal) = cortiq_engine::router::calibrate(&model, target_fpr) else {
        return Ok(None);
    };
    let mut header = model.header.clone();
    header.routing = Some(cal.clone());
    let specs: Vec<TensorSpec> = model
        .tensors
        .iter()
        .map(|t| TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model
                .tensor_bytes(&t.name)
                .map(|b| b.to_vec())
                .unwrap_or_default(),
        })
        .collect();
    let tmp = path.with_extension("cmf.cal");
    CmfModel::write(
        &tmp,
        &header,
        &specs,
        if model.masks.masks.is_empty() {
            None
        } else {
            Some(&model.masks)
        },
        model.vocab.as_deref(),
    )?;
    drop(model);
    std::fs::rename(&tmp, path)?;
    Ok(Some(cal))
}

/// Bake: returns (replacement tensors [(runtime tensor name, shape, data)],
/// selection descriptor, kept fraction per layer, held-out losses (base,
/// best A, best B)).
#[allow(clippy::type_complexity)]
pub fn bake(
    ck: &Checkpoint,
    corpus: &Shard,
    a: &BakeArgs,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<(
    Vec<(String, Vec<usize>, Vec<f32>)>,
    SelectionDescriptor,
    Vec<f32>,
    (f32, f32, f32),
)> {
    let cfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    let i = cfg.inter;
    let mut gpu = EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &ck.params)
        .ok_or_else(|| anyhow::anyhow!("no GPU device (Metal / Vulkan)"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false); // the genome's routing state is frozen
    let c = gpu.ctx();
    gpu.skill = Some(SkillState::new(
        c,
        a.layers.clone(),
        i,
        SKILL_INIT_LOGIT,
        a.tau,
    ));
    // corpus split: last 10% held out (never trained on)
    let n = corpus.tokens.len();
    anyhow::ensure!(n > 20 * (a.seq + 2), "skill corpus too small: {n} tokens");
    let cut = n - n / 10;
    let train = Shard {
        tokens: corpus.tokens[..cut].to_vec(),
    };
    let held = Shard {
        tokens: corpus.tokens[cut..].to_vec(),
    };
    let mut sampler = Sampler::new(a.batch, a.seq, a.seed);
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    let m = a.batch * a.seq;
    let nval = (held.tokens.len() / (m + 1)).clamp(1, 4);
    let eval = |gpu: &EmbryoGpu, tk: &mut Vec<u32>, tg: &mut Vec<u32>| -> f32 {
        let mut s = 0.0;
        for k in 0..nval {
            Sampler::fixed_batch(&held, a.batch, a.seq, k, tk, tg);
            s += gpu.eval_loss(tk, tg);
        }
        s / nval as f32
    };
    let t0 = Instant::now();
    // base held-out loss (mask all-on: logits +3 → σ ≈ 0.95; use hard=true for the true base)
    gpu.skill.as_ref().unwrap().hard.set(true);
    let base_loss = eval(&gpu, &mut tk, &mut tg);
    gpu.skill.as_ref().unwrap().hard.set(false);
    eprintln!(
        "skill '{}': base held-out loss {base_loss:.4} (ppl {:.1}); layers {:?}",
        a.id,
        base_loss.exp(),
        a.layers
    );
    // ---- phase A: masks to the denoising bottom ----
    let mut best_a = (f32::MAX, gpu.skill.as_ref().unwrap().logits.to_vec());
    for step in 0..a.steps_a {
        if should_stop() {
            anyhow::bail!("preempted");
        }
        let l1 = a.l1 * (step as f32 / a.steps_a.max(1) as f32); // progressive L1
        gpu.skill.as_ref().unwrap().l1.set(l1);
        sampler.batch(&train, &mut tk, &mut tg);
        let (loss, gn) = gpu.train_step_skill(&tk, &tg, a.lr_a, 0.0, 1.0, false);
        if (step + 1) % a.eval_every == 0 || step + 1 == a.steps_a {
            let sk = gpu.skill.as_ref().unwrap();
            sk.hard.set(true);
            let vl = eval(&gpu, &mut tk, &mut tg);
            sk.hard.set(false);
            let masks = sk.hard_masks();
            let kept: Vec<f32> = masks
                .iter()
                .map(|mk| mk.iter().filter(|&&b| b).count() as f32 / mk.len() as f32)
                .collect();
            eprintln!(
                "  A step {:>4} loss {loss:.4} |g| {gn:.3} l1 {l1:.2e} held-out(hard) {vl:.4} kept {:?} [{:.0} s]",
                step + 1,
                kept.iter().map(|k| format!("{k:.2}")).collect::<Vec<_>>(),
                t0.elapsed().as_secs_f64()
            );
            if vl < best_a.0 {
                best_a = (vl, sk.logits.to_vec());
            }
        }
    }
    // restore the best mask, freeze it hard
    {
        let sk = gpu.skill.as_ref().unwrap();
        sk.logits.write_from(&best_a.1);
        sk.hard.set(true);
        sk.l1.set(0.0);
    }
    let kept: Vec<f32> = gpu
        .skill
        .as_ref()
        .unwrap()
        .hard_masks()
        .iter()
        .map(|mk| mk.iter().filter(|&&b| b).count() as f32 / mk.len() as f32)
        .collect();
    eprintln!(
        "phase A best held-out {:.4} (ppl {:.1}); kept {:?}",
        best_a.0,
        best_a.0.exp(),
        kept
    );
    // ---- phase B: polish of the selected FFNs under the hard mask ----
    let ffn_ranges = ffn_ranges(&cfg, &lay, &a.layers);
    let snapshot = |gpu: &EmbryoGpu| -> Vec<Vec<f32>> {
        let p = gpu.params_host();
        ffn_ranges
            .iter()
            .map(|&(o, n)| p[o..o + n].to_vec())
            .collect()
    };
    let mut best_b = (best_a.0, snapshot(&gpu));
    for step in 0..a.steps_b {
        if should_stop() {
            anyhow::bail!("preempted");
        }
        sampler.batch(&train, &mut tk, &mut tg);
        let lr = a.lr_b
            * 0.5
            * (1.0 + (std::f32::consts::PI * step as f32 / a.steps_b.max(1) as f32).cos());
        let (loss, gn) = gpu.train_step_skill(&tk, &tg, lr, 0.0, 1.0, true);
        if (step + 1) % a.eval_every == 0 || step + 1 == a.steps_b {
            let vl = eval(&gpu, &mut tk, &mut tg);
            eprintln!(
                "  B step {:>4} loss {loss:.4} |g| {gn:.3} lr {lr:.2e} held-out {vl:.4} [{:.0} s]",
                step + 1,
                t0.elapsed().as_secs_f64()
            );
            if vl < best_b.0 {
                best_b = (vl, snapshot(&gpu));
            }
        }
    }
    eprintln!(
        "phase B best held-out {:.4} (ppl {:.1}) vs base {:.4} (ppl {:.1})",
        best_b.0,
        best_b.0.exp(),
        base_loss,
        base_loss.exp()
    );
    // ---- fold the mask into the best tensors, name them for the runtime ----
    let masks = gpu.skill.as_ref().unwrap().hard_masks();
    let out = fold_mask(&cfg, &a.id, &a.layers, &best_b.1, &masks);
    // ---- routing descriptor: φ = mean-pooled hidden AFTER phi_layer over corpus windows ----
    let mut phis: Vec<Vec<f32>> = Vec::new();
    let mut ps = Sampler::new(a.batch, a.seq, a.seed + 7);
    for _ in 0..24 {
        ps.batch(&train, &mut tk, &mut tg);
        phis.extend(gpu.probe_phi(&tk, a.phi_layer, a.phi_len));
    }
    let sel = fit_selection(&phis, a.phi_layer, a.rank);
    Ok((out, sel, kept, (base_loss, best_a.0, best_b.0)))
}

/// Append the skill to a .cmf: every existing tensor copied byte-for-byte,
/// the skill tensors added, the registry record pushed. Returns the number
/// of unchanged tensors (all of the old ones).
pub fn append_to_cmf(
    base: &Path,
    out: &Path,
    id: &str,
    layers: &[usize],
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    sel: SelectionDescriptor,
    quality: serde_json::Value,
) -> anyhow::Result<usize> {
    let model = CmfModel::open(base)?;
    refuse_knowledge_file(&model, base, "append_to_cmf (legacy skill append)")?;
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len() + tensors.len());
    let mut kept = 0usize;
    for t in &model.tensors {
        if t.name.starts_with(&format!("skill.{id}.")) {
            continue;
        }
        specs.push(TensorSpec {
            name: t.name.clone(),
            dtype: t.dtype,
            shape: t.shape.clone(),
            data: model.tensor_bytes(&t.name)?.to_vec(),
        });
        kept += 1;
    }
    for (name, shape, data) in tensors {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        specs.push(TensorSpec {
            name: name.clone(),
            dtype: TensorDtype::F32,
            shape: shape.clone(),
            data: bytes,
        });
    }
    let mut header = model.header.clone();
    header.skills.retain(|s| s.id != id);
    header.skills.push(SkillRecord {
        id: id.to_string(),
        name: None,
        layers: layers.to_vec(),
        selection: Some(sel),
        input_mask_task: None,
        quality: Some(quality),
        base_dir_hash: None,
        kind: None,
        overrides: Vec::new(),
        bound: None,
        state_effect: None,
        status: None,
        gate: None,
        prompt_contract: None,
        origin: None,
        experts: None,
        lookup: None,
        base_arch: Some(model.header.arch.arch_name.clone()),
        task: None,
        provenance: Some(serde_json::json!({"producer": "cortiq-embryo skill-bake", "recipe": "DTG-MA mask (phase A, L1 to the denoising bottom) + polish under the hard mask (phase B), mask folded"})),
    });
    let vocab = model.vocab.clone();
    CmfModel::write(
        out,
        &header,
        &specs,
        if model.masks.masks.is_empty() {
            None
        } else {
            Some(&model.masks)
        },
        vocab.as_deref(),
    )?;
    Ok(kept)
}

// ───────────────────────── shared by both paths ─────────────────────────

/// Arena ranges `(offset, len)` of the shared FFN (gate, up, down) of each
/// selected layer, in `layers` order — the trainable set of phase B.
pub fn ffn_ranges(cfg: &EmbryoCfg, lay: &Layout, layers: &[usize]) -> Vec<(usize, usize)> {
    let (h, i) = (cfg.hidden, cfg.inter);
    layers
        .iter()
        .flat_map(|&l| {
            let ffn = match &lay.layers[l] {
                LayerOffs::Mixer { ffn, .. }
                | LayerOffs::Anchor { ffn, .. }
                | LayerOffs::Gdn { ffn, .. } => ffn,
            };
            [(ffn.wg, i * h), (ffn.wu, i * h), (ffn.wd, h * i)]
        })
        .collect()
}

/// Runtime name (without the `skill.{id}.` prefix) of the shared-FFN tensor
/// `proj` (`gate_proj` | `up_proj` | `down_proj`) of layer `l`.
pub fn ffn_tensor_name(cfg: &EmbryoCfg, l: usize, proj: &str) -> String {
    if cfg.experts > 0 {
        format!("model.layers.{l}.mlp.shared_expert.{proj}.weight")
    } else {
        format!("model.layers.{l}.mlp.{proj}.weight")
    }
}

/// Fold the hard masks into the FFN snapshots (`snaps` = 3 per layer, the
/// [`ffn_ranges`] order): a dropped neuron j zeroes row j of gate/up and
/// column j of down. Returns `(skill.{id}.X, shape, data)` per tensor.
pub fn fold_mask(
    cfg: &EmbryoCfg,
    id: &str,
    layers: &[usize],
    snaps: &[Vec<f32>],
    masks: &[Vec<bool>],
) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let (h, i) = (cfg.hidden, cfg.inter);
    let mut out = Vec::with_capacity(layers.len() * 3);
    for (li, &l) in layers.iter().enumerate() {
        let (mut wg, mut wu, mut wd) = (
            snaps[li * 3].clone(),
            snaps[li * 3 + 1].clone(),
            snaps[li * 3 + 2].clone(),
        );
        for j in 0..i {
            if !masks[li][j] {
                wg[j * h..(j + 1) * h].fill(0.0);
                wu[j * h..(j + 1) * h].fill(0.0);
                for r in 0..h {
                    wd[r * i + j] = 0.0;
                }
            }
        }
        let name = |proj: &str| format!("skill.{id}.{}", ffn_tensor_name(cfg, l, proj));
        out.push((name("gate_proj"), vec![i, h], wg));
        out.push((name("up_proj"), vec![i, h], wu));
        out.push((name("down_proj"), vec![h, i], wd));
    }
    out
}

fn kept_fractions(masks: &[Vec<bool>]) -> Vec<f32> {
    masks
        .iter()
        .map(|mk| mk.iter().filter(|&&b| b).count() as f32 / mk.len().max(1) as f32)
        .collect()
}

// ───────────────────────── v2: records over a frozen genome ─────────────────────────

/// The prompt template the v2 records are trained and routed under.
pub const PROMPT_CONTRACT: &str = "cmf-im-v1";
/// `origin.recipe` of a v2 record.
pub const RECIPE_V2: &str = "DTG-MA mask (phase A) + polish under the hard mask (phase B)";
/// Fewest prompts per φ class (skill / backbone) a descriptor is fitted on.
pub const PHI_MIN_PROMPTS: usize = 8;

/// Hyper-parameters of [`bake_v2`].
#[derive(Clone, Debug)]
pub struct BakeV2Args {
    pub id: String,
    /// layers whose shared FFN the record replaces (sorted, distinct)
    pub layers: Vec<usize>,
    pub steps_a: usize,
    pub steps_b: usize,
    pub lr_a: f32,
    pub lr_b: f32,
    /// final L1 pressure on σ(m) (ramps from 0 over phase A); ≥ 0
    pub l1: f32,
    /// hard-mask threshold on σ(m): 0 < τ < σ([`SKILL_INIT_LOGIT`]), so the
    /// all-on start is the exact genome and a mask can drop a neuron
    pub tau: f32,
    pub eval_every: usize,
    /// rows per batch (SFT rows + raw-LM rows)
    pub batch: usize,
    /// dev evaluation size: 0 = every record of `--sft-dev` (default);
    /// N > 0 = N·batch records spread evenly over the WHOLE shard
    /// ([`eval_records`], never a prefix). Raw-LM dev: N batches of windows
    /// ([`LM_DEV_BATCHES`] when 0)
    pub dev_batches: usize,
    /// fraction of each training batch's rows drawn from `--lm-train`
    pub lm_frac: f32,
    /// φ = hidden AFTER this layer; must be < min(layers)
    pub phi_layer: usize,
    /// prompts per φ class (deterministic sample)
    pub phi_max: usize,
    /// prompts per φ probe batch (rows)
    pub phi_batch: usize,
    /// longest canonical φ input (prefix + q + suffix) probed; longer skipped
    pub phi_max_len: usize,
    pub rank: usize,
    /// a skill must beat the backbone by this much in unit error; `None`
    /// keeps the margin of the router `--base` already carries (0 for the
    /// first record)
    pub route_margin: Option<f32>,
    /// refit the backbone descriptor from `--general-prompts` although
    /// `--base` already routes (otherwise the base's descriptor is kept:
    /// it is fitted once, at the first record)
    pub refit_base: bool,
    pub target_fpr: f32,
    pub seed: u64,
}

/// Files of a v2 bake. Every data file's sha256 lands in `origin`.
#[derive(Clone, Debug)]
pub struct BakeV2Inputs {
    pub ckpt: PathBuf,
    pub base: PathBuf,
    pub out: PathBuf,
    /// must equal the base's VOCAB section when given
    pub tokenizer: Option<PathBuf>,
    pub sft_train: PathBuf,
    pub sft_dev: PathBuf,
    pub sft_final: Option<PathBuf>,
    /// the `sft-prepare --messages` manifest of the three shards: verifies
    /// the group split (group → split map) and binds it to the shards
    pub sft_manifest: Option<PathBuf>,
    pub lm_train: Option<PathBuf>,
    pub lm_dev: Option<PathBuf>,
    pub phi_prompts: PathBuf,
    /// backbone φ class; required for the first record (a `--base` without
    /// a router) and with `refit_base`, otherwise unused
    pub general_prompts: Option<PathBuf>,
}

/// The token data of a v2 bake.
pub struct BakeV2Data {
    pub sft_train: SftShard,
    pub sft_dev: SftShard,
    pub sft_final: Option<SftShard>,
    pub lm_train: Option<Shard>,
    pub lm_dev: Option<Shard>,
}

/// Dev numbers at one point of the bake (response-only answer NLL on the
/// group-disjoint dev shard; raw-LM NLL on `--lm-dev` when given).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevPoint {
    pub answer_nll: f32,
    pub lm_nll: Option<f32>,
    /// phase step the point was measured at (0 = before training)
    pub step: usize,
}

/// Raw-LM rows per training batch: `round(batch · lm_frac)`, at least one
/// SFT row kept; 0 without an LM shard.
pub fn lm_rows(batch: usize, lm_frac: f32, has_lm: bool) -> usize {
    if has_lm {
        ((batch as f32 * lm_frac).round() as usize).min(batch.saturating_sub(1))
    } else {
        0
    }
}

/// What the GPU part of the v2 bake produced.
pub struct BakeV2Trained {
    /// `(skill.{id}.X, shape, f32 data)`, mask folded
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub kept: Vec<f32>,
    /// kept-neuron bits per layer (the folded mask)
    pub masks: Vec<Vec<bool>>,
    pub base: DevPoint,
    pub best_a: DevPoint,
    pub best_b: DevPoint,
    /// terminal `--sft-final` answer NLL (base, skill); never used for selection
    pub final_nll: Option<(f32, f32)>,
    /// sequence length and raw-LM rows of the training batches
    pub seq: usize,
    pub lm_rows: usize,
    /// dev records every evaluation covered, of the shard's records
    pub dev_records: (usize, usize),
    /// `--sft-final` records evaluated (always the whole shard)
    pub final_records: Option<usize>,
    /// MoE capacity of training and every evaluation: `"dropless"` (every
    /// token keeps its routed expert, as in the runtime) or `"none"` (no
    /// routed experts)
    pub moe_capacity: &'static str,
}

/// Canonical φ frame of cmf-im-v1 (spec §4): the ids before and after the
/// user text — `encode("<|im_start|>user\n")` and
/// `encode("<|im_end|>\n<|im_start|>assistant\n")` with the added tokens.
/// This is exactly the single-turn prompt of the SFT records (sft.rs).
pub fn phi_spec(bpe: &Bpe, layer: usize) -> anyhow::Result<PhiSpec> {
    for s in ["<|im_start|>", "<|im_end|>"] {
        anyhow::ensure!(bpe.special_id(s).is_some(), "tokenizer has no {s}");
    }
    let mut cache = HashMap::new();
    let (mut prefix_ids, mut suffix_ids) = (Vec::new(), Vec::new());
    bpe.encode_with_specials("<|im_start|>user\n", &mut cache, &mut prefix_ids);
    bpe.encode_with_specials(
        "<|im_end|>\n<|im_start|>assistant\n",
        &mut cache,
        &mut suffix_ids,
    );
    Ok(PhiSpec {
        layer,
        pool: "span_mean".into(),
        norm: "unit".into(),
        prefix_ids,
        suffix_ids,
    })
}

/// `"prompt"` of every JSONL line (lines without one are skipped).
pub fn read_prompts(path: &Path) -> anyhow::Result<Vec<String>> {
    Ok(read_prompts_lines(path)?
        .into_iter()
        .map(|(_, p)| p)
        .collect())
}

/// [`read_prompts`] with the 1-based line number of every prompt — what
/// `origin.phi` names the calibration holdouts by.
pub fn read_prompts_lines(path: &Path) -> anyhow::Result<Vec<(usize, String)>> {
    let f = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("open prompts {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (no, line) in std::io::BufRead::lines(std::io::BufReader::new(f)).enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| anyhow::anyhow!("{}:{}: {e}", path.display(), no + 1))?;
        if let Some(p) = v.get("prompt").and_then(|p| p.as_str()) {
            if !p.trim().is_empty() {
                out.push((no + 1, p.to_string()));
            }
        }
    }
    Ok(out)
}

/// The tokenizer the RUNTIME builds from a file's VOCAB section
/// (`cortiq_engine::tokenizer::Tokenizer`, added tokens split out of raw
/// text) — the one φ's user text is encoded with (spec §4: exactly
/// `Tokenizer::encode(q)`, the call `route_request` makes).
pub fn runtime_tokenizer(vocab: &[u8]) -> anyhow::Result<RuntimeTokenizer> {
    RuntimeTokenizer::from_bytes(vocab).map_err(|e| anyhow::anyhow!("VOCAB tokenizer: {e}"))
}

/// Ids of a user text for φ: `Tokenizer::encode(q)` of the runtime (no BOS;
/// a literal `<|im_end|>` in the text is the one special id, as served).
pub fn user_text_ids(tok: &RuntimeTokenizer, q: &str) -> Vec<u32> {
    tok.encode(q)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic sample of at most `max` distinct prompts, shuffled (so the
/// descriptor's 20 % holdout tail is a random subset, not the file's tail).
pub fn sample_prompts(prompts: &[String], max: usize, seed: u64) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut v: Vec<String> = prompts
        .iter()
        .filter(|p| seen.insert(p.as_str()))
        .cloned()
        .collect();
    let mut st = seed ^ 0x5048_4953_414d_504c;
    for k in (1..v.len()).rev() {
        let j = (splitmix(&mut st) % (k as u64 + 1)) as usize;
        v.swap(k, j);
    }
    v.truncate(max);
    v
}

/// Streaming SHA-256 of a file, lowercase hex.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let mut f =
        std::fs::File::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = std::io::Read::read(&mut f, &mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `--out` must be a new file distinct from `--base` (compared
/// canonically): a bake never rewrites or overwrites its genome file. The
/// work file is a fresh private temp ([`create_bake_tmp`], never an
/// existing file — so never `--base`, whatever its name) and the result is
/// published by [`publish_new_file`], which refuses an `--out` that
/// appeared meanwhile.
pub fn check_out_path(base: &Path, out: &Path) -> anyhow::Result<()> {
    let base_c = std::fs::canonicalize(base)
        .map_err(|e| anyhow::anyhow!("--base {}: {e}", base.display()))?;
    let name = out
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("--out {} names no file", out.display()))?;
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let parent_c = std::fs::canonicalize(&parent)
        .map_err(|e| anyhow::anyhow!("--out directory {}: {e}", parent.display()))?;
    let out_c = parent_c.join(name);
    anyhow::ensure!(
        out_c != base_c,
        "refusing: --out {} is --base {} (the genome file is never rewritten; write a new file)",
        out.display(),
        base.display()
    );
    anyhow::ensure!(
        !out.exists(),
        "refusing to overwrite existing --out {}",
        out.display()
    );
    Ok(())
}

/// Refuse when two roles name the same file (held-out sets are separate
/// files, never the train file itself).
fn distinct_files(files: &[(&str, Option<&PathBuf>)]) -> anyhow::Result<()> {
    let mut seen: Vec<(&str, PathBuf)> = Vec::new();
    for &(flag, p) in files {
        let Some(p) = p else { continue };
        let c =
            std::fs::canonicalize(p).map_err(|e| anyhow::anyhow!("{flag} {}: {e}", p.display()))?;
        if let Some((other, _)) = seen.iter().find(|(_, q)| *q == c) {
            anyhow::bail!("{flag} and {other} are the same file ({})", p.display());
        }
        seen.push((flag, c));
    }
    Ok(())
}

/// The checkpoint ↔ base binding of a v2 bake.
pub struct CkptBinding {
    /// trunk tensors verified equal (name, dtype, shape, hash64)
    pub trunk_tensors: usize,
    /// hash64 of the f32 master bytes per exported tensor name (the
    /// `overrides[].base_hash` of a record)
    pub master_hashes: HashMap<String, u64>,
}

/// Re-export `ck` in memory with the base genome's encoding and refuse
/// unless every trunk tensor of `--base` (name, dtype, shape, hash64) is
/// exactly the checkpoint's, and the f32 re-export hashes to the genome's
/// `master_trunk_hash` (which also covers arch, vocab, tokenizer config).
pub fn bind_ckpt_to_base(ck: &Checkpoint, base: &CmfModel) -> anyhow::Result<CkptBinding> {
    let g = base.header.genome.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "--base carries no GENOME: export the genome with `export --genome-id <id> \
             --genome-status <status>` first (a v2 skill binds to a frozen genome)"
        )
    })?;
    let vocab = base
        .vocab
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("--base has no VOCAB section (tokenizer)"))?;
    let enc = match g.encoding.as_str() {
        "f32" => TensorDtype::F32,
        "f16" => TensorDtype::F16,
        e => anyhow::bail!(
            "genome '{}' is encoded as {e}; a bake binds to the f32/f16 export of its checkpoint",
            g.id
        ),
    };
    let (hdr32, specs32) = crate::export::build_export(ck, vocab, TensorDtype::F32, None)?;
    let master_hashes: HashMap<String, u64> = specs32
        .iter()
        .map(|t| (t.name.clone(), cortiq_core::hash64(&t.data)))
        .collect();
    let specs_enc = if enc == TensorDtype::F32 {
        None
    } else {
        Some(crate::export::build_export(ck, vocab, enc, None)?.1)
    };
    let specs = specs_enc.as_ref().unwrap_or(&specs32);
    let by_name: HashMap<&str, &TensorSpec> = specs.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut bad: Vec<String> = Vec::new();
    let mut n = 0usize;
    for e in base.tensors.iter().filter(|t| is_trunk_tensor(&t.name)) {
        n += 1;
        match by_name.get(e.name.as_str()) {
            None => bad.push(format!("{}: not produced by the checkpoint", e.name)),
            Some(s) if s.dtype != e.dtype || s.shape != e.shape => bad.push(format!(
                "{}: {:?}{:?} in the checkpoint, {:?}{:?} in the base",
                e.name, s.dtype, s.shape, e.dtype, e.shape
            )),
            Some(s) if cortiq_core::hash64(&s.data) != e.hash => {
                bad.push(format!("{}: bytes differ", e.name))
            }
            Some(_) => {}
        }
    }
    for s in specs {
        if base.tensor(&s.name).is_none() {
            bad.push(format!("{}: absent from the base", s.name));
        }
    }
    anyhow::ensure!(
        bad.is_empty(),
        "refusing: --ckpt is not the checkpoint of --base genome '{}' ({} of {n} trunk tensors \
         differ; first: {})",
        g.id,
        bad.len(),
        bad[0]
    );
    let master = trunk_hash(&hdr32, &crate::export::spec_entries(&specs32), Some(vocab));
    anyhow::ensure!(
        hex64(master) == g.master_trunk_hash,
        "refusing: the checkpoint's f32 trunk hashes to {} but genome '{}' declares \
         master_trunk_hash {} (arch / tokenizer / vocab differ)",
        hex64(master),
        g.id,
        g.master_trunk_hash
    );
    Ok(CkptBinding {
        trunk_tensors: n,
        master_hashes,
    })
}

/// Canonical φ of each prompt (`q_ids` = `Bpe::encode(q)`): the backbone
/// run on `prefix ++ q ++ suffix`, mean of the hidden AFTER `spec.layer`
/// over the q positions — one prompt per row, right padding, on a dropless
/// probe instance (no expert-capacity coupling between rows). Prompts whose
/// canonical input exceeds `max_len` or whose q is empty are skipped.
/// Returns the raw (unnormalized) φ of the kept prompts, in order, and the
/// indices of the skipped ones.
pub fn probe_prompts(
    ck: &Checkpoint,
    spec: &PhiSpec,
    q_ids: &[Vec<u32>],
    rows: usize,
    max_len: usize,
    pad: u32,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<usize>)> {
    let frame = spec.prefix_ids.len() + spec.suffix_ids.len();
    let mut keep = Vec::new();
    let mut skipped = Vec::new();
    for (k, q) in q_ids.iter().enumerate() {
        if q.is_empty() || frame + q.len() > max_len {
            skipped.push(k);
        } else {
            keep.push(k);
        }
    }
    if keep.is_empty() {
        return Ok((Vec::new(), skipped));
    }
    let longest = keep.iter().map(|&k| frame + q_ids[k].len()).max().unwrap();
    let t = longest.div_ceil(64).max(1) * 64;
    let rows = rows.max(1);
    let gpu = EmbryoGpu::new_eval_dropless(ck.cfg.clone(), rows, t, &ck.params)
        .ok_or_else(|| anyhow::anyhow!("no GPU device for the φ probe (Metal / Vulkan)"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false);
    gpu.anchor_fixed_window.set(true);
    let mut out = Vec::with_capacity(keep.len());
    let mut tokens = vec![pad; rows * t];
    for chunk in keep.chunks(rows) {
        tokens.fill(pad);
        let mut spans = vec![0..0; rows];
        for (r, &k) in chunk.iter().enumerate() {
            let (ids, span) = phi_span_ids(spec, &q_ids[k]);
            tokens[r * t..r * t + ids.len()].copy_from_slice(&ids);
            spans[r] = span;
        }
        let phis = gpu.probe_phi_span(&tokens, spec.layer, &spans);
        out.extend(phis.into_iter().take(chunk.len()));
    }
    drop(gpu);
    Ok((out, skipped))
}

/// `prefix ++ q ++ suffix` and the q span — the runtime's own function, so
/// trainer and runtime cannot disagree on the canonical φ input.
pub use cortiq_engine::router::phi_span_ids;

/// Deterministic mixed batches: `b − lm_rows` response-only SFT records
/// (IGNORE targets on prompt/padding) then `lm_rows` raw-LM windows.
pub struct MixSampler {
    pub b: usize,
    pub t: usize,
    pub lm_rows: usize,
    state: u64,
}

impl MixSampler {
    pub fn new(b: usize, t: usize, lm_rows: usize, seed: u64) -> MixSampler {
        assert!(lm_rows < b, "at least one SFT row per batch");
        MixSampler {
            b,
            t,
            lm_rows,
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    pub fn batch(
        &mut self,
        sft: &SftShard,
        lm: Option<&Shard>,
        tokens: &mut Vec<u32>,
        targets: &mut Vec<u32>,
    ) {
        assert_eq!(sft.seq, self.t);
        tokens.clear();
        targets.clear();
        for _ in 0..self.b - self.lm_rows {
            let (tok, tgt) = sft.record((splitmix(&mut self.state) % sft.records as u64) as usize);
            tokens.extend(tok[..self.t].iter().map(|&x| x as u32));
            targets.extend(
                tgt.iter()
                    .map(|&x| if x == IGNORE { u32::MAX } else { x as u32 }),
            );
        }
        if self.lm_rows > 0 {
            let lm = lm.expect("lm rows need an LM shard");
            let n = lm.tokens.len();
            for _ in 0..self.lm_rows {
                let start = (splitmix(&mut self.state) % (n - self.t - 1) as u64) as usize;
                let w = &lm.tokens[start..start + self.t + 1];
                tokens.extend(w[..self.t].iter().map(|&x| x as u32));
                targets.extend(w[1..].iter().map(|&x| x as u32));
            }
        }
    }
}

/// Raw-LM dev batches per evaluation when `dev_batches == 0`.
pub const LM_DEV_BATCHES: usize = 8;

/// Records a held-out SFT evaluation covers: every record (`cap == 0` or
/// `cap ≥ records`), else `cap` records spread evenly over the WHOLE shard
/// (`⌊k·records/cap⌋`, deterministic, each once) — never a prefix of the
/// file, whose records `sft-prepare` keeps grouped in input order.
pub fn eval_records(records: usize, cap: usize) -> Vec<usize> {
    if cap == 0 || cap >= records {
        (0..records).collect()
    } else {
        (0..cap).map(|k| k * records / cap).collect()
    }
}

/// The GPU instance a v2 bake trains and evaluates on: DROPLESS expert
/// capacity (every token keeps its routed expert, exactly the runtime's
/// operator). The capacity-2 training instance drops the tokens an
/// over-subscribed expert has no slot for — right-padded SFT rows fill a
/// deep-layer expert with pad tokens — so the skill would be polished and
/// selected on a model the file never executes, and a row's NLL would
/// depend on the other rows of its batch.
pub fn bake_gpu(cfg: EmbryoCfg, b: usize, t: usize, params: &[f32]) -> Option<EmbryoGpu> {
    EmbryoGpu::new_eval_dropless(cfg, b, t, params)
}

/// Token-weighted response-only NLL over exactly the records `rows`, each
/// once: batches of `batch` rows in `rows` order, the tail batch completed
/// with padding rows whose targets are all IGNORE (no weight; on the
/// dropless instance they share no capacity with the real rows). NaN when
/// the records carry no answer token.
pub fn eval_answer_nll(gpu: &EmbryoGpu, sft: &SftShard, batch: usize, rows: &[usize]) -> f32 {
    let t = sft.seq;
    let batch = batch.max(1);
    let (mut tk, mut tg) = (Vec::with_capacity(batch * t), Vec::with_capacity(batch * t));
    let (mut sum, mut cnt) = (0f64, 0usize);
    for chunk in rows.chunks(batch) {
        tk.clear();
        tg.clear();
        for &r in chunk {
            let (tok, tgt) = sft.record(r);
            tk.extend(tok[..t].iter().map(|&x| x as u32));
            tg.extend(
                tgt.iter()
                    .map(|&x| if x == IGNORE { u32::MAX } else { x as u32 }),
            );
        }
        for _ in chunk.len()..batch {
            tk.extend(std::iter::repeat_n(0u32, t));
            tg.extend(std::iter::repeat_n(u32::MAX, t));
        }
        let v = tg.iter().filter(|&&x| x != u32::MAX).count();
        if v == 0 {
            continue;
        }
        sum += gpu.eval_loss(&tk, &tg) as f64 * v as f64;
        cnt += v;
    }
    if cnt == 0 {
        f32::NAN
    } else {
        (sum / cnt as f64) as f32
    }
}

fn eval_lm_nll(gpu: &EmbryoGpu, lm: &Shard, batch: usize, t: usize, batches: usize) -> f32 {
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    let mut s = 0f64;
    for k in 0..batches {
        Sampler::fixed_batch(lm, batch, t, k, &mut tk, &mut tg);
        s += gpu.eval_loss(&tk, &tg) as f64;
    }
    (s / batches.max(1) as f64) as f32
}

/// The GPU part of the v2 bake: phase A (mask logits to the denoising
/// bottom under progressive L1, best hard-mask point on the dev answer
/// NLL), phase B (the selected shared FFNs polished under the frozen hard
/// mask with a fresh AdamW, best point on the dev answer NLL), then the
/// mask folded into the best tensors. Training rows mix response-only SFT
/// records with `lm_frac` raw-LM rows; the anchor trains on its served
/// window; routing descriptors and every other tensor stay frozen.
pub fn bake_v2_train(
    ck: &Checkpoint,
    data: &BakeV2Data,
    a: &BakeV2Args,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<BakeV2Trained> {
    let cfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    let t = data.sft_train.seq;
    anyhow::ensure!(
        data.sft_dev.seq == t && data.sft_final.as_ref().is_none_or(|f| f.seq == t),
        "SFT train/dev/final shards must share one sequence length"
    );
    anyhow::ensure!(t % 64 == 0, "SFT sequence {t} must be a multiple of 64");
    anyhow::ensure!(
        data.sft_train.records > 0 && data.sft_dev.records > 0,
        "empty SFT train/dev shard"
    );
    anyhow::ensure!(
        data.sft_train.valid_tokens() > 0 && data.sft_dev.valid_tokens() > 0,
        "SFT train/dev shards carry no answer tokens"
    );
    if let Some(lm) = &data.lm_train {
        anyhow::ensure!(
            lm.tokens.len() > t + 1,
            "--lm-train shorter than one window"
        );
        anyhow::ensure!(
            (0.0..1.0).contains(&a.lm_frac),
            "--lm-frac must be in [0, 1)"
        );
    }
    let lm_rows = lm_rows(a.batch, a.lm_frac, data.lm_train.is_some());
    if let Some(lm) = &data.lm_dev {
        anyhow::ensure!(
            lm.tokens.len() > a.batch * (t + 1),
            "--lm-dev shorter than one batch of windows"
        );
    }
    // every id the GPU gathers must lie inside the genome's vocabulary
    let v = cfg.vocab;
    let shards = [
        ("--sft-train", Some(&data.sft_train)),
        ("--sft-dev", Some(&data.sft_dev)),
        ("--sft-final", data.sft_final.as_ref()),
    ];
    for (flag, sft) in shards {
        if let Some(sft) = sft {
            let bad = sft.tokens.iter().any(|&x| x as usize >= v)
                || sft.targets.iter().any(|&x| x != IGNORE && x as usize >= v);
            anyhow::ensure!(!bad, "{flag}: token id ≥ vocab {v} (another tokenizer?)");
        }
    }
    for (flag, lm) in [("--lm-train", &data.lm_train), ("--lm-dev", &data.lm_dev)] {
        if let Some(lm) = lm {
            anyhow::ensure!(
                lm.tokens.iter().all(|&x| (x as usize) < v),
                "{flag}: token id ≥ vocab {v} (another tokenizer?)"
            );
        }
    }
    let i = cfg.inter;
    let mut gpu = bake_gpu(cfg.clone(), a.batch, t, &ck.params)
        .ok_or_else(|| anyhow::anyhow!("no GPU device (Metal / Vulkan)"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false); // the genome's routing state is frozen
    gpu.anchor_fixed_window.set(true); // train on the served window
    let c = gpu.ctx();
    gpu.skill = Some(SkillState::new(
        c,
        a.layers.clone(),
        i,
        SKILL_INIT_LOGIT,
        a.tau,
    ));
    // the whole dev shard (or an even stride over all of it), each record once
    let dev_rows = eval_records(data.sft_dev.records, a.dev_batches.saturating_mul(a.batch));
    let final_rows = data.sft_final.as_ref().map(|f| eval_records(f.records, 0));
    let lm_batches = data.lm_dev.as_ref().map(|lm| {
        let want = if a.dev_batches == 0 {
            LM_DEV_BATCHES
        } else {
            a.dev_batches
        };
        want.min((lm.tokens.len() / (a.batch * (t + 1))).max(1))
    });
    let dev = |gpu: &EmbryoGpu, step: usize| DevPoint {
        answer_nll: eval_answer_nll(gpu, &data.sft_dev, a.batch, &dev_rows),
        lm_nll: data
            .lm_dev
            .as_ref()
            .map(|lm| eval_lm_nll(gpu, lm, a.batch, t, lm_batches.unwrap())),
        step,
    };
    let fin = |gpu: &EmbryoGpu| {
        data.sft_final
            .as_ref()
            .map(|f| eval_answer_nll(gpu, f, a.batch, final_rows.as_deref().unwrap()))
    };
    let fmt_lm = |p: &DevPoint| {
        p.lm_nll
            .map(|x| format!(" lm-dev {x:.4}"))
            .unwrap_or_default()
    };
    let t0 = Instant::now();
    // the base: the all-on hard mask is the exact genome (σ(3) > τ)
    gpu.skill.as_ref().unwrap().hard.set(true);
    let base = dev(&gpu, 0);
    let base_final = fin(&gpu);
    gpu.skill.as_ref().unwrap().hard.set(false);
    anyhow::ensure!(base.answer_nll.is_finite(), "non-finite base dev NLL");
    eprintln!(
        "skill '{}' v2: base dev answer NLL {:.4}{} ({} of {} dev records); layers {:?}; \
         batch {} = {} SFT + {lm_rows} LM rows × {t}; MoE dropless",
        a.id,
        base.answer_nll,
        fmt_lm(&base),
        dev_rows.len(),
        data.sft_dev.records,
        a.layers,
        a.batch,
        a.batch - lm_rows
    );
    let mut sampler = MixSampler::new(a.batch, t, lm_rows, a.seed);
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    // ---- phase A: masks to the denoising bottom ----
    let mut best_a = (base, gpu.skill.as_ref().unwrap().logits.to_vec());
    for step in 0..a.steps_a {
        anyhow::ensure!(!should_stop(), "preempted");
        let l1 = a.l1 * (step as f32 / a.steps_a.max(1) as f32);
        gpu.skill.as_ref().unwrap().l1.set(l1);
        sampler.batch(&data.sft_train, data.lm_train.as_ref(), &mut tk, &mut tg);
        let (loss, gn) = gpu.train_step_skill(&tk, &tg, a.lr_a, 0.0, 1.0, false);
        if (step + 1) % a.eval_every.max(1) == 0 || step + 1 == a.steps_a {
            let sk = gpu.skill.as_ref().unwrap();
            sk.hard.set(true);
            let p = dev(&gpu, step + 1);
            sk.hard.set(false);
            let kept = kept_fractions(&sk.hard_masks());
            eprintln!(
                "  A step {:>5} loss {loss:.4} |g| {gn:.3} l1 {l1:.2e} dev(hard) {:.4}{} kept {:?} [{:.0} s]",
                step + 1,
                p.answer_nll,
                fmt_lm(&p),
                kept.iter().map(|k| format!("{k:.2}")).collect::<Vec<_>>(),
                t0.elapsed().as_secs_f64()
            );
            if p.answer_nll < best_a.0.answer_nll {
                best_a = (p, sk.logits.to_vec());
            }
        }
    }
    {
        let sk = gpu.skill.as_ref().unwrap();
        sk.logits.write_from(&best_a.1);
        sk.hard.set(true);
        sk.l1.set(0.0);
    }
    let masks = gpu.skill.as_ref().unwrap().hard_masks();
    let kept = kept_fractions(&masks);
    eprintln!(
        "phase A best dev {:.4} at step {}; kept {kept:?}",
        best_a.0.answer_nll, best_a.0.step
    );
    // ---- phase B: polish the selected FFNs under the hard mask ----
    let ranges = ffn_ranges(&cfg, &lay, &a.layers);
    let snapshot = |gpu: &EmbryoGpu| -> Vec<Vec<f32>> {
        let p = gpu.params_host();
        ranges.iter().map(|&(o, n)| p[o..o + n].to_vec()).collect()
    };
    // step 0 = no phase-B point beat phase A (the record is A's mask over
    // the genome FFN)
    let mut best_b = (
        DevPoint {
            step: 0,
            ..best_a.0
        },
        snapshot(&gpu),
    );
    gpu.step = 0; // fresh AdamW bias correction for the polished ranges
    for step in 0..a.steps_b {
        anyhow::ensure!(!should_stop(), "preempted");
        sampler.batch(&data.sft_train, data.lm_train.as_ref(), &mut tk, &mut tg);
        let lr = a.lr_b
            * 0.5
            * (1.0 + (std::f32::consts::PI * step as f32 / a.steps_b.max(1) as f32).cos());
        let (loss, gn) = gpu.train_step_skill(&tk, &tg, lr, 0.0, 1.0, true);
        if (step + 1) % a.eval_every.max(1) == 0 || step + 1 == a.steps_b {
            let p = dev(&gpu, step + 1);
            eprintln!(
                "  B step {:>5} loss {loss:.4} |g| {gn:.3} lr {lr:.2e} dev {:.4}{} [{:.0} s]",
                step + 1,
                p.answer_nll,
                fmt_lm(&p),
                t0.elapsed().as_secs_f64()
            );
            if p.answer_nll < best_b.0.answer_nll {
                best_b = (p, snapshot(&gpu));
            }
        }
    }
    eprintln!(
        "phase B best dev {:.4} (step {}) vs base {:.4}",
        best_b.0.answer_nll, best_b.0.step, base.answer_nll
    );
    // terminal final split: the chosen tensors in place, evaluated once
    let final_nll = match (base_final, &data.sft_final) {
        (Some(b0), Some(_)) => {
            let mut p = gpu.params_host();
            for (&(o, n), snap) in ranges.iter().zip(&best_b.1) {
                p[o..o + n].copy_from_slice(&snap[..n]);
            }
            gpu.set_params(&p);
            fin(&gpu).map(|s| (b0, s))
        }
        _ => None,
    };
    let tensors = fold_mask(&cfg, &a.id, &a.layers, &best_b.1, &masks);
    Ok(BakeV2Trained {
        tensors,
        kept,
        masks,
        base,
        best_a: best_a.0,
        best_b: best_b.0,
        final_nll,
        seq: t,
        lm_rows,
        dev_records: (dev_rows.len(), data.sft_dev.records),
        final_records: final_rows.as_ref().map(Vec::len),
        moe_capacity: if cfg.experts > 0 { "dropless" } else { "none" },
    })
}

fn pack_bits_b64(bits: &[bool]) -> String {
    use base64::Engine;
    let mut bytes = vec![0u8; bits.len().div_ceil(8)];
    for (j, &b) in bits.iter().enumerate() {
        if b {
            bytes[j / 8] |= 1 << (j % 8);
        }
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn file_role(role: &str, path: &Path) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::json!({
        "role": role,
        "file": path.file_name().map(|n| n.to_string_lossy().to_string()),
        "sha256": sha256_file(path)?,
    }))
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// hash64 of every record's tokens (rendered prompt + answer + padding).
pub fn record_hashes(sft: &SftShard) -> Vec<u64> {
    (0..sft.records)
        .map(|r| {
            let (tok, _) = sft.record(r);
            let bytes: Vec<u8> = tok.iter().flat_map(|x| x.to_le_bytes()).collect();
            cortiq_core::hash64(&bytes)
        })
        .collect()
}

/// What the held-out shards are VERIFIED to be (the record's `quality`
/// states this, not a constant): refuses when a dev or final record also
/// occurs in train (identical tokens); with the `sft-prepare --messages`
/// manifest, checks its group → split map against the group rule and binds
/// it to the shards (per-split record and answer-token counts).
pub fn check_held_out(
    train: &SftShard,
    dev: &SftShard,
    fin: Option<&SftShard>,
    manifest: Option<&Path>,
) -> anyhow::Result<serde_json::Value> {
    let train_h: HashSet<u64> = record_hashes(train).into_iter().collect();
    let dev_h = record_hashes(dev);
    let dev_in_train = dev_h.iter().filter(|h| train_h.contains(h)).count();
    anyhow::ensure!(
        dev_in_train == 0,
        "refusing: {dev_in_train} of {} --sft-dev records also occur in --sft-train (identical \
         tokens) — best-A/best-B would be selected on seen data; split with `sft-prepare` (group \
         split) instead",
        dev.records
    );
    let mut fin_json = serde_json::Value::Null;
    if let Some(f) = fin {
        let f_h = record_hashes(f);
        let in_train = f_h.iter().filter(|h| train_h.contains(h)).count();
        anyhow::ensure!(
            in_train == 0,
            "refusing: {in_train} of {} --sft-final records also occur in --sft-train (identical \
             tokens)",
            f.records
        );
        let dev_set: HashSet<u64> = dev_h.iter().copied().collect();
        let in_dev = f_h.iter().filter(|h| dev_set.contains(h)).count();
        if in_dev > 0 {
            eprintln!("warning: {in_dev} --sft-final records also occur in --sft-dev");
        }
        fin_json = serde_json::json!({"records": f.records, "in_train": 0, "in_dev": in_dev});
    }
    let groups = match manifest {
        None => serde_json::json!({"status": "unverified (no --sft-manifest)"}),
        Some(p) => check_manifest(p, train, dev, fin)?,
    };
    Ok(serde_json::json!({
        "records_disjoint": format!(
            "verified: 0 of {} dev records occur in train (token hash64)",
            dev.records
        ),
        "final": fin_json,
        "group_disjoint": groups,
    }))
}

fn check_manifest(
    path: &Path,
    train: &SftShard,
    dev: &SftShard,
    fin: Option<&SftShard>,
) -> anyhow::Result<serde_json::Value> {
    let bad = |what: String| anyhow::anyhow!("refusing: --sft-manifest {}: {what}", path.display());
    let raw = std::fs::read(path).map_err(|e| bad(e.to_string()))?;
    let m: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| bad(e.to_string()))?;
    if m["sequence"].as_u64() != Some(train.seq as u64) {
        return Err(bad(format!(
            "sequence {} differs from the shards' {}",
            m["sequence"], train.seq
        )));
    }
    for (name, sh) in [("train", Some(train)), ("dev", Some(dev)), ("final", fin)] {
        let Some(sh) = sh else { continue };
        let sp = &m["splits"][name];
        if sp["examples"].as_u64() != Some(sh.records as u64)
            || sp["valid_answer_tokens"].as_u64() != Some(sh.valid_tokens() as u64)
        {
            return Err(bad(format!(
                "split {name} declares {} records / {} answer tokens, the shard has {} / {} \
                 (shards of another prepare run?)",
                sp["examples"],
                sp["valid_answer_tokens"],
                sh.records,
                sh.valid_tokens()
            )));
        }
    }
    let groups = m["groups"].as_object().ok_or_else(|| {
        bad(
            "no `groups` map (group → split): not written by `sft-prepare --messages`, the \
             group split cannot be verified"
                .into(),
        )
    })?;
    let mut per: BTreeMap<String, usize> = BTreeMap::new();
    for (g, sp) in groups {
        let sp = sp.as_str().unwrap_or("");
        let rule = crate::sft::group_split(g);
        if sp != rule {
            return Err(bad(format!(
                "group '{g}' is listed in split '{sp}', the group rule puts it in '{rule}'"
            )));
        }
        *per.entry(sp.to_string()).or_default() += 1;
    }
    for (name, n) in &per {
        if let Some(tr) = m["splits"][name.as_str()]["trees"].as_u64() {
            if tr != *n as u64 {
                return Err(bad(format!(
                    "split {name}: {tr} groups declared, {n} in the group map"
                )));
            }
        }
    }
    Ok(serde_json::json!({
        "status": "verified by --sft-manifest (each group in one split by the group rule; \
                   split sizes match the shards)",
        "manifest_sha256": sha256_hex(&raw),
        "groups": groups.len(),
        "groups_per_split": per,
    }))
}

/// One φ class of a bake: the probed prompts in sample order (the
/// descriptor's holdout is the last [`holdout_count`] of them).
struct PhiClass {
    phis: Vec<Vec<f32>>,
    texts: Vec<String>,
    q_ids: Vec<Vec<u32>>,
    /// 1-based line of each prompt in its JSONL file
    lines: Vec<usize>,
    info: serde_json::Value,
}

impl PhiClass {
    fn holdout(&self) -> std::ops::Range<usize> {
        let n = self.phis.len();
        n - holdout_count(n)..n
    }

    /// `info` + the calibration holdout by line number and the sha256 of
    /// its texts joined by `\n` — what `route-eval` needs to replay the
    /// calibration measurement.
    fn origin_json(&self) -> serde_json::Value {
        let r = self.holdout();
        let mut j = self.info.clone();
        j["holdout"] = serde_json::json!({
            "n": r.len(),
            "lines": &self.lines[r.clone()],
            "sha256": sha256_hex(self.texts[r].join("\n").as_bytes()),
        });
        j
    }
}

#[allow(clippy::too_many_arguments)]
fn probe_class(
    ck: &Checkpoint,
    spec: &PhiSpec,
    tok: &RuntimeTokenizer,
    path: &Path,
    what: &str,
    a: &BakeV2Args,
    seed: u64,
    pad: u32,
) -> anyhow::Result<PhiClass> {
    let all = read_prompts_lines(path)?;
    let mut first_line: HashMap<&str, usize> = HashMap::new();
    for (l, p) in &all {
        first_line.entry(p.as_str()).or_insert(*l);
    }
    let texts: Vec<String> = all.iter().map(|(_, p)| p.clone()).collect();
    let sample = sample_prompts(&texts, a.phi_max, seed);
    let q: Vec<Vec<u32>> = sample.iter().map(|p| user_text_ids(tok, p)).collect();
    let (phis, skipped) = probe_prompts(ck, spec, &q, a.phi_batch, a.phi_max_len, pad)?;
    anyhow::ensure!(
        phis.len() >= PHI_MIN_PROMPTS,
        "{what}: {} usable prompts in {} (need ≥ {PHI_MIN_PROMPTS}; {} skipped as empty or \
         longer than {} tokens)",
        phis.len(),
        path.display(),
        skipped.len(),
        a.phi_max_len
    );
    let skip: HashSet<usize> = skipped.iter().copied().collect();
    let kept: Vec<usize> = (0..sample.len()).filter(|k| !skip.contains(k)).collect();
    Ok(PhiClass {
        lines: kept
            .iter()
            .map(|&k| first_line[sample[k].as_str()])
            .collect(),
        q_ids: kept.iter().map(|&k| q[k].clone()).collect(),
        texts: kept.iter().map(|&k| sample[k].clone()).collect(),
        info: serde_json::json!({
            "lines": all.len(), "sampled": sample.len(), "probed": phis.len(),
            "skipped_too_long_or_empty": skipped.len(),
        }),
        phis,
    })
}

/// Tolerance of the trainer ↔ runtime unit-φ parity (max |Δ| per component).
pub const PHI_PARITY_TOL: f32 = 1e-5;
/// Holdout prompts per φ class the post-append runtime check replays.
pub const RUNTIME_CHECK_PER_CLASS: usize = 64;

/// One prompt of [`runtime_route_check`]: its text and what the trainer
/// computed for it (q ids, raw φ).
pub struct PhiCase<'a> {
    pub text: &'a str,
    pub q_ids: &'a [u32],
    pub phi: &'a [f32],
}

fn unit_vec(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Restores `CMF_GPU` when the runtime check ends.
struct GpuEnvGuard(Option<std::ffi::OsString>);

impl Drop for GpuEnvGuard {
    fn drop(&mut self) {
        // SAFETY: as at the set below — no other thread of the bake reads
        // the environment while the check runs.
        unsafe {
            match &self.0 {
                Some(v) => std::env::set_var("CMF_GPU", v),
                None => std::env::remove_var("CMF_GPU"),
            }
        }
    }
}

/// Before a v2 file is published the RUNTIME replays the calibration
/// holdouts: the CPU reference pipeline (`CMF_GPU=0`, the G2 protocol) on
/// the written file encodes each text with its own tokenizer (must give the
/// trainer's ids), computes φ with `probe_phi_span` (unit φ within `tol` of
/// the trainer's) and decides through `route_request_ids` with quarantined
/// skills scored; the decision must be the one the trainer's φ takes under
/// the same header. Any difference refuses: `router.measured` would
/// describe a router the file does not execute.
pub fn runtime_route_check(
    path: &Path,
    cases: &[PhiCase],
    tol: f32,
) -> anyhow::Result<serde_json::Value> {
    use cortiq_engine::router::{RouteOptions, phi_span_ids, route_policy_with, route_request_ids};
    let _restore = GpuEnvGuard(std::env::var_os("CMF_GPU"));
    // SAFETY: the bake is between phases — the trainer's GPU work is done
    // and the engine pipeline (and its pool) is created only below, so no
    // thread of this process reads the environment concurrently.
    unsafe { std::env::set_var("CMF_GPU", "0") };
    cortiq_engine::gpu::cpu_scope(|| -> anyhow::Result<serde_json::Value> {
        let model = std::sync::Arc::new(CmfModel::open(path)?);
        let policy =
            model.header.router.clone().ok_or_else(|| {
                anyhow::anyhow!("runtime check: {} has no router", path.display())
            })?;
        let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(
            &model,
            cortiq_engine::sampler::SamplerConfig::default(),
        )?;
        let opts = RouteOptions {
            include_quarantine: true,
        };
        let (mut worst, mut to_skill) = (0f32, 0usize);
        for (k, c) in cases.iter().enumerate() {
            let q = pipe.tokenizer.encode(c.text);
            anyhow::ensure!(
                q == c.q_ids,
                "runtime check: holdout prompt #{k} encodes to {} ids in the runtime, {} in the \
                 trainer ({:?})",
                q.len(),
                c.q_ids.len(),
                c.text
            );
            let (ids, span) = phi_span_ids(&policy.phi, &q);
            let phi_rt = pipe.probe_phi_span(&ids, policy.phi.layer, span);
            let d = unit_vec(&phi_rt)
                .iter()
                .zip(&unit_vec(c.phi))
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            anyhow::ensure!(
                d.is_finite() && d <= tol,
                "runtime check: holdout prompt #{k} — the runtime's unit φ differs from the \
                 trainer's by {d:.3e} (> {tol:.0e}): the descriptors and the calibration would \
                 describe another φ than the served one ({:?})",
                c.text
            );
            worst = worst.max(d);
            let rt = route_request_ids(&model, &mut pipe, &q, opts);
            let tr = route_policy_with(&model.header, c.phi, opts);
            anyhow::ensure!(
                rt.target == tr.target,
                "runtime check: holdout prompt #{k} — the runtime routes to {} ({}), the \
                 trainer's φ to {} ({}) ({:?})",
                rt.target_label(),
                rt.reason,
                tr.target_label(),
                tr.reason,
                c.text
            );
            to_skill += usize::from(rt.skill().is_some());
        }
        Ok(serde_json::json!({
            "prompts": cases.len(),
            "decisions_equal": cases.len(),
            "to_skill": to_skill,
            "to_backbone": cases.len() - to_skill,
            "max_unit_phi_delta": worst,
            "tol": tol,
            "reference": "runtime CPU pipeline (CMF_GPU=0), route_request_ids, quarantine scored",
        }))
    })
}

/// A new private temp file next to `out`, created with `create_new` under a
/// unique name (pid + time + counter): it can never be an existing file —
/// not `--base`, not another bake's temp — so the bake removes only what it
/// created.
pub fn create_bake_tmp(out: &Path) -> anyhow::Result<(PathBuf, std::fs::File)> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let name = out
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("--out {} names no file", out.display()))?
        .to_string_lossy()
        .to_string();
    for _ in 0..64 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let nonce = t ^ SEQ.fetch_add(1, Ordering::Relaxed).rotate_left(40);
        let tmp = out.with_file_name(format!(
            ".{name}.bake-{}-{nonce:016x}.tmp",
            std::process::id()
        ));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(f) => return Ok((tmp, f)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => anyhow::bail!("create {}: {e}", tmp.display()),
        }
    }
    anyhow::bail!(
        "could not create a unique temp file next to {}",
        out.display()
    )
}

/// Publish the finished `tmp` as `out` WITHOUT ever replacing an existing
/// `out`: a hard link (fails if `out` exists, atomically) then the temp name
/// is dropped. If `out` appeared meanwhile the bake refuses and KEEPS the
/// result at `tmp` (the error names it). Filesystems without hard links fall
/// back to a rename after re-checking that `out` is absent.
pub fn publish_new_file(tmp: &Path, out: &Path) -> anyhow::Result<()> {
    let exists = || {
        anyhow::anyhow!(
            "refusing to overwrite --out {} (it appeared while the bake ran); the finished file \
             is kept at {}",
            out.display(),
            tmp.display()
        )
    };
    match std::fs::hard_link(tmp, out) {
        Ok(()) => {
            if let Err(e) = std::fs::remove_file(tmp) {
                eprintln!(
                    "warning: published {}; temp name {} not removed: {e}",
                    out.display(),
                    tmp.display()
                );
            }
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Err(exists()),
        Err(e) => {
            if std::fs::symlink_metadata(out).is_ok() {
                return Err(exists());
            }
            eprintln!("warning: no hard link on this filesystem ({e}); publishing by rename");
            std::fs::rename(tmp, out)?;
            Ok(())
        }
    }
}

/// The v2 bake, end to end (spec §3–§5): refuse unless `--out` is a new
/// file ≠ `--base`, `phi_layer < min(layers)`, 0 < τ < σ(init logit),
/// `--base` carries a genome and `--ckpt` is exactly its checkpoint, and the
/// held-out shards share no record with train; probe φ on the SERVED trunk
/// (f16-rounded for an f16 genome) with the runtime's tokenizer; fit the
/// skill descriptor (and the backbone descriptor at the first record, or
/// with `refit_base` — otherwise the base's router keeps its backbone
/// descriptor and margin); bake on the dropless instance, selecting on the
/// whole dev shard; calibrate router v2 over the backbone + every v2 skill;
/// copy the base into a private temp, tail-append the `ffn_replace` record
/// (status `quarantine`), move every other `active` v2 skill to
/// `stale_regate` (its gate was measured under the previous router) with a
/// `recalibrate` lineage event, verify G1, replay the holdouts through the
/// runtime, and publish without overwriting. Returns the summary JSON
/// `skill-bake` prints.
pub fn bake_v2(
    inp: &BakeV2Inputs,
    a: &BakeV2Args,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<serde_json::Value> {
    // ---- cheap refusals first ----
    check_out_path(&inp.base, &inp.out)?;
    anyhow::ensure!(!a.layers.is_empty(), "--layers is empty");
    let mut sorted = a.layers.clone();
    sorted.sort_unstable();
    sorted.dedup();
    anyhow::ensure!(
        sorted == a.layers,
        "--layers must be distinct and ascending, got {:?}",
        a.layers
    );
    let lmin = a.layers[0];
    anyhow::ensure!(
        a.phi_layer < lmin,
        "refusing: --phi-layer {} must be < min(--layers) {lmin} (φ is read from the backbone \
         before the first replaced FFN, so routing never depends on the active skill)",
        a.phi_layer
    );
    anyhow::ensure!(
        !a.id.is_empty() && !a.id.contains('.') && a.id != BACKBONE_CLASS_ID,
        "skill id '{}' must be non-empty, contain no '.', and not be {BACKBONE_CLASS_ID}",
        a.id
    );
    anyhow::ensure!(
        a.batch >= 1 && a.rank >= 1,
        "--batch and --rank must be ≥ 1"
    );
    if let Some(m) = a.route_margin {
        anyhow::ensure!(
            m.is_finite() && m >= 0.0,
            "--route-margin must be finite and ≥ 0"
        );
    }
    anyhow::ensure!(
        a.target_fpr > 0.0 && a.target_fpr < 1.0,
        "--target-fpr must be in (0, 1)"
    );
    let tau_max = tau_ceiling();
    anyhow::ensure!(
        a.tau > 0.0 && a.tau < tau_max,
        "refusing: --tau {} must lie in (0, σ({SKILL_INIT_LOGIT}) = {tau_max:.4}): the all-on \
         hard mask is the exact genome only for τ below σ(init logit) — the base dev NLL and \
         phase A start there — and τ ≤ 0 never drops a neuron",
        a.tau
    );
    anyhow::ensure!(
        a.l1.is_finite() && a.l1 >= 0.0,
        "--l1 must be finite and ≥ 0"
    );
    distinct_files(&[
        ("--sft-train", Some(&inp.sft_train)),
        ("--sft-dev", Some(&inp.sft_dev)),
        ("--sft-final", inp.sft_final.as_ref()),
        ("--sft-manifest", inp.sft_manifest.as_ref()),
    ])?;
    distinct_files(&[
        ("--lm-train", inp.lm_train.as_ref()),
        ("--lm-dev", inp.lm_dev.as_ref()),
    ])?;
    distinct_files(&[
        ("--phi-prompts", Some(&inp.phi_prompts)),
        ("--general-prompts", inp.general_prompts.as_ref()),
    ])?;
    let base = CmfModel::open(&inp.base)?;
    // the file the header / binding / calibration were read from (a tail
    // append by another process while the bake runs is refused at the copy)
    let base_len = std::fs::metadata(&inp.base)?.len();
    let genome = base.header.genome.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "refusing: --base {} carries no GENOME — export it with `export --genome-id <id> \
             --genome-status <status>`",
            inp.base.display()
        )
    })?;
    anyhow::ensure!(
        base.header.skills.iter().all(|s| s.id != a.id),
        "skill '{}' already exists in --base (re-bake under a new id)",
        a.id
    );
    let old_router = base.header.router.clone();
    if let Some(r) = &old_router {
        anyhow::ensure!(
            r.phi.layer == a.phi_layer,
            "--base routes on φ after layer {} (its skills' descriptors); --phi-layer {} differs",
            r.phi.layer,
            a.phi_layer
        );
    }
    // the backbone descriptor is fitted once (first record) or on request
    let refit_base = old_router.is_none() || a.refit_base;
    let general_prompts = if refit_base {
        Some(inp.general_prompts.as_ref().ok_or_else(|| {
            if old_router.is_none() {
                anyhow::anyhow!(
                    "refusing: the first record over genome '{}' needs --general-prompts (the \
                     backbone φ class the router is fitted on)",
                    genome.id
                )
            } else {
                anyhow::anyhow!("--refit-base needs --general-prompts")
            }
        })?)
    } else {
        if let Some(p) = &inp.general_prompts {
            eprintln!(
                "warning: --general-prompts {} is not used: --base already routes and keeps its \
                 backbone descriptor (--refit-base refits it; every active skill then needs a new \
                 gate)",
                p.display()
            );
        }
        None
    };
    let margin = a
        .route_margin
        .or(old_router.as_ref().map(|r| r.margin))
        .unwrap_or(0.0);
    // ---- checkpoint ↔ base ----
    let mut ck = crate::train::load_checkpoint(&inp.ckpt)?;
    let nl = ck.cfg.layers;
    anyhow::ensure!(
        *a.layers.last().unwrap() < nl,
        "--layers {:?} out of range for a {nl}-layer genome",
        a.layers
    );
    let binding = bind_ckpt_to_base(&ck, &base)?;
    eprintln!(
        "skill '{}' v2: --ckpt bound to genome '{}' gen {} ({} trunk tensors identical, master {})",
        a.id, genome.id, genome.generation, binding.trunk_tensors, genome.master_trunk_hash
    );
    let vocab = base.vocab.clone().expect("checked by the binding");
    if let Some(tp) = &inp.tokenizer {
        let tj = std::fs::read(tp)?;
        anyhow::ensure!(
            tj == vocab,
            "--tokenizer {} differs from the VOCAB section of --base",
            tp.display()
        );
    }
    let bpe = Bpe::from_json(&vocab)?;
    let spec = phi_spec(&bpe, a.phi_layer)?;
    // φ's user text is encoded exactly as the runtime does (added tokens
    // split out of raw text); the frame must agree too
    let rt_tok = runtime_tokenizer(&vocab)?;
    anyhow::ensure!(
        rt_tok.encode("<|im_start|>user\n") == spec.prefix_ids
            && rt_tok.encode("<|im_end|>\n<|im_start|>assistant\n") == spec.suffix_ids,
        "the runtime tokenizer renders the cmf-im-v1 frame differently from the trainer's"
    );
    if let Some(r) = &old_router {
        anyhow::ensure!(
            r.phi == spec,
            "--base routes on another φ frame than cmf-im-v1 with this tokenizer"
        );
    }
    // ---- the trunk the runtime serves: φ and the polish run on it ----
    let trunk_params = match genome.encoding.as_str() {
        "f16" => {
            ck.params = crate::export::served_params(&ck, &vocab, TensorDtype::F16)?;
            "f16 (served trunk: every f16-stored matrix rounded f32→f16→f32)"
        }
        _ => "f32",
    };
    // ---- data ----
    let data = BakeV2Data {
        sft_train: SftShard::load(&inp.sft_train)?,
        sft_dev: SftShard::load(&inp.sft_dev)?,
        sft_final: inp.sft_final.as_deref().map(SftShard::load).transpose()?,
        lm_train: inp.lm_train.as_deref().map(Shard::load).transpose()?,
        lm_dev: inp.lm_dev.as_deref().map(Shard::load).transpose()?,
    };
    let held_out = check_held_out(
        &data.sft_train,
        &data.sft_dev,
        data.sft_final.as_ref(),
        inp.sft_manifest.as_deref(),
    )?;
    let mut roles = vec![
        file_role("sft_train", &inp.sft_train)?,
        file_role("sft_dev", &inp.sft_dev)?,
    ];
    if let Some(p) = &inp.sft_final {
        roles.push(file_role("sft_final", p)?);
    }
    if let Some(p) = &inp.sft_manifest {
        roles.push(file_role("sft_manifest", p)?);
    }
    if let Some(p) = &inp.lm_train {
        roles.push(file_role("lm_train", p)?);
    }
    if let Some(p) = &inp.lm_dev {
        roles.push(file_role("lm_dev", p)?);
    }
    let phi_role = file_role("phi_prompts", &inp.phi_prompts)?;
    roles.push(phi_role.clone());
    let gen_role = general_prompts
        .map(|p| file_role("general_prompts", p))
        .transpose()?;
    if let Some(r) = &gen_role {
        roles.push(r.clone());
    }
    // ---- φ: skill class and (first record / refit) backbone class ----
    let pad = bpe
        .special_id("<|pad|>")
        .or_else(|| bpe.special_id(crate::tokenizer::EOT))
        .unwrap_or(0);
    let skill_cls = probe_class(
        &ck,
        &spec,
        &rt_tok,
        &inp.phi_prompts,
        "--phi-prompts",
        a,
        a.seed ^ 0x51,
        pad,
    )?;
    let gen_cls = general_prompts
        .map(|p| {
            probe_class(
                &ck,
                &spec,
                &rt_tok,
                p,
                "--general-prompts",
                a,
                a.seed ^ 0x6e,
                pad,
            )
        })
        .transpose()?;
    let class_overlap = match general_prompts {
        Some(g) => {
            let sk: HashSet<String> = read_prompts(&inp.phi_prompts)?.into_iter().collect();
            let n = read_prompts(g)?.iter().filter(|p| sk.contains(*p)).count();
            if n > 0 {
                eprintln!(
                    "warning: {n} prompts appear in both --phi-prompts and --general-prompts"
                );
            }
            Some(n)
        }
        None => None,
    };
    let sel = fit_selection(&skill_cls.phis, a.phi_layer, a.rank);
    let base_sel = match (&gen_cls, &old_router) {
        (Some(g), _) => fit_selection(&g.phis, a.phi_layer, a.rank),
        (None, Some(r)) => r.base.clone(),
        (None, None) => unreachable!("the first record always fits the backbone"),
    };
    eprintln!(
        "φ after layer {}: skill {} prompts (rank {}), backbone {} (rank {})",
        a.phi_layer,
        skill_cls.phis.len(),
        sel.rank,
        match &gen_cls {
            Some(g) => format!("{} prompts", g.phis.len()),
            None => "descriptor kept from --base".into(),
        },
        base_sel.rank
    );
    // ---- bake ----
    let trained = bake_v2_train(&ck, &data, a, should_stop)?;
    // ---- the record ----
    let prefix = format!("skill.{}.", a.id);
    let mut overrides = Vec::with_capacity(trained.tensors.len());
    let mut specs = Vec::with_capacity(trained.tensors.len());
    for (name, shape, data) in &trained.tensors {
        let x = name.strip_prefix(&prefix).expect("fold_mask names");
        let mh = binding
            .master_hashes
            .get(x)
            .ok_or_else(|| anyhow::anyhow!("{x}: not a trunk tensor of the checkpoint"))?;
        overrides.push(SkillOverride {
            name: x.to_string(),
            base_hash: hex64(*mh),
        });
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        specs.push(TensorSpec {
            name: name.clone(),
            dtype: TensorDtype::F32,
            shape: shape.clone(),
            data: bytes,
        });
    }
    let state_effect = ffn_replace_state_effect(&base.header.arch, &a.layers);
    let mask_json: serde_json::Map<String, serde_json::Value> = a
        .layers
        .iter()
        .zip(trained.masks.iter().zip(&trained.kept))
        .map(|(l, (bits, k))| {
            (
                l.to_string(),
                serde_json::json!({"kept": k, "bits_b64": pack_bits_b64(bits), "neurons": bits.len()}),
            )
        })
        .collect();
    let (dev_used, dev_total) = trained.dev_records;
    let dev_records = serde_json::json!({
        "used": dev_used,
        "total": dev_total,
        "selection": if dev_used == dev_total { "all" } else { "stride over the whole shard" },
    });
    let gen_origin = match &gen_cls {
        Some(g) => g.origin_json(),
        None => serde_json::json!({"descriptor": "kept from the router of --base"}),
    };
    let origin = serde_json::json!({
        "trigger": "user_corpus",
        "dataset_sha256": roles.iter().map(|r| r["sha256"].clone()).collect::<Vec<_>>(),
        "inputs": roles,
        "recipe": RECIPE_V2,
        "steps": {"a": a.steps_a, "b": a.steps_b},
        "lr": {"a": a.lr_a, "b": a.lr_b},
        "l1": a.l1, "tau": a.tau,
        "batch": a.batch, "seq": trained.seq, "lm_rows": trained.lm_rows, "lm_frac": a.lm_frac,
        "seed": a.seed,
        "moe_capacity": trained.moe_capacity,
        "trunk_params": trunk_params,
        "kept": trained.kept,
        // the mask as an object (P2): the core allows only override tensors
        // under skill.{id}., so the bits ride here (LSB-first, per layer)
        "mask": mask_json,
        "dev_records": dev_records,
        "final_records": trained.final_records,
        "dev_answer_nll": {
            "base": trained.base.answer_nll,
            "best_a": trained.best_a.answer_nll,
            "best_b": trained.best_b.answer_nll,
            "best_a_step": trained.best_a.step,
            "best_b_step": trained.best_b.step,
        },
        "lm_dev_nll": trained.base.lm_nll.map(|b| serde_json::json!({
            "base": b, "best_a": trained.best_a.lm_nll, "best_b": trained.best_b.lm_nll,
        })),
        "final_answer_nll": trained.final_nll.map(|(b, s)| serde_json::json!({"base": b, "skill": s})),
        "phi": {
            "layer": a.phi_layer,
            "tokenizer": "runtime Tokenizer::encode (VOCAB of --base)",
            "skill": skill_cls.origin_json(),
            "general": gen_origin,
            "rank": sel.rank, "class_overlap": class_overlap,
        },
        "ckpt_step": ck.step,
    });
    let record = SkillRecord {
        id: a.id.clone(),
        layers: a.layers.clone(),
        selection: Some(sel),
        quality: Some(serde_json::json!({
            "set": "sft_dev (response-only)",
            "dev_answer_nll": {"base": trained.base.answer_nll, "skill": trained.best_b.answer_nll},
            "dev_records": dev_records,
            "held_out": held_out,
        })),
        base_arch: Some(base.header.arch.arch_name.clone()),
        provenance: Some(
            serde_json::json!({"producer": "cortiq-embryo skill-bake v2", "recipe": RECIPE_V2}),
        ),
        kind: Some(skill_kind::FFN_REPLACE.into()),
        overrides,
        bound: Some(SkillBound {
            genome_id: genome.id.clone(),
            generation: genome.generation,
            master_trunk_hash: genome.master_trunk_hash.clone(),
        }),
        state_effect: Some(state_effect.clone()),
        status: Some("quarantine".into()),
        gate: None,
        prompt_contract: Some(PROMPT_CONTRACT.into()),
        origin: Some(origin),
        ..Default::default()
    };
    // ---- router v2 + calibration over the backbone and every v2 skill ----
    let mut router = RouterPolicy {
        version: 2,
        policy: "backbone_gated".into(),
        granularity: "request".into(),
        phi: spec,
        base: base_sel,
        margin,
        skills_hash: String::new(),
        measured: None,
    };
    let mut hdr = base.header.clone();
    hdr.skills.push(record.clone());
    hdr.router = Some(router.clone());
    let (cal, mut measured) =
        cortiq_engine::router::calibrate_v2(&hdr, a.target_fpr).map_err(anyhow::Error::msg)?;
    router.skills_hash = hex64(cortiq_engine::router::skills_hash(&hdr));
    measured["in_sha256"] = phi_role["sha256"].clone();
    measured["general_sha256"] = match (&gen_role, &old_router) {
        (Some(r), _) => r["sha256"].clone(),
        (None, Some(r)) => r
            .measured
            .as_ref()
            .map(|m| m["general_sha256"].clone())
            .unwrap_or(serde_json::Value::Null),
        (None, None) => serde_json::Value::Null,
    };
    router.measured = Some(measured.clone());
    // every other active v2 skill was gated under the previous router (its
    // G3 false-accept belongs to another calibration): it must be re-gated
    let stale: Vec<String> = base
        .header
        .skills
        .iter()
        .filter(|s| s.is_v2() && s.status.as_deref() == Some("active"))
        .map(|s| s.id.clone())
        .collect();
    let recalibrate = old_router.as_ref().map(|old| {
        serde_json::json!({
            "cause": format!("skill_committed {}", a.id),
            "stale_regate": stale,
            "base_descriptor": if refit_base { "refitted (--refit-base)" } else { "kept" },
            "margin": {"old": old.margin, "new": margin},
            "skills_hash": {"old": old.skills_hash, "new": router.skills_hash},
            "temperature": cal.temperature, "novelty_theta": cal.novelty_theta,
        })
    });
    drop(base);
    // the runtime replays the calibration holdouts of the classes fitted here
    let mut cases: Vec<PhiCase> = Vec::new();
    for cls in std::iter::once(&skill_cls).chain(gen_cls.as_ref()) {
        for k in cls.holdout().take(RUNTIME_CHECK_PER_CLASS) {
            cases.push(PhiCase {
                text: &cls.texts[k],
                q_ids: &cls.q_ids[k],
                phi: &cls.phis[k],
            });
        }
    }
    // ---- private copy + true tail append (+ recalibration), G1, runtime ----
    let (tmp, file) = create_bake_tmp(&inp.out)?;
    let staged = (|| -> anyhow::Result<(u64, serde_json::Value)> {
        let mut file = file;
        let copied = std::io::copy(&mut std::fs::File::open(&inp.base)?, &mut file)?;
        anyhow::ensure!(
            copied == base_len,
            "--base {} changed while the bake ran ({base_len} → {copied} bytes): its header, \
             binding and calibration inputs are stale — bake again",
            inp.base.display()
        );
        file.sync_all()?;
        drop(file);
        CmfModel::append_skill(&tmp, record, &specs, Some(router), Some(cal.clone()), None)?;
        if let Some(detail) = recalibrate {
            CmfModel::update_header_append(&tmp, |h| {
                for s in h.skills.iter_mut() {
                    if stale.contains(&s.id) {
                        s.status = Some("stale_regate".into());
                    }
                }
                let seq = h.lineage.last().map(|e| e.seq + 1).unwrap_or(0);
                h.lineage
                    .push(LineageEvent::now(seq, "recalibrate", detail));
            })?;
        }
        verify_append(&inp.base, &tmp)?;
        let check = runtime_route_check(&tmp, &cases, PHI_PARITY_TOL)?;
        let grown = std::fs::metadata(&tmp)?.len() - base_len;
        Ok((grown, check))
    })();
    let (appended_bytes, runtime_check) = match staged {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp); // our own temp (create_new)
            return Err(e);
        }
    };
    publish_new_file(&tmp, &inp.out)?;
    Ok(serde_json::json!({
        "id": a.id,
        "layers": a.layers,
        "phi_layer": a.phi_layer,
        "base_dev_nll": trained.base.answer_nll,
        "bestA": trained.best_a.answer_nll,
        "bestB": trained.best_b.answer_nll,
        "dev_records": dev_records,
        "lm_dev_nll": {"base": trained.base.lm_nll, "bestA": trained.best_a.lm_nll, "bestB": trained.best_b.lm_nll},
        "final_nll": trained.final_nll.map(|(b, s)| serde_json::json!({"base": b, "skill": s})),
        "kept": trained.kept,
        "moe_capacity": trained.moe_capacity,
        "trunk_params": trunk_params,
        "calib": {
            "temperature": cal.temperature,
            "theta": cal.novelty_theta,
            "samples": cal.samples,
            "target_fpr": cal.target_fpr,
            "measured": measured,
        },
        "router": {
            "base_descriptor": if gen_cls.is_some() { "fitted" } else { "kept from --base" },
            "margin": margin,
        },
        "stale_regate": stale,
        "runtime_check": runtime_check,
        "state_effect": state_effect,
        "status": "quarantine",
        "genome": {"id": genome.id, "generation": genome.generation, "trunk_hash": genome.trunk_hash},
        "out": inp.out.display().to_string(),
        "appended_bytes": appended_bytes,
    }))
}

/// G1 on the result of a tail append: same trunk hash, every base entry
/// byte-identical in the directory, bytes `[128, len(base))` unchanged.
pub fn verify_append(base: &Path, out: &Path) -> anyhow::Result<()> {
    let m0 = CmfModel::open(base)?;
    let m1 = CmfModel::open(out)?;
    anyhow::ensure!(
        m0.trunk_hash() == m1.trunk_hash(),
        "append changed the trunk hash"
    );
    for t in &m0.tensors {
        anyhow::ensure!(
            m1.tensor(&t.name) == Some(t),
            "append changed the directory entry of {}",
            t.name
        );
    }
    drop((m0, m1));
    let (mut f0, mut f1) = (std::fs::File::open(base)?, std::fs::File::open(out)?);
    let len0 = f0.metadata()?.len();
    anyhow::ensure!(
        f1.metadata()?.len() >= len0,
        "the output is shorter than the base"
    );
    use std::io::{Read, Seek, SeekFrom};
    f0.seek(SeekFrom::Start(128))?;
    f1.seek(SeekFrom::Start(128))?;
    let (mut b0, mut b1) = (vec![0u8; 4 << 20], vec![0u8; 4 << 20]);
    let mut left = len0.saturating_sub(128);
    while left > 0 {
        let n = (left as usize).min(b0.len());
        f0.read_exact(&mut b0[..n])?;
        f1.read_exact(&mut b1[..n])?;
        anyhow::ensure!(
            b0[..n] == b1[..n],
            "the output's prefix bytes differ from the base"
        );
        left -= n as u64;
    }
    Ok(())
}
