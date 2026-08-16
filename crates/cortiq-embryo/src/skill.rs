//! Skill bake for the Embryo (DTG-MA / P2 + P15 record): from a genome
//! checkpoint and a task corpus, train a neuron mask over the shared FFN of
//! selected layers to its denoising bottom (phase A, L1 pressure, held-out
//! gate), FCD-polish those FFNs under the hard mask (phase B), fold the
//! mask into the tensors, fit the recon-argmin routing descriptor (P1) and
//! append the record `skill.{id}.*` to the .cmf — every pre-existing tensor
//! byte-for-byte unchanged (the directory hashes prove it).

use crate::model::{EmbryoGpu, LayerOffs, Layout, SkillState};
use crate::train::{Checkpoint, Sampler, Shard};
use cortiq_core::format::{CmfModel, SelectionDescriptor, SkillRecord, TensorSpec};
use cortiq_core::types::TensorDtype;
use std::path::Path;
use std::time::Instant;

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
        let round = (m >> 13) as u16 + if (m & 0x1fff) > 0x1000 || ((m & 0x1fff) == 0x1000 && (m & 0x2000) != 0) { 1 } else { 0 };
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

/// Fit the P1 selection descriptor from φ samples: mean + rank principal
/// directions (orthonormal rows).
pub fn fit_selection(phis: &[Vec<f32>], phi_layer: usize, rank: usize) -> SelectionDescriptor {
    let n = phis.len();
    let h = phis[0].len();
    let mut mean = vec![0f32; h];
    for p in phis {
        for (m, v) in mean.iter_mut().zip(p) {
            *m += v / n as f32;
        }
    }
    let rank = rank.min(n.saturating_sub(1)).max(1).min(16);
    // covariance and top eigenvectors
    let mut cov = vec![0f32; h * h];
    for p in phis {
        let c: Vec<f32> = p.iter().zip(&mean).map(|(v, m)| v - m).collect();
        for i in 0..h {
            if c[i] == 0.0 {
                continue;
            }
            for j in 0..h {
                cov[i * h + j] += c[i] * c[j] / n as f32;
            }
        }
    }
    let basis = crate::model::top_eigenvectors(&cov, h, rank, 40, 99);
    SelectionDescriptor { metric: "mse".into(), phi_layer, mean: f16_base64(&mean), basis: f16_base64(&basis), rank }
}

/// Bake: returns (replacement tensors [(runtime tensor name, shape, data)],
/// selection descriptor, kept fraction per layer, held-out losses (base,
/// best A, best B)).
#[allow(clippy::type_complexity)]
pub fn bake(
    ck: &Checkpoint,
    corpus: &Shard,
    a: &BakeArgs,
) -> anyhow::Result<(Vec<(String, Vec<usize>, Vec<f32>)>, SelectionDescriptor, Vec<f32>, (f32, f32, f32))> {
    let cfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    let (h, i) = (cfg.hidden, cfg.inter);
    let mut gpu = EmbryoGpu::new(cfg.clone(), a.batch, a.seq, &ck.params).ok_or_else(|| anyhow::anyhow!("no Metal"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_updates.set(false); // the genome's routing state is frozen
    let c = gpu.ctx();
    gpu.skill = Some(SkillState::new(c, a.layers.clone(), i, 3.0, a.tau));
    // corpus split: last 10% held out (never trained on)
    let n = corpus.tokens.len();
    anyhow::ensure!(n > 20 * (a.seq + 2), "skill corpus too small: {n} tokens");
    let cut = n - n / 10;
    let train = Shard { tokens: corpus.tokens[..cut].to_vec() };
    let held = Shard { tokens: corpus.tokens[cut..].to_vec() };
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
    eprintln!("skill '{}': base held-out loss {base_loss:.4} (ppl {:.1}); layers {:?}", a.id, base_loss.exp(), a.layers);
    // ---- phase A: masks to the denoising bottom ----
    let mut best_a = (f32::MAX, gpu.skill.as_ref().unwrap().logits.to_vec());
    for step in 0..a.steps_a {
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
            let kept: Vec<f32> = masks.iter().map(|mk| mk.iter().filter(|&&b| b).count() as f32 / mk.len() as f32).collect();
            eprintln!("  A step {:>4} loss {loss:.4} |g| {gn:.3} l1 {l1:.2e} held-out(hard) {vl:.4} kept {:?} [{:.0} s]", step + 1, kept.iter().map(|k| format!("{k:.2}")).collect::<Vec<_>>(), t0.elapsed().as_secs_f64());
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
    let kept: Vec<f32> = gpu.skill.as_ref().unwrap().hard_masks().iter().map(|mk| mk.iter().filter(|&&b| b).count() as f32 / mk.len() as f32).collect();
    eprintln!("phase A best held-out {:.4} (ppl {:.1}); kept {:?}", best_a.0, best_a.0.exp(), kept);
    // ---- phase B: FCD polish of the selected FFNs under the hard mask ----
    let ffn_ranges: Vec<(usize, usize)> = a
        .layers
        .iter()
        .flat_map(|&l| {
            let ffn = match &lay.layers[l] {
                LayerOffs::Mixer { ffn, .. } | LayerOffs::Anchor { ffn, .. } => ffn,
            };
            [(ffn.wg, i * h), (ffn.wu, i * h), (ffn.wd, h * i)]
        })
        .collect();
    let snapshot = |gpu: &EmbryoGpu| -> Vec<Vec<f32>> {
        let p = gpu.params_host();
        ffn_ranges.iter().map(|&(o, n)| p[o..o + n].to_vec()).collect()
    };
    let mut best_b = (best_a.0, snapshot(&gpu));
    for step in 0..a.steps_b {
        sampler.batch(&train, &mut tk, &mut tg);
        let lr = a.lr_b * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / a.steps_b.max(1) as f32).cos());
        let (loss, gn) = gpu.train_step_skill(&tk, &tg, lr, 0.0, 1.0, true);
        if (step + 1) % a.eval_every == 0 || step + 1 == a.steps_b {
            let vl = eval(&gpu, &mut tk, &mut tg);
            eprintln!("  B step {:>4} loss {loss:.4} |g| {gn:.3} lr {lr:.2e} held-out {vl:.4} [{:.0} s]", step + 1, t0.elapsed().as_secs_f64());
            if vl < best_b.0 {
                best_b = (vl, snapshot(&gpu));
            }
        }
    }
    eprintln!("phase B best held-out {:.4} (ppl {:.1}) vs base {:.4} (ppl {:.1})", best_b.0, best_b.0.exp(), base_loss, base_loss.exp());
    // ---- fold the mask into the best tensors, name them for the runtime ----
    let masks = gpu.skill.as_ref().unwrap().hard_masks();
    let mut out = Vec::new();
    for (li, &l) in a.layers.iter().enumerate() {
        let (mut wg, mut wu, mut wd) = (best_b.1[li * 3].clone(), best_b.1[li * 3 + 1].clone(), best_b.1[li * 3 + 2].clone());
        for j in 0..i {
            if !masks[li][j] {
                wg[j * h..(j + 1) * h].fill(0.0);
                wu[j * h..(j + 1) * h].fill(0.0);
                for r in 0..h {
                    wd[r * i + j] = 0.0;
                }
            }
        }
        let pf = if cfg.experts > 0 { format!("skill.{}.model.layers.{l}.mlp.shared_expert.", a.id) } else { format!("skill.{}.model.layers.{l}.mlp.", a.id) };
        out.push((format!("{pf}gate_proj.weight"), vec![i, h], wg));
        out.push((format!("{pf}up_proj.weight"), vec![i, h], wu));
        out.push((format!("{pf}down_proj.weight"), vec![h, i], wd));
    }
    // ---- routing descriptor: φ = mean-pooled hidden entering phi_layer over corpus windows ----
    let mut phis: Vec<Vec<f32>> = Vec::new();
    let mut ps = Sampler::new(a.batch, a.seq, a.seed + 7);
    for _ in 0..8 {
        ps.batch(&train, &mut tk, &mut tg);
        phis.extend(gpu.probe_phi(&tk, a.phi_layer));
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
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(model.tensors.len() + tensors.len());
    let mut kept = 0usize;
    for t in &model.tensors {
        if t.name.starts_with(&format!("skill.{id}.")) {
            continue;
        }
        specs.push(TensorSpec { name: t.name.clone(), dtype: t.dtype, shape: t.shape.clone(), data: model.tensor_bytes(&t.name)?.to_vec() });
        kept += 1;
    }
    for (name, shape, data) in tensors {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        specs.push(TensorSpec { name: name.clone(), dtype: TensorDtype::F32, shape: shape.clone(), data: bytes });
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
        base_arch: Some(model.header.arch.arch_name.clone()),
        task: None,
        provenance: Some(serde_json::json!({"producer": "cortiq-embryo skill-bake", "recipe": "DTG-MA mask (phase A, L1 to the denoising bottom) + FCD polish (phase B), mask folded"})),
    });
    let vocab = model.vocab.clone();
    CmfModel::write(out, &header, &specs, if model.masks.masks.is_empty() { None } else { Some(&model.masks) }, vocab.as_deref())?;
    Ok(kept)
}
