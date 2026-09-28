//! Data + the step loop for the birth: token shards, batches, schedule,
//! checkpoints. Everything host-side and portable; the GPU work is in
//! `model::EmbryoGpu`.

use std::io::{Read, Write};
use std::path::Path;

/// A flat token stream (u16 little-endian on disk — any tokenizer with a
/// vocabulary ≤ 65536; the Embryo vocab is 32768).
pub struct Shard {
    pub tokens: Vec<u16>,
}

impl Shard {
    pub fn load(path: &Path) -> anyhow::Result<Shard> {
        let mut f = std::fs::File::open(path)?;
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() % 2 == 0,
            "shard {}: odd byte length",
            path.display()
        );
        let tokens = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Shard { tokens })
    }
    /// Bytes as tokens (vocab 256) — the zero-dependency smoke corpus.
    pub fn from_bytes(text: &[u8]) -> Shard {
        Shard {
            tokens: text.iter().map(|b| *b as u16).collect(),
        }
    }
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut f = std::fs::File::create(path)?;
        let mut bytes = Vec::with_capacity(self.tokens.len() * 2);
        for t in &self.tokens {
            bytes.extend_from_slice(&t.to_le_bytes());
        }
        f.write_all(&bytes)?;
        Ok(())
    }
}

/// Deterministic batch sampler: B random windows of T+1 tokens.
pub struct Sampler {
    pub b: usize,
    pub t: usize,
    state: u64,
}

impl Sampler {
    pub fn new(b: usize, t: usize, seed: u64) -> Sampler {
        Sampler {
            b,
            t,
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Fills tokens/targets ([B·T] each) from the shard.
    pub fn batch(&mut self, shard: &Shard, tokens: &mut Vec<u32>, targets: &mut Vec<u32>) {
        let n = shard.tokens.len();
        assert!(n > self.t + 1, "shard shorter than one window");
        tokens.clear();
        targets.clear();
        for _ in 0..self.b {
            let start = (self.next_u64() % (n - self.t - 1) as u64) as usize;
            let w = &shard.tokens[start..start + self.t + 1];
            tokens.extend(w[..self.t].iter().map(|x| *x as u32));
            targets.extend(w[1..].iter().map(|x| *x as u32));
        }
    }
    /// Fixed evenly spaced windows for a deterministic validation set.
    pub fn fixed_batch(
        shard: &Shard,
        b: usize,
        t: usize,
        index: usize,
        tokens: &mut Vec<u32>,
        targets: &mut Vec<u32>,
    ) {
        let n = shard.tokens.len();
        tokens.clear();
        targets.clear();
        for i in 0..b {
            let k = index * b + i;
            let start = (k * 7919) % (n - t - 1);
            let w = &shard.tokens[start..start + t + 1];
            tokens.extend(w[..t].iter().map(|x| *x as u32));
            targets.extend(w[1..].iter().map(|x| *x as u32));
        }
    }
}

/// Stream sampler for state carry-over (plan S6b, `--carry`): every batch
/// row follows ONE document stream — consecutive T-token windows of one
/// shard — and the call reports which rows (re)started a stream, i.e. whose
/// carried state must be cleared. A row restarts at the first call, after
/// `reset_every` windows (0 = never), when its shard runs out, and — with
/// `eot` set — in the window after one that contained the end-of-text id
/// (the runtime concatenates documents the same way inside a window).
pub struct StreamSampler {
    pub b: usize,
    pub t: usize,
    state: u64,
    cursors: Vec<Option<(usize, usize)>>, // (shard, position)
    windows: Vec<usize>,
    pending_reset: Vec<bool>,
    pub reset_every: usize,
    pub eot: Option<u32>,
    single: Option<Mix>,
}

impl StreamSampler {
    pub fn new(b: usize, t: usize, seed: u64, reset_every: usize, eot: Option<u32>) -> StreamSampler {
        StreamSampler {
            b,
            t,
            state: seed ^ 0x5354_5245_414D_0000 ^ 0x9E37_79B9_7F4A_7C15,
            cursors: vec![None; b],
            windows: vec![0; b],
            pending_reset: vec![false; b],
            reset_every,
            eot,
            single: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Next window of every row; returns `reset[b]` (true = fresh stream).
    pub fn batch_mix(&mut self, mix: &Mix, tokens: &mut Vec<u32>, targets: &mut Vec<u32>) -> Vec<bool> {
        tokens.clear();
        targets.clear();
        let mut reset = vec![false; self.b];
        for row in 0..self.b {
            let restart = match self.cursors[row] {
                None => true,
                Some((si, pos)) => {
                    self.pending_reset[row]
                        || (self.reset_every > 0 && self.windows[row] >= self.reset_every)
                        || pos + self.t + 1 > mix.shards[si].tokens.len()
                }
            };
            if restart {
                // shard by weight, then a random start inside it
                let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let mut acc = 0.0;
                let mut si = mix.shards.len() - 1;
                for (i, w) in mix.weights.iter().enumerate() {
                    acc += w;
                    if u < acc {
                        si = i;
                        break;
                    }
                }
                let n = mix.shards[si].tokens.len();
                assert!(n > self.t + 1, "shard shorter than one window");
                let start = (self.next_u64() % (n - self.t - 1) as u64) as usize;
                self.cursors[row] = Some((si, start));
                self.windows[row] = 0;
                self.pending_reset[row] = false;
                reset[row] = true;
            }
            let (si, pos) = self.cursors[row].unwrap();
            let w = &mix.shards[si].tokens[pos..pos + self.t + 1];
            tokens.extend(w[..self.t].iter().map(|x| *x as u32));
            targets.extend(w[1..].iter().map(|x| *x as u32));
            if let Some(e) = self.eot {
                if w[..self.t].iter().any(|&x| x as u32 == e) {
                    self.pending_reset[row] = true;
                }
            }
            self.cursors[row] = Some((si, pos + self.t));
            self.windows[row] += 1;
        }
        reset
    }
    /// Windows served so far in every row's current stream (1 = fresh).
    pub fn depths(&self) -> Vec<usize> {
        self.windows.clone()
    }
    /// Single-shard variant.
    pub fn batch(&mut self, shard: &Shard, tokens: &mut Vec<u32>, targets: &mut Vec<u32>) -> Vec<bool> {
        if self.single.as_ref().map_or(true, |m| m.shards[0].tokens.len() != shard.tokens.len()) {
            self.single = Some(Mix {
                shards: vec![Shard {
                    tokens: shard.tokens.clone(),
                }],
                weights: vec![1.0],
            });
        }
        let mix = self.single.take().unwrap();
        let r = self.batch_mix(&mix, tokens, targets);
        self.single = Some(mix);
        r
    }
}

/// Warmup + cosine learning-rate schedule.
pub fn lr_at(step: usize, total: usize, warmup: usize, peak: f32, floor: f32) -> f32 {
    if step < warmup {
        return peak * (step + 1) as f32 / warmup as f32;
    }
    let p = ((step - warmup) as f32 / (total.saturating_sub(warmup)).max(1) as f32).min(1.0);
    floor + 0.5 * (peak - floor) * (1.0 + (std::f32::consts::PI * p).cos())
}

/// Checkpoint: config JSON + raw f32 params (+ optional m/v) + named extra
/// blobs (the expert descriptors) — the plain trainer format; `.cmf` export
/// is the runtime's container.
pub fn save_checkpoint(
    path: &Path,
    cfg: &crate::model::EmbryoCfg,
    step: u32,
    params: &[f32],
    m: Option<&[f32]>,
    v: Option<&[f32]>,
    extras: &[(&str, &[f32])],
) -> anyhow::Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::File::create(&tmp)?;
    let ex: Vec<serde_json::Value> = extras
        .iter()
        .map(|(n, x)| serde_json::json!([n, x.len()]))
        .collect();
    let hdr = serde_json::json!({ "cfg": cfg, "step": step, "n": params.len(), "opt": m.is_some(), "extras": ex });
    let hs = serde_json::to_vec(&hdr)?;
    f.write_all(&(hs.len() as u64).to_le_bytes())?;
    f.write_all(&hs)?;
    let w = |f: &mut std::fs::File, x: &[f32]| -> anyhow::Result<()> {
        let bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) };
        f.write_all(bytes)?;
        Ok(())
    };
    w(&mut f, params)?;
    if let (Some(m), Some(v)) = (m, v) {
        w(&mut f, m)?;
        w(&mut f, v)?;
    }
    for (_, x) in extras {
        w(&mut f, x)?;
    }
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub struct Checkpoint {
    pub cfg: crate::model::EmbryoCfg,
    pub step: u32,
    pub params: Vec<f32>,
    pub m: Option<Vec<f32>>,
    pub v: Option<Vec<f32>>,
    pub extras: Vec<(String, Vec<f32>)>,
}

/// One tensor copied from the all-softmax donor during the layer-4 anchor
/// graft.  Shapes are explicit so a report cannot accidentally turn a
/// name-based copy into an undocumented positional migration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnchorGraftTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub source: &'static str,
}

/// Result of the deterministic second-anchor graft.  `checkpoint` contains
/// the candidate arena; all tensors not listed in `donor_tensors` are copied
/// by exact name and shape from the student checkpoint.
pub struct AnchorGraft {
    pub checkpoint: Checkpoint,
    pub student_params: usize,
    pub candidate_params: usize,
    pub student_tensors: usize,
    pub donor_tensors: Vec<AnchorGraftTensor>,
}

/// Provenance record for one tensor in the additive layer-4 GQA lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GqaLaneTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub source: &'static str,
}

/// Result of appending the layer-4 additive GQA lane.  The candidate keeps
/// the complete student arena as an exact prefix; only q/k/v are copied from
/// the all-softmax donor and the new output projection starts at zero.
pub struct GqaLaneGraft {
    pub checkpoint: Checkpoint,
    pub donor_tensors: Vec<GqaLaneTensor>,
}

/// Graft the compatible all-softmax twin's layer 4 (index 3) GQA projection
/// tensors into the one-anchor student, producing an `anchor_every=4`
/// candidate.  This is deliberately host-only: no Metal model is created and
/// no checkpoint or optimizer state is mutated in place.
///
/// Every candidate tensor is resolved by its stable `Layout::names` entry.
/// Matching student records (including optimizer moments) are copied exactly;
/// the only newly introduced records are layer-4 q/k/v/o from the donor and
/// their optimizer moments are zero.  Descriptor extras, step, and all other
/// checkpoint metadata remain student-owned.
pub fn graft_layer4_anchor_checkpoint(
    student: &Checkpoint,
    donor: &Checkpoint,
) -> anyhow::Result<AnchorGraft> {
    use std::collections::{HashMap, HashSet};

    anyhow::ensure!(
        student.cfg.anchor_every == 8,
        "student anchor_every must be 8 (got {})",
        student.cfg.anchor_every
    );
    anyhow::ensure!(
        !student.cfg.gqa_lane,
        "student must not already have gqa_lane enabled"
    );
    anyhow::ensure!(
        donor.cfg.anchor_every == 1,
        "donor anchor_every must be 1 (got {})",
        donor.cfg.anchor_every
    );
    anyhow::ensure!(student.cfg.layers > 3, "student must have a layer 4");
    anyhow::ensure!(donor.cfg.layers > 3, "donor must have a layer 4");

    // The donor and student may differ in anchor cadence and short-conv
    // implementation (the donor has no conv on its all-anchor path), but all
    // dimensions participating in the graft must agree exactly.
    macro_rules! same {
        ($field:ident) => {
            anyhow::ensure!(
                student.cfg.$field == donor.cfg.$field,
                "student/donor geometry mismatch for {}: {:?} vs {:?}",
                stringify!($field),
                student.cfg.$field,
                donor.cfg.$field
            );
        };
    }
    same!(vocab);
    same!(hidden);
    same!(layers);
    same!(heads);
    same!(nphase);
    same!(dv);
    same!(horizon_min);
    same!(horizon_max);
    same!(kappa_bias);
    same!(anchor_q_heads);
    same!(anchor_kv_heads);
    same!(anchor_hd);
    same!(rope_base);
    same!(experts);
    same!(inter);
    same!(head_clusters);
    same!(mtp_heads);
    same!(seq);
    same!(norm_eps);
    same!(learn_decay);
    same!(gdn_lane);

    let mut candidate_cfg = student.cfg.clone();
    candidate_cfg.anchor_every = 4;
    let student_lay = crate::model::Layout::new(&student.cfg);
    let donor_lay = crate::model::Layout::new(&donor.cfg);
    let candidate_lay = crate::model::Layout::new(&candidate_cfg);

    anyhow::ensure!(
        student.params.len() == student_lay.total,
        "student params length {} != layout {}",
        student.params.len(),
        student_lay.total
    );
    anyhow::ensure!(
        donor.params.len() == donor_lay.total,
        "donor params length {} != layout {}",
        donor.params.len(),
        donor_lay.total
    );
    anyhow::ensure!(
        student.m.is_some() == student.v.is_some(),
        "student optimizer state must contain both m and v or neither"
    );
    anyhow::ensure!(
        donor.m.is_some() == donor.v.is_some(),
        "donor optimizer state must contain both m and v or neither"
    );
    if let Some(m) = &student.m {
        anyhow::ensure!(
            m.len() == student_lay.total,
            "student m length {} != layout {}",
            m.len(),
            student_lay.total
        );
    }
    if let Some(v) = &student.v {
        anyhow::ensure!(
            v.len() == student_lay.total,
            "student v length {} != layout {}",
            v.len(),
            student_lay.total
        );
    }
    if let Some(m) = &donor.m {
        anyhow::ensure!(
            m.len() == donor_lay.total,
            "donor m length {} != layout {}",
            m.len(),
            donor_lay.total
        );
    }
    if let Some(v) = &donor.v {
        anyhow::ensure!(
            v.len() == donor_lay.total,
            "donor v length {} != layout {}",
            v.len(),
            donor_lay.total
        );
    }

    let smap: HashMap<&str, (usize, usize)> = student_lay
        .names
        .iter()
        .map(|(name, off, len)| (name.as_str(), (*off, *len)))
        .collect();
    let dmap: HashMap<&str, (usize, usize)> = donor_lay
        .names
        .iter()
        .map(|(name, off, len)| (name.as_str(), (*off, *len)))
        .collect();
    let expected_donor: [&str; 4] = [
        "layers.3.attn.q",
        "layers.3.attn.k",
        "layers.3.attn.v",
        "layers.3.attn.o",
    ];
    let expected_donor_set: HashSet<&str> = expected_donor.iter().copied().collect();
    let expected_student_removed: [&str; 6] = [
        "layers.3.hk.thq",
        "layers.3.hk.thk",
        "layers.3.hk.v",
        "layers.3.hk.kappa",
        "layers.3.hk.o",
        "layers.3.hk.conv",
    ];
    let expected_student_removed_set: HashSet<&str> =
        expected_student_removed.iter().copied().collect();

    let candidate_names: HashSet<&str> = candidate_lay
        .names
        .iter()
        .map(|(name, _, _)| name.as_str())
        .collect();
    let student_names: HashSet<&str> = smap.keys().copied().collect();
    let candidate_only: HashSet<&str> = candidate_names
        .difference(&student_names)
        .copied()
        .collect();
    anyhow::ensure!(
        candidate_only == expected_donor_set,
        "candidate introduced unexpected tensors: {:?}",
        candidate_only
    );
    let student_only: HashSet<&str> = student_names
        .difference(&candidate_names)
        .copied()
        .collect();
    anyhow::ensure!(
        student_only == expected_student_removed_set,
        "candidate removed unexpected student tensors: {:?}",
        student_only
    );

    let donor_shapes = |name: &str| -> Option<Vec<usize>> {
        let h = candidate_cfg.hidden;
        let qh = candidate_cfg.anchor_q_heads;
        let kvh = candidate_cfg.anchor_kv_heads;
        let hd = candidate_cfg.anchor_hd;
        match name {
            "layers.3.attn.q" => Some(vec![qh * hd, h]),
            "layers.3.attn.k" | "layers.3.attn.v" => Some(vec![kvh * hd, h]),
            "layers.3.attn.o" => Some(vec![h, qh * hd]),
            _ => None,
        }
    };

    // Start with a zeroed arena so every destination is proven to be filled
    // by one of the two explicit source maps below.
    let mut params = vec![0.0f32; candidate_lay.total];
    let mut donor_tensors = Vec::with_capacity(expected_donor.len());
    let mut copied_student = 0usize;
    let mut filled = vec![false; candidate_lay.total];
    for (name, no, nlen) in &candidate_lay.names {
        if let Some((so, slen)) = smap.get(name.as_str()) {
            anyhow::ensure!(
                *slen == *nlen,
                "same-name student shape mismatch for {name}: {slen} vs {nlen}"
            );
            params[*no..*no + *nlen].copy_from_slice(&student.params[*so..*so + *slen]);
            filled[*no..*no + *nlen].fill(true);
            copied_student += 1;
            continue;
        }
        anyhow::ensure!(
            expected_donor_set.contains(name.as_str()),
            "candidate tensor {name} has no student source and is not an authorized donor tensor"
        );
        let (doff, dlen) = dmap
            .get(name.as_str())
            .copied()
            .ok_or_else(|| anyhow::anyhow!("donor is missing required tensor {name}"))?;
        anyhow::ensure!(
            dlen == *nlen,
            "donor tensor {name} length mismatch: {dlen} vs candidate {nlen}"
        );
        let shape = donor_shapes(name)
            .ok_or_else(|| anyhow::anyhow!("missing explicit shape for donor tensor {name}"))?;
        anyhow::ensure!(
            shape.iter().product::<usize>() == *nlen,
            "donor tensor {name} shape {:?} does not match length {nlen}",
            shape
        );
        params[*no..*no + *nlen].copy_from_slice(&donor.params[doff..doff + dlen]);
        filled[*no..*no + *nlen].fill(true);
        donor_tensors.push(AnchorGraftTensor {
            name: name.clone(),
            shape,
            source: "donor.layer4.gqa",
        });
    }
    anyhow::ensure!(
        filled.iter().all(|x| *x),
        "candidate arena has unfilled offsets"
    );
    anyhow::ensure!(
        donor_tensors.len() == expected_donor.len(),
        "expected {} donor tensors, copied {}",
        expected_donor.len(),
        donor_tensors.len()
    );
    donor_tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let copy_moment = |src: &Option<Vec<f32>>| -> anyhow::Result<Option<Vec<f32>>> {
        let Some(src) = src else { return Ok(None) };
        anyhow::ensure!(
            src.len() == student_lay.total,
            "student optimizer moment length {} != layout {}",
            src.len(),
            student_lay.total
        );
        let mut dst = vec![0.0f32; candidate_lay.total];
        for (name, no, nlen) in &candidate_lay.names {
            if let Some((so, slen)) = smap.get(name.as_str()) {
                anyhow::ensure!(
                    *slen == *nlen,
                    "moment shape mismatch for {name}: {slen} vs {nlen}"
                );
                dst[*no..*no + *nlen].copy_from_slice(&src[*so..*so + *slen]);
            }
            // Authorized donor-only moments intentionally remain exact zero.
        }
        Ok(Some(dst))
    };

    anyhow::ensure!(
        params.iter().all(|x| x.is_finite()),
        "grafted params contain non-finite values"
    );
    let checkpoint = Checkpoint {
        cfg: candidate_cfg,
        step: student.step,
        params,
        m: copy_moment(&student.m)?,
        v: copy_moment(&student.v)?,
        extras: student.extras.clone(),
    };
    Ok(AnchorGraft {
        student_params: student_lay.total,
        candidate_params: candidate_lay.total,
        student_tensors: copied_student,
        donor_tensors,
        checkpoint,
    })
}

/// Append the phase-2 GDN correction tail to a checkpoint while preserving
/// every legacy parameter and AdamW moment bit-for-bit.  The returned
/// checkpoint is ready for `EmbryoGpu::new`; new lane moments are zero and the
/// deterministic lane initialization comes from `seed` (the experimental
/// lane uses exact-zero output projection plus unit residual gain so its
/// first backward pass is not suppressed).
pub fn append_gdn_lane_checkpoint(ck: &Checkpoint, seed: u64) -> anyhow::Result<Checkpoint> {
    anyhow::ensure!(!ck.cfg.gdn_lane, "checkpoint already has gdn_lane enabled");
    let old_lay = crate::model::Layout::new(&ck.cfg);
    let mut cfg = ck.cfg.clone();
    cfg.gdn_lane = true;
    let new_lay = crate::model::Layout::new(&cfg);
    anyhow::ensure!(
        ck.params.len() == old_lay.total,
        "checkpoint parameter length does not match its layout"
    );
    let mut params = crate::model::init_params(&cfg, &new_lay, seed);
    let old_map: std::collections::HashMap<&str, (usize, usize)> = old_lay
        .names
        .iter()
        .map(|(n, o, nlen)| (n.as_str(), (*o, *nlen)))
        .collect();
    for (name, no, nlen) in &new_lay.names {
        if let Some((oo, olen)) = old_map.get(name.as_str()) {
            if *olen == *nlen {
                params[*no..*no + *nlen].copy_from_slice(&ck.params[*oo..*oo + *olen]);
            }
        }
    }
    let copy_moment = |src: &Option<Vec<f32>>| -> anyhow::Result<Option<Vec<f32>>> {
        let Some(src) = src else { return Ok(None) };
        anyhow::ensure!(
            src.len() == old_lay.total,
            "optimizer moment length does not match checkpoint layout"
        );
        let mut dst = vec![0.0f32; new_lay.total];
        for (name, no, nlen) in &new_lay.names {
            if let Some((oo, olen)) = old_map.get(name.as_str()) {
                if *olen == *nlen {
                    dst[*no..*no + *nlen].copy_from_slice(&src[*oo..*oo + *olen]);
                }
            }
        }
        Ok(Some(dst))
    };
    Ok(Checkpoint {
        cfg,
        step: ck.step,
        params,
        m: copy_moment(&ck.m)?,
        v: copy_moment(&ck.v)?,
        extras: ck.extras.clone(),
    })
}

/// Continue a legacy (full-causal) checkpoint under the bounded anchor
/// `swa_sink_v1` (the S4 probe): switch the config to `window`/`sink`/
/// `train_windows` and, when sinks are requested, extend the arena with the
/// new `layers.{l}.attn.sink_k`/`sink_v` tensors.  Every legacy tensor and
/// both AdamW moments are copied by name (the layout re-numbers offsets
/// after each anchor's projections, so this is a name-based migration like
/// growth, not a positional one); the sinks take their deterministic
/// `init_params` draw (`sink_k ~ N(0, 0.02)`, `sink_v = 0`) with zero
/// moments and the optimizer step is preserved.  A checkpoint that already
/// carries sinks may only keep its sink count (the tensors would otherwise
/// change shape).
pub fn append_anchor_sinks_checkpoint(
    ck: &Checkpoint,
    window: usize,
    sink: usize,
    train_windows: &[usize],
    seed: u64,
) -> anyhow::Result<Checkpoint> {
    anyhow::ensure!(window > 0, "a bounded anchor needs window >= 1");
    anyhow::ensure!(
        ck.cfg.anchor_sink == 0 || ck.cfg.anchor_sink == sink,
        "checkpoint already has {} sinks per KV head; cannot change to {sink}",
        ck.cfg.anchor_sink
    );
    let old_lay = crate::model::Layout::new(&ck.cfg);
    anyhow::ensure!(
        ck.params.len() == old_lay.total,
        "checkpoint parameter length does not match its layout"
    );
    let mut cfg = ck.cfg.clone();
    cfg.anchor_window = window;
    cfg.anchor_sink = sink;
    cfg.anchor_train_windows = train_windows.to_vec();
    cfg.check_anchor().map_err(|e| anyhow::anyhow!(e))?;
    let new_lay = crate::model::Layout::new(&cfg);
    let mut params = crate::model::init_params(&cfg, &new_lay, seed);
    let old_map: std::collections::HashMap<&str, (usize, usize)> = old_lay
        .names
        .iter()
        .map(|(n, o, nlen)| (n.as_str(), (*o, *nlen)))
        .collect();
    let copy_by_name = |src: &[f32], dst: &mut [f32]| -> usize {
        let mut copied = 0usize;
        for (name, no, nlen) in &new_lay.names {
            if let Some((oo, olen)) = old_map.get(name.as_str()) {
                if *olen == *nlen {
                    dst[*no..*no + *nlen].copy_from_slice(&src[*oo..*oo + *olen]);
                    copied += 1;
                }
            }
        }
        copied
    };
    let copied = copy_by_name(&ck.params, &mut params);
    anyhow::ensure!(
        copied == old_lay.names.len(),
        "only {copied} of {} legacy tensors found in the bounded layout",
        old_lay.names.len()
    );
    let copy_moment = |src: &Option<Vec<f32>>| -> anyhow::Result<Option<Vec<f32>>> {
        let Some(src) = src else { return Ok(None) };
        anyhow::ensure!(
            src.len() == old_lay.total,
            "optimizer moment length does not match checkpoint layout"
        );
        let mut dst = vec![0.0f32; new_lay.total];
        copy_by_name(src, &mut dst);
        Ok(Some(dst))
    };
    Ok(Checkpoint {
        cfg,
        step: ck.step,
        params,
        m: copy_moment(&ck.m)?,
        v: copy_moment(&ck.v)?,
        extras: ck.extras.clone(),
    })
}

/// Append the exact-identity layer-4 additive GQA lane.  The donor must be
/// the compatible all-softmax twin (`anchor_every=1`); its layer-4 q/k/v are
/// copied by explicit name and shape into the new `layers.3.gqa.*` tail.  The
/// output projection is new and exact zero, so the original hybrid path and
/// logits remain unchanged before training.  Legacy parameters, moments,
/// descriptor extras, and optimizer step are copied byte-for-byte.
pub fn append_gqa_lane_checkpoint(
    student: &Checkpoint,
    donor: &Checkpoint,
) -> anyhow::Result<GqaLaneGraft> {
    use std::collections::HashMap;
    anyhow::ensure!(
        !student.cfg.gqa_lane,
        "student already has gqa_lane enabled"
    );
    anyhow::ensure!(!donor.cfg.gqa_lane, "donor must not have gqa_lane enabled");
    anyhow::ensure!(
        student.cfg.anchor_every == 8,
        "student anchor_every must be 8"
    );
    anyhow::ensure!(donor.cfg.anchor_every == 1, "donor anchor_every must be 1");
    anyhow::ensure!(
        student.cfg.layers > 3 && donor.cfg.layers > 3,
        "both checkpoints need layer 4"
    );
    anyhow::ensure!(
        !student.cfg.is_anchor(3),
        "student layer 4 must remain hybrid"
    );
    macro_rules! same {
        ($field:ident) => {
            anyhow::ensure!(
                student.cfg.$field == donor.cfg.$field,
                "geometry mismatch for {}",
                stringify!($field)
            );
        };
    }
    same!(vocab);
    same!(hidden);
    same!(layers);
    same!(heads);
    same!(nphase);
    same!(dv);
    same!(horizon_min);
    same!(horizon_max);
    same!(kappa_bias);
    same!(anchor_q_heads);
    same!(anchor_kv_heads);
    same!(anchor_hd);
    same!(rope_base);
    same!(experts);
    same!(inter);
    same!(head_clusters);
    same!(mtp_heads);
    same!(seq);
    same!(norm_eps);
    // The all-softmax donor has no hybrid short-conv path; conv_k is
    // irrelevant to its layer-4 q/k/v geometry and the student path remains
    // unchanged.
    same!(learn_decay);
    same!(gdn_lane);
    let old_lay = crate::model::Layout::new(&student.cfg);
    let donor_lay = crate::model::Layout::new(&donor.cfg);
    let mut cfg = student.cfg.clone();
    cfg.gqa_lane = true;
    let new_lay = crate::model::Layout::new(&cfg);
    anyhow::ensure!(
        student.params.len() == old_lay.total,
        "student params/layout mismatch"
    );
    anyhow::ensure!(
        donor.params.len() == donor_lay.total,
        "donor params/layout mismatch"
    );
    anyhow::ensure!(
        student.m.is_some() == student.v.is_some(),
        "student optimizer state incomplete"
    );
    anyhow::ensure!(
        donor.m.is_some() == donor.v.is_some(),
        "donor optimizer state incomplete"
    );
    if let Some(m) = &student.m {
        anyhow::ensure!(m.len() == old_lay.total, "student m/layout mismatch");
    }
    if let Some(v) = &student.v {
        anyhow::ensure!(v.len() == old_lay.total, "student v/layout mismatch");
    }
    let old_map: HashMap<&str, (usize, usize)> = old_lay
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    let donor_map: HashMap<&str, (usize, usize)> = donor_lay
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    let mut params = vec![0.0f32; new_lay.total];
    let mut donor_tensors = Vec::new();
    for (name, no, nlen) in &new_lay.names {
        if let Some((oo, olen)) = old_map.get(name.as_str()) {
            anyhow::ensure!(*olen == *nlen, "legacy tensor shape changed for {name}");
            params[*no..*no + *nlen].copy_from_slice(&student.params[*oo..*oo + *nlen]);
            continue;
        }
        let donor_name = match name.as_str() {
            "layers.3.gqa.q" => "layers.3.attn.q",
            "layers.3.gqa.k" => "layers.3.attn.k",
            "layers.3.gqa.v" => "layers.3.attn.v",
            "layers.3.gqa.o" => {
                params[*no..*no + *nlen].fill(0.0);
                continue;
            }
            _ => return Err(anyhow::anyhow!("unexpected candidate-only tensor {name}")),
        };
        let (doff, dlen) = donor_map
            .get(donor_name)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("donor missing {donor_name}"))?;
        anyhow::ensure!(dlen == *nlen, "donor tensor shape mismatch for {name}");
        params[*no..*no + *nlen].copy_from_slice(&donor.params[doff..doff + dlen]);
        let shape = match name.as_str() {
            "layers.3.gqa.q" => vec![cfg.anchor_q_heads * cfg.anchor_hd, cfg.hidden],
            _ => vec![cfg.anchor_kv_heads * cfg.anchor_hd, cfg.hidden],
        };
        donor_tensors.push(GqaLaneTensor {
            name: name.clone(),
            shape,
            source: "donor.layer4.gqa",
        });
    }
    anyhow::ensure!(donor_tensors.len() == 3, "expected donor q/k/v only");
    let copy_moment = |src: &Option<Vec<f32>>| -> anyhow::Result<Option<Vec<f32>>> {
        let Some(src) = src else { return Ok(None) };
        anyhow::ensure!(
            src.len() == old_lay.total,
            "optimizer moment/layout mismatch"
        );
        let mut dst = vec![0.0f32; new_lay.total];
        for (name, no, nlen) in &new_lay.names {
            if let Some((oo, olen)) = old_map.get(name.as_str()) {
                anyhow::ensure!(*olen == *nlen, "moment shape changed for {name}");
                dst[*no..*no + *nlen].copy_from_slice(&src[*oo..*oo + *nlen]);
            }
        }
        Ok(Some(dst))
    };
    anyhow::ensure!(
        params.iter().all(|x| x.is_finite()),
        "gqa lane params non-finite"
    );
    Ok(GqaLaneGraft {
        checkpoint: Checkpoint {
            cfg,
            step: student.step,
            params,
            m: copy_moment(&student.m)?,
            v: copy_moment(&student.v)?,
            extras: student.extras.clone(),
        },
        donor_tensors,
    })
}

pub fn load_checkpoint(path: &Path) -> anyhow::Result<Checkpoint> {
    let mut f = std::fs::File::open(path)?;
    let mut b8 = [0u8; 8];
    f.read_exact(&mut b8)?;
    let hl = u64::from_le_bytes(b8) as usize;
    let mut hs = vec![0u8; hl];
    f.read_exact(&mut hs)?;
    let hdr: serde_json::Value = serde_json::from_slice(&hs)?;
    let cfg: crate::model::EmbryoCfg = serde_json::from_value(hdr["cfg"].clone())?;
    let step = hdr["step"].as_u64().unwrap_or(0) as u32;
    let n = hdr["n"].as_u64().unwrap_or(0) as usize;
    let opt = hdr["opt"].as_bool().unwrap_or(false);
    let rd = |f: &mut std::fs::File, n: usize| -> anyhow::Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        f.read_exact(&mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let params = rd(&mut f, n)?;
    let (m, v) = if opt {
        (Some(rd(&mut f, n)?), Some(rd(&mut f, n)?))
    } else {
        (None, None)
    };
    let mut extras = Vec::new();
    if let Some(list) = hdr["extras"].as_array() {
        for e in list {
            let name = e[0].as_str().unwrap_or("").to_string();
            let len = e[1].as_u64().unwrap_or(0) as usize;
            extras.push((name, rd(&mut f, len)?));
        }
    }
    Ok(Checkpoint {
        cfg,
        step,
        params,
        m,
        v,
        extras,
    })
}

/// Several shards mixed by weight (each sequence of a batch is drawn from
/// one shard, chosen by weight) — the birth's corpus mix (en / ru / code / math).
pub struct Mix {
    pub shards: Vec<Shard>,
    pub weights: Vec<f64>,
}

impl Mix {
    /// Parse `path[:weight]` specs and load; splits the last `holdout`
    /// fraction of every shard into a validation shard.
    pub fn load(specs: &[String], holdout: f64, seq: usize) -> anyhow::Result<(Mix, Shard)> {
        let mut shards = Vec::new();
        let mut weights = Vec::new();
        let mut val = Vec::new();
        for s in specs {
            let (path, w) = match s.rsplit_once(':') {
                Some((p, w)) if w.parse::<f64>().is_ok() && !p.is_empty() => {
                    (p.to_string(), w.parse::<f64>().unwrap())
                }
                _ => (s.clone(), 1.0),
            };
            let mut sh = Shard::load(Path::new(&path))?;
            let n = sh.tokens.len();
            let cut = n - ((n as f64 * holdout) as usize).max(seq + 2).min(n / 2);
            val.extend_from_slice(&sh.tokens[cut..]);
            sh.tokens.truncate(cut);
            eprintln!(
                "shard {path}: {:.1} M train tokens, weight {w}",
                sh.tokens.len() as f64 / 1e6
            );
            shards.push(sh);
            weights.push(w);
        }
        let tot: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= tot;
        }
        Ok((Mix { shards, weights }, Shard { tokens: val }))
    }
    pub fn total_tokens(&self) -> usize {
        self.shards.iter().map(|s| s.tokens.len()).sum()
    }
}

impl Sampler {
    /// B windows, each from a shard chosen by weight.
    pub fn batch_mix(&mut self, mix: &Mix, tokens: &mut Vec<u32>, targets: &mut Vec<u32>) {
        tokens.clear();
        targets.clear();
        for _ in 0..self.b {
            let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let mut acc = 0.0;
            let mut si = mix.shards.len() - 1;
            for (i, w) in mix.weights.iter().enumerate() {
                acc += w;
                if u < acc {
                    si = i;
                    break;
                }
            }
            let sh = &mix.shards[si];
            let n = sh.tokens.len();
            let start = (self.next_u64() % (n - self.t - 1) as u64) as usize;
            let w = &sh.tokens[start..start + self.t + 1];
            tokens.extend(w[..self.t].iter().map(|x| *x as u32));
            targets.extend(w[1..].iter().map(|x| *x as u32));
        }
    }
}
