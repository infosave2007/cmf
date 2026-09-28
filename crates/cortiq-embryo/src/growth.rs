//! Growth as records (docs §2 "Рост", §4.3; SPEC_GROWTH_RECORDS §3 +
//! addendum): the genome gains `K` expert slots in the GROWN layers — the
//! k-th new expert of a layer is a copy of that layer's k-th hottest TRUNK
//! expert plus noise — for a record growth "hottest" is measured ON THE
//! GROWTH CORPUS (the trunk expert winning the most of its tokens there,
//! [`sources_by_corpus_wins`]); the legacy order is the most negative
//! balancing bias ([`hottest_trunk_order`]); never a grown one.
//! Its DESCRIPTOR is initialised from the growth corpus ([`cluster_inits`]):
//! `μ_new` = the mean of the layer's MoE inputs the source wins there,
//! `U_new` = the top-k eigenvectors of their covariance (a second copy of
//! the same source takes its own half of that cluster along its principal
//! direction). The runtime score `bias − ‖(x−μ)⊥U‖²` is invariant to a
//! shift of μ inside span(U), so a copy with the source's own U and a μ
//! shifted along `u₀` scored every token exactly like the source — the
//! legacy path (no corpus witness: the sleep daemon) now shifts along a
//! random direction OUTSIDE span(U). `--source-mode novel` ([`SourceMode`])
//! takes the other road — P1 novelty → P3/P9 OOD cluster → growth: a
//! token is novel for the trunk when its best trunk reconstruction error
//! exceeds `τ_l`, the `--novel-quantile` of that error on the GENERAL
//! shard ([`trunk_min_errors`], [`novel_taus`]); the novel tokens of the
//! growth corpus ([`novel_sets`]) are K-means-clustered and copy `k` takes
//! cluster k's mean / principal subspace and the weights of the trunk
//! expert hottest on that cluster ([`novel_inits`]); its descriptor stays
//! frozen through training by default ([`DescMode`]) and the record
//! reports `novel_coverage` = the share of the novel corpus tokens it wins
//! ([`novel_coverage`]). Measured on the real genome, the `hottest` copy
//! adapted to a corpus that is mostly ordinary language became a second
//! general expert (30–34 % of the general tokens without a shell).
//! Only the new experts (weights, μ EMA unless frozen)
//! then train on the growth corpus on a DROPLESS instance (every token
//! keeps its routed expert, as at runtime), gated on held-out loss against
//! the pre-growth GENOME; their balancing bias is pinned for the whole
//! training ([`BiasMode`]: 0, or the source's balancing bias so the copy
//! ties with its source instead of out-scoring it) — the value the record
//! stores — so the record routes at runtime exactly as it trained. Every
//! old tensor and descriptor stays byte-identical.
//!
//! After training the host recomputes the routing of every token of the
//! growth corpus with EXACTLY the runtime formula
//! (`cortiq_engine::pipeline::Resonance::scores`: `bias_e − ‖(x−μ_e)⊥U_e‖²`,
//! argmax, lower index on ties) from the layer's MoE input
//! (`EmbryoGpu::ffn_input_host`): the `shell` of a grown expert = the
//! `shell_quantile` of the reconstruction errors of the tokens it wins
//! ([`ShellMode::WonQuantile`]) or, calibrated on the general shard, the
//! quantile that admits at most `shell_target_shift` of the layer's
//! general tokens ([`ShellMode::GeneralTarget`], [`shells_general_target`]);
//! the `routing_shift` on a general shard = the fraction of tokens a grown
//! expert wins (without / with the shell); the `coverage` on the growth
//! held-out = the fraction of tokens inside at least one shell. `reshell`
//! redoes this tail from a saved trained checkpoint ([`shrink_experts`]
//! recovers the pre-growth checkpoint it binds with).
//!
//! [`write_growth_record`] appends the grown experts as an `expert_append`
//! record (status `quarantine`) to a COPY of the genome file — a true tail
//! append through `CmfModel::append_skill`: trunk hash, directory entries
//! and prefix bytes of the base are unchanged (verified). The legacy
//! full-genome growth ([`grow_experts`], [`train_new_experts`], `--export`)
//! stays for the sleep daemon: it changes `arch.moe.num_experts` and is NOT
//! a genome record.

use crate::model::{EmbryoCfg, EmbryoGpu, FfnOffs, LayerOffs, Layout, MOE_K, gauss_vec};
use crate::train::{Checkpoint, Sampler, Shard};
use cortiq_core::format::{CmfHeader, CmfModel, SkillRecord, TensorSpec};
use cortiq_core::knowledge::{
    ExpertAppend, SkillBound, expert_append_base, expert_append_layout,
    expert_append_state_effect, expert_leaf, genome_moe_experts, moe_layers, skill_kind,
    trunk_expert_rank,
};
use cortiq_core::types::TensorDtype;
use std::path::Path;
use std::time::Instant;

/// Balancing bias of an INERT grown slot: the trainer's arena has the same
/// number of experts in every layer, so a layer outside `--layers` receives
/// the slots too; with this bias their score `raw − bias` is ~1e30 and
/// they never win (the record never carries them; a `--export`ed full
/// genome keeps them inert through `Resonance::scores` = `bias − err`).
pub const DEAD_BIAS: f32 = -1.0e30;

/// The runtime's resident graph handles at most this many experts per
/// layer (`cortiq-engine` refuses the graph above it and falls back to the
/// slow per-op path): `E0 + K` must not exceed it.
pub const MAX_RUNTIME_EXPERTS: usize = 8;

/// Growth recipe name written into the record's origin.
pub const RECIPE_GROWTH: &str =
    "expert_append: K copies of trunk experts (`source_mode`: `hottest` = the K experts hottest on the growth corpus, \
     descriptors μ = mean / U = top-k eigenvectors of the tokens they win; `novel` = K-means clusters of the corpus \
     tokens novel for the trunk (best trunk reconstruction error above τ, the `novel_quantile` of that error on the \
     general shard), descriptors from the clusters, weights of the expert hottest on each cluster), new experts \
     trained dropless (bias pinned: 0 or the source's, `bias_mode`; descriptors adapting or frozen, `desc_mode`), \
     gated against the genome, shells = quantile of won errors (`won-quantile`) or calibrated on the general shard \
     (`general-target`, `shell_mode`)";

/// What a grown expert's shell is calibrated on (`--shell-mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShellMode {
    /// The `shell_quantile` of the reconstruction errors of ALL the growth
    /// tokens the expert wins — as wide as the source's whole cluster when
    /// the copy out-scores its source.
    WonQuantile,
    /// Calibrated on the GENERAL shard: an expert that captures more than
    /// `shell_target_shift` of the layer's general tokens (no shell) gets
    /// the `(target / share)`-quantile of the errors of those captured
    /// tokens, so the shell admits at most `target` of them; one under the
    /// target keeps its won-quantile shell. Requires `--general`.
    GeneralTarget,
}

impl ShellMode {
    pub fn parse(s: &str) -> anyhow::Result<ShellMode> {
        match s {
            "won-quantile" => Ok(ShellMode::WonQuantile),
            "general-target" => Ok(ShellMode::GeneralTarget),
            _ => anyhow::bail!("--shell-mode {s}: want won-quantile | general-target"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            ShellMode::WonQuantile => "won-quantile",
            ShellMode::GeneralTarget => "general-target",
        }
    }
}

/// The balancing bias a grown expert carries, frozen for the whole
/// training and written into the record (`--bias-mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BiasMode {
    /// `desc.bias = 0`: against the NEGATIVE balancing biases of the trunk
    /// the copy out-scores its source on the source's whole cluster.
    Zero,
    /// `desc.bias` = the SOURCE trunk expert's balancing bias: the copy
    /// ties with its source at insertion instead of out-scoring it.
    Source,
}

impl BiasMode {
    pub fn parse(s: &str) -> anyhow::Result<BiasMode> {
        match s {
            "zero" => Ok(BiasMode::Zero),
            "source" => Ok(BiasMode::Source),
            _ => anyhow::bail!("--bias-mode {s}: want zero | source"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            BiasMode::Zero => "zero",
            BiasMode::Source => "source",
        }
    }
}

/// Where the K copies of a grown layer come from and what their
/// descriptors are initialised with (`--source-mode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceMode {
    /// Copy `k` = the trunk expert with the k-th most wins on the growth
    /// corpus; its descriptor = the mean / principal subspace of the
    /// tokens that source wins there ([`sources_by_corpus_wins`],
    /// [`cluster_inits`]). Measured on the real genome: on a corpus that
    /// is mostly ordinary language the copy becomes a second GENERAL
    /// expert (30–34 % of the general tokens without a shell).
    Hottest,
    /// The P1 → P3/P9 path: a token is NOVEL for the trunk in layer `l`
    /// when its best trunk reconstruction error `min_e ‖(x−μ_e)⊥U_e‖²`
    /// exceeds `τ_l`, the `--novel-quantile` of that error on the GENERAL
    /// shard ([`trunk_min_errors`], [`novel_taus`]); the novel tokens of
    /// the growth corpus ([`novel_sets`]) are clustered (K-means, K
    /// clusters) and copy `k` takes cluster k's mean / principal subspace
    /// as its descriptor and the weights of the trunk expert hottest ON
    /// THAT CLUSTER ([`novel_inits`]). Needs `--general`.
    Novel,
}

impl SourceMode {
    pub fn parse(s: &str) -> anyhow::Result<SourceMode> {
        match s {
            "hottest" => Ok(SourceMode::Hottest),
            "novel" => Ok(SourceMode::Novel),
            _ => anyhow::bail!("--source-mode {s}: want hottest | novel"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            SourceMode::Hottest => "hottest",
            SourceMode::Novel => "novel",
        }
    }
}

/// Whether the grown experts' descriptors move during training
/// (`--desc-mode`). The balancing bias is pinned either way
/// ([`BiasMode`]); the trunk descriptors never move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DescMode {
    /// The μ of every grown expert follows the EMA of the tokens it wins
    /// while training (the trainer's online descriptor update) — on a
    /// corpus that is mostly ordinary language it drifts to the general
    /// tokens the copy wins. The default of [`SourceMode::Hottest`].
    Adapt,
    /// μ / U (and the bias) of the grown experts stay at their
    /// initialisation for the whole training: the record's descriptor is
    /// exactly the novel cluster's. The default of [`SourceMode::Novel`].
    Frozen,
}

impl DescMode {
    pub fn parse(s: &str) -> anyhow::Result<DescMode> {
        match s {
            "adapt" => Ok(DescMode::Adapt),
            "frozen" => Ok(DescMode::Frozen),
            _ => anyhow::bail!("--desc-mode {s}: want adapt | frozen"),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            DescMode::Adapt => "adapt",
            DescMode::Frozen => "frozen",
        }
    }
    /// The default of a source mode: `adapt` for `hottest`, `frozen` for
    /// `novel`.
    pub fn default_for(m: SourceMode) -> DescMode {
        match m {
            SourceMode::Hottest => DescMode::Adapt,
            SourceMode::Novel => DescMode::Frozen,
        }
    }
}

fn ffn_offs(lo: &LayerOffs) -> &FfnOffs {
    match lo {
        LayerOffs::Mixer { ffn, .. } | LayerOffs::Anchor { ffn, .. } | LayerOffs::Gdn { ffn, .. } => {
            ffn
        }
    }
}

// ───────────────────────── surgery ─────────────────────────

/// What to grow.
#[derive(Clone, Debug)]
pub struct GrowSpec {
    /// `K ≥ 1` new experts per grown layer.
    pub experts: usize,
    /// The grown layers (any non-empty subset of the MoE layers; every
    /// layer of the Embryo carries routed experts).
    pub layers: Vec<usize>,
    pub noise: f32,
    pub shift: f32,
    pub seed: u64,
    /// Grown experts start at bias 0 (and stay there under
    /// `bias_frozen_from` in [`train_grown_experts`]) — the value their
    /// record stores ([`BiasMode::Zero`]). `false` = the copy inherits the
    /// source's bias: frozen it is [`BiasMode::Source`], trained it is the
    /// legacy full-genome growth.
    pub zero_bias: bool,
}

/// Distinct ascending layers inside `0..n_layers`, non-empty.
pub fn check_layers(n_layers: usize, layers: &[usize]) -> anyhow::Result<Vec<usize>> {
    anyhow::ensure!(!layers.is_empty(), "--layers is empty");
    let mut v = layers.to_vec();
    v.sort_unstable();
    for w in v.windows(2) {
        anyhow::ensure!(w[0] != w[1], "duplicate layer {} in {layers:?}", w[0]);
    }
    if let Some(&l) = v.last() {
        anyhow::ensure!(
            l < n_layers,
            "layer {l} out of range: the genome has {n_layers} layers"
        );
    }
    Ok(v)
}

/// Trunk experts of one layer ordered hottest first (most negative
/// balancing bias; lower index on ties).
pub fn hottest_trunk_order(bias: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..bias.len()).collect();
    order.sort_by(|&a, &b| {
        bias[a]
            .partial_cmp(&bias[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// The data-driven descriptor of one grown expert ([`cluster_inits`]):
/// the mean and the top-`MOE_K` subspace (orthonormal rows, `[MOE_K, H]`)
/// of the growth-corpus tokens its trunk source wins in that layer.
/// `rows == 0` = no witness (the surgery falls back to the legacy copy).
#[derive(Clone, Debug, Default)]
pub struct ClusterInit {
    pub mu: Vec<f32>,
    pub u: Vec<f32>,
    /// Tokens the mean / covariance were taken from.
    pub rows: usize,
}

/// The trunk sources of a growth, per GROWN layer (ascending) and copy:
/// expert `E0 + k` copies the k-th hottest trunk expert (`k mod E0` when
/// `K > E0`).
pub fn grown_sources(ck: &Checkpoint, spec: &GrowSpec) -> anyhow::Result<Vec<Vec<usize>>> {
    let e0 = ck.cfg.experts;
    anyhow::ensure!(e0 >= 1, "the genome has no routed experts to grow from");
    anyhow::ensure!(spec.experts >= 1, "--experts must be ≥ 1");
    let layers = check_layers(ck.cfg.layers, &spec.layers)?;
    let b0 = ck
        .extras
        .iter()
        .find(|(n, _)| n == "desc.bias")
        .map(|(_, x)| x.clone())
        .unwrap_or_else(|| vec![0.0; ck.cfg.layers * e0]);
    anyhow::ensure!(
        b0.len() == ck.cfg.layers * e0,
        "checkpoint desc.bias has {} values for {} layers × {e0} experts",
        b0.len(),
        ck.cfg.layers
    );
    Ok(layers
        .iter()
        .map(|&l| {
            let order = hottest_trunk_order(&b0[l * e0..(l + 1) * e0]);
            (0..spec.experts).map(|kk| order[kk % e0]).collect()
        })
        .collect())
}

/// A unit vector OUTSIDE span(U): a seeded Gaussian direction with its
/// projection on every (non-zero) row of `u` (`[k, H]`) removed. The
/// runtime score is invariant to a shift of μ inside span(U), so only this
/// component moves a copy's reconstruction error away from its source's.
fn direction_outside_subspace(seed: u64, u: &[f32], k: usize, h: usize) -> Vec<f32> {
    for attempt in 0..8u64 {
        let mut dir = gauss_vec(seed.wrapping_add(1000 + attempt), h);
        for i in 0..k {
            let row = &u[i * h..(i + 1) * h];
            let nn: f32 = row.iter().map(|x| x * x).sum();
            if nn < 1e-12 {
                continue;
            }
            let dot: f32 = dir.iter().zip(row).map(|(a, b)| a * b).sum();
            for (d, r) in dir.iter_mut().zip(row) {
                *d -= dot / nn * r;
            }
        }
        let n2: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n2 > 1e-6 {
            for d in &mut dir {
                *d /= n2;
            }
            return dir;
        }
    }
    vec![0.0; h]
}

/// Host-side surgery: E0 → E0+K experts in every layer of the arena.
/// Returns the grown checkpoint and, per layer, the trunk sources copied
/// (empty for a layer outside `spec.layers`, whose slots are inert:
/// [`DEAD_BIAS`]). Expert `E0 + k` of a grown layer copies the k-th
/// hottest TRUNK expert (`k mod E0` when `K > E0`); its descriptor is the
/// legacy copy (μ shifted outside span(U), U copied) — a record growth
/// passes the corpus witness to [`grow_experts_init`] instead.
pub fn grow_experts_k(ck: &Checkpoint, spec: &GrowSpec) -> anyhow::Result<(Checkpoint, Vec<Vec<usize>>)> {
    grow_experts_init(ck, spec, None)
}

/// [`grow_experts_k`] with the descriptors of the new experts initialised
/// from the growth corpus: `inits[grown layer index][k]` (see
/// [`cluster_inits`]); an entry with `rows == 0` (or a missing one) falls
/// back to the legacy copy of that expert.
pub fn grow_experts_init(
    ck: &Checkpoint,
    spec: &GrowSpec,
    inits: Option<&[Vec<ClusterInit>]>,
) -> anyhow::Result<(Checkpoint, Vec<Vec<usize>>)> {
    let sources = grown_sources(ck, spec)?;
    grow_experts_from(ck, spec, &sources, inits)
}

/// [`grow_experts_init`] with EXPLICIT trunk sources: `sources[grown layer
/// index][k]` is the trunk expert that copy `E0 + k` starts from (a record
/// growth takes the experts hottest ON THE GROWTH CORPUS —
/// [`sources_by_corpus_wins`]; the legacy order is the balancing bias).
pub fn grow_experts_from(
    ck: &Checkpoint,
    spec: &GrowSpec,
    sources_in: &[Vec<usize>],
    inits: Option<&[Vec<ClusterInit>]>,
) -> anyhow::Result<(Checkpoint, Vec<Vec<usize>>)> {
    let cfg0 = ck.cfg.clone();
    let e0 = cfg0.experts;
    anyhow::ensure!(e0 >= 1, "the genome has no routed experts to grow from");
    anyhow::ensure!(spec.experts >= 1, "--experts must be ≥ 1");
    let layers = check_layers(cfg0.layers, &spec.layers)?;
    let kn = spec.experts;
    anyhow::ensure!(
        e0 + kn <= MAX_RUNTIME_EXPERTS,
        "E0 {e0} + K {kn} > {MAX_RUNTIME_EXPERTS}: the runtime graph handles at most \
         {MAX_RUNTIME_EXPERTS} experts per layer (above it the file runs per-op, many times slower)"
    );
    anyhow::ensure!(
        sources_in.len() == layers.len() && sources_in.iter().all(|v| v.len() == kn && v.iter().all(|&e| e < e0)),
        "trunk sources must be [{} grown layers][{kn}] of experts < {e0}, got {sources_in:?}",
        layers.len()
    );
    if let Some(inits) = inits {
        anyhow::ensure!(
            inits.len() == layers.len() && inits.iter().all(|v| v.len() == kn),
            "descriptor inits must be [{} grown layers][{kn}], got {:?}",
            layers.len(),
            inits.iter().map(|v| v.len()).collect::<Vec<_>>()
        );
    }
    let mut cfg = cfg0.clone();
    cfg.experts += kn;
    let e1 = cfg.experts;
    let (h, i, k) = (cfg.hidden, cfg.inter, MOE_K);
    let lay0 = Layout::new(&cfg0);
    let lay1 = Layout::new(&cfg);
    let mut p = vec![0.0f32; lay1.total];
    // copy every named tensor by name (offsets differ; names are stable)
    let old_by_name: std::collections::HashMap<&str, (usize, usize)> = lay0
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    for (name, off, len) in &lay1.names {
        if let Some((o0, l0)) = old_by_name.get(name.as_str()) {
            debug_assert_eq!(l0, len);
            p[*off..*off + len].copy_from_slice(&ck.params[*o0..*o0 + l0]);
        }
    }
    let ex = |name: &str| {
        ck.extras
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, x)| x.clone())
    };
    let mu0 = ex("desc.mu").unwrap_or_else(|| vec![0.0; cfg0.layers * e0 * h]);
    let u0 = ex("desc.u").unwrap_or_else(|| vec![0.0; cfg0.layers * e0 * k * h]);
    let b0 = ex("desc.bias").unwrap_or_else(|| vec![0.0; cfg0.layers * e0]);
    anyhow::ensure!(
        mu0.len() == cfg0.layers * e0 * h && u0.len() == cfg0.layers * e0 * k * h && b0.len() == cfg0.layers * e0,
        "checkpoint descriptors do not match its config (mu {}, u {}, bias {} for {} layers × {e0} experts × {h})",
        mu0.len(),
        u0.len(),
        b0.len(),
        cfg0.layers
    );
    let mut mu1 = vec![0.0f32; cfg.layers * e1 * h];
    let mut u1 = vec![0.0f32; cfg.layers * e1 * k * h];
    let mut b1 = vec![0.0f32; cfg.layers * e1];
    let mut sources = vec![Vec::new(); cfg.layers];
    let mut m1 = ck.m.as_ref().map(|_| vec![0.0f32; lay1.total]);
    let mut v1 = ck.v.as_ref().map(|_| vec![0.0f32; lay1.total]);
    if let (Some(m), Some(mo)) = (m1.as_mut(), ck.m.as_ref()) {
        for (name, off, len) in &lay1.names {
            if let Some((o0, _)) = old_by_name.get(name.as_str()) {
                m[*off..*off + len].copy_from_slice(&mo[*o0..*o0 + len]);
            }
        }
    }
    if let (Some(v), Some(vo)) = (v1.as_mut(), ck.v.as_ref()) {
        for (name, off, len) in &lay1.names {
            if let Some((o0, _)) = old_by_name.get(name.as_str()) {
                v[*off..*off + len].copy_from_slice(&vo[*o0..*o0 + len]);
            }
        }
    }
    let ew = 3 * h * i;
    for l in 0..cfg.layers {
        // old descriptors copied verbatim
        mu1[l * e1 * h..l * e1 * h + e0 * h].copy_from_slice(&mu0[l * e0 * h..(l + 1) * e0 * h]);
        u1[l * e1 * k * h..l * e1 * k * h + e0 * k * h]
            .copy_from_slice(&u0[l * e0 * k * h..(l + 1) * e0 * k * h]);
        b1[l * e1..l * e1 + e0].copy_from_slice(&b0[l * e0..(l + 1) * e0]);
        let grown = layers.iter().position(|&x| x == l);
        // an undeclared layer's inert slots copy the bias-hottest trunk
        // expert (never executed: DEAD_BIAS)
        let order = hottest_trunk_order(&b0[l * e0..(l + 1) * e0]);
        let ffn = ffn_offs(&lay1.layers[l]);
        for kk in 0..kn {
            let src = match grown {
                Some(li) => sources_in[li][kk],
                None => order[kk % e0],
            };
            let et = e0 + kk;
            if grown.is_some() {
                sources[l].push(src);
            }
            let (s_off, n_off) = (ffn.experts + src * ew, ffn.experts + et * ew);
            let seed_lk = spec.seed.wrapping_add((l * kn + kk) as u64);
            let noise_v = gauss_vec(seed_lk, ew);
            for j in 0..ew {
                p[n_off + j] = p[s_off + j] + spec.noise * noise_v[j];
            }
            let mu_src = &mu0[(l * e0 + src) * h..(l * e0 + src + 1) * h];
            let u_src = &u0[(l * e0 + src) * k * h..(l * e0 + src + 1) * k * h];
            let base = (l * e1 + et) * h;
            let ub = (l * e1 + et) * k * h;
            let init = grown
                .and_then(|li| inits.and_then(|v| v.get(li)))
                .and_then(|v| v.get(kk))
                .filter(|c| c.rows > 0);
            match init {
                Some(c) => {
                    // the corpus witness: μ = the cluster mean, U = its
                    // principal subspace
                    anyhow::ensure!(
                        c.mu.len() == h && c.u.len() == k * h,
                        "descriptor init of layer {l} expert {et}: mu {} / u {} (want {h} / {})",
                        c.mu.len(),
                        c.u.len(),
                        k * h
                    );
                    mu1[base..base + h].copy_from_slice(&c.mu);
                    u1[ub..ub + k * h].copy_from_slice(&c.u);
                }
                None => {
                    // legacy copy: μ_new = μ_src ± shift·‖μ_src‖·dir with dir
                    // OUTSIDE span(U_src) (inside it the runtime score would
                    // not change); a second copy of the same source (K > E0)
                    // shifts the other way; U copied
                    let dir = direction_outside_subspace(seed_lk, u_src, k, h);
                    let sign = if (kk / e0) % 2 == 1 { -1.0 } else { 1.0 };
                    let mn: f32 = mu_src.iter().map(|x| x * x).sum::<f32>().sqrt();
                    for j in 0..h {
                        mu1[base + j] = mu_src[j] + sign * spec.shift * mn * dir[j];
                    }
                    u1[ub..ub + k * h].copy_from_slice(u_src);
                }
            }
            b1[l * e1 + et] = if grown.is_none() {
                DEAD_BIAS
            } else if spec.zero_bias {
                0.0
            } else {
                b0[l * e0 + src]
            };
        }
    }
    let extras = vec![
        ("desc.mu".to_string(), mu1),
        ("desc.u".to_string(), u1),
        ("desc.bias".to_string(), b1),
    ];
    Ok((
        Checkpoint {
            cfg,
            step: ck.step,
            params: p,
            m: m1,
            v: v1,
            extras,
        },
        sources,
    ))
}

/// Legacy full-genome growth (the sleep daemon): E → E+1 in every layer,
/// the copy inherits the source's bias. Returns the grown checkpoint and,
/// per layer, the source expert copied.
pub fn grow_experts(ck: &Checkpoint, noise: f32, shift: f32, seed: u64) -> (Checkpoint, Vec<usize>) {
    let spec = GrowSpec {
        experts: 1,
        layers: (0..ck.cfg.layers).collect(),
        noise,
        shift,
        seed,
        zero_bias: false,
    };
    let (grown, sources) = grow_experts_k(ck, &spec).expect("legacy growth: one expert per layer");
    (grown, sources.into_iter().map(|s| s[0]).collect())
}

/// The inverse surgery: the `E0`-expert checkpoint a grown checkpoint
/// (`E1 > E0` experts in every layer) was grown from — every tensor of the
/// `E0` layout copied by name (the trunk experts `e < E0` of every layer,
/// everything else) and the descriptors of the first `E0` experts per
/// layer. Exact for an f32 genome; for an f16 genome the trunk of a
/// grown checkpoint is the SERVED (rounded) arena, not the master.
pub fn shrink_experts(ck: &Checkpoint, e0: usize) -> anyhow::Result<Checkpoint> {
    let e1 = ck.cfg.experts;
    anyhow::ensure!(
        e0 >= 1 && e0 < e1,
        "shrink: e0 {e0} must lie in 1..{e1} (the checkpoint's expert count)"
    );
    let mut cfg = ck.cfg.clone();
    cfg.experts = e0;
    let lay0 = Layout::new(&cfg);
    let lay1 = Layout::new(&ck.cfg);
    let by_name1: std::collections::HashMap<&str, (usize, usize)> = lay1
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    let mut p = vec![0.0f32; lay0.total];
    for (name, off, len) in &lay0.names {
        let (o1, l1) = by_name1
            .get(name.as_str())
            .ok_or_else(|| anyhow::anyhow!("shrink: the grown layout has no '{name}'"))?;
        anyhow::ensure!(l1 == len, "shrink: '{name}' has {l1} values in the grown layout, {len} in the E0 layout");
        p[*off..*off + len].copy_from_slice(&ck.params[*o1..*o1 + l1]);
    }
    let (l, h, k) = (cfg.layers, cfg.hidden, MOE_K);
    let mut extras = Vec::with_capacity(ck.extras.len());
    for (name, x) in &ck.extras {
        let per = match name.as_str() {
            "desc.mu" => h,
            "desc.u" => k * h,
            "desc.bias" => 1,
            _ => {
                extras.push((name.clone(), x.clone()));
                continue;
            }
        };
        anyhow::ensure!(
            x.len() == l * e1 * per,
            "shrink: {name} has {} values for {l} layers × {e1} experts × {per}",
            x.len()
        );
        let mut y = Vec::with_capacity(l * e0 * per);
        for li in 0..l {
            y.extend_from_slice(&x[li * e1 * per..(li * e1 + e0) * per]);
        }
        extras.push((name.clone(), y));
    }
    Ok(Checkpoint {
        cfg,
        step: ck.step,
        params: p,
        m: None,
        v: None,
        extras,
    })
}

/// The grown layers of a grown checkpoint: the layers whose experts `e ≥
/// e0` are live (any balancing bias other than [`DEAD_BIAS`]); a layer
/// outside `--layers` received inert slots. Refuses a layer with a mix.
pub fn grown_layers_of(ck: &Checkpoint, e0: usize) -> anyhow::Result<Vec<usize>> {
    let e1 = ck.cfg.experts;
    anyhow::ensure!(e0 >= 1 && e0 < e1, "e0 {e0} must lie in 1..{e1}");
    let bias = ck
        .extras
        .iter()
        .find(|(n, _)| n == "desc.bias")
        .map(|(_, x)| x.as_slice())
        .ok_or_else(|| anyhow::anyhow!("the checkpoint has no desc.bias"))?;
    anyhow::ensure!(bias.len() == ck.cfg.layers * e1, "desc.bias has {} values for {} layers × {e1}", bias.len(), ck.cfg.layers);
    let mut layers = Vec::new();
    for l in 0..ck.cfg.layers {
        let live = (e0..e1).filter(|&e| bias[l * e1 + e] != DEAD_BIAS).count();
        if live == e1 - e0 {
            layers.push(l);
        } else {
            anyhow::ensure!(live == 0, "layer {l}: {live} of {} grown slots are live, the rest inert", e1 - e0);
        }
    }
    anyhow::ensure!(!layers.is_empty(), "the checkpoint has no live grown experts (every grown slot is inert)");
    Ok(layers)
}

/// The balancing bias every grown expert must carry — `[grown layer
/// index][k]`: 0.0 under [`BiasMode::Zero`], the SOURCE trunk expert's
/// bias of the pre-growth checkpoint `ck0` under [`BiasMode::Source`].
pub fn expected_grown_bias(
    ck0: &Checkpoint,
    layers: &[usize],
    sources: &[Vec<usize>],
    mode: BiasMode,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let e0 = ck0.cfg.experts;
    anyhow::ensure!(sources.len() == layers.len(), "one source list per grown layer");
    let b0 = ck0
        .extras
        .iter()
        .find(|(n, _)| n == "desc.bias")
        .map(|(_, x)| x.clone())
        .unwrap_or_else(|| vec![0.0; ck0.cfg.layers * e0]);
    anyhow::ensure!(b0.len() == ck0.cfg.layers * e0, "pre-growth desc.bias has {} values for {} layers × {e0}", b0.len(), ck0.cfg.layers);
    Ok(layers
        .iter()
        .zip(sources)
        .map(|(&l, srcs)| {
            srcs.iter()
                .map(|&src| match mode {
                    BiasMode::Zero => 0.0,
                    BiasMode::Source => b0[l * e0 + src],
                })
                .collect()
        })
        .collect())
}

/// The grown biases a grown checkpoint carries — `[grown layer index][k]`.
pub fn grown_bias_of(ck: &Checkpoint, e0: usize, layers: &[usize]) -> anyhow::Result<Vec<Vec<f32>>> {
    let e1 = ck.cfg.experts;
    anyhow::ensure!(e0 >= 1 && e0 < e1, "e0 {e0} must lie in 1..{e1}");
    let bias = ck
        .extras
        .iter()
        .find(|(n, _)| n == "desc.bias")
        .map(|(_, x)| x.as_slice())
        .ok_or_else(|| anyhow::anyhow!("the checkpoint has no desc.bias"))?;
    anyhow::ensure!(bias.len() == ck.cfg.layers * e1, "desc.bias has {} values for {} layers × {e1}", bias.len(), ck.cfg.layers);
    Ok(layers.iter().map(|&l| bias[l * e1 + e0..(l + 1) * e1].to_vec()).collect())
}

// ───────────────────────── training ─────────────────────────

pub struct GrowArgs {
    pub steps: usize,
    pub lr: f32,
    pub batch: usize,
    pub seq: usize,
    pub eval_every: usize,
    pub seed: u64,
    /// Held-out budget: at most this many `[batch, seq]` batches of
    /// consecutive non-overlapping windows (0 = [`DEFAULT_HELD_BATCHES`]).
    pub held_batches: usize,
}

/// Default held-out budget in batches (`--held-batches`): 16 batches of
/// 4 × 512 = 32k tokens at the CLI defaults.
pub const DEFAULT_HELD_BATCHES: usize = 16;

/// The data and the trainable set of a growth.
pub struct GrowTrain<'a> {
    /// Trunk experts per layer (experts `e ≥ e0` train).
    pub e0: usize,
    /// The grown layers (only their new experts train).
    pub layers: &'a [usize],
    pub train: &'a Shard,
    pub held: &'a Shard,
    /// Pin the balancing bias of every expert `e ≥ e0` at its starting
    /// value (record growth: the record stores that value — 0 or the
    /// source's — and the expert must train under it).
    pub freeze_bias: bool,
    /// Keep the descriptors (μ EMA) of every expert `e ≥ e0` at their
    /// starting value too ([`DescMode::Frozen`]): the record's descriptor
    /// is exactly the initialisation (the novel cluster's).
    pub freeze_desc: bool,
    /// The PRE-growth genome checkpoint (`E0` experts): its held-out loss
    /// on the same windows is [`HeldOut::genome`], what the gate compares
    /// against. None = legacy (gate against the untrained grown model).
    pub genome: Option<&'a Checkpoint>,
}

/// Held-out losses of a growth — all on the SAME consecutive windows
/// ([`held_windows`]) and all dropless (every token keeps its routed
/// expert, as at runtime).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct HeldOut {
    /// The pre-growth genome (`GrowTrain::genome`).
    pub genome: Option<f32>,
    /// The grown model before any training (copies at bias 0 already
    /// take tokens from their sources).
    pub untrained: f32,
    /// The best checkpoint.
    pub after: f32,
    pub windows: usize,
    pub batches: usize,
}

impl HeldOut {
    /// What the gate measures: `(before − after) / before` with `before`
    /// = the genome's loss when known, else the untrained grown model's.
    pub fn improvement(&self) -> f32 {
        let before = self.genome.unwrap_or(self.untrained);
        (before - self.after) / before.abs().max(1e-6)
    }
}

/// The held-out batches: consecutive non-overlapping windows of `t + 1`
/// tokens (`tokens = w[..t]`, `targets = w[1..]`) from the start of the
/// shard, `b` per batch, at most `max_batches` batches (0 =
/// [`DEFAULT_HELD_BATCHES`]); when the shard holds fewer than `b` windows
/// the rows wrap around. Every evaluation of a growth (genome, untrained,
/// every checkpoint) reads exactly these.
pub fn held_windows(
    held: &Shard,
    b: usize,
    t: usize,
    max_batches: usize,
) -> anyhow::Result<Vec<(Vec<u32>, Vec<u32>)>> {
    anyhow::ensure!(b >= 1 && t >= 1, "batch ≥ 1 and seq ≥ 1");
    let n = held.tokens.len();
    anyhow::ensure!(
        n > t + 1,
        "growth held-out shorter than one window ({n} tokens, seq {t})"
    );
    let n_w = (n - 1) / t;
    let max_b = if max_batches == 0 { DEFAULT_HELD_BATCHES } else { max_batches };
    let n_b = (n_w / b).clamp(1, max_b);
    let mut out = Vec::with_capacity(n_b);
    for bi in 0..n_b {
        let mut tk = Vec::with_capacity(b * t);
        let mut tg = Vec::with_capacity(b * t);
        for r in 0..b {
            let w = (bi * b + r) % n_w;
            let s = w * t;
            tk.extend(held.tokens[s..s + t].iter().map(|&x| x as u32));
            tg.extend(held.tokens[s + 1..s + t + 1].iter().map(|&x| x as u32));
        }
        out.push((tk, tg));
    }
    Ok(out)
}

/// Mean eval loss over [`held_windows`] batches.
pub fn eval_windows(gpu: &EmbryoGpu, batches: &[(Vec<u32>, Vec<u32>)]) -> f32 {
    let mut s = 0.0f32;
    for (tk, tg) in batches {
        s += gpu.eval_loss(tk, tg);
    }
    s / batches.len().max(1) as f32
}

/// The 10 % tail split of a flat corpus (the fallback when no `--held`
/// files are given): `(train, held)`.
pub fn split_tail(corpus: &Shard, seq: usize) -> anyhow::Result<(Shard, Shard)> {
    let n = corpus.tokens.len();
    anyhow::ensure!(n > 20 * (seq + 2), "growth corpus too small: {n} tokens");
    let cut = n - n / 10;
    Ok((
        Shard {
            tokens: corpus.tokens[..cut].to_vec(),
        },
        Shard {
            tokens: corpus.tokens[cut..].to_vec(),
        },
    ))
}

/// Train ONLY the grown experts (`e ≥ e0`, arena ranges of the grown
/// layers) with `desc_frozen_below = e0` (old descriptors never move) and,
/// with `freeze_bias`, `bias_frozen_from = e0` in the Metal / WGSL
/// descriptor-update kernels. The instance is DROPLESS
/// (`EmbryoGpu::new_eval_dropless`): a grown copy of the hottest trunk
/// expert at bias 0 routinely takes more tokens than a capacity-2 slot
/// holds, and a dropped token neither trains the expert nor is scored the
/// way the runtime scores it — so the training trajectory, the held-out
/// losses, the gate and the best checkpoint all describe the model the
/// runtime executes. Returns (best held-out checkpoint, held-out losses).
pub fn train_grown_experts(
    ck: &Checkpoint,
    gt: &GrowTrain,
    a: &GrowArgs,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<(Checkpoint, HeldOut)> {
    let cfg: EmbryoCfg = ck.cfg.clone();
    let lay = Layout::new(&cfg);
    let (h, i) = (cfg.hidden, cfg.inter);
    anyhow::ensure!(
        gt.e0 >= 1 && gt.e0 < cfg.experts,
        "e0 {} must lie in 1..{} (the checkpoint's expert count)",
        gt.e0,
        cfg.experts
    );
    let layers = check_layers(cfg.layers, gt.layers)?;
    let kn = cfg.experts - gt.e0;
    anyhow::ensure!(
        gt.train.tokens.len() > a.seq + 1,
        "growth train corpus shorter than one window ({} tokens, seq {})",
        gt.train.tokens.len(),
        a.seq
    );
    let batches = held_windows(gt.held, a.batch, a.seq, a.held_batches)?;
    // the genome's loss on the same windows (the gate's reference)
    let held_genome = match gt.genome {
        Some(g) => {
            anyhow::ensure!(
                g.cfg.experts == gt.e0,
                "the genome checkpoint has {} experts, e0 = {}",
                g.cfg.experts,
                gt.e0
            );
            let g0 = EmbryoGpu::new_eval_dropless(g.cfg.clone(), a.batch, a.seq, &g.params)
                .ok_or_else(|| anyhow::anyhow!("no GPU backend (Metal on macOS / --features vulkan)"))?;
            g0.set_desc(&g.extras);
            g0.desc_updates.set(false);
            let l = eval_windows(&g0, &batches);
            drop(g0);
            Some(l)
        }
        None => None,
    };
    let mut gpu = EmbryoGpu::new_eval_dropless(cfg.clone(), a.batch, a.seq, &ck.params)
        .ok_or_else(|| anyhow::anyhow!("no GPU backend (Metal on macOS / --features vulkan)"))?;
    gpu.set_desc(&ck.extras);
    gpu.desc_frozen_below.set(gt.e0);
    if gt.freeze_bias {
        gpu.bias_frozen_from.set(gt.e0);
    }
    if gt.freeze_desc {
        // With `desc_frozen_below = e0` (the kernels return for `e < e0`)
        // and `bias_frozen_from = e0` (no bias step for `e ≥ e0`) the
        // descriptor-update kernels (Metal `moe_update_f32`, WGSL op 36)
        // move nothing but the μ EMA of the grown experts, and the U
        // refresh (`update_subspaces`) is never called by a growth: turning
        // the online updates off freezes exactly the grown descriptors —
        // the symmetric "frozen from" cell would gate the same kernels to
        // the same no-op, so none is added (the bias is frozen too, as
        // under `freeze_bias`).
        gpu.desc_updates.set(false);
    }
    let l_untrained = eval_windows(&gpu, &batches);
    // trainable ranges: the K new experts of every grown layer (contiguous)
    let ew = 3 * h * i;
    let ranges: Vec<(usize, usize)> = layers
        .iter()
        .map(|&l| (ffn_offs(&lay.layers[l]).experts + gt.e0 * ew, kn * ew))
        .collect();
    let mut sampler = Sampler::new(a.batch, a.seq, a.seed);
    let (mut tk, mut tg) = (Vec::new(), Vec::new());
    let t0 = Instant::now();
    let eval_every = a.eval_every.max(1);
    let mut best = (l_untrained, gpu.params_host(), gpu.desc_host());
    for step in 0..a.steps {
        if should_stop() {
            anyhow::bail!("preempted");
        }
        sampler.batch(gt.train, &mut tk, &mut tg);
        let lr =
            a.lr * 0.5 * (1.0 + (std::f32::consts::PI * step as f32 / a.steps.max(1) as f32).cos());
        let (loss, gn) = gpu.train_step_ranges(&tk, &tg, lr, 0.0, 1.0, &ranges);
        if (step + 1) % eval_every == 0 || step + 1 == a.steps {
            let vl = eval_windows(&gpu, &batches);
            eprintln!(
                "  grow step {:>4} loss {loss:.4} |g| {gn:.3} lr {lr:.2e} held-out {vl:.4} (genome {}, untrained {l_untrained:.4}) [{:.0} s]",
                step + 1,
                held_genome.map(|g| format!("{g:.4}")).unwrap_or_else(|| "n/a".into()),
                t0.elapsed().as_secs_f64()
            );
            if vl < best.0 {
                best = (vl, gpu.params_host(), gpu.desc_host());
            }
        }
    }
    let extras: Vec<(String, Vec<f32>)> = best
        .2
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    Ok((
        Checkpoint {
            cfg,
            step: ck.step,
            params: best.1,
            m: None,
            v: None,
            extras,
        },
        HeldOut {
            genome: held_genome,
            untrained: l_untrained,
            after: best.0,
            windows: batches.len() * a.batch,
            batches: batches.len(),
        },
    ))
}

/// Legacy: train the newest expert of every layer (index E−1) on `corpus`
/// with its 10 % tail held out; the bias trains too. Returns (trained,
/// held-out of the untrained grown model, after).
pub fn train_new_experts(
    ck: &Checkpoint,
    corpus: &Shard,
    a: &GrowArgs,
    should_stop: &dyn Fn() -> bool,
) -> anyhow::Result<(Checkpoint, f32, f32)> {
    let (train, held) = split_tail(corpus, a.seq)?;
    let layers: Vec<usize> = (0..ck.cfg.layers).collect();
    let (trained, ho) = train_grown_experts(
        ck,
        &GrowTrain {
            e0: ck.cfg.experts - 1,
            layers: &layers,
            train: &train,
            held: &held,
            freeze_bias: false,
            freeze_desc: false,
            genome: None,
        },
        a,
        should_stop,
    )?;
    Ok((trained, ho.untrained, ho.after))
}

// ───────────────────────── the runtime formula on the host ─────────────────────────

/// EXACTLY `cortiq_engine::pipeline::Resonance::scores` (same operations in
/// the same order): for every expert `e`, `err_e = Σ_j (x_j−μ_ej)² −
/// Σ_i (Σ_j (x_j−μ_ej)·u_eij)²`, `out[e] = bias_e − err_e`; with a shell,
/// `err_e > shell_e` → `−∞`. `mu` is `[E, H]`, `u` is `[E, k, H]`, `err`
/// receives the reconstruction errors.
pub fn resonance_scores(
    x: &[f32],
    mu: &[f32],
    u: &[f32],
    k: usize,
    bias: &[f32],
    shell: Option<&[f32]>,
    out: &mut [f32],
    err: &mut [f32],
) {
    let h = x.len();
    let ne = out.len();
    for e in 0..ne {
        let mu_e = &mu[e * h..(e + 1) * h];
        let mut d2 = 0.0f32;
        for j in 0..h {
            let d = x[j] - mu_e[j];
            d2 += d * d;
        }
        let mut proj = 0.0f32;
        for i in 0..k {
            let ue = &u[(e * k + i) * h..(e * k + i + 1) * h];
            let mut p = 0.0f32;
            for j in 0..h {
                p += (x[j] - mu_e[j]) * ue[j];
            }
            proj += p * p;
        }
        let er = d2 - proj;
        err[e] = er;
        out[e] = bias.get(e).copied().unwrap_or(0.0) - er;
        if let Some(sh) = shell {
            if er > sh.get(e).copied().unwrap_or(f32::INFINITY) {
                out[e] = f32::NEG_INFINITY;
            }
        }
    }
}

/// The runtime's top-1 over resonance scores (`moe_route`: descending by
/// score, lower index wins ties): the first index of the maximum.
pub fn resonance_winner(scores: &[f32]) -> usize {
    let mut best = 0usize;
    for (e, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = e;
        }
    }
    best
}

/// Nearest-rank quantile of an ASCENDING sample (`q` in `[0, 1]`).
pub fn quantile_sorted(sorted: &[f32], q: f32) -> f32 {
    assert!(!sorted.is_empty());
    let n = sorted.len();
    let rank = (q as f64 * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

/// The descriptors of a grown checkpoint (`[L, E, H]`, `[L, E, K, H]`,
/// `[L, E]`) — what the record stores and the runtime scores with.
pub struct GrowthDesc {
    pub e0: usize,
    pub e: usize,
    pub h: usize,
    pub layers_total: usize,
    pub mu: Vec<f32>,
    pub u: Vec<f32>,
    pub bias: Vec<f32>,
}

impl GrowthDesc {
    pub fn from_checkpoint(ck: &Checkpoint, e0: usize) -> anyhow::Result<GrowthDesc> {
        let (l, e, h) = (ck.cfg.layers, ck.cfg.experts, ck.cfg.hidden);
        anyhow::ensure!(e0 <= e, "e0 {e0} > experts {e}");
        let ex = |name: &str| -> anyhow::Result<Vec<f32>> {
            ck.extras
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, x)| x.clone())
                .ok_or_else(|| anyhow::anyhow!("checkpoint has no {name}"))
        };
        let mu = ex("desc.mu")?;
        let u = ex("desc.u").unwrap_or_else(|_| vec![0.0; l * e * MOE_K * h]);
        let bias = ex("desc.bias")?;
        anyhow::ensure!(
            mu.len() == l * e * h && u.len() == l * e * MOE_K * h && bias.len() == l * e,
            "descriptor sizes do not match the config"
        );
        Ok(GrowthDesc {
            e0,
            e,
            h,
            layers_total: l,
            mu,
            u,
            bias,
        })
    }
    /// Grown experts per layer.
    pub fn k(&self) -> usize {
        self.e - self.e0
    }
}

/// The routing witness of one grown layer over a shard: per token the best
/// TRUNK expert and its score, and the score / reconstruction error of
/// every grown expert (`[tokens, K]`). Everything the shell, shift and
/// coverage need follows from it without another forward.
pub struct LayerTrace {
    pub layer: usize,
    pub e0: usize,
    pub k: usize,
    pub tokens: usize,
    pub trunk_best: Vec<u32>,
    pub trunk_best_score: Vec<f32>,
    pub grown_score: Vec<f32>,
    pub grown_err: Vec<f32>,
}

impl LayerTrace {
    /// The runtime winner of token `t`: the best trunk expert unless a
    /// grown expert (in index order, inside its shell when `shells` is
    /// given) scores strictly higher — the same first-maximum rule as
    /// [`resonance_winner`] over the full score vector.
    pub fn winner(&self, t: usize, shells: Option<&[f32]>) -> usize {
        let mut best = self.trunk_best[t] as usize;
        let mut best_s = self.trunk_best_score[t];
        for kk in 0..self.k {
            let s = self.grown_score[t * self.k + kk];
            if let Some(sh) = shells {
                if self.grown_err[t * self.k + kk] > sh[kk] {
                    continue; // −∞ at runtime
                }
            }
            if s > best_s {
                best = self.e0 + kk;
                best_s = s;
            }
        }
        best
    }
    /// Is token `t` inside the shell of at least one grown expert?
    pub fn inside_any(&self, t: usize, shells: &[f32]) -> bool {
        (0..self.k).any(|kk| self.grown_err[t * self.k + kk] <= shells[kk])
    }
}

/// Eval forwards over `shard` in fixed non-overlapping windows of the
/// instance's `[B, T]` (the tail that does not fill a window is dropped;
/// padding rows of the last batch are excluded), reading the MoE input of
/// every layer in `layers` and scoring it on the host with the runtime
/// formula. The instance should be dropless (`EmbryoGpu::new_eval_dropless`)
/// so no token is dropped by expert capacity, as at runtime. Note the
/// forward itself routes without shells (the trainer kernel has none): the
/// witness is the no-shell trajectory, scored per layer with and without
/// the shell.
pub fn trace_routes(gpu: &EmbryoGpu, shard: &Shard, layers: &[usize], desc: &GrowthDesc) -> anyhow::Result<Vec<LayerTrace>> {
    let t = gpu.t;
    let n_w = shard.tokens.len() / t;
    anyhow::ensure!(n_w >= 1, "shard shorter than one window ({} tokens, seq {t})", shard.tokens.len());
    let rows: Vec<Vec<u32>> = (0..n_w)
        .map(|w| shard.tokens[w * t..(w + 1) * t].iter().map(|&x| x as u32).collect())
        .collect();
    let spans = vec![0..t; n_w];
    trace_routes_rows(gpu, &rows, &spans, layers, desc)
}

/// Split a flat token stream into documents at `eot` (the separator is
/// dropped; empty documents are skipped).
pub fn split_docs_at_eot(tokens: &[u16], eot: u16) -> Vec<Vec<u16>> {
    tokens
        .split(|&x| x == eot)
        .filter(|d| !d.is_empty())
        .map(|d| d.to_vec())
        .collect()
}

/// The document-wise witness rows: every document rendered as
/// `prefix ++ doc ++ suffix` (the cmf-im-v1 frame when given, else the
/// bare document), ONE per row, right-padded with `pad` to `t` — the way
/// the runtime sees a prompt (fresh state at position 0, the frame in
/// front, no other document's tokens before it). A document longer than
/// `t − frame` is split into consecutive chunks, each framed. Returns the
/// rows and, per row, the positions of the document's own tokens (the
/// frame and the padding are never scored).
pub fn frame_docs(
    docs: &[Vec<u16>],
    prefix: &[u32],
    suffix: &[u32],
    t: usize,
    pad: u32,
) -> anyhow::Result<(Vec<Vec<u32>>, Vec<std::ops::Range<usize>>)> {
    let frame = prefix.len() + suffix.len();
    anyhow::ensure!(
        frame + 1 <= t,
        "the frame ({frame} tokens) leaves no room for a document in seq {t}"
    );
    let body = t - frame;
    let mut rows = Vec::new();
    let mut spans = Vec::new();
    for d in docs {
        if d.is_empty() {
            continue;
        }
        for chunk in d.chunks(body) {
            let mut row = Vec::with_capacity(t);
            row.extend_from_slice(prefix);
            let a = row.len();
            row.extend(chunk.iter().map(|&x| x as u32));
            let b = row.len();
            row.extend_from_slice(suffix);
            row.resize(t, pad);
            rows.push(row);
            spans.push(a..b);
        }
    }
    anyhow::ensure!(!rows.is_empty(), "no documents to witness");
    Ok((rows, spans))
}

/// [`trace_routes`] over explicit rows: `rows[r]` is one `[T]` input of
/// the instance and only the positions `spans[r]` of it are scored
/// (padding rows of the last batch and positions outside the spans are
/// excluded). Rows are forwarded `B` at a time in order.
pub fn trace_routes_rows(
    gpu: &EmbryoGpu,
    rows: &[Vec<u32>],
    spans: &[std::ops::Range<usize>],
    layers: &[usize],
    desc: &GrowthDesc,
) -> anyhow::Result<Vec<LayerTrace>> {
    let (b, t) = (gpu.b, gpu.t);
    let (h, e, e0, k) = (desc.h, desc.e, desc.e0, desc.k());
    anyhow::ensure!(h == gpu.cfg.hidden && e == gpu.cfg.experts, "descriptor / instance mismatch");
    anyhow::ensure!(!rows.is_empty(), "no rows to witness");
    anyhow::ensure!(rows.len() == spans.len(), "one span per row");
    for (r, (row, sp)) in rows.iter().zip(spans).enumerate() {
        anyhow::ensure!(row.len() == t, "row {r} has {} tokens, the instance {t}", row.len());
        anyhow::ensure!(sp.end <= t && sp.start <= sp.end, "row {r}: span {sp:?} outside 0..{t}");
    }
    let layers = check_layers(desc.layers_total, layers)?;
    let mut traces: Vec<LayerTrace> = layers
        .iter()
        .map(|&l| LayerTrace {
            layer: l,
            e0,
            k,
            tokens: 0,
            trunk_best: Vec::new(),
            trunk_best_score: Vec::new(),
            grown_score: Vec::new(),
            grown_err: Vec::new(),
        })
        .collect();
    let mut tokens = vec![0u32; b * t];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 64);
    let mut start = 0usize;
    while start < rows.len() {
        let valid = (rows.len() - start).min(b);
        // the positions scored in this batch, row-major
        let mut sel: Vec<usize> = Vec::new();
        for r in 0..b {
            let w = if r < valid { start + r } else { 0 };
            tokens[r * t..(r + 1) * t].copy_from_slice(&rows[w]);
            if r < valid {
                sel.extend(spans[w].clone().map(|pos| r * t + pos));
            }
        }
        let _ = gpu.forward_hidden(&tokens);
        for tr in traces.iter_mut() {
            let l = tr.layer;
            let x2 = gpu.ffn_input_host(l);
            let mu = &desc.mu[l * e * h..(l + 1) * e * h];
            let u = &desc.u[l * e * MOE_K * h..(l + 1) * e * MOE_K * h];
            let bias = &desc.bias[l * e..(l + 1) * e];
            // rows are independent: score them in parallel chunks, in order
            let chunk = sel.len().div_ceil(threads).max(1);
            let parts: Vec<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>)> = std::thread::scope(|sc| {
                let handles: Vec<_> = sel
                    .chunks(chunk)
                    .map(|part| {
                        let x2 = &x2;
                        sc.spawn(move || {
                            let mut scores = vec![0.0f32; e];
                            let mut errs = vec![0.0f32; e];
                            let n = part.len();
                            let mut tb_v = Vec::with_capacity(n);
                            let mut ts_v = Vec::with_capacity(n);
                            let mut gs_v = Vec::with_capacity(n * k);
                            let mut ge_v = Vec::with_capacity(n * k);
                            for &row in part {
                                let x = &x2[row * h..(row + 1) * h];
                                resonance_scores(x, mu, u, MOE_K, bias, None, &mut scores, &mut errs);
                                let tb = resonance_winner(&scores[..e0]);
                                tb_v.push(tb as u32);
                                ts_v.push(scores[tb]);
                                gs_v.extend_from_slice(&scores[e0..]);
                                ge_v.extend_from_slice(&errs[e0..]);
                            }
                            (tb_v, ts_v, gs_v, ge_v)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("route probe thread")).collect()
            });
            for (tb_v, ts_v, gs_v, ge_v) in parts {
                tr.tokens += tb_v.len();
                tr.trunk_best.extend(tb_v);
                tr.trunk_best_score.extend(ts_v);
                tr.grown_score.extend(gs_v);
                tr.grown_err.extend(ge_v);
            }
        }
        start += valid;
    }
    Ok(traces)
}

// ───────────────────────── descriptor init from the corpus ─────────────────────────

/// Rows kept per (layer, source) for the covariance of [`cluster_inits`]
/// (a deterministic reservoir over the won tokens; the mean uses every
/// won token).
pub const INIT_COV_ROWS: usize = 4096;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Mean and top-`MOE_K` eigenvectors (orthonormal rows) of `rows` (`[n,
/// H]`) — the descriptor of a cluster. `n == 0` gives an empty init.
fn cluster_descriptor(rows: &[f32], n: usize, h: usize, seed: u64) -> ClusterInit {
    if n == 0 {
        return ClusterInit::default();
    }
    let mut mu = vec![0.0f32; h];
    for r in 0..n {
        for j in 0..h {
            mu[j] += rows[r * h + j];
        }
    }
    for v in &mut mu {
        *v /= n as f32;
    }
    let mut cov = vec![0.0f32; h * h];
    let mut d = vec![0.0f32; h];
    for r in 0..n {
        for j in 0..h {
            d[j] = rows[r * h + j] - mu[j];
        }
        for a in 0..h {
            let da = d[a];
            if da == 0.0 {
                continue;
            }
            let row = &mut cov[a * h..(a + 1) * h];
            for b in 0..h {
                row[b] += da * d[b];
            }
        }
    }
    for v in &mut cov {
        *v /= n as f32;
    }
    let u = crate::model::top_eigenvectors(&cov, h, MOE_K, 24, seed);
    ClusterInit { mu, u, rows: n }
}

/// Trunk wins on the growth corpus: forward `shard` on the PRE-growth
/// instance (`E0` experts, dropless) and count, per grown layer, the
/// tokens every trunk expert wins with the runtime formula (the argmax of
/// `bias − err`, no shell) — `[grown layer index][E0]`.
pub fn trunk_corpus_wins(
    gpu: &EmbryoGpu,
    shard: &Shard,
    layers: &[usize],
    desc0: &GrowthDesc,
) -> anyhow::Result<Vec<Vec<usize>>> {
    let (b, t) = (gpu.b, gpu.t);
    let (h, e0) = (desc0.h, desc0.e0);
    anyhow::ensure!(desc0.e == e0, "the pre-growth descriptors must have E0 == E experts");
    anyhow::ensure!(h == gpu.cfg.hidden && e0 == gpu.cfg.experts, "descriptor / instance mismatch");
    let layers = check_layers(desc0.layers_total, layers)?;
    let n_w = shard.tokens.len() / t;
    anyhow::ensure!(n_w >= 1, "growth corpus shorter than one window ({} tokens, seq {t})", shard.tokens.len());
    let mut wins = vec![vec![0usize; e0]; layers.len()];
    let mut tokens = vec![0u32; b * t];
    let mut scores = vec![0.0f32; e0];
    let mut errs = vec![0.0f32; e0];
    let mut start = 0usize;
    while start < n_w {
        let valid = (n_w - start).min(b);
        for r in 0..b {
            let w = if r < valid { start + r } else { 0 };
            for pos in 0..t {
                tokens[r * t + pos] = shard.tokens[w * t + pos] as u32;
            }
        }
        let _ = gpu.forward_hidden(&tokens);
        for (li, &l) in layers.iter().enumerate() {
            let x2 = gpu.ffn_input_host(l);
            let mu = &desc0.mu[l * e0 * h..(l + 1) * e0 * h];
            let u = &desc0.u[l * e0 * MOE_K * h..(l + 1) * e0 * MOE_K * h];
            let bias = &desc0.bias[l * e0..(l + 1) * e0];
            for row in 0..valid * t {
                let x = &x2[row * h..(row + 1) * h];
                resonance_scores(x, mu, u, MOE_K, bias, None, &mut scores, &mut errs);
                wins[li][resonance_winner(&scores)] += 1;
            }
        }
        start += valid;
    }
    Ok(wins)
}

/// The trunk sources of a RECORD growth, per grown layer and copy: expert
/// `E0 + k` copies the trunk expert with the k-th most wins on the growth
/// corpus in that layer (`k mod E0` when `K > E0`; ties → the hotter
/// balancing bias, then the lower index). A copy of an expert that wins
/// nothing of the corpus would keep the genome's descriptor (μ / U of the
/// genome's own tokens) and a shell measured on the foreign tokens it
/// steals — wide enough to cover the genome's distribution; the corpus'
/// own hottest expert gives every copy a witness ([`cluster_inits`]).
pub fn sources_by_corpus_wins(wins: &[Vec<usize>], bias: &[f32], e0: usize, layers: &[usize], k: usize) -> anyhow::Result<Vec<Vec<usize>>> {
    anyhow::ensure!(wins.len() == layers.len() && wins.iter().all(|w| w.len() == e0), "wins must be [grown layers][E0]");
    anyhow::ensure!(k >= 1 && e0 >= 1);
    Ok(layers
        .iter()
        .zip(wins)
        .map(|(&l, w)| {
            let b = &bias[l * e0..(l + 1) * e0];
            let mut order: Vec<usize> = (0..e0).collect();
            order.sort_by(|&a, &c| {
                w[c].cmp(&w[a])
                    .then(b[a].partial_cmp(&b[c]).unwrap_or(std::cmp::Ordering::Equal))
                    .then(a.cmp(&c))
            });
            (0..k).map(|kk| order[kk % e0]).collect()
        })
        .collect())
}

/// The corpus witness of the descriptors of the new experts: forward the
/// growth TRAIN shard on the PRE-growth instance (`E0` experts, dropless)
/// and, per grown layer and copy `k` with trunk source `sources[li][k]`,
/// take the mean of the layer's MoE inputs the source wins (the runtime
/// formula, argmax over the trunk) and the top-`MOE_K` eigenvectors of
/// their covariance (over a reservoir of [`INIT_COV_ROWS`] rows). Copies
/// of the SAME source (K > E0) split its cluster along its principal
/// direction: copy `c` of `n` takes the c-th n-quantile bin of the
/// projections. A source that wins nothing gives `rows == 0` (the surgery
/// then falls back to the legacy copy).
pub fn cluster_inits(
    gpu: &EmbryoGpu,
    shard: &Shard,
    layers: &[usize],
    desc0: &GrowthDesc,
    sources: &[Vec<usize>],
    seed: u64,
) -> anyhow::Result<Vec<Vec<ClusterInit>>> {
    let (b, t) = (gpu.b, gpu.t);
    let (h, e0) = (desc0.h, desc0.e0);
    anyhow::ensure!(desc0.e == e0, "the pre-growth descriptors must have E0 == E experts");
    anyhow::ensure!(h == gpu.cfg.hidden && e0 == gpu.cfg.experts, "descriptor / instance mismatch");
    let layers = check_layers(desc0.layers_total, layers)?;
    anyhow::ensure!(sources.len() == layers.len(), "one source list per grown layer");
    let n_w = shard.tokens.len() / t;
    anyhow::ensure!(n_w >= 1, "growth corpus shorter than one window ({} tokens, seq {t})", shard.tokens.len());
    // per grown layer, per trunk source: (count, sum[H], reservoir rows, reservoir count)
    struct Acc {
        n: usize,
        sum: Vec<f32>,
        res: Vec<f32>,
        res_n: usize,
    }
    let mut acc: Vec<Vec<Acc>> = layers
        .iter()
        .map(|_| {
            (0..e0)
                .map(|_| Acc {
                    n: 0,
                    sum: vec![0.0; h],
                    res: Vec::new(),
                    res_n: 0,
                })
                .collect()
        })
        .collect();
    let mut rng = seed ^ 0x5DEE_CE66_D1CE_4E5B;
    let mut tokens = vec![0u32; b * t];
    let mut scores = vec![0.0f32; e0];
    let mut errs = vec![0.0f32; e0];
    let mut start = 0usize;
    while start < n_w {
        let valid = (n_w - start).min(b);
        for r in 0..b {
            let w = if r < valid { start + r } else { 0 };
            for pos in 0..t {
                tokens[r * t + pos] = shard.tokens[w * t + pos] as u32;
            }
        }
        let _ = gpu.forward_hidden(&tokens);
        for (li, &l) in layers.iter().enumerate() {
            let x2 = gpu.ffn_input_host(l);
            let mu = &desc0.mu[l * e0 * h..(l + 1) * e0 * h];
            let u = &desc0.u[l * e0 * MOE_K * h..(l + 1) * e0 * MOE_K * h];
            let bias = &desc0.bias[l * e0..(l + 1) * e0];
            for row in 0..valid * t {
                let x = &x2[row * h..(row + 1) * h];
                resonance_scores(x, mu, u, MOE_K, bias, None, &mut scores, &mut errs);
                let w = resonance_winner(&scores);
                if !sources[li].contains(&w) {
                    continue;
                }
                let a = &mut acc[li][w];
                for j in 0..h {
                    a.sum[j] += x[j];
                }
                let j = a.n;
                a.n += 1;
                if a.res_n < INIT_COV_ROWS {
                    a.res.extend_from_slice(x);
                    a.res_n += 1;
                } else {
                    let r = (splitmix(&mut rng) % (j as u64 + 1)) as usize;
                    if r < INIT_COV_ROWS {
                        a.res[r * h..(r + 1) * h].copy_from_slice(x);
                    }
                }
            }
        }
        start += valid;
    }
    // descriptors: one job per (grown layer, copy), in parallel
    let jobs: Vec<(usize, usize)> = (0..layers.len())
        .flat_map(|li| (0..sources[li].len()).map(move |kk| (li, kk)))
        .collect();
    let acc = &acc;
    let results: Vec<ClusterInit> = std::thread::scope(|sc| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|&(li, kk)| {
                let src = sources[li][kk];
                let copies: Vec<usize> = (0..sources[li].len()).filter(|&q| sources[li][q] == src).collect();
                let c = copies.iter().position(|&q| q == kk).unwrap_or(0);
                let n_c = copies.len();
                let seed_lk = seed.wrapping_add((li * 64 + kk) as u64);
                sc.spawn(move || {
                    let a = &acc[li][src];
                    if a.n == 0 {
                        return ClusterInit::default();
                    }
                    if n_c == 1 {
                        let mut init = cluster_descriptor(&a.res, a.res_n, h, seed_lk);
                        // the mean over EVERY won token, not only the reservoir
                        for j in 0..h {
                            init.mu[j] = a.sum[j] / a.n as f32;
                        }
                        init.rows = a.n;
                        return init;
                    }
                    // split the cluster along its principal direction
                    let full = cluster_descriptor(&a.res, a.res_n, h, seed_lk);
                    let v1 = &full.u[..h];
                    let mut proj: Vec<(f32, usize)> = (0..a.res_n)
                        .map(|r| {
                            let x = &a.res[r * h..(r + 1) * h];
                            (
                                (0..h).map(|j| (x[j] - full.mu[j]) * v1[j]).sum::<f32>(),
                                r,
                            )
                        })
                        .collect();
                    proj.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal).then(p.1.cmp(&q.1)));
                    let lo = c * a.res_n / n_c;
                    let hi = (c + 1) * a.res_n / n_c;
                    let mut rows = Vec::with_capacity((hi - lo) * h);
                    for &(_, r) in &proj[lo..hi] {
                        rows.extend_from_slice(&a.res[r * h..(r + 1) * h]);
                    }
                    cluster_descriptor(&rows, hi - lo, h, seed_lk.wrapping_add(7))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("descriptor init thread")).collect()
    });
    let mut out: Vec<Vec<ClusterInit>> = layers.iter().map(|_| Vec::new()).collect();
    for ((li, _), init) in jobs.into_iter().zip(results) {
        out[li].push(init);
    }
    Ok(out)
}

// ───────────────────────── novelty (P1) → OOD clusters (P3/P9) ─────────────────────────

/// A grown layer must hold at least this many novel tokens per new expert
/// ([`novel_inits`] refuses below it).
pub const NOVEL_MIN_PER_EXPERT: usize = 64;

/// Rows of novel tokens kept per grown layer for the clustering and the
/// covariances (a deterministic reservoir; the K = 1 mean uses every
/// novel token).
pub const NOVEL_RES_ROWS: usize = 8192;

/// Lloyd iterations of the K-means of [`novel_inits`].
pub const KMEANS_ITERS: usize = 12;

/// Forward `shard` on the PRE-growth instance (`E0` experts, dropless,
/// fixed non-overlapping windows of `[B, T]` in order) and, per grown
/// layer and token, hand `f(layer index, token index, x, trunk winner,
/// min trunk error)` the layer's MoE input, the trunk expert winning it
/// with the runtime formula (argmax of `bias − err`, no shell) and the
/// smallest reconstruction error over the trunk experts (no bias). Rows
/// are scored in parallel; `f` runs sequentially in token order.
fn scan_trunk<F: FnMut(usize, usize, &[f32], usize, f32)>(
    gpu: &EmbryoGpu,
    shard: &Shard,
    layers: &[usize],
    desc0: &GrowthDesc,
    mut f: F,
) -> anyhow::Result<usize> {
    let (b, t) = (gpu.b, gpu.t);
    let (h, e0) = (desc0.h, desc0.e0);
    anyhow::ensure!(desc0.e == e0, "the pre-growth descriptors must have E0 == E experts");
    anyhow::ensure!(h == gpu.cfg.hidden && e0 == gpu.cfg.experts, "descriptor / instance mismatch");
    let layers = check_layers(desc0.layers_total, layers)?;
    let n_w = shard.tokens.len() / t;
    anyhow::ensure!(n_w >= 1, "shard shorter than one window ({} tokens, seq {t})", shard.tokens.len());
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 64);
    let mut tokens = vec![0u32; b * t];
    let mut start = 0usize;
    let mut idx = 0usize;
    while start < n_w {
        let valid = (n_w - start).min(b);
        for r in 0..b {
            let w = if r < valid { start + r } else { 0 };
            for pos in 0..t {
                tokens[r * t + pos] = shard.tokens[w * t + pos] as u32;
            }
        }
        let _ = gpu.forward_hidden(&tokens);
        let rows = valid * t;
        for (li, &l) in layers.iter().enumerate() {
            let x2 = gpu.ffn_input_host(l);
            let mu = &desc0.mu[l * e0 * h..(l + 1) * e0 * h];
            let u = &desc0.u[l * e0 * MOE_K * h..(l + 1) * e0 * MOE_K * h];
            let bias = &desc0.bias[l * e0..(l + 1) * e0];
            let chunk = rows.div_ceil(threads).max(1);
            let parts: Vec<(Vec<u32>, Vec<f32>)> = std::thread::scope(|sc| {
                let handles: Vec<_> = (0..rows)
                    .collect::<Vec<_>>()
                    .chunks(chunk)
                    .map(|part| {
                        let x2 = &x2;
                        let part = part.to_vec();
                        sc.spawn(move || {
                            let mut scores = vec![0.0f32; e0];
                            let mut errs = vec![0.0f32; e0];
                            let mut wv = Vec::with_capacity(part.len());
                            let mut mv = Vec::with_capacity(part.len());
                            for row in part {
                                let x = &x2[row * h..(row + 1) * h];
                                resonance_scores(x, mu, u, MOE_K, bias, None, &mut scores, &mut errs);
                                wv.push(resonance_winner(&scores) as u32);
                                mv.push(errs.iter().copied().fold(f32::INFINITY, f32::min));
                            }
                            (wv, mv)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("trunk scan thread")).collect()
            });
            let mut row = 0usize;
            for (wv, mv) in parts {
                for (w, m) in wv.into_iter().zip(mv) {
                    f(li, idx + row, &x2[row * h..(row + 1) * h], w as usize, m);
                    row += 1;
                }
            }
        }
        idx += rows;
        start += valid;
    }
    Ok(idx)
}

/// P1 novelty of every token of `shard` (the GENERAL shard) for the
/// trunk, per grown layer: `min_e ‖(x−μ_e)⊥U_e‖²` over the trunk experts
/// (runtime formula, no bias, no shell) — `[grown layer index][token]`,
/// tokens in window order. On the pre-growth instance.
pub fn trunk_min_errors(
    gpu: &EmbryoGpu,
    shard: &Shard,
    layers: &[usize],
    desc0: &GrowthDesc,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let n = check_layers(desc0.layers_total, layers)?.len();
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); n];
    scan_trunk(gpu, shard, layers, desc0, |li, _, _, _, m| out[li].push(m))?;
    Ok(out)
}

/// `τ_l` per grown layer: the nearest-rank `q` quantile of the general
/// shard's min trunk errors ([`trunk_min_errors`]) — at most `1 − q` of
/// the general tokens are novel for the trunk in that layer.
pub fn novel_taus(errs: &[Vec<f32>], q: f32) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        q.is_finite() && (0.0..=1.0).contains(&q),
        "--novel-quantile {q} must lie in [0, 1]"
    );
    errs.iter()
        .enumerate()
        .map(|(li, v)| {
            anyhow::ensure!(!v.is_empty(), "no general tokens witnessed for grown layer index {li}");
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            Ok(quantile_sorted(&s, q))
        })
        .collect()
}

/// The novel tokens of the growth corpus in one grown layer
/// ([`novel_sets`]): every token whose min trunk error exceeds `tau`,
/// with a reservoir of their MoE inputs (and the trunk expert winning
/// each) for the clustering.
#[derive(Clone, Debug)]
pub struct NovelSet {
    pub layer: usize,
    pub tau: f32,
    /// Tokens of the corpus trace.
    pub tokens: usize,
    /// Novel ones.
    pub novel: usize,
    /// Per trace token (window order): novel?
    pub mask: Vec<bool>,
    /// `[H]` sum of every novel token's MoE input.
    pub sum: Vec<f32>,
    /// `[res_n, H]` reservoir of novel MoE inputs.
    pub rows: Vec<f32>,
    pub res_n: usize,
    /// The trunk winner (runtime formula) of every reservoir row.
    pub winners: Vec<u32>,
}

impl NovelSet {
    /// The share of the corpus trace that is novel in this layer.
    pub fn share(&self) -> f32 {
        self.novel as f32 / self.tokens.max(1) as f32
    }
}

/// The novel set of the growth TRAIN trace per grown layer: forward on
/// the PRE-growth instance, a token is novel in layer `l` when its min
/// trunk error exceeds `taus[li]` ([`novel_taus`] on the general shard).
/// Deterministic reservoir of [`NOVEL_RES_ROWS`] rows per layer (seeded).
pub fn novel_sets(
    gpu: &EmbryoGpu,
    shard: &Shard,
    layers: &[usize],
    desc0: &GrowthDesc,
    taus: &[f32],
    seed: u64,
) -> anyhow::Result<Vec<NovelSet>> {
    let layers_v = check_layers(desc0.layers_total, layers)?;
    anyhow::ensure!(taus.len() == layers_v.len(), "one τ per grown layer");
    let h = desc0.h;
    let mut sets: Vec<NovelSet> = layers_v
        .iter()
        .zip(taus)
        .map(|(&l, &tau)| NovelSet {
            layer: l,
            tau,
            tokens: 0,
            novel: 0,
            mask: Vec::new(),
            sum: vec![0.0; h],
            rows: Vec::new(),
            res_n: 0,
            winners: Vec::new(),
        })
        .collect();
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    scan_trunk(gpu, shard, layers, desc0, |li, _, x, w, m| {
        let s = &mut sets[li];
        s.tokens += 1;
        let is_novel = m > s.tau;
        s.mask.push(is_novel);
        if !is_novel {
            return;
        }
        for j in 0..h {
            s.sum[j] += x[j];
        }
        let j = s.novel;
        s.novel += 1;
        if s.res_n < NOVEL_RES_ROWS {
            s.rows.extend_from_slice(x);
            s.winners.push(w as u32);
            s.res_n += 1;
        } else {
            let r = (splitmix(&mut rng) % (j as u64 + 1)) as usize;
            if r < NOVEL_RES_ROWS {
                s.rows[r * h..(r + 1) * h].copy_from_slice(x);
                s.winners[r] = w as u32;
            }
        }
    })?;
    Ok(sets)
}

/// Squared distance between two `[H]` rows.
fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// K-means (k-means++ seeding, [`KMEANS_ITERS`] Lloyd iterations, an
/// empty cluster takes the row farthest from its centroid) over `rows`
/// (`[n, H]`), seeded: the label of every row.
pub fn kmeans(rows: &[f32], n: usize, h: usize, k: usize, iters: usize, seed: u64) -> Vec<u32> {
    assert!(k >= 1 && n >= k, "k-means: {n} rows for {k} clusters");
    let row = |r: usize| &rows[r * h..(r + 1) * h];
    let mut rng = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut cent = vec![0.0f32; k * h];
    let first = (splitmix(&mut rng) % n as u64) as usize;
    cent[..h].copy_from_slice(row(first));
    let mut d2 = vec![f32::INFINITY; n];
    for c in 1..k {
        let prev = &cent[(c - 1) * h..c * h].to_vec();
        for r in 0..n {
            let d = dist2(row(r), prev);
            if d < d2[r] {
                d2[r] = d;
            }
        }
        let total: f64 = d2.iter().map(|&x| x as f64).sum();
        let mut pick = (splitmix(&mut rng) >> 11) as f64 / (1u64 << 53) as f64 * total;
        let mut chosen = n - 1;
        for r in 0..n {
            pick -= d2[r] as f64;
            if pick <= 0.0 {
                chosen = r;
                break;
            }
        }
        cent[c * h..(c + 1) * h].copy_from_slice(row(chosen));
    }
    let mut labels = vec![0u32; n];
    let mut own = vec![0.0f32; n];
    for _ in 0..iters {
        for r in 0..n {
            let (mut best, mut bd) = (0usize, f32::INFINITY);
            for c in 0..k {
                let d = dist2(row(r), &cent[c * h..(c + 1) * h]);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            labels[r] = best as u32;
            own[r] = bd;
        }
        let mut counts = vec![0usize; k];
        for &l in &labels {
            counts[l as usize] += 1;
        }
        for c in 0..k {
            if counts[c] == 0 {
                // the row farthest from its own centroid founds the cluster
                let far = (0..n)
                    .filter(|&r| counts[labels[r] as usize] > 1)
                    .max_by(|&a, &b| own[a].partial_cmp(&own[b]).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0);
                counts[labels[far] as usize] -= 1;
                labels[far] = c as u32;
                counts[c] = 1;
                own[far] = 0.0;
            }
        }
        cent.iter_mut().for_each(|v| *v = 0.0);
        for r in 0..n {
            let c = labels[r] as usize;
            for j in 0..h {
                cent[c * h + j] += rows[r * h + j];
            }
        }
        for c in 0..k {
            let m = counts[c].max(1) as f32;
            for j in 0..h {
                cent[c * h + j] /= m;
            }
        }
    }
    labels
}

/// The trunk sources and descriptors of a [`SourceMode::Novel`] growth,
/// per grown layer (in `sets` order): K = 1 → μ = the mean of EVERY novel
/// token, U = the top-`MOE_K` eigenvectors of the reservoir's covariance,
/// the weights of the trunk expert winning the most novel tokens; K > 1 →
/// [`kmeans`] with K clusters over the reservoir (seeded), per cluster
/// its mean / principal subspace and the trunk expert hottest ON THAT
/// CLUSTER (most wins among its rows; lower index on ties). Refuses a
/// layer with fewer than [`NOVEL_MIN_PER_EXPERT`]·K novel tokens.
/// Returns `(sources[li][k], inits[li][k], cluster_rows[li][k])` — the
/// cluster sizes over the reservoir (K = 1: every reservoir row).
#[allow(clippy::type_complexity)]
pub fn novel_inits(
    sets: &[NovelSet],
    k: usize,
    e0: usize,
    seed: u64,
) -> anyhow::Result<(Vec<Vec<usize>>, Vec<Vec<ClusterInit>>, Vec<Vec<usize>>)> {
    anyhow::ensure!(k >= 1 && e0 >= 1);
    for s in sets {
        anyhow::ensure!(
            s.novel >= NOVEL_MIN_PER_EXPERT * k,
            "layer {}: only {} of {} corpus tokens are novel for the trunk (min trunk error > τ = {:.4}), fewer \
             than {} × K {k} = {} — the growth corpus holds too little the trunk cannot already reconstruct in \
             this layer; grow other layers (--layers), lower --novel-quantile, or give more corpus",
            s.layer,
            s.novel,
            s.tokens,
            s.tau,
            NOVEL_MIN_PER_EXPERT,
            NOVEL_MIN_PER_EXPERT * k
        );
        anyhow::ensure!(s.res_n >= k, "layer {}: {} reservoir rows for K {k}", s.layer, s.res_n);
    }
    let results: Vec<anyhow::Result<(Vec<usize>, Vec<ClusterInit>, Vec<usize>)>> = std::thread::scope(|sc| {
        let handles: Vec<_> = sets
            .iter()
            .enumerate()
            .map(|(li, s)| {
                let seed_l = seed.wrapping_add((li * 64) as u64);
                sc.spawn(move || -> anyhow::Result<(Vec<usize>, Vec<ClusterInit>, Vec<usize>)> {
                    let h = s.sum.len();
                    let hottest = |members: &[usize]| -> usize {
                        let mut wins = vec![0usize; e0];
                        for &r in members {
                            wins[s.winners[r] as usize] += 1;
                        }
                        (0..e0).max_by(|&a, &b| wins[a].cmp(&wins[b]).then(b.cmp(&a))).unwrap_or(0)
                    };
                    if k == 1 {
                        let mut init = cluster_descriptor(&s.rows, s.res_n, h, seed_l);
                        for j in 0..h {
                            init.mu[j] = s.sum[j] / s.novel as f32;
                        }
                        init.rows = s.novel;
                        let all: Vec<usize> = (0..s.res_n).collect();
                        return Ok((vec![hottest(&all)], vec![init], vec![s.res_n]));
                    }
                    let labels = kmeans(&s.rows, s.res_n, h, k, KMEANS_ITERS, seed_l);
                    let mut sources = Vec::with_capacity(k);
                    let mut inits = Vec::with_capacity(k);
                    let mut sizes = Vec::with_capacity(k);
                    for c in 0..k {
                        let members: Vec<usize> = (0..s.res_n).filter(|&r| labels[r] as usize == c).collect();
                        anyhow::ensure!(
                            !members.is_empty(),
                            "layer {}: novel cluster {c} of {k} is empty",
                            s.layer
                        );
                        let mut rows = Vec::with_capacity(members.len() * h);
                        for &r in &members {
                            rows.extend_from_slice(&s.rows[r * h..(r + 1) * h]);
                        }
                        inits.push(cluster_descriptor(&rows, members.len(), h, seed_l.wrapping_add(c as u64 + 1)));
                        sources.push(hottest(&members));
                        sizes.push(members.len());
                    }
                    Ok((sources, inits, sizes))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("novel init thread")).collect()
    });
    let mut sources = Vec::with_capacity(sets.len());
    let mut inits = Vec::with_capacity(sets.len());
    let mut sizes = Vec::with_capacity(sets.len());
    for r in results {
        let (s, i, z) = r?;
        sources.push(s);
        inits.push(i);
        sizes.push(z);
    }
    Ok((sources, inits, sizes))
}

/// What the novelty pass witnessed (the record's origin / the summary):
/// `τ` per grown layer, the novel share of the corpus trace, the
/// reservoir and cluster sizes. `masks` (per trace token) feed
/// [`novel_coverage`] and are not serialised.
#[derive(Clone, Debug, serde::Serialize)]
pub struct NovelWitness {
    /// `--novel-quantile`.
    pub quantile: f32,
    pub layers: Vec<usize>,
    pub tau: Vec<f32>,
    /// Tokens of the general shard the quantile was taken over.
    pub general_tokens: usize,
    /// Tokens of the corpus trace.
    pub corpus_tokens: usize,
    pub novel_tokens: Vec<usize>,
    /// `novel_tokens / corpus_tokens` per grown layer.
    pub share_corpus: Vec<f32>,
    pub reservoir_rows: Vec<usize>,
    /// Cluster sizes over the reservoir (`source-mode novel` only).
    pub cluster_rows: Option<Vec<Vec<usize>>>,
    #[serde(skip)]
    pub masks: Vec<Vec<bool>>,
}

impl NovelWitness {
    pub fn from_sets(quantile: f32, general_tokens: usize, sets: &[NovelSet], cluster_rows: Option<Vec<Vec<usize>>>) -> NovelWitness {
        NovelWitness {
            quantile,
            layers: sets.iter().map(|s| s.layer).collect(),
            tau: sets.iter().map(|s| s.tau).collect(),
            general_tokens,
            corpus_tokens: sets.first().map(|s| s.tokens).unwrap_or(0),
            novel_tokens: sets.iter().map(|s| s.novel).collect(),
            share_corpus: sets.iter().map(|s| s.share()).collect(),
            reservoir_rows: sets.iter().map(|s| s.res_n).collect(),
            cluster_rows,
            masks: sets.iter().map(|s| s.mask.clone()).collect(),
        }
    }
}

/// `novel_coverage`: of the corpus-trace tokens novel for the trunk
/// ([`NovelWitness::masks`]), the fraction a grown expert wins — per
/// grown layer, with the shell (what the runtime routes to the record)
/// and without; overall = Σ wins / Σ novel over the layers.
#[derive(Clone, Debug, serde::Serialize)]
pub struct NovelCoverageReport {
    pub layers: Vec<usize>,
    pub novel_tokens: Vec<usize>,
    pub per_layer_shell: Vec<f32>,
    pub per_layer_noshell: Vec<f32>,
    pub overall_shell: f32,
    pub overall_noshell: f32,
}

/// [`NovelCoverageReport`] of `traces` (the growth TRAIN trace on the
/// trained instance, the same windows in the same order as the novelty
/// pass) under `shells`.
pub fn novel_coverage(traces: &[LayerTrace], masks: &[Vec<bool>], shells: &[Vec<f32>]) -> anyhow::Result<NovelCoverageReport> {
    anyhow::ensure!(
        traces.len() == masks.len() && traces.len() == shells.len(),
        "novel coverage: {} traces, {} masks, {} shell rows",
        traces.len(),
        masks.len(),
        shells.len()
    );
    let (mut tot_n, mut tot_sh, mut tot_no) = (0usize, 0usize, 0usize);
    let mut per_sh = Vec::with_capacity(traces.len());
    let mut per_no = Vec::with_capacity(traces.len());
    let mut novel = Vec::with_capacity(traces.len());
    for ((tr, mask), sh) in traces.iter().zip(masks).zip(shells) {
        anyhow::ensure!(
            tr.tokens == mask.len(),
            "novel coverage: layer {} trace has {} tokens, the novelty mask {}",
            tr.layer,
            tr.tokens,
            mask.len()
        );
        let (mut n, mut c_sh, mut c_no) = (0usize, 0usize, 0usize);
        for t in 0..tr.tokens {
            if !mask[t] {
                continue;
            }
            n += 1;
            if tr.winner(t, Some(sh.as_slice())) >= tr.e0 {
                c_sh += 1;
            }
            if tr.winner(t, None) >= tr.e0 {
                c_no += 1;
            }
        }
        novel.push(n);
        per_sh.push(c_sh as f32 / n.max(1) as f32);
        per_no.push(c_no as f32 / n.max(1) as f32);
        tot_n += n;
        tot_sh += c_sh;
        tot_no += c_no;
    }
    Ok(NovelCoverageReport {
        layers: traces.iter().map(|t| t.layer).collect(),
        novel_tokens: novel,
        per_layer_shell: per_sh,
        per_layer_noshell: per_no,
        overall_shell: tot_sh as f32 / tot_n.max(1) as f32,
        overall_noshell: tot_no as f32 / tot_n.max(1) as f32,
    })
}

/// Shells from the growth TRAIN witness: per grown layer and expert, the
/// `q` quantile of the reconstruction errors of the tokens it wins (argmax
/// without shell); an expert that wins nothing gets a closed shell (0.0).
/// Returns `(shells[layer][k], wins[layer][k])` in `traces` order.
pub fn shells_from_traces(traces: &[LayerTrace], q: f32) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let mut shells = Vec::with_capacity(traces.len());
    let mut wins = Vec::with_capacity(traces.len());
    for tr in traces {
        let mut errs: Vec<Vec<f32>> = vec![Vec::new(); tr.k];
        for t in 0..tr.tokens {
            let w = tr.winner(t, None);
            if w >= tr.e0 {
                errs[w - tr.e0].push(tr.grown_err[t * tr.k + (w - tr.e0)]);
            }
        }
        let mut sh = Vec::with_capacity(tr.k);
        let mut wn = Vec::with_capacity(tr.k);
        for v in errs.iter_mut() {
            wn.push(v.len());
            if v.is_empty() {
                sh.push(0.0);
            } else {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                sh.push(quantile_sorted(v, q));
            }
        }
        shells.push(sh);
        wins.push(wn);
    }
    (shells, wins)
}

/// The shells of a growth and how each was calibrated (`[grown layer
/// index][k]` throughout, in trace order).
#[derive(Clone, Debug, serde::Serialize)]
pub struct ShellWitness {
    pub mode: ShellMode,
    /// `--shell-quantile`.
    pub quantile: f32,
    /// `--shell-target-shift` (general-target only).
    pub target_shift: Option<f32>,
    pub shells: Vec<Vec<f32>>,
    /// Tokens of the growth TRAIN trace each expert wins (no shell).
    pub wins: Vec<Vec<usize>>,
    /// Tokens of the general trace each expert captures (no shell) and
    /// their share of the layer's general tokens (general-target only).
    pub general_captured: Option<Vec<Vec<usize>>>,
    pub general_share: Option<Vec<Vec<f32>>>,
    /// The quantile each shell is (`quantile` of the won errors, or
    /// `target / share` of the captured general errors).
    pub applied_quantile: Vec<Vec<f32>>,
    /// The rule each shell came from: `won-quantile` | `general-target`.
    pub rule: Vec<Vec<&'static str>>,
}

/// [`shells_from_traces`] as a witness ([`ShellMode::WonQuantile`]).
pub fn shells_won_quantile(train: &[LayerTrace], q: f32) -> ShellWitness {
    let (shells, wins) = shells_from_traces(train, q);
    let applied_quantile = shells.iter().map(|s| vec![q; s.len()]).collect();
    let rule = shells.iter().map(|s| vec!["won-quantile"; s.len()]).collect();
    ShellWitness {
        mode: ShellMode::WonQuantile,
        quantile: q,
        target_shift: None,
        shells,
        wins,
        general_captured: None,
        general_share: None,
        applied_quantile,
        rule,
    }
}

/// Shells calibrated on the GENERAL trace ([`ShellMode::GeneralTarget`]):
/// per grown layer and expert `e`, `share_e` = the fraction of ALL the
/// layer's general tokens whose no-shell winner is `e`; `share_e ≤ target`
/// → the won-quantile shell (`q` of the won errors on the TRAIN trace);
/// else the `(target / share_e)`-quantile (nearest rank) of the
/// reconstruction errors of those captured general tokens, so the shell
/// admits at most `target` of the layer's general tokens (`+ 1 / tokens`
/// from the rank rounding; ties admit their equals). `target ≤ 0` closes
/// the shell of a capturing expert (0.0). Both traces must witness the
/// same layers and `K`.
pub fn shells_general_target(
    train: &[LayerTrace],
    general: &[LayerTrace],
    q: f32,
    target: f32,
) -> anyhow::Result<ShellWitness> {
    anyhow::ensure!(
        train.len() == general.len()
            && train
                .iter()
                .zip(general)
                .all(|(a, b)| a.layer == b.layer && a.k == b.k && a.e0 == b.e0),
        "the train and general traces must witness the same layers and K"
    );
    anyhow::ensure!(
        target.is_finite() && target <= 1.0,
        "--shell-target-shift {target} must be a finite fraction ≤ 1"
    );
    let won = shells_won_quantile(train, q);
    let mut shells = Vec::with_capacity(train.len());
    let mut captured_all = Vec::with_capacity(train.len());
    let mut share_all = Vec::with_capacity(train.len());
    let mut applied = Vec::with_capacity(train.len());
    let mut rule = Vec::with_capacity(train.len());
    for (li, tr) in general.iter().enumerate() {
        let mut errs: Vec<Vec<f32>> = vec![Vec::new(); tr.k];
        for t in 0..tr.tokens {
            let w = tr.winner(t, None);
            if w >= tr.e0 {
                errs[w - tr.e0].push(tr.grown_err[t * tr.k + (w - tr.e0)]);
            }
        }
        let n = tr.tokens.max(1) as f32;
        let mut sh = Vec::with_capacity(tr.k);
        let mut cap = Vec::with_capacity(tr.k);
        let mut share = Vec::with_capacity(tr.k);
        let mut ap = Vec::with_capacity(tr.k);
        let mut ru = Vec::with_capacity(tr.k);
        for (kk, v) in errs.iter_mut().enumerate() {
            let s = v.len() as f32 / n;
            cap.push(v.len());
            share.push(s);
            if v.is_empty() || s <= target {
                sh.push(won.shells[li][kk]);
                ap.push(q);
                ru.push("won-quantile");
                continue;
            }
            let qq = target / s;
            let shell = if qq <= 0.0 {
                0.0
            } else {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                quantile_sorted(v, qq.min(1.0))
            };
            sh.push(shell);
            ap.push(qq.max(0.0));
            ru.push("general-target");
        }
        shells.push(sh);
        captured_all.push(cap);
        share_all.push(share);
        applied.push(ap);
        rule.push(ru);
    }
    Ok(ShellWitness {
        mode: ShellMode::GeneralTarget,
        quantile: q,
        target_shift: Some(target),
        shells,
        wins: won.wins,
        general_captured: Some(captured_all),
        general_share: Some(share_all),
        applied_quantile: applied,
        rule,
    })
}

/// `routing_shift`: the fraction of tokens a grown expert wins, per layer
/// (in `traces` order) and overall (a grown winner in at least one grown
/// layer), without and with the shell.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ShiftReport {
    pub tokens: usize,
    pub layers: Vec<usize>,
    pub per_layer_noshell: Vec<f32>,
    pub per_layer_shell: Vec<f32>,
    pub overall_noshell: f32,
    pub overall_shell: f32,
}

pub fn routing_shift(traces: &[LayerTrace], shells: &[Vec<f32>]) -> ShiftReport {
    let n = traces.first().map(|t| t.tokens).unwrap_or(0);
    let mut any_no = vec![false; n];
    let mut any_sh = vec![false; n];
    let mut per_no = Vec::new();
    let mut per_sh = Vec::new();
    for (tr, sh) in traces.iter().zip(shells) {
        let (mut c_no, mut c_sh) = (0usize, 0usize);
        for t in 0..tr.tokens {
            if tr.winner(t, None) >= tr.e0 {
                c_no += 1;
                any_no[t] = true;
            }
            if tr.winner(t, Some(sh.as_slice())) >= tr.e0 {
                c_sh += 1;
                any_sh[t] = true;
            }
        }
        per_no.push(c_no as f32 / tr.tokens.max(1) as f32);
        per_sh.push(c_sh as f32 / tr.tokens.max(1) as f32);
    }
    ShiftReport {
        tokens: n,
        layers: traces.iter().map(|t| t.layer).collect(),
        per_layer_noshell: per_no,
        per_layer_shell: per_sh,
        overall_noshell: any_no.iter().filter(|&&x| x).count() as f32 / n.max(1) as f32,
        overall_shell: any_sh.iter().filter(|&&x| x).count() as f32 / n.max(1) as f32,
    }
}

/// `coverage`: the fraction of tokens inside at least one grown expert's
/// shell, per layer (in `traces` order) and overall (any grown layer).
#[derive(Clone, Debug, serde::Serialize)]
pub struct CoverageReport {
    pub tokens: usize,
    pub layers: Vec<usize>,
    pub per_layer: Vec<f32>,
    pub overall: f32,
}

pub fn coverage(traces: &[LayerTrace], shells: &[Vec<f32>]) -> CoverageReport {
    let n = traces.first().map(|t| t.tokens).unwrap_or(0);
    let mut any = vec![false; n];
    let mut per = Vec::new();
    for (tr, sh) in traces.iter().zip(shells) {
        let mut c = 0usize;
        for t in 0..tr.tokens {
            if tr.inside_any(t, sh) {
                c += 1;
                any[t] = true;
            }
        }
        per.push(c as f32 / tr.tokens.max(1) as f32);
    }
    CoverageReport {
        tokens: n,
        layers: traces.iter().map(|t| t.layer).collect(),
        per_layer: per,
        overall: any.iter().filter(|&&x| x).count() as f32 / n.max(1) as f32,
    }
}

// ───────────────────────── the record ─────────────────────────

/// The `expert_append` records of `header` that are not retired and touch
/// any of `layers` (record index, id, status, common layers).
pub fn growth_records_in_layers(
    header: &CmfHeader,
    layers: &[usize],
) -> Vec<(usize, String, String, Vec<usize>)> {
    header
        .skills
        .iter()
        .enumerate()
        .filter(|(_, r)| r.kind.as_deref() == Some(skill_kind::EXPERT_APPEND))
        .filter(|(_, r)| r.status.as_deref() != Some("retired"))
        .filter_map(|(i, r)| {
            let common: Vec<usize> = r.layers.iter().copied().filter(|l| layers.contains(l)).collect();
            (!common.is_empty()).then(|| {
                (
                    i,
                    r.id.clone(),
                    r.status.clone().unwrap_or_else(|| "?".into()),
                    common,
                )
            })
        })
        .collect()
}

/// Refuse to grow on a base whose (non-retired) `expert_append` records
/// already occupy any of the grown layers: the trainer's arena holds only
/// the `E0` trunk experts, so the new experts would train and be
/// witnessed (shells, coverage, routing shift) in a routing WITHOUT the
/// experts the runtime mounts next to them (`CMF_GROWTH=active|all`).
/// Grow from the genome file F0, or from layers those records do not
/// touch.
pub fn check_base_growth_records(header: &CmfHeader, layers: &[usize]) -> anyhow::Result<()> {
    let busy = growth_records_in_layers(header, layers);
    anyhow::ensure!(
        busy.is_empty(),
        "refusing: --base already carries expert_append records in the grown layers ({}) — the \
         trainer routes with the E0 trunk experts only, so their shells / coverage / routing \
         shift would be measured without the experts the runtime mounts alongside; grow from \
         the genome file F0 or choose layers those records do not touch",
        busy.iter()
            .map(|(i, id, st, ls)| format!("#{i} '{id}' [{st}] layers {ls:?}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

/// Inputs of [`write_growth_record`].
pub struct RecordArgs<'a> {
    /// The genome file `F0` (never rewritten).
    pub base: &'a Path,
    /// The new file `F1` (must not exist; never `base`).
    pub out: &'a Path,
    pub id: &'a str,
    /// The PRE-growth checkpoint — must be exactly the checkpoint of `base`
    /// (every trunk tensor byte-identical, `master_trunk_hash` equal).
    pub ck0: &'a Checkpoint,
    /// The trained grown checkpoint (`experts == e0 + K`).
    pub trained: &'a Checkpoint,
    pub e0: usize,
    /// The grown layers (ascending).
    pub layers: &'a [usize],
    pub shell_quantile: f32,
    /// `shells[layer index in `layers`][k]`.
    pub shells: &'a [Vec<f32>],
    /// How the grown bias was chosen (`--bias-mode`), for the refusal
    /// message; the value itself is `grown_bias`.
    pub bias_mode: BiasMode,
    /// The balancing bias every grown expert must carry —
    /// `grown_bias[layer index in `layers`][k]` ([`expected_grown_bias`]):
    /// the trained checkpoint's bias must equal it bit for bit (it was
    /// frozen: `bias_frozen_from = E0`) and the record stores it.
    pub grown_bias: &'a [Vec<f32>],
    pub origin: serde_json::Value,
    pub quality: serde_json::Value,
}

/// Append the grown experts as an `expert_append` record (status
/// `quarantine`) to a COPY of `base`: refuse unless `out` is a new file ≠
/// `base`, `base` carries a resonance-MoE genome with `E0 == e0`, `ck0` is
/// exactly its checkpoint, every grown layer is an MoE layer of the trunk
/// and the descriptor rank matches; tensors are exactly the format's
/// layout plan (`skill.{id}.model.layers.{l}.mlp.experts.{E0+n+k}.*`,
/// the bias written as `grown_bias` — refused if the trained bias differs
/// from it — and the shell); then `CmfModel::append_skill` on a private
/// temp, G1 verification (trunk hash, directory entries, prefix bytes)
/// and a no-overwrite publish. Returns a summary (record position, expert
/// indices per layer, appended bytes).
pub fn write_growth_record(a: &RecordArgs) -> anyhow::Result<serde_json::Value> {
    crate::skill::check_out_path(a.base, a.out)?;
    anyhow::ensure!(
        !a.id.is_empty() && !a.id.contains('.') && !a.id.chars().any(|c| c.is_whitespace()),
        "record id '{}' must be non-empty without '.' or whitespace",
        a.id
    );
    anyhow::ensure!(
        a.shell_quantile.is_finite() && (0.0..=1.0).contains(&a.shell_quantile),
        "--shell-quantile {} must lie in [0, 1]",
        a.shell_quantile
    );
    let trained = a.trained;
    let e0 = a.e0;
    let e1 = trained.cfg.experts;
    anyhow::ensure!(
        e0 >= 1 && e1 > e0,
        "the trained checkpoint has {e1} experts, e0 = {e0}: nothing grown"
    );
    anyhow::ensure!(
        a.ck0.cfg.experts == e0,
        "the pre-growth checkpoint has {} experts, e0 = {e0}",
        a.ck0.cfg.experts
    );
    let kn = e1 - e0;
    let layers = check_layers(trained.cfg.layers, a.layers)?;
    anyhow::ensure!(
        layers == a.layers,
        "layers must be ascending (shells are given in that order), got {:?}",
        a.layers
    );
    anyhow::ensure!(
        a.shells.len() == layers.len() && a.shells.iter().all(|s| s.len() == kn),
        "shells must be [{} layers][{kn}]",
        layers.len()
    );
    for (li, sh) in a.shells.iter().enumerate() {
        for (kk, s) in sh.iter().enumerate() {
            anyhow::ensure!(
                s.is_finite(),
                "shell of layer {} expert {} is {s} — must be finite",
                layers[li],
                e0 + kk
            );
        }
    }
    anyhow::ensure!(
        a.grown_bias.len() == layers.len() && a.grown_bias.iter().all(|b| b.len() == kn),
        "grown_bias must be [{} layers][{kn}]",
        layers.len()
    );
    for (li, bs) in a.grown_bias.iter().enumerate() {
        for (kk, b) in bs.iter().enumerate() {
            anyhow::ensure!(
                b.is_finite() && (a.bias_mode != BiasMode::Zero || *b == 0.0),
                "grown bias of layer {} expert {} is {b} under --bias-mode {}",
                layers[li],
                e0 + kk,
                a.bias_mode.as_str()
            );
        }
    }
    let base = CmfModel::open(a.base)?;
    let genome = base.header.genome.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "--base {} carries no GENOME: export the genome with `export --genome-id <id> \
             --genome-status <status>` first (a growth record binds to a frozen genome)",
            a.base.display()
        )
    })?;
    let e0_file = genome_moe_experts(&base.header)?;
    anyhow::ensure!(
        e0_file == e0,
        "--base genome '{}' has E0 = {e0_file} trunk experts, the checkpoint {e0}",
        genome.id
    );
    let trunk_moe = moe_layers(&base.tensors);
    for &l in &layers {
        anyhow::ensure!(
            trunk_moe.contains(&l),
            "layer {l} is not an MoE layer of the trunk (MoE layers: {trunk_moe:?})"
        );
    }
    let binding = crate::skill::bind_ckpt_to_base(a.ck0, &base)?;
    let rank = trunk_expert_rank(&base.header, &base.tensors, layers[0])?;
    let at = base.header.skills.len();
    let record = SkillRecord {
        id: a.id.to_string(),
        layers: layers.clone(),
        base_arch: Some(base.header.arch.arch_name.clone()),
        provenance: Some(serde_json::json!({
            "producer": "cortiq-embryo grow",
            "recipe": RECIPE_GROWTH,
        })),
        quality: Some(a.quality.clone()),
        kind: Some(skill_kind::EXPERT_APPEND.into()),
        experts: Some(ExpertAppend {
            count: kn,
            shell_quantile: a.shell_quantile,
            rank,
        }),
        bound: Some(SkillBound {
            genome_id: genome.id.clone(),
            generation: genome.generation,
            master_trunk_hash: genome.master_trunk_hash.clone(),
        }),
        state_effect: Some(expert_append_state_effect(&layers)),
        status: Some("quarantine".into()),
        origin: Some(a.origin.clone()),
        ..Default::default()
    };
    let plan = expert_append_layout(&base.header, &base.tensors, at, &record)?;
    // ---- payloads from the trained checkpoint, exactly the plan ----
    let lay1 = Layout::new(&trained.cfg);
    let (h, i) = (trained.cfg.hidden, trained.cfg.inter);
    let ew = 3 * h * i;
    let ex = |name: &str| -> anyhow::Result<&[f32]> {
        trained
            .extras
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, x)| x.as_slice())
            .ok_or_else(|| anyhow::anyhow!("trained checkpoint has no {name}"))
    };
    let mu1 = ex("desc.mu")?;
    let u1 = ex("desc.u")?;
    let b1 = ex("desc.bias")?;
    anyhow::ensure!(
        mu1.len() == trained.cfg.layers * e1 * h
            && u1.len() == trained.cfg.layers * e1 * MOE_K * h
            && b1.len() == trained.cfg.layers * e1,
        "trained descriptors do not match the config"
    );
    let mut specs: Vec<TensorSpec> = Vec::with_capacity(plan.len());
    let mut indices: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for p in &plan {
        let l = p.layer;
        let li = layers.iter().position(|&x| x == l).expect("plan layers ⊆ record layers");
        let base_l = expert_append_base(&base.header, at, l)?;
        anyhow::ensure!(
            p.expert >= base_l && p.expert - base_l < kn,
            "plan expert {} of layer {l} outside [{base_l}, {})",
            p.expert,
            base_l + kn
        );
        let kk = p.expert - base_l;
        let et = e0 + kk;
        let ffn = ffn_offs(&lay1.layers[l]);
        let ffn_slice = |off: usize, n: usize| &trained.params[off..off + n];
        let (shape, data): (Vec<usize>, Vec<f32>) = match p.leaf {
            expert_leaf::GATE => (vec![i, h], ffn_slice(ffn.experts + et * ew, i * h).to_vec()),
            expert_leaf::UP => (vec![i, h], ffn_slice(ffn.experts + et * ew + h * i, i * h).to_vec()),
            expert_leaf::DOWN => (vec![h, i], ffn_slice(ffn.experts + et * ew + 2 * h * i, h * i).to_vec()),
            expert_leaf::MU => (vec![h], mu1[(l * e1 + et) * h..(l * e1 + et + 1) * h].to_vec()),
            expert_leaf::U => (
                vec![MOE_K, h],
                u1[(l * e1 + et) * MOE_K * h..(l * e1 + et + 1) * MOE_K * h].to_vec(),
            ),
            expert_leaf::BIAS => {
                let b = b1[l * e1 + et];
                let want = a.grown_bias[li][kk];
                anyhow::ensure!(
                    b.to_bits() == want.to_bits(),
                    "grown expert {et} of layer {l} carries bias {b}, --bias-mode {} wants {want} \
                     (the bias is frozen for the whole training: bias_frozen_from = E0)",
                    a.bias_mode.as_str()
                );
                (vec![1], vec![want])
            }
            expert_leaf::SHELL => (vec![1], vec![a.shells[li][kk]]),
            other => anyhow::bail!("unknown expert leaf '{other}' in the layout plan"),
        };
        anyhow::ensure!(
            shape == p.shape,
            "{}: the checkpoint's shape {shape:?} != the trunk's {:?}",
            p.name,
            p.shape
        );
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in &data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        specs.push(TensorSpec {
            name: p.name.clone(),
            dtype: TensorDtype::F32,
            shape,
            data: bytes,
        });
        let v = indices.entry(l).or_default();
        if !v.contains(&p.expert) {
            v.push(p.expert);
        }
    }
    let base_len = std::fs::metadata(a.base)?.len();
    drop(base);
    // ---- private copy + true tail append, G1, publish ----
    let (tmp, file) = crate::skill::create_bake_tmp(a.out)?;
    let staged = (|| -> anyhow::Result<u64> {
        let mut file = file;
        let copied = std::io::copy(&mut std::fs::File::open(a.base)?, &mut file)?;
        anyhow::ensure!(
            copied == base_len,
            "--base {} changed while the growth ran ({base_len} → {copied} bytes)",
            a.base.display()
        );
        file.sync_all()?;
        drop(file);
        CmfModel::append_skill(&tmp, record, &specs, None, None, None)?;
        crate::skill::verify_append(a.base, &tmp)?;
        Ok(std::fs::metadata(&tmp)?.len() - base_len)
    })();
    let appended = match staged {
        Ok(v) => v,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    crate::skill::publish_new_file(&tmp, a.out)?;
    Ok(serde_json::json!({
        "id": a.id,
        "out": a.out.display().to_string(),
        "record_index": at,
        "trunk_tensors_bound": binding.trunk_tensors,
        "experts": indices,
        "rank": rank,
        "status": "quarantine",
        "appended_bytes": appended,
        "genome": {"id": genome.id, "generation": genome.generation, "trunk_hash": genome.trunk_hash},
    }))
}
