//! Format v2 "knowledge without forgetting" (spec §9.2–§9.5): a FROZEN
//! genome, skill records v2 that own their tensors over it, and the
//! backbone-gated router policy.
//!
//! Three `required_features` bits carry the semantics, each DERIVED by the
//! writers from header content and checked both ways on open:
//!
//! * `GENOME` ⇔ `header.genome` — the reader recomputes [`trunk_hash`] from
//!   the directory and refuses a file whose trunk changed without a new
//!   genome;
//! * `SKILLS_V2` ⇔ some `skills[i].kind` — every v2 record is validated
//!   against the genome it is bound to and against the directory;
//! * `ROUTER_V2` ⇔ `header.router` — only the declared `backbone_gated`
//!   policy may auto-route; the legacy argmin-of-skills is not a fallback.
//!
//! Everything here is pure metadata logic (no I/O): `format.rs` calls
//! [`validate_knowledge`] from `open()`, from every writer and from the
//! tail append, so a file the writer produced is a file the reader accepts.

use crate::format::{
    CmfError, CmfHeader, SelectionDescriptor, SkillRecord, TensorEntry, TensorSpec, features,
};
use crate::hash::hash64;
use crate::types::{LayerType, ModelArch, TensorDtype};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};

/// `genome.status` values.
pub const GENOME_STATUSES: &[&str] = &["pre_chat", "sealed", "candidate", "rejected"];
/// `skills[i].status` values (v2 records).
pub const SKILL_STATUSES: &[&str] = &["quarantine", "active", "stale_regate", "retired"];
/// `segments[i].kind` values.
pub const SEGMENT_KINDS: &[&str] = &["genome", "base", "skill"];
/// The metric every router-v2 descriptor uses (unit-normalized φ).
pub const METRIC_MSE_UNIT: &str = "mse_unit";
/// Row id of the backbone class in router-v2 decisions. No skill may use it.
pub const BACKBONE_CLASS_ID: &str = "@backbone";
/// Ids a v2 skill may not take: the CLI and the route reports spell the
/// backbone and the automatic decision with them (`--skill none`,
/// `--route backbone|auto`, `target: "backbone"`), so a skill with such an
/// id would be unreachable or indistinguishable from the backbone.
pub const RESERVED_SKILL_IDS: &[&str] = &["backbone", "none", "auto"];

/// Skill-record kinds (`SkillRecord.kind`).
pub mod skill_kind {
    /// Full-shape replacement of FFN tensors of some layers (the Patent-15
    /// per-skill record), switched per request by the router.
    pub const FFN_REPLACE: &str = "ffn_replace";
    /// New MoE experts appended after the trunk's `E0` in the grown layers
    /// (DTG-MA new edges / P3 growth — not a P15 record): the Embryo's
    /// built-in resonance router picks them per token inside the layer, so
    /// the record is part of the organism and is never switched per
    /// request. Layout and rules: [`super::ExpertAppend`],
    /// [`super::expert_append_layout`].
    pub const EXPERT_APPEND: &str = "expert_append";
    /// An explicit reference memory (spec §9.5.2): a key → card table
    /// appended after the sealed genome under `skill.{id}.lookup.*`,
    /// picked per request by the `backbone_gated` router like any v2
    /// record and answered from the table — exact facts by construction,
    /// no forgetting by construction (the network's computation and state
    /// are untouched), O(1) per query. Layout and rules:
    /// [`super::LookupInfo`], [`super::lookup_leaf`].
    pub const LOOKUP: &str = "lookup";
    /// Kinds this reader implements.
    pub const IMPLEMENTED: &[&str] = &[FFN_REPLACE, EXPERT_APPEND, LOOKUP];
    /// Kinds that are named (so files can be planned against them) but
    /// REFUSED until a reader implements them: `anchor_sinks`, `mask_only`,
    /// `fcd_vector` (FCD-JMLR Tucker v_t / P1 cl.7).
    pub const RESERVED: &[&str] = &["anchor_sinks", "mask_only", "fcd_vector"];
}

/// `StateEffect.switch` values.
pub mod state_switch {
    /// Switching the skill mid-sequence leaves every recurrent / ring state
    /// exact: no layer after the first replaced FFN carries sequence state.
    pub const TRANSPARENT: &str = "transparent";
    /// The skill may only change at a sequence start (state is reset).
    pub const SEQUENCE_START: &str = "sequence_start";
    /// Ordered from most to least permissive.
    pub const ALL: &[&str] = &[TRANSPARENT, SEQUENCE_START];
}

// ───────────────────────── genome + lineage ─────────────────────────

/// The frozen genome the file carries (`header.genome`, bit `GENOME`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeInfo {
    /// Stable family id, e.g. `"embryo-o1-fam-a"`. Gradient training of a
    /// sealed trunk is a NEW genome id (a new birth), not generation+1.
    pub id: String,
    /// 0 at birth.
    pub generation: u32,
    /// `pre_chat` | `sealed` | `candidate` | `rejected`.
    pub status: String,
    /// hex hash64 of the trunk AS ENCODED in this file ([`trunk_hash`]).
    /// A writer fills an empty value; a non-empty one must match the
    /// content or the writer refuses.
    #[serde(default)]
    pub trunk_hash: String,
    /// `"f32"` | `"f16"` | `"q4tp"` | … (informational).
    pub encoding: String,
    /// trunk_hash of the f32 master (== `trunk_hash` when `encoding` is
    /// `"f32"`; the writer fills it then). Skill records bind to THIS, so a
    /// requant does not orphan them.
    #[serde(default)]
    pub master_trunk_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<GenomeParent>,
    /// Measured reference battery `{name: {value, dataset_sha256}}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<serde_json::Value>,
    /// `E0`: trunk experts per MoE layer of a resonance-routed genome
    /// (`arch.moe.router_resonance`). The writer fills an empty value from
    /// `arch.moe.num_experts` when it seals the genome; the reader refuses
    /// a value that disagrees with the arch. `expert_append` records grow
    /// every layer from here ([`expert_append_base`]). Absent on dense /
    /// gated-MoE genomes and on files written before the field existed
    /// ([`genome_moe_experts`] falls back to the arch).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe_experts: Option<usize>,
}

impl GenomeInfo {
    /// A genome at birth; the writer fills `trunk_hash` (and the master
    /// hash for `f32`) and `moe_experts` from the content it writes.
    pub fn birth(
        id: impl Into<String>,
        status: impl Into<String>,
        encoding: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            generation: 0,
            status: status.into(),
            trunk_hash: String::new(),
            encoding: encoding.into(),
            master_trunk_hash: String::new(),
            parent: None,
            reference: None,
            moe_experts: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenomeParent {
    pub id: String,
    pub generation: u32,
    pub master_trunk_hash: String,
}

/// One journal entry (`header.lineage`): `birth` | `skill_committed` |
/// `skill_retired` | `requant` | `recalibrate` | `compact` | `rollback`.
/// `seq` is strictly increasing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LineageEvent {
    pub seq: u32,
    pub event: String,
    /// RFC 3339 UTC.
    pub ts: String,
    #[serde(default)]
    pub detail: serde_json::Value,
}

impl LineageEvent {
    /// An event stamped with the current UTC time.
    pub fn now(seq: u32, event: impl Into<String>, detail: serde_json::Value) -> Self {
        Self {
            seq,
            event: event.into(),
            ts: utc_now_rfc3339(),
            detail,
        }
    }
}

/// Data-range ownership after tail appends (`header.segments`). Offsets
/// are relative to `data_off`, like directory offsets. A full rewrite is a
/// compaction and clears the table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    /// `genome` | `base` | `skill`.
    pub kind: String,
    pub id: String,
    pub data_start: u64,
    pub data_end: u64,
}

// ───────────────────────── skill records v2 ─────────────────────────

/// A base tensor a v2 skill replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillOverride {
    /// Trunk tensor name (`X` of `skill.{id}.X`).
    pub name: String,
    /// hex hash64 of the base entry the skill was trained against (the f32
    /// master's entry; comparable with this file only when
    /// `genome.encoding == "f32"`).
    pub base_hash: String,
}

/// The genome a v2 skill is bound to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillBound {
    pub genome_id: String,
    pub generation: u32,
    pub master_trunk_hash: String,
}

/// What switching this skill does to the O(1) sequence state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateEffect {
    pub first_affected_layer: usize,
    /// `transparent` | `sequence_start` ([`state_switch`]).
    pub switch: String,
    /// Extra per-sequence state the record adds (0 for `ffn_replace`).
    pub state_bytes_added: u64,
}

/// Does this layer type carry sequence state (KV, ring, recurrent)?
/// Exhaustive on purpose: a new operator must decide here.
pub fn layer_type_is_stateful(t: LayerType) -> bool {
    match t {
        LayerType::FullAttention
        | LayerType::SlidingAttention
        | LayerType::LinearAttention
        | LayerType::ShortConv
        | LayerType::Kda
        | LayerType::BoundedAttention => true,
    }
}

/// The state effect of replacing the FFN of `layers`, computed by the
/// WRITER from `arch.layer_types`: the FFN output of layer l feeds every
/// layer j > l, so switching is `transparent` iff no layer after
/// `min(layers)` carries sequence state (and the stack is not looped —
/// a loop feeds layer l back into layer 0). The MTP head is not counted:
/// its state only steers drafts that the main model verifies.
pub fn ffn_replace_state_effect(arch: &ModelArch, layers: &[usize]) -> StateEffect {
    let first = layers.iter().copied().min().unwrap_or(0);
    let later_stateful = arch.num_loops > 1
        || arch
            .layer_types
            .iter()
            .enumerate()
            .any(|(j, t)| j > first && layer_type_is_stateful(*t));
    StateEffect {
        first_affected_layer: first,
        switch: if later_stateful {
            state_switch::SEQUENCE_START
        } else {
            state_switch::TRANSPARENT
        }
        .into(),
        state_bytes_added: 0,
    }
}

// ───────────────────────── expert_append ─────────────────────────

/// Parameters of a `kind = "expert_append"` record (`SkillRecord.experts`):
/// `count` new experts in EACH of `SkillRecord.layers` (the grown MoE
/// layers, any non-empty subset), appended after the experts already
/// present in that layer. The per-layer index of the new experts is not
/// stored — the reader computes it ([`expert_append_base`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertAppend {
    /// New experts per grown layer (≥ 1).
    pub count: usize,
    /// The quantile (in `[0, 1]`) of the reconstruction errors of the tokens
    /// each expert won on the growth corpus that the trainer wrote into its
    /// `desc.shell` — provenance of the shell, not used by the reader.
    pub shell_quantile: f32,
    /// Rank of every new expert's `desc.u` — must equal the trunk's expert 0
    /// rank in every grown layer ([`trunk_expert_rank`]); 0 when the trunk
    /// carries no `desc.u` (then the record carries none either).
    pub rank: usize,
}

/// Per-expert tensor leaves of an MoE layer
/// (`model.layers.{l}.mlp.experts.{e}.{leaf}`), in the order a record lists
/// them.
pub mod expert_leaf {
    pub const GATE: &str = "gate_proj.weight";
    pub const UP: &str = "up_proj.weight";
    pub const DOWN: &str = "down_proj.weight";
    /// Resonance descriptor mean `[hidden]`.
    pub const MU: &str = "desc.mu";
    /// Resonance descriptor subspace, same shape as the trunk's expert 0
    /// (`[rank, hidden]` as the Embryo exports it); absent when `rank == 0`.
    pub const U: &str = "desc.u";
    /// `[1]` f32 finite — the grown expert's balancing bias, frozen by
    /// the trainer (0.0, or its source trunk expert's; the record's origin
    /// says which as `bias_mode`).
    pub const BIAS: &str = "desc.bias";
    /// `[1]` f32 finite — reconstruction-error threshold outside which the
    /// expert never wins (grown experts only; the trunk has no shell).
    pub const SHELL: &str = "desc.shell";
    /// Every leaf a grown expert carries (`U` only when `rank > 0`).
    pub const ALL: &[&str] = &[GATE, UP, DOWN, MU, U, BIAS, SHELL];
}

/// `model.layers.{layer}.mlp.experts.{expert}.{leaf}` — a trunk expert
/// tensor; a record's copy is `skill.{id}.` + this
/// ([`expert_append_tensor_name`]).
pub fn expert_tensor_name(layer: usize, expert: usize, leaf: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{leaf}")
}

/// `skill.{id}.model.layers.{layer}.mlp.experts.{expert}.{leaf}`.
pub fn expert_append_tensor_name(id: &str, layer: usize, expert: usize, leaf: &str) -> String {
    format!("skill.{id}.{}", expert_tensor_name(layer, expert, leaf))
}

/// The layers whose trunk carries expert tensors
/// (`model.layers.{l}.mlp.experts.0.gate_proj.weight`), ascending — the
/// layers an `expert_append` record may grow (the trainer's default: all).
pub fn moe_layers(tensors: &[TensorEntry]) -> Vec<usize> {
    let mut out: Vec<usize> = tensors
        .iter()
        .filter_map(|t| {
            let l = layer_of(&t.name)?;
            (t.name == expert_tensor_name(l, 0, expert_leaf::GATE)).then_some(l)
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// `E0` of a genome: `genome.moe_experts`, or — on a file written before
/// the field existed — `arch.moe.num_experts` of a resonance-routed MoE
/// (the value the writer would fill). Errors: no genome, or the genome is
/// not a resonance-routed MoE (a gated MoE cannot take appended experts:
/// its gate tensor fixes `E`).
pub fn genome_moe_experts(header: &CmfHeader) -> Result<usize, CmfError> {
    let g = header
        .genome
        .as_ref()
        .ok_or_else(|| perr("expert_append: the file has no genome".into()))?;
    let moe = header
        .arch
        .moe
        .as_ref()
        .filter(|m| m.router_resonance)
        .ok_or_else(|| {
            perr(format!(
                "genome '{}': expert_append needs a resonance-routed MoE trunk \
                 (arch.moe.router_resonance); a gated MoE fixes its expert count in the gate",
                g.id
            ))
        })?;
    Ok(g.moe_experts.unwrap_or(moe.num_experts))
}

/// Rank of the trunk's expert 0 descriptor subspace in `layer`
/// (`experts.0.desc.u` elements / `hidden_size`), 0 when the trunk has no
/// `desc.u` there. Errors: `layer` is not an MoE layer of the trunk, or it
/// has no per-expert descriptors (`experts.0.desc.mu`) — the legacy
/// stacked `mlp.desc.*` form cannot be grown per expert.
pub fn trunk_expert_rank(
    header: &CmfHeader,
    tensors: &[TensorEntry],
    layer: usize,
) -> Result<usize, CmfError> {
    let by_name: HashMap<&str, &TensorEntry> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    trunk_expert_rank_in(header, &by_name, layer)
}

fn trunk_expert_rank_in(
    header: &CmfHeader,
    by_name: &HashMap<&str, &TensorEntry>,
    layer: usize,
) -> Result<usize, CmfError> {
    let trunk = |leaf: &str| {
        by_name
            .get(expert_tensor_name(layer, 0, leaf).as_str())
            .copied()
            .filter(|t| is_trunk_tensor(&t.name))
    };
    if trunk(expert_leaf::GATE).is_none() {
        return Err(perr(format!(
            "layer {layer} is not an MoE layer of the trunk (no '{}')",
            expert_tensor_name(layer, 0, expert_leaf::GATE)
        )));
    }
    if trunk(expert_leaf::MU).is_none() {
        return Err(perr(format!(
            "layer {layer}: the trunk has no per-expert descriptors ('{}'); an \
             expert_append record needs the per-expert form, not the stacked mlp.desc.*",
            expert_tensor_name(layer, 0, expert_leaf::MU)
        )));
    }
    let Some(u) = trunk(expert_leaf::U) else {
        return Ok(0);
    };
    let hidden = header.arch.hidden_size.max(1);
    let n = u.n_elems();
    if n == 0 || n % hidden != 0 {
        return Err(perr(format!(
            "'{}': {n} elements is not a multiple of hidden {hidden}",
            u.name
        )));
    }
    Ok(n / hidden)
}

/// Grown experts of the `expert_append` records before `at` that grow
/// `layer` (every status), and the id of the first retired one.
fn expert_chain_before(header: &CmfHeader, at: usize, layer: usize) -> (usize, Option<&str>) {
    let mut n = 0usize;
    let mut retired = None;
    for s in header.skills.iter().take(at) {
        if s.kind.as_deref() != Some(skill_kind::EXPERT_APPEND) || !s.layers.contains(&layer) {
            continue;
        }
        n += s.experts.as_ref().map(|e| e.count).unwrap_or(0);
        if retired.is_none() && s.status.as_deref() == Some("retired") {
            retired = Some(s.id.as_str());
        }
    }
    (n, retired)
}

/// First expert index the `expert_append` record at position `at` of
/// `header.skills` owns in `layer` — the per-layer chain rule:
/// `E0 + Σ count` of the expert_append records BEFORE `at` that grow
/// `layer`. Its experts are `base + k`, `k in 0..count`; a layer the record
/// does not grow keeps its experts. `at == header.skills.len()` plans a
/// record about to be appended.
///
/// Indices are fixed when a record is written and never move: a record's
/// status (quarantine / active / stale_regate) does not change the chain.
/// Retiring is allowed only from the TAIL of a layer's chain — a live
/// record behind a retired one in the same layer is refused (here and by
/// `open()`), so a retired record cannot shift the indices of a later one;
/// once every later record of the layer is retired, the earlier one may be.
pub fn expert_append_base(header: &CmfHeader, at: usize, layer: usize) -> Result<usize, CmfError> {
    let e0 = genome_moe_experts(header)?;
    let (n, retired) = expert_chain_before(header, at, layer);
    if let Some(r) = retired {
        return Err(perr(format!(
            "expert_append chain at layer {layer} is broken: record '{r}' before position {at} \
             is retired — retire records from the tail of the chain (the last appended first)"
        )));
    }
    Ok(e0 + n)
}

/// One tensor an `expert_append` record carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertTensorSpec {
    /// Full directory name `skill.{id}.model.layers.{layer}.mlp.experts.{expert}.{leaf}`.
    pub name: String,
    pub shape: Vec<usize>,
    pub layer: usize,
    /// Absolute expert index in the layer (`base + k`).
    pub expert: usize,
    /// One of [`expert_leaf`].
    pub leaf: &'static str,
}

/// Every tensor (name and shape) the `expert_append` record `record` must
/// carry when it sits at position `at` of `header.skills` — exactly these,
/// no others: per grown layer `l` and `k in 0..count`, expert
/// `e = expert_append_base(l) + k`: `gate_proj.weight`, `up_proj.weight`,
/// `down_proj.weight` shaped like the trunk's expert 0 of `l`, `desc.mu
/// [hidden]`, `desc.u` shaped like the trunk's expert 0 `desc.u` (absent
/// when `rank == 0`), `desc.bias [1]` and `desc.shell [1]` (both f32).
///
/// Errors name the first rule broken: no genome / not a resonance MoE, no
/// `experts`, `count == 0`, empty or duplicate `layers`, a layer that is
/// not MoE or has no per-expert descriptors, `rank` != the trunk's, a
/// broken chain. The trainer builds its `TensorSpec`s from this plan with
/// `at = header.skills.len()`; the loader recovers the names of a mounted
/// record with its position.
pub fn expert_append_layout(
    header: &CmfHeader,
    tensors: &[TensorEntry],
    at: usize,
    record: &SkillRecord,
) -> Result<Vec<ExpertTensorSpec>, CmfError> {
    let by_name: HashMap<&str, &TensorEntry> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    expert_append_layout_in(header, &by_name, at, record, false)
}

/// [`expert_append_layout`]; `retired_self` = the record is retired, so a
/// retired predecessor does not break its chain (its indices were fixed
/// when every predecessor was live).
fn expert_append_layout_in(
    header: &CmfHeader,
    by_name: &HashMap<&str, &TensorEntry>,
    at: usize,
    record: &SkillRecord,
    retired_self: bool,
) -> Result<Vec<ExpertTensorSpec>, CmfError> {
    let id = &record.id;
    let e0 = genome_moe_experts(header).map_err(|e| perr(format!("skill '{id}': {e}")))?;
    let ea = record
        .experts
        .as_ref()
        .ok_or_else(|| perr(format!("skill '{id}': expert_append record has no `experts`")))?;
    if ea.count == 0 {
        return Err(perr(format!(
            "skill '{id}': experts.count must be ≥ 1"
        )));
    }
    if !(ea.shell_quantile.is_finite() && (0.0..=1.0).contains(&ea.shell_quantile)) {
        return Err(perr(format!(
            "skill '{id}': experts.shell_quantile {} must be finite and in [0, 1]",
            ea.shell_quantile
        )));
    }
    if record.layers.is_empty() {
        return Err(perr(format!(
            "skill '{id}': expert_append record grows no layers (`layers` is empty)"
        )));
    }
    let distinct: BTreeSet<usize> = record.layers.iter().copied().collect();
    if distinct.len() != record.layers.len() {
        return Err(perr(format!(
            "skill '{id}': duplicate layer in {:?}",
            record.layers
        )));
    }
    let hidden = header.arch.hidden_size;
    let mut out = Vec::with_capacity(distinct.len() * ea.count * expert_leaf::ALL.len());
    for &l in &distinct {
        if l >= header.arch.num_layers {
            return Err(perr(format!(
                "skill '{id}': layer {l} of {} layers",
                header.arch.num_layers
            )));
        }
        let rank = trunk_expert_rank_in(header, by_name, l)
            .map_err(|e| perr(format!("skill '{id}': {e}")))?;
        if rank != ea.rank {
            return Err(perr(format!(
                "skill '{id}': experts.rank {} != the trunk expert 0's desc.u rank {rank} at \
                 layer {l}",
                ea.rank
            )));
        }
        let (n_before, retired) = expert_chain_before(header, at, l);
        if let (Some(r), false) = (retired, retired_self) {
            return Err(perr(format!(
                "skill '{id}': expert_append chain at layer {l} is broken — the earlier record \
                 '{r}' is retired while this one is live; retire records from the tail of the \
                 chain (the last appended first)"
            )));
        }
        let base = e0 + n_before;
        let trunk_shape = |leaf: &str| -> Vec<usize> {
            by_name
                .get(expert_tensor_name(l, 0, leaf).as_str())
                .map(|t| t.shape.clone())
                .unwrap_or_default()
        };
        for leaf in [expert_leaf::GATE, expert_leaf::UP, expert_leaf::DOWN] {
            if trunk_shape(leaf).is_empty() {
                return Err(perr(format!(
                    "skill '{id}': the trunk has no '{}'",
                    expert_tensor_name(l, 0, leaf)
                )));
            }
        }
        for k in 0..ea.count {
            let e = base + k;
            let mut push = |leaf: &'static str, shape: Vec<usize>| {
                out.push(ExpertTensorSpec {
                    name: expert_append_tensor_name(id, l, e, leaf),
                    shape,
                    layer: l,
                    expert: e,
                    leaf,
                })
            };
            push(expert_leaf::GATE, trunk_shape(expert_leaf::GATE));
            push(expert_leaf::UP, trunk_shape(expert_leaf::UP));
            push(expert_leaf::DOWN, trunk_shape(expert_leaf::DOWN));
            push(expert_leaf::MU, vec![hidden]);
            if rank > 0 {
                push(expert_leaf::U, trunk_shape(expert_leaf::U));
            }
            push(expert_leaf::BIAS, vec![1]);
            push(expert_leaf::SHELL, vec![1]);
        }
    }
    Ok(out)
}

/// The state effect of an `expert_append` record: the grown experts are
/// part of the organism from the first grown layer on and are never
/// switched, so `sequence_start` (the least permissive) and no state added.
pub fn expert_append_state_effect(layers: &[usize]) -> StateEffect {
    StateEffect {
        first_affected_layer: layers.iter().copied().min().unwrap_or(0),
        switch: state_switch::SEQUENCE_START.into(),
        state_bytes_added: 0,
    }
}

/// First f32 (LE) of a payload.
pub fn f32_le_head(bytes: &[u8]) -> Option<f32> {
    bytes
        .get(..4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// The value rules of `expert_append` descriptors, which the directory
/// cannot express: every `desc.bias` is finite (the trainer freezes it —
/// 0.0 or the source's balancing bias, `origin.bias_mode`; NaN/inf would
/// buy or lose every token) and every `desc.shell` is finite (+inf would
/// mean "no shell", NaN would never fire). `f32_of` yields the first f32
/// of an entry's payload (`None` = not available → refused).
/// Called by `open()` (mmap) and by every writer (the payload it writes)
/// after [`validate_knowledge`], which fixed the dtype to f32.
pub fn validate_expert_append_values(
    header: &CmfHeader,
    tensors: &[TensorEntry],
    mut f32_of: impl FnMut(&TensorEntry) -> Option<f32>,
) -> Result<(), CmfError> {
    for s in header
        .skills
        .iter()
        .filter(|s| s.kind.as_deref() == Some(skill_kind::EXPERT_APPEND))
    {
        let prefix = format!("skill.{}.", s.id);
        for t in tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
            let bias = t.name.ends_with(".desc.bias");
            let shell = t.name.ends_with(".desc.shell");
            if !(bias || shell) {
                continue;
            }
            let v = f32_of(t).ok_or_else(|| {
                perr(format!(
                    "skill '{}': payload of '{}' is not available for the value check",
                    s.id, t.name
                ))
            })?;
            if bias && !v.is_finite() {
                return Err(perr(format!(
                    "skill '{}': '{}' is {v} — a grown expert's desc.bias must be a finite f32",
                    s.id, t.name
                )));
            }
            if shell && !v.is_finite() {
                return Err(perr(format!(
                    "skill '{}': '{}' is {v} — desc.shell must be a finite f32 threshold",
                    s.id, t.name
                )));
            }
        }
    }
    Ok(())
}

// ───────────────────────── lookup ─────────────────────────

/// The pre-release key normalisation (NFC → lowercase → every
/// non-alphanumeric char → space): it split a word at a combining stress
/// mark or a soft hyphen (`Рома́шка` → `рома шка`) and kept Latin
/// diacritics (`Pínus` ≠ `Pinus`). No table was released under it; this
/// reader refuses the id with a rebuild hint.
pub const KEY_NORM_V1: &str = "cmf-key-v1";
/// `cmf-key-v2`: [`normalize_key`] — the rule every table is built and
/// read with. A new rule is a new id, never a silent change.
pub const KEY_NORM_V2: &str = "cmf-key-v2";
/// The key normalisation a `lookup` record declares (`LookupInfo.key_norm`):
/// [`KEY_NORM_V2`], [`normalize_key`]. This reader knows exactly one.
pub const KEY_NORM: &str = KEY_NORM_V2;
/// Longest key the runtime can match, in words of the normalised form:
/// the user message is searched as n-grams of `1..=LOOKUP_MAX_NGRAM`
/// words (`cortiq_engine::lookup::MAX_NGRAM` is this value), so the
/// builder drops a longer key as unreachable instead of storing it.
pub const LOOKUP_MAX_NGRAM: usize = 4;

/// Parameters of a `kind = "lookup"` record (`SkillRecord.lookup`, spec
/// §9.5.2): `entries` cards, each with one text slot per language of
/// `langs` (slot `entry · L + lang`), reached through `keys` normalised,
/// hashed keys. The table itself is the four tensors of [`lookup_leaf`];
/// a lookup record has no `overrides`, no `experts`, no `layers` and the
/// transparent [`lookup_state_effect`] — it never touches the network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LookupInfo {
    /// `E ≥ 1` cards.
    pub entries: usize,
    /// `K ≥ 1` keys; several keys may name one entry (synonyms, languages).
    pub keys: usize,
    /// [`KEY_NORM`] — how a key is normalised before it is hashed.
    pub key_norm: String,
    /// `L ≥ 1` languages in slot order, unique, e.g. `["ru", "en"]`.
    pub langs: Vec<String>,
    /// Names of the fields the cards carry (informational; unique).
    #[serde(default)]
    pub fields: Vec<String>,
    /// How a request reaches the record ([`lookup_policy`]):
    /// `router_and_key` — the φ router sends the request, the key is then
    /// looked up in it (also when the field is absent); `key_first` — a
    /// STRONG key in the message (≥ 2 words: an exact or stem n-gram, a
    /// Latin binomial) sends the request to the record even when the
    /// router picked the backbone, a one-word key still needs the router.
    /// Additive: files written before 26.09.2026 carry no field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
}

/// Values of `LookupInfo.policy` (spec §9.5.2).
pub mod lookup_policy {
    /// The router decides; the table is consulted for what it sent.
    pub const ROUTER_AND_KEY: &str = "router_and_key";
    /// A strong key in the message takes the request past a backbone
    /// decision of the router.
    pub const KEY_FIRST: &str = "key_first";
    /// Every known value.
    pub const ALL: &[&str] = &[ROUTER_AND_KEY, KEY_FIRST];
    /// The policy of a record without the field.
    pub const DEFAULT: &str = ROUTER_AND_KEY;
}

/// `(skill id, value)` of every lookup record whose `lookup.policy` this
/// reader does not know ([`lookup_policy::ALL`]). `open()` accepts such a
/// file and warns; the runtime reads the record as
/// [`lookup_policy::DEFAULT`] — the conservative reading, the one a reader
/// from before the field existed applies (review KF-6).
pub fn unknown_lookup_policies(header: &CmfHeader) -> Vec<(String, String)> {
    header
        .skills
        .iter()
        .filter(|s| s.kind.as_deref() == Some(skill_kind::LOOKUP))
        .filter_map(|s| {
            let p = s.lookup.as_ref()?.policy.as_ref()?;
            (!lookup_policy::ALL.contains(&p.as_str())).then(|| (s.id.clone(), p.clone()))
        })
        .collect()
}

/// The writers' rule: every lookup record's `lookup.policy`, when present,
/// is one this writer knows — `append_skill`, `update_header_append` and
/// the full writers refuse anything else before a byte is written
/// (`open()` only warns, [`unknown_lookup_policies`]).
pub fn check_lookup_policies(header: &CmfHeader) -> Result<(), CmfError> {
    match unknown_lookup_policies(header).first() {
        None => Ok(()),
        Some((id, p)) => Err(perr(format!(
            "skill '{id}': lookup.policy '{p}' (expected {})",
            lookup_policy::ALL.join(" | ")
        ))),
    }
}

impl LookupInfo {
    /// The declared policy, [`lookup_policy::DEFAULT`] when absent.
    pub fn policy_label(&self) -> &str {
        self.policy.as_deref().unwrap_or(lookup_policy::DEFAULT)
    }

    /// `E · L` — slots of the text blob (`None` on overflow).
    pub fn slots(&self) -> Option<usize> {
        self.entries.checked_mul(self.langs.len())
    }

    /// Slot index of `entry` in language `lang` (`None` when out of range).
    pub fn slot(&self, entry: usize, lang: usize) -> Option<usize> {
        (entry < self.entries && lang < self.langs.len()).then(|| entry * self.langs.len() + lang)
    }
}

/// Tensor leaves of a lookup record (`skill.{id}.` + leaf).
pub mod lookup_leaf {
    /// `[K]` u64 — hash64 of every normalised key, sorted ascending, unique
    /// (the runtime binary-searches it).
    pub const KEYS_HASH: &str = "lookup.keys.hash";
    /// `[K]` u32 — the entry each key names (`< E`), parallel to `keys.hash`.
    pub const KEYS_ENTRY: &str = "lookup.keys.entry";
    /// `[E·L + 1]` u64 — byte offsets into `text`: slot `s` is
    /// `text[off[s]..off[s+1]]`, monotone, all within the blob.
    pub const ENTRIES_OFF: &str = "lookup.entries.off";
    /// `[N]` u8 — UTF-8 blob; every slot is a JSON object
    /// `{"card": "...", "fields": {field: text}}`.
    pub const TEXT: &str = "lookup.text";
    /// Every leaf a lookup record carries — exactly these.
    pub const ALL: &[&str] = &[KEYS_HASH, KEYS_ENTRY, ENTRIES_OFF, TEXT];
}

/// `skill.{id}.{leaf}` for one of [`lookup_leaf`].
pub fn lookup_tensor_name(id: &str, leaf: &str) -> String {
    format!("skill.{id}.{leaf}")
}

/// `cmf-key-v2` ([`KEY_NORM`]): NFC → every combining mark and every
/// invisible format character is dropped WITHOUT ending the word (a
/// stress mark over a Cyrillic vowel has no precomposed form, so NFC
/// leaves U+0301 standing: `Рома́шка` → `ромашка`; soft hyphen,
/// zero-width space / joiner / non-joiner, word joiner, BOM: `ро­машка`
/// → `ромашка`) → a precomposed LATIN letter loses its diacritics
/// (`Pínus` → `pinus`, `école` → `ecole`; a Cyrillic letter keeps its
/// mark — `й`, `ё`, `ў` are letters, not accents) → lowercase → every
/// other char that is not alphanumeric (Unicode) becomes a space → runs
/// of spaces collapse → trim. Shared by the builder (keys) and the
/// runtime (n-grams of the user message), so
/// `normalize_key("Пихта бальзамическая (Abies balsamea)")` ==
/// `"пихта бальзамическая abies balsamea"`. Idempotent.
pub fn normalize_key(s: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    use unicode_normalization::char::{decompose_canonical, is_combining_mark};
    /// Invisible format characters that sit inside a word.
    fn is_format(c: char) -> bool {
        matches!(
            c,
            '\u{00AD}' | '\u{200B}'..='\u{200F}' | '\u{2060}'..='\u{2064}' | '\u{FEFF}'
        )
    }
    /// Latin-1 Supplement, Latin Extended-A/B, Latin Extended Additional:
    /// where the precomposed Latin letters with diacritics live.
    fn is_latin_precomposed(c: char) -> bool {
        matches!(c, '\u{00C0}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}')
    }
    fn emit(out: &mut String, gap: &mut bool, c: char) {
        for l in c.to_lowercase() {
            if is_combining_mark(l) {
                continue;
            }
            if l.is_alphanumeric() {
                if *gap && !out.is_empty() {
                    out.push(' ');
                }
                *gap = false;
                out.push(l);
            } else {
                *gap = true;
            }
        }
    }
    let mut out = String::with_capacity(s.len());
    let mut gap = false;
    for c in s.nfc() {
        if is_combining_mark(c) || is_format(c) {
            continue;
        }
        if is_latin_precomposed(c) {
            decompose_canonical(c, |d| {
                if !is_combining_mark(d) {
                    emit(&mut out, &mut gap, d);
                }
            });
        } else {
            emit(&mut out, &mut gap, c);
        }
    }
    out
}

/// hash64 of the UTF-8 bytes of an ALREADY normalised key (the runtime
/// normalises the whole message once and hashes its n-grams).
pub fn normalized_key_hash(norm: &str) -> u64 {
    hash64(norm.as_bytes())
}

/// hash64 of [`normalize_key`]`(key)` — the value stored in `keys.hash`.
pub fn key_hash(key: &str) -> u64 {
    normalized_key_hash(&normalize_key(key))
}

/// The state effect of a lookup record: it never touches the network, so
/// `first_affected_layer 0`, `transparent`, no state added — and the
/// validator requires exactly this.
pub fn lookup_state_effect() -> StateEffect {
    StateEffect {
        first_affected_layer: 0,
        switch: state_switch::TRANSPARENT.into(),
        state_bytes_added: 0,
    }
}

/// Little-endian `u64` words of a raw payload (`u64` tensors).
pub fn read_u64_le(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Little-endian `u32` words of a raw payload (`u32` tensors).
pub fn read_u32_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// The four tensors of the lookup record `id` from hashed keys and slot
/// texts: `keys` = `(key_hash, entry)` in any order, `slots[entry · L +
/// lang]` = the JSON object text of that card in that language (`E · L`
/// of them, `L = info.langs.len()`). Sorts the keys by hash and refuses
/// a duplicate hash (two keys that normalise alike — the builder drops
/// and reports them BEFORE calling this), an entry `≥ E`, a key count or
/// slot count that disagrees with `info`, a slot that is not a JSON
/// object. What comes back passes the record validation by construction.
pub fn lookup_tensors(
    id: &str,
    info: &LookupInfo,
    keys: &[(u64, u32)],
    slots: &[&str],
) -> Result<Vec<TensorSpec>, CmfError> {
    let n_slots = info
        .slots()
        .ok_or_else(|| perr(format!("lookup '{id}': entries × langs overflows")))?;
    if info.entries == 0 || info.langs.is_empty() {
        return Err(perr(format!(
            "lookup '{id}': entries {} and langs {:?} must both be non-empty",
            info.entries, info.langs
        )));
    }
    if keys.is_empty() || keys.len() != info.keys {
        return Err(perr(format!(
            "lookup '{id}': {} keys given, lookup.keys says {} (≥ 1)",
            keys.len(),
            info.keys
        )));
    }
    if slots.len() != n_slots {
        return Err(perr(format!(
            "lookup '{id}': {} slots given, entries {} × langs {} = {n_slots}",
            slots.len(),
            info.entries,
            info.langs.len()
        )));
    }
    let mut sorted: Vec<(u64, u32)> = keys.to_vec();
    sorted.sort_unstable();
    for w in sorted.windows(2) {
        if w[0].0 == w[1].0 {
            return Err(perr(format!(
                "lookup '{id}': duplicate key hash {} (entries {} and {})",
                hex64(w[0].0),
                w[0].1,
                w[1].1
            )));
        }
    }
    for &(h, e) in &sorted {
        if e as usize >= info.entries {
            return Err(perr(format!(
                "lookup '{id}': key {} points at entry {e} of {}",
                hex64(h),
                info.entries
            )));
        }
    }
    let mut text = Vec::new();
    let mut off = Vec::with_capacity(n_slots + 1);
    off.push(0u64);
    for (s, slot) in slots.iter().enumerate() {
        let v: serde_json::Value = serde_json::from_str(slot)
            .map_err(|e| perr(format!("lookup '{id}': slot {s} is not JSON: {e}")))?;
        if !v.is_object() {
            return Err(perr(format!(
                "lookup '{id}': slot {s} is not a JSON object"
            )));
        }
        text.extend_from_slice(slot.as_bytes());
        off.push(text.len() as u64);
    }
    let u64s = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let hashes: Vec<u64> = sorted.iter().map(|k| k.0).collect();
    let entries: Vec<u8> = sorted.iter().flat_map(|k| k.1.to_le_bytes()).collect();
    Ok(vec![
        TensorSpec {
            name: lookup_tensor_name(id, lookup_leaf::KEYS_HASH),
            dtype: TensorDtype::U64,
            shape: vec![hashes.len()],
            data: u64s(&hashes),
        },
        TensorSpec {
            name: lookup_tensor_name(id, lookup_leaf::KEYS_ENTRY),
            dtype: TensorDtype::U32,
            shape: vec![sorted.len()],
            data: entries,
        },
        TensorSpec {
            name: lookup_tensor_name(id, lookup_leaf::ENTRIES_OFF),
            dtype: TensorDtype::U64,
            shape: vec![off.len()],
            data: u64s(&off),
        },
        TensorSpec {
            name: lookup_tensor_name(id, lookup_leaf::TEXT),
            dtype: TensorDtype::U8,
            shape: vec![text.len()],
            data: text,
        },
    ])
}

/// The value rules of lookup tables, which the directory cannot express:
/// `keys.hash` strictly ascending (sorted, unique — the runtime
/// binary-searches it), every `keys.entry < E`, `entries.off` monotone
/// and within the blob, the blob valid UTF-8 with every slot boundary on
/// a character boundary, every slot shaped as a JSON object (first
/// non-blank byte `{`, last `}`). With `deep` every slot is also parsed
/// as JSON: what a WRITER does with the payload it is about to commit;
/// `open()` passes `false` — the runtime reads a slot lazily and reports
/// a slot that does not parse at that request, so parsing every card of
/// a table of hundreds of MB on every open of the file would only cost
/// time. `bytes_of` yields an entry's payload (`None` = not available →
/// refused). Called after [`validate_knowledge`], which fixed dtypes and
/// shapes. One pass over the table: O(K + N).
pub fn validate_lookup_values<'a>(
    header: &CmfHeader,
    tensors: &[TensorEntry],
    deep: bool,
    mut bytes_of: impl FnMut(&TensorEntry) -> Option<Cow<'a, [u8]>>,
) -> Result<(), CmfError> {
    for s in header
        .skills
        .iter()
        .filter(|s| s.kind.as_deref() == Some(skill_kind::LOOKUP))
    {
        let id = &s.id;
        let Some(info) = s.lookup.as_ref() else {
            continue; // refused by validate_knowledge
        };
        let mut get = |leaf: &str| -> Result<Cow<'a, [u8]>, CmfError> {
            let name = lookup_tensor_name(id, leaf);
            let t = tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| perr(format!("skill '{id}': missing tensor '{name}'")))?;
            bytes_of(t).ok_or_else(|| {
                perr(format!(
                    "skill '{id}': payload of '{name}' is not available for the value check"
                ))
            })
        };
        let hashes = read_u64_le(&get(lookup_leaf::KEYS_HASH)?);
        let entries = read_u32_le(&get(lookup_leaf::KEYS_ENTRY)?);
        let offs = read_u64_le(&get(lookup_leaf::ENTRIES_OFF)?);
        let text = get(lookup_leaf::TEXT)?;
        for (i, w) in hashes.windows(2).enumerate() {
            if w[1] == w[0] {
                return Err(perr(format!(
                    "skill '{id}': duplicate key hash {} in '{}' (keys {i} and {})",
                    hex64(w[0]),
                    lookup_leaf::KEYS_HASH,
                    i + 1
                )));
            }
            if w[1] < w[0] {
                return Err(perr(format!(
                    "skill '{id}': '{}' is not sorted ascending at key {} ({} after {}) — the \
                     runtime binary-searches it",
                    lookup_leaf::KEYS_HASH,
                    i + 1,
                    hex64(w[1]),
                    hex64(w[0])
                )));
            }
        }
        for (i, &e) in entries.iter().enumerate() {
            if e as usize >= info.entries {
                return Err(perr(format!(
                    "skill '{id}': key {i} points at entry {e} of {} (lookup.entries)",
                    info.entries
                )));
            }
        }
        for (i, w) in offs.windows(2).enumerate() {
            if w[1] < w[0] {
                return Err(perr(format!(
                    "skill '{id}': '{}' is not monotone at slot {i} ({} then {})",
                    lookup_leaf::ENTRIES_OFF,
                    w[0],
                    w[1]
                )));
            }
        }
        if let Some(&last) = offs.last() {
            if last > text.len() as u64 {
                return Err(perr(format!(
                    "skill '{id}': '{}'[{}] = {last} is beyond the text blob ({} bytes)",
                    lookup_leaf::ENTRIES_OFF,
                    offs.len() - 1,
                    text.len()
                )));
            }
        }
        let l = info.langs.len().max(1);
        let where_ = |slot: usize| {
            format!(
                "slot {slot} (entry {}, lang '{}')",
                slot / l,
                info.langs.get(slot % l).map(String::as_str).unwrap_or("?")
            )
        };
        // One UTF-8 pass over the whole blob; a failure is reported for
        // the slot that holds the offending byte.
        let text_str = std::str::from_utf8(&text).map_err(|e| {
            let at = e.valid_up_to() as u64;
            let slot = offs.partition_point(|&o| o <= at).saturating_sub(1);
            perr(format!("skill '{id}': {} is not UTF-8: {e}", where_(slot)))
        })?;
        for (slot, w) in offs.windows(2).enumerate() {
            let (a, b) = (w[0] as usize, w[1] as usize);
            if !text_str.is_char_boundary(a) || !text_str.is_char_boundary(b) {
                return Err(perr(format!(
                    "skill '{id}': {} does not start and end on a UTF-8 character boundary \
                     ({a}..{b})",
                    where_(slot)
                )));
            }
            let txt = &text_str[a..b];
            let shaped = txt.trim_start().as_bytes().first() == Some(&b'{')
                && txt.trim_end().as_bytes().last() == Some(&b'}');
            if !shaped {
                return Err(perr(format!(
                    "skill '{id}': {} is not a JSON object",
                    where_(slot)
                )));
            }
            if deep {
                let v: serde_json::Value = serde_json::from_str(txt).map_err(|e| {
                    perr(format!(
                        "skill '{id}': {} is not a JSON object: {e}",
                        where_(slot)
                    ))
                })?;
                if !v.is_object() {
                    return Err(perr(format!(
                        "skill '{id}': {} is not a JSON object",
                        where_(slot)
                    )));
                }
            }
        }
    }
    Ok(())
}

impl SkillRecord {
    /// A v2 record (`kind` set) — validated by the SKILLS_V2 rules.
    pub fn is_v2(&self) -> bool {
        self.kind.is_some()
    }

    /// May the router-v2 policy auto-route to this skill? Requires a v2
    /// record with a selection descriptor, `status == "active"` and a
    /// measured gate (`gate.status == "measured"`).
    pub fn is_auto_routable(&self) -> bool {
        self.is_v2()
            && self.selection.is_some()
            && self.status.as_deref() == Some("active")
            && self
                .gate
                .as_ref()
                .and_then(|g| g.get("status"))
                .and_then(|s| s.as_str())
                == Some("measured")
    }

    fn has_v2_fields(&self) -> bool {
        !self.overrides.is_empty()
            || self.bound.is_some()
            || self.state_effect.is_some()
            || self.status.is_some()
            || self.gate.is_some()
            || self.prompt_contract.is_some()
            || self.origin.is_some()
            || self.experts.is_some()
            || self.lookup.is_some()
    }
}

// ───────────────────────── router policy v2 ─────────────────────────

/// `header.router` (bit `ROUTER_V2`): the per-request backbone-gated
/// policy. The backbone is the default; a skill runs only when the
/// calibrated decision says so (`cortiq-engine` `router::decide_backbone_gated`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterPolicy {
    /// 2.
    pub version: u32,
    /// `"backbone_gated"` (the only value).
    pub policy: String,
    /// `"request"` (the only value today).
    pub granularity: String,
    /// How φ is computed — identical in trainer and runtime.
    pub phi: PhiSpec,
    /// Descriptor of the BACKBONE class (general prompts), same schema and
    /// metric as the skills'.
    pub base: SelectionDescriptor,
    /// A skill must beat the backbone by this much in unit error E.
    pub margin: f32,
    /// hex hash64 over the base descriptor + every calibrated skill
    /// descriptor (`cortiq-engine` `router::skills_hash`). A mismatch means
    /// the calibration is stale → the backbone runs.
    pub skills_hash: String,
    /// `{in_scope_recall, false_accept, false_accept_upper95, n_in,
    /// n_general, general_sha256, in_sha256}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<serde_json::Value>,
}

/// Canonical φ(q) of a user message q: `ids = prefix_ids ++ encode(q) ++
/// suffix_ids` (encode without BOS or special handling), run the backbone
/// through `layer`, take the hidden AFTER that layer, mean over the
/// positions of `encode(q)` only (`span_mean`), unit-normalize (`unit`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhiSpec {
    pub layer: usize,
    /// `"span_mean"`.
    pub pool: String,
    /// `"unit"`.
    pub norm: String,
    #[serde(default)]
    pub prefix_ids: Vec<u32>,
    #[serde(default)]
    pub suffix_ids: Vec<u32>,
}

// ───────────────────────── trunk hash ─────────────────────────

/// Is this directory entry part of the trunk (not a skill or router
/// tensor)?
pub fn is_trunk_tensor(name: &str) -> bool {
    !(name.starts_with("skill.") || name.starts_with("route."))
}

/// hash64 of the genome's trunk. Input, in order:
///
/// 1. `b"cmf-trunk-v1\0"`;
/// 2. every directory entry outside `skill.` / `route.`, sorted by name
///    (bytewise): `name 0x00 dtype-u8 0x00 ndim-u8 shape(u64 LE each) 0x00
///    hash(u64 LE)`;
/// 3. `0x1e` + canonical JSON of the `arch` object ([`canonical_json`]:
///    object keys sorted, no whitespace);
/// 4. `0x1e` + hash64 of the vocab section (u64 LE, 0 when absent);
/// 5. `0x1e` + canonical JSON of `tokenizer_config` (`null` when absent).
///
/// The JSON parts are taken AS WRITTEN: this function serializes the
/// records exactly as the header writer does and canonicalizes the parsed
/// result, and `open()` canonicalizes the raw header JSON of the file
/// ([`trunk_hash_json`]) — the two agree byte for byte, and an arch field
/// a newer writer added (additive evolution) is covered instead of silently
/// dropped by an older struct.
///
/// Directory-level: payload bytes enter through the entry hashes, so
/// `open()` recomputes it cheaply and `verify` (which re-hashes payloads)
/// catches a payload changed under an unchanged directory.
///
/// This form covers a file WITHOUT mask / sparse-index sections; a file
/// that carries them is hashed by [`trunk_hash_exec`] (the masks are part
/// of how the trunk executes: `run`/`serve` apply the catalog's fallback
/// mask to every request, the backbone's included).
pub fn trunk_hash(header: &CmfHeader, tensors: &[TensorEntry], vocab: Option<&[u8]>) -> u64 {
    trunk_hash_exec(header, tensors, vocab, ExecHashes::default())
}

/// hash64 of the execution-relevant sections outside the directory: the
/// DTG-MA mask catalog and the sparse index derived from it (`None` =
/// the section is absent).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecHashes {
    pub masks: Option<u64>,
    pub index: Option<u64>,
}

impl ExecHashes {
    /// From the raw section bytes (empty or absent section → `None`).
    pub fn of(masks: Option<&[u8]>, index: Option<&[u8]>) -> Self {
        let h = |b: Option<&[u8]>| b.filter(|b| !b.is_empty()).map(hash64);
        Self {
            masks: h(masks),
            index: h(index),
        }
    }

    fn is_empty(&self) -> bool {
        self.masks.is_none() && self.index.is_none()
    }
}

/// [`trunk_hash`] with the mask / sparse-index sections (NF-1): when the
/// file carries either, `0x1e b"exec" 0x00` + per section (tag u8 + hash64
/// LE) is appended to the hash input. A file with neither hashes exactly
/// as [`trunk_hash`] (so genomes born without masks keep their hash), and
/// a mask catalog added to a genome later changes the trunk hash — the
/// reader refuses it unless it is a new genome.
pub fn trunk_hash_exec(
    header: &CmfHeader,
    tensors: &[TensorEntry],
    vocab: Option<&[u8]>,
    exec: ExecHashes,
) -> u64 {
    let as_written = |bytes: Vec<u8>| -> serde_json::Value {
        serde_json::from_slice(&bytes).expect("serialized JSON parses")
    };
    let arch = as_written(serde_json::to_vec(&header.arch).expect("ModelArch serializes"));
    let tok = as_written(serde_json::to_vec(&header.tokenizer_config).expect("bundle serializes"));
    trunk_hash_json_exec(&arch, &tok, tensors, vocab, exec)
}

/// [`trunk_hash`] over the raw `arch` / `tokenizer_config` JSON values of a
/// header (`Value::Null` for an absent bundle) — the reader's form.
pub fn trunk_hash_json(
    arch: &serde_json::Value,
    tokenizer_config: &serde_json::Value,
    tensors: &[TensorEntry],
    vocab: Option<&[u8]>,
) -> u64 {
    trunk_hash_json_exec(arch, tokenizer_config, tensors, vocab, ExecHashes::default())
}

/// [`trunk_hash_json`] with the mask / sparse-index sections
/// ([`trunk_hash_exec`]).
pub fn trunk_hash_json_exec(
    arch: &serde_json::Value,
    tokenizer_config: &serde_json::Value,
    tensors: &[TensorEntry],
    vocab: Option<&[u8]>,
    exec: ExecHashes,
) -> u64 {
    let mut trunk: Vec<&TensorEntry> = tensors
        .iter()
        .filter(|t| is_trunk_tensor(&t.name))
        .collect();
    trunk.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));
    let mut buf = Vec::with_capacity(trunk.len() * 96 + 4096);
    buf.extend_from_slice(b"cmf-trunk-v1\0");
    for t in trunk {
        buf.extend_from_slice(t.name.as_bytes());
        buf.push(0);
        buf.push(t.dtype.id());
        buf.push(0);
        buf.push(t.shape.len() as u8);
        for &d in &t.shape {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        buf.push(0);
        buf.extend_from_slice(&t.hash.to_le_bytes());
    }
    buf.push(0x1e);
    buf.extend(canonical_json(arch));
    buf.push(0x1e);
    buf.extend_from_slice(&vocab.map(hash64).unwrap_or(0).to_le_bytes());
    buf.push(0x1e);
    buf.extend(canonical_json(tokenizer_config));
    if !exec.is_empty() {
        buf.push(0x1e);
        buf.extend_from_slice(b"exec\0");
        for h in [exec.masks, exec.index] {
            match h {
                Some(v) => {
                    buf.push(1);
                    buf.extend_from_slice(&v.to_le_bytes());
                }
                None => buf.push(0),
            }
        }
    }
    hash64(&buf)
}

/// Canonical JSON bytes of a value: object keys sorted bytewise at every
/// depth, no whitespace. Independent of map insertion order and of
/// serde_json's `preserve_order` feature (which the engine enables and a
/// core-only build does not).
pub fn canonical_json(v: &serde_json::Value) -> Vec<u8> {
    fn walk(v: &serde_json::Value, out: &mut Vec<u8>) {
        match v {
            serde_json::Value::Object(m) => {
                let mut keys: Vec<&String> = m.keys().collect();
                keys.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
                out.push(b'{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.extend(serde_json::to_vec(k).expect("string serializes"));
                    out.push(b':');
                    walk(&m[k], out);
                }
                out.push(b'}');
            }
            serde_json::Value::Array(a) => {
                out.push(b'[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    walk(x, out);
                }
                out.push(b']');
            }
            other => out.extend(serde_json::to_vec(other).expect("scalar serializes")),
        }
    }
    let mut out = Vec::new();
    walk(v, &mut out);
    out
}

/// `{:016x}` of a hash64.
pub fn hex64(h: u64) -> String {
    format!("{h:016x}")
}

/// Parse a hex hash64 (as written by [`hex64`]).
pub fn parse_hex64(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 16 {
        return None;
    }
    u64::from_str_radix(s, 16).ok()
}

/// Writer side of the genome contract: fill an empty `trunk_hash` (and an
/// empty master hash when `encoding == "f32"`) from the content being
/// written; refuse a non-empty one that disagrees — a trunk that changed
/// must become a new genome (or a `requant` with a rewritten hash), never
/// the old one with different bytes.
pub(crate) fn seal_genome(
    header: &mut CmfHeader,
    tensors: &[TensorEntry],
    vocab: Option<&[u8]>,
    exec: ExecHashes,
) -> Result<(), CmfError> {
    let computed = trunk_hash_exec(header, tensors, vocab, exec);
    // E0 of a resonance-routed MoE genome, filled once (the arch is hashed;
    // the reader refuses a disagreeing value).
    let e0 = header
        .arch
        .moe
        .as_ref()
        .filter(|m| m.router_resonance)
        .map(|m| m.num_experts);
    let Some(g) = header.genome.as_mut() else {
        return Ok(());
    };
    if g.moe_experts.is_none() {
        g.moe_experts = e0;
    }
    if g.trunk_hash.is_empty() {
        g.trunk_hash = hex64(computed);
    } else {
        let stored = parse_hex64(&g.trunk_hash).ok_or_else(|| {
            CmfError::Parse(format!(
                "genome.trunk_hash '{}' is not a hex hash64",
                g.trunk_hash
            ))
        })?;
        if stored != computed {
            return Err(CmfError::Parse(format!(
                "refusing to write genome '{}': genome.trunk_hash {} != trunk content {} — \
                 the trunk changed without a new genome (a new id/generation with a lineage \
                 event, or a requant that rewrites trunk_hash/encoding)",
                g.id,
                g.trunk_hash,
                hex64(computed)
            )));
        }
    }
    if g.master_trunk_hash.is_empty() && g.encoding == "f32" {
        g.master_trunk_hash = g.trunk_hash.clone();
    }
    Ok(())
}

// ───────────────────────── validation ─────────────────────────

/// Knowledge bits (all three are derived from content).
pub const KNOWLEDGE_BITS: u32 = features::GENOME | features::SKILLS_V2 | features::ROUTER_V2;

/// The knowledge bits this header's content requires.
pub fn knowledge_bits(header: &CmfHeader) -> u32 {
    let mut bits = 0;
    if header.genome.is_some() {
        bits |= features::GENOME;
    }
    if header.skills.iter().any(|s| s.kind.is_some()) {
        bits |= features::SKILLS_V2;
    }
    if header.router.is_some() {
        bits |= features::ROUTER_V2;
    }
    bits
}

fn perr(msg: String) -> CmfError {
    CmfError::Parse(msg)
}

/// `model.layers.{i}.` → i.
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.layers.")?;
    let (li, _) = rest.split_once('.')?;
    li.parse().ok()
}

/// Every knowledge rule (§9.2–§9.5) over a header + directory: the bits ⇔
/// content in both directions, the genome's trunk hash, the lineage order,
/// segment bounds, every v2 skill record and the router policy. Called by
/// `open()` (with the envelope's bits) and by every writer before a byte
/// is written. Files carrying none of the new bits and no v2 content pass
/// untouched.
///
/// `raw_trunk_hash`: the trunk hash `open()` computed from the file's raw
/// header JSON ([`trunk_hash_json_exec`]); `None` = compute it from
/// `header` (writers, whose header is exactly what they write). `exec`:
/// the mask / sparse-index sections the file carries ([`ExecHashes`]).
pub fn validate_knowledge(
    header: &CmfHeader,
    tensors: &[TensorEntry],
    vocab: Option<&[u8]>,
    required_features: u32,
    data_len: Option<u64>,
    raw_trunk_hash: Option<u64>,
    exec: ExecHashes,
) -> Result<(), CmfError> {
    // 1. bit ⇔ content, both ways (like BOUNDED_STATE).
    let content = knowledge_bits(header);
    for (bit, name, what) in [
        (features::GENOME, "GENOME", "genome record"),
        (features::SKILLS_V2, "SKILLS_V2", "v2 skill record (kind)"),
        (features::ROUTER_V2, "ROUTER_V2", "router policy"),
    ] {
        let declared = content & bit != 0;
        let set = required_features & bit != 0;
        if declared != set {
            return Err(perr(format!(
                "{what} ({}) and {name} feature bit ({}) disagree",
                if declared { "present" } else { "absent" },
                if set { "set" } else { "clear" }
            )));
        }
    }
    // v2 fields on a record without `kind` would be invisible to the bit.
    for s in &header.skills {
        if s.kind.is_none() && s.has_v2_fields() {
            return Err(perr(format!(
                "skill '{}': v2 fields (overrides/bound/state_effect/status/gate/\
                 prompt_contract/origin/experts/lookup) on a record without `kind`",
                s.id
            )));
        }
    }
    let new_file = content != 0 || !header.segments.is_empty() || !header.lineage.is_empty();
    if !new_file {
        return Ok(());
    }

    // 2. genome.
    if let Some(g) = &header.genome {
        if header.shard.is_some() {
            return Err(perr(
                "a genome file cannot be sharded (the trunk hash covers one directory)".into(),
            ));
        }
        if g.id.is_empty() {
            return Err(perr("genome.id is empty".into()));
        }
        if !GENOME_STATUSES.contains(&g.status.as_str()) {
            return Err(perr(format!(
                "genome.status '{}' is not one of {GENOME_STATUSES:?}",
                g.status
            )));
        }
        if g.encoding.is_empty() {
            return Err(perr("genome.encoding is empty".into()));
        }
        // The trunk's tokenizer is part of the genome: without an embedded
        // VOCAB the hash covers "no vocab" and a reader would pick up
        // whatever tokenizer.json lies beside the file (NF-5).
        if vocab.is_none_or(|v| v.is_empty()) {
            return Err(perr(format!(
                "genome '{}': a genome file must embed its tokenizer (VOCAB section) — the \
                 trunk hash binds it; a sidecar tokenizer.json is not part of the genome",
                g.id
            )));
        }
        // O(1) attention of a genome is set by its arch (hashed); a
        // provenance hint would switch the trunk's operator without a new
        // genome (NF-6).
        if header
            .provenance
            .as_ref()
            .and_then(|p| p.get("o1_attn"))
            .is_some()
        {
            return Err(perr(format!(
                "genome '{}': provenance.o1_attn is refused on a genome file — the trunk's \
                 attention operator is part of the genome (arch), not a provenance hint",
                g.id
            )));
        }
        let stored = parse_hex64(&g.trunk_hash).ok_or_else(|| {
            perr(format!(
                "genome.trunk_hash '{}' is not a hex hash64",
                g.trunk_hash
            ))
        })?;
        let master = parse_hex64(&g.master_trunk_hash).ok_or_else(|| {
            perr(format!(
                "genome.master_trunk_hash '{}' is not a hex hash64",
                g.master_trunk_hash
            ))
        })?;
        if g.encoding == "f32" && master != stored {
            return Err(perr(format!(
                "genome.encoding is f32 but master_trunk_hash {} != trunk_hash {}",
                g.master_trunk_hash, g.trunk_hash
            )));
        }
        // E0 is the arch's expert count (the arch is hashed; the field is
        // the growth chain's origin and may not drift from it).
        if let Some(n) = g.moe_experts {
            match &header.arch.moe {
                Some(m) if m.num_experts == n => {}
                Some(m) => {
                    return Err(perr(format!(
                        "genome '{}': moe_experts {n} != arch.moe.num_experts {}",
                        g.id, m.num_experts
                    )));
                }
                None => {
                    return Err(perr(format!(
                        "genome '{}': moe_experts {n} on an arch without a MoE block",
                        g.id
                    )));
                }
            }
        }
        let computed =
            raw_trunk_hash.unwrap_or_else(|| trunk_hash_exec(header, tensors, vocab, exec));
        if computed != stored {
            return Err(perr(format!(
                "trunk_hash mismatch: the genome's bytes changed without a new genome \
                 (lineage) — genome '{}' gen {} declares {}, the directory hashes to {}",
                g.id,
                g.generation,
                g.trunk_hash,
                hex64(computed)
            )));
        }
    }

    // 3. lineage order.
    for w in header.lineage.windows(2) {
        if w[1].seq <= w[0].seq {
            return Err(perr(format!(
                "lineage seq must strictly increase ({} then {})",
                w[0].seq, w[1].seq
            )));
        }
    }
    for e in &header.lineage {
        if e.event.is_empty() {
            return Err(perr(format!("lineage event #{} has an empty name", e.seq)));
        }
    }

    // 4. segments.
    for s in &header.segments {
        if !SEGMENT_KINDS.contains(&s.kind.as_str()) {
            return Err(perr(format!(
                "segment kind '{}' is not one of {SEGMENT_KINDS:?}",
                s.kind
            )));
        }
        let over = data_len.is_some_and(|dl| s.data_end > dl);
        if s.data_start > s.data_end || over {
            return Err(perr(format!(
                "segment {}:{} [{}, {}) is outside the data section",
                s.kind, s.id, s.data_start, s.data_end
            )));
        }
    }

    // 5. skill records.
    let by_name: HashMap<&str, &TensorEntry> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut ids = BTreeSet::new();
    for (at, s) in header.skills.iter().enumerate() {
        if !ids.insert(s.id.as_str()) {
            return Err(perr(format!("duplicate skill id '{}'", s.id)));
        }
        if s.id == BACKBONE_CLASS_ID {
            return Err(perr(format!(
                "skill id '{BACKBONE_CLASS_ID}' is reserved for the backbone class"
            )));
        }
        if s.kind.is_some() {
            validate_skill_v2(header, at, s, tensors, &by_name)?;
        }
    }
    // every skill tensor belongs to a record
    for t in tensors {
        if let Some(rest) = t.name.strip_prefix("skill.") {
            let owned = header.skills.iter().any(|s| {
                rest.strip_prefix(s.id.as_str())
                    .is_some_and(|r| r.starts_with('.'))
            });
            if !owned {
                return Err(perr(format!(
                    "tensor '{}' belongs to no skill record",
                    t.name
                )));
            }
        }
    }

    // 6. router policy.
    if let Some(r) = &header.router {
        validate_router(header, r)?;
    }
    Ok(())
}

fn validate_skill_v2(
    header: &CmfHeader,
    at: usize,
    s: &SkillRecord,
    tensors: &[TensorEntry],
    by_name: &HashMap<&str, &TensorEntry>,
) -> Result<(), CmfError> {
    let id = &s.id;
    if id.is_empty() || id.contains('.') {
        return Err(perr(format!(
            "skill '{id}': a v2 skill id must be non-empty and contain no '.' \
             (it prefixes its tensors: skill.{{id}}.X)"
        )));
    }
    if RESERVED_SKILL_IDS.contains(&id.as_str()) {
        return Err(perr(format!(
            "skill '{id}': the id is reserved ({RESERVED_SKILL_IDS:?} name the backbone / the \
             automatic decision in --skill, --route and route reports)"
        )));
    }
    let kind = s.kind.as_deref().unwrap_or_default();
    if skill_kind::RESERVED.contains(&kind) {
        return Err(perr(format!(
            "skill '{id}': kind '{kind}' is reserved and not implemented by this reader"
        )));
    }
    if !skill_kind::IMPLEMENTED.contains(&kind) {
        return Err(perr(format!("skill '{id}': unknown kind '{kind}'")));
    }
    if let Some(st) = &s.status {
        if !SKILL_STATUSES.contains(&st.as_str()) {
            return Err(perr(format!(
                "skill '{id}': status '{st}' is not one of {SKILL_STATUSES:?}"
            )));
        }
    }
    // Bound to exactly this genome (the f32 master's hash).
    let bound = s
        .bound
        .as_ref()
        .ok_or_else(|| perr(format!("skill '{id}': v2 record has no `bound`")))?;
    let g = header.genome.as_ref().ok_or_else(|| {
        perr(format!(
            "skill '{id}': v2 record in a file without a genome"
        ))
    })?;
    if bound.genome_id != g.id {
        return Err(perr(format!(
            "skill '{id}': bound.genome_id '{}' != genome.id '{}'",
            bound.genome_id, g.id
        )));
    }
    if bound.generation != g.generation {
        return Err(perr(format!(
            "skill '{id}': bound.generation {} != genome.generation {}",
            bound.generation, g.generation
        )));
    }
    let bm = parse_hex64(&bound.master_trunk_hash);
    if bm.is_none() || bm != parse_hex64(&g.master_trunk_hash) {
        return Err(perr(format!(
            "skill '{id}': bound.master_trunk_hash {} != genome.master_trunk_hash {}",
            bound.master_trunk_hash, g.master_trunk_hash
        )));
    }
    match kind {
        skill_kind::EXPERT_APPEND => validate_expert_append(header, at, s, tensors, by_name),
        skill_kind::LOOKUP => validate_lookup(s, tensors, by_name),
        _ => validate_ffn_replace(header, s, tensors, by_name),
    }
}

/// `lookup` (spec §9.5.2): `lookup` present and consistent, exactly the
/// four tensors of [`lookup_leaf`] with their dtypes and shapes, no
/// overrides / experts / layers, the transparent state effect. Values
/// (sorted unique hashes, entries in range, monotone offsets, JSON
/// slots) are checked by [`validate_lookup_values`] where the payload is
/// at hand.
fn validate_lookup(
    s: &SkillRecord,
    tensors: &[TensorEntry],
    by_name: &HashMap<&str, &TensorEntry>,
) -> Result<(), CmfError> {
    let id = &s.id;
    let info = s
        .lookup
        .as_ref()
        .ok_or_else(|| perr(format!("skill '{id}': lookup record has no `lookup`")))?;
    if s.experts.is_some() {
        return Err(perr(format!(
            "skill '{id}': `experts` belongs to an expert_append record, not to lookup"
        )));
    }
    if !s.overrides.is_empty() {
        return Err(perr(format!(
            "skill '{id}': a lookup record replaces nothing — `overrides` must be empty"
        )));
    }
    if !s.layers.is_empty() {
        return Err(perr(format!(
            "skill '{id}': a lookup record touches no layer — `layers` must be empty (is {:?})",
            s.layers
        )));
    }
    if info.key_norm == KEY_NORM_V1 {
        return Err(perr(format!(
            "skill '{id}': lookup.key_norm '{KEY_NORM_V1}' is the pre-release rule — rebuild \
             the table with `cortiq lookup-build` (this reader knows {KEY_NORM})"
        )));
    }
    if info.key_norm != KEY_NORM {
        return Err(perr(format!(
            "skill '{id}': lookup.key_norm '{}' (this reader knows {KEY_NORM})",
            info.key_norm
        )));
    }
    // `lookup.policy` is NOT checked here — this rule set is also what
    // `open()` applies, and a value this reader does not know (a newer
    // writer's policy) must not make the whole genome unreadable: the
    // runtime reads it as `router_and_key` and `open()` warns
    // ([`unknown_lookup_policies`]); the writers refuse it
    // ([`check_lookup_policies`]).
    if info.entries == 0 {
        return Err(perr(format!("skill '{id}': lookup.entries must be ≥ 1")));
    }
    if info.keys == 0 {
        return Err(perr(format!("skill '{id}': lookup.keys must be ≥ 1")));
    }
    if info.langs.is_empty() {
        return Err(perr(format!("skill '{id}': lookup.langs is empty")));
    }
    for (what, list) in [("langs", &info.langs), ("fields", &info.fields)] {
        let mut seen = BTreeSet::new();
        for x in list {
            if x.is_empty() {
                return Err(perr(format!(
                    "skill '{id}': lookup.{what} has an empty name"
                )));
            }
            if !seen.insert(x.as_str()) {
                return Err(perr(format!(
                    "skill '{id}': duplicate '{x}' in lookup.{what}"
                )));
            }
        }
    }
    let n_off = info
        .slots()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| perr(format!("skill '{id}': lookup.entries × langs overflows")))?;
    let prefix = format!("skill.{id}.");
    for t in tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
        if !lookup_leaf::ALL.contains(&&t.name[prefix.len()..]) {
            return Err(perr(format!(
                "skill '{id}': tensor '{}' is not part of a lookup record (expected exactly \
                 {prefix}{{{}}})",
                t.name,
                lookup_leaf::ALL.join(", ")
            )));
        }
    }
    let want = [
        (lookup_leaf::KEYS_HASH, TensorDtype::U64, Some((info.keys, "lookup.keys"))),
        (lookup_leaf::KEYS_ENTRY, TensorDtype::U32, Some((info.keys, "lookup.keys"))),
        (
            lookup_leaf::ENTRIES_OFF,
            TensorDtype::U64,
            Some((n_off, "lookup.entries × langs + 1")),
        ),
        (lookup_leaf::TEXT, TensorDtype::U8, None),
    ];
    for (leaf, dtype, len) in want {
        let name = lookup_tensor_name(id, leaf);
        let Some(t) = by_name.get(name.as_str()) else {
            return Err(perr(format!("skill '{id}': missing tensor '{name}'")));
        };
        if t.dtype != dtype {
            return Err(perr(format!(
                "skill '{id}': '{name}' must be {} (is {})",
                dtype.name(),
                t.dtype.name()
            )));
        }
        if t.shape.len() != 1 {
            return Err(perr(format!(
                "skill '{id}': '{name}' must be 1-D (shape {:?})",
                t.shape
            )));
        }
        if let Some((n, why)) = len {
            if t.shape[0] != n {
                return Err(perr(format!(
                    "skill '{id}': '{name}' shape {:?} != [{n}] ({why})",
                    t.shape
                )));
            }
        }
    }
    let se = s
        .state_effect
        .as_ref()
        .ok_or_else(|| perr(format!("skill '{id}': v2 record has no `state_effect`")))?;
    let want = lookup_state_effect();
    if *se != want {
        return Err(perr(format!(
            "skill '{id}': state_effect {{first_affected_layer {}, switch '{}', \
             state_bytes_added {}}} — a lookup record never touches the network's state: \
             first_affected_layer 0, switch '{}', state_bytes_added 0",
            se.first_affected_layer, se.switch, se.state_bytes_added, want.switch
        )));
    }
    Ok(())
}

/// `expert_append` (spec §9.5.1): exactly the tensors of
/// [`expert_append_layout`] with its shapes, f32 `desc.bias` / `desc.shell`,
/// no overrides, no per-request selection, the least permissive state
/// effect. Values (finite bias, finite shell) are checked by
/// [`validate_expert_append_values`] where the payload is at hand.
fn validate_expert_append(
    header: &CmfHeader,
    at: usize,
    s: &SkillRecord,
    tensors: &[TensorEntry],
    by_name: &HashMap<&str, &TensorEntry>,
) -> Result<(), CmfError> {
    let id = &s.id;
    if s.lookup.is_some() {
        return Err(perr(format!(
            "skill '{id}': `lookup` belongs to a lookup record, not to expert_append"
        )));
    }
    if !s.overrides.is_empty() {
        return Err(perr(format!(
            "skill '{id}': an expert_append record replaces nothing — `overrides` must be empty"
        )));
    }
    if s.selection.is_some() {
        return Err(perr(format!(
            "skill '{id}': an expert_append record carries no `selection` — its experts are \
             chosen per token by the layer's resonance router, never per request"
        )));
    }
    let retired = s.status.as_deref() == Some("retired");
    let plan = expert_append_layout_in(header, by_name, at, s, retired)?;
    let expected: HashMap<&str, &ExpertTensorSpec> =
        plan.iter().map(|p| (p.name.as_str(), p)).collect();
    let prefix = format!("skill.{id}.");
    for t in tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
        let Some(p) = expected.get(t.name.as_str()) else {
            return Err(perr(format!(
                "skill '{id}': tensor '{}' is not part of the expert_append layout (layers {:?}, \
                 count {})",
                t.name,
                s.layers,
                s.experts.as_ref().map(|e| e.count).unwrap_or(0)
            )));
        };
        if t.shape != p.shape {
            return Err(perr(format!(
                "skill '{id}': '{}' shape {:?} != {:?} (the trunk's expert 0 of layer {})",
                t.name, t.shape, p.shape, p.layer
            )));
        }
        if matches!(p.leaf, expert_leaf::BIAS | expert_leaf::SHELL) && t.dtype != TensorDtype::F32
        {
            return Err(perr(format!(
                "skill '{id}': '{}' must be f32 (is {:?})",
                t.name, t.dtype
            )));
        }
    }
    for p in &plan {
        if !by_name.contains_key(p.name.as_str()) {
            return Err(perr(format!(
                "skill '{id}': missing tensor '{}' (layer {}, expert {})",
                p.name, p.layer, p.expert
            )));
        }
    }
    let se = s
        .state_effect
        .as_ref()
        .ok_or_else(|| perr(format!("skill '{id}': v2 record has no `state_effect`")))?;
    let computed = expert_append_state_effect(&s.layers);
    if se.first_affected_layer != computed.first_affected_layer {
        return Err(perr(format!(
            "skill '{id}': state_effect.first_affected_layer {} != min(layers) {}",
            se.first_affected_layer, computed.first_affected_layer
        )));
    }
    if se.switch != computed.switch {
        return Err(perr(format!(
            "skill '{id}': state_effect.switch '{}' — grown experts are part of the organism, \
             the record's switch is '{}'",
            se.switch, computed.switch
        )));
    }
    if se.state_bytes_added != 0 {
        return Err(perr(format!(
            "skill '{id}': state_effect.state_bytes_added {} — an expert_append record adds \
             no per-sequence state",
            se.state_bytes_added
        )));
    }
    Ok(())
}

/// `ffn_replace`: tensors ⇔ overrides, same shape as the trunk entry.
fn validate_ffn_replace(
    header: &CmfHeader,
    s: &SkillRecord,
    tensors: &[TensorEntry],
    by_name: &HashMap<&str, &TensorEntry>,
) -> Result<(), CmfError> {
    let id = &s.id;
    let g = header.genome.as_ref().expect("checked by the caller");
    if s.experts.is_some() {
        return Err(perr(format!(
            "skill '{id}': `experts` belongs to an expert_append record, not to ffn_replace"
        )));
    }
    if s.lookup.is_some() {
        return Err(perr(format!(
            "skill '{id}': `lookup` belongs to a lookup record, not to ffn_replace"
        )));
    }
    let prefix = format!("skill.{id}.");
    let mut seen = BTreeSet::new();
    for o in &s.overrides {
        if !seen.insert(o.name.as_str()) {
            return Err(perr(format!(
                "skill '{id}': duplicate override '{}'",
                o.name
            )));
        }
    }
    let mut layers_named = BTreeSet::new();
    let mut n_tensors = 0usize;
    for t in tensors.iter().filter(|t| t.name.starts_with(&prefix)) {
        n_tensors += 1;
        let x = &t.name[prefix.len()..];
        let Some(o) = s.overrides.iter().find(|o| o.name == x) else {
            return Err(perr(format!(
                "skill '{id}': tensor '{}' has no override entry",
                t.name
            )));
        };
        let li =
            layer_of(x).filter(|&l| x[format!("model.layers.{l}.").len()..].starts_with("mlp."));
        let Some(li) = li else {
            return Err(perr(format!(
                "skill '{id}': ffn_replace tensor '{}' is not an FFN tensor (model.layers.N.mlp.*)",
                t.name
            )));
        };
        if li >= header.arch.num_layers {
            return Err(perr(format!(
                "skill '{id}': '{}' names layer {li} of {}",
                t.name, header.arch.num_layers
            )));
        }
        layers_named.insert(li);
        let Some(base) = by_name.get(x).copied().filter(|b| is_trunk_tensor(&b.name)) else {
            return Err(perr(format!(
                "skill '{id}': override '{x}' replaces no trunk tensor"
            )));
        };
        if base.shape != t.shape {
            return Err(perr(format!(
                "skill '{id}': '{}' shape {:?} != trunk '{x}' shape {:?}",
                t.name, t.shape, base.shape
            )));
        }
        let bh = parse_hex64(&o.base_hash).ok_or_else(|| {
            perr(format!(
                "skill '{id}': override '{x}' base_hash '{}' is not a hex hash64",
                o.base_hash
            ))
        })?;
        if g.encoding == "f32" && bh != base.hash {
            return Err(perr(format!(
                "skill '{id}': override '{x}' base_hash {} != trunk entry hash {} \
                 (the skill was trained against other trunk bytes)",
                o.base_hash,
                hex64(base.hash)
            )));
        }
    }
    for o in &s.overrides {
        if by_name
            .get(format!("{prefix}{}", o.name).as_str())
            .is_none()
        {
            return Err(perr(format!(
                "skill '{id}': override '{}' has no tensor '{prefix}{}'",
                o.name, o.name
            )));
        }
    }
    if n_tensors == 0 {
        return Err(perr(format!(
            "skill '{id}': ffn_replace record carries no tensors"
        )));
    }
    let declared: BTreeSet<usize> = s.layers.iter().copied().collect();
    if declared != layers_named || declared.len() != s.layers.len() {
        return Err(perr(format!(
            "skill '{id}': layers {:?} != layers named by its tensors {:?}",
            s.layers, layers_named
        )));
    }
    let se = s
        .state_effect
        .as_ref()
        .ok_or_else(|| perr(format!("skill '{id}': v2 record has no `state_effect`")))?;
    let first = *layers_named.iter().next().unwrap();
    if se.first_affected_layer != first {
        return Err(perr(format!(
            "skill '{id}': state_effect.first_affected_layer {} != min(layers) {first}",
            se.first_affected_layer
        )));
    }
    let rank = |sw: &str| state_switch::ALL.iter().position(|v| *v == sw);
    let Some(claimed) = rank(&se.switch) else {
        return Err(perr(format!(
            "skill '{id}': state_effect.switch '{}' is not one of {:?}",
            se.switch,
            state_switch::ALL
        )));
    };
    let computed = ffn_replace_state_effect(&header.arch, &s.layers);
    if claimed < rank(&computed.switch).unwrap() {
        return Err(perr(format!(
            "skill '{id}': state_effect.switch '{}' is more permissive than the layer schedule \
             allows ('{}')",
            se.switch, computed.switch
        )));
    }
    Ok(())
}

fn validate_router(header: &CmfHeader, r: &RouterPolicy) -> Result<(), CmfError> {
    if r.version != 2 {
        return Err(perr(format!(
            "router.version {} (this reader knows 2)",
            r.version
        )));
    }
    if r.policy != "backbone_gated" {
        return Err(perr(format!(
            "router.policy '{}' (known: backbone_gated)",
            r.policy
        )));
    }
    if r.granularity != "request" {
        return Err(perr(format!(
            "router.granularity '{}' (known: request)",
            r.granularity
        )));
    }
    if r.phi.pool != "span_mean" || r.phi.norm != "unit" {
        return Err(perr(format!(
            "router.phi pool/norm '{}'/'{}' (known: span_mean/unit)",
            r.phi.pool, r.phi.norm
        )));
    }
    if r.phi.layer >= header.arch.num_layers {
        return Err(perr(format!(
            "router.phi.layer {} of {} layers",
            r.phi.layer, header.arch.num_layers
        )));
    }
    if !(r.margin.is_finite() && r.margin >= 0.0) {
        return Err(perr(format!(
            "router.margin {} must be finite and ≥ 0",
            r.margin
        )));
    }
    if parse_hex64(&r.skills_hash).is_none() {
        return Err(perr(format!(
            "router.skills_hash '{}' is not a hex hash64",
            r.skills_hash
        )));
    }
    let check_desc = |who: &str, d: &SelectionDescriptor| -> Result<(), CmfError> {
        if d.metric != METRIC_MSE_UNIT {
            return Err(perr(format!(
                "{who}: selection.metric '{}' (router v2 requires {METRIC_MSE_UNIT})",
                d.metric
            )));
        }
        if d.phi_layer != r.phi.layer {
            return Err(perr(format!(
                "{who}: selection.phi_layer {} != router.phi.layer {}",
                d.phi_layer, r.phi.layer
            )));
        }
        Ok(())
    };
    check_desc("router.base", &r.base)?;
    for s in header.skills.iter().filter(|s| s.is_v2()) {
        if let Some(sel) = &s.selection {
            check_desc(&format!("skill '{}'", s.id), sel)?;
            if let Some(&lmin) = s.layers.iter().min() {
                if r.phi.layer >= lmin {
                    return Err(perr(format!(
                        "skill '{}': router.phi.layer {} must be < min(layers) {lmin} \
                         (φ must not depend on the active skill)",
                        s.id, r.phi.layer
                    )));
                }
            }
        }
    }
    Ok(())
}

// ───────────────────────── time ─────────────────────────

/// Current UTC time as RFC 3339 (`YYYY-MM-DDTHH:MM:SSZ`), no dependency.
pub fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (H. Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        sod / 3600,
        (sod / 60) % 60,
        sod % 60
    )
}
