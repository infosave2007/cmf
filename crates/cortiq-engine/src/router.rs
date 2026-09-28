//! Recon-argmin skill routing (spec §9, P1 signal-consistency): the
//! container's selection descriptors define per-skill affine subspaces
//! over φ(x); the winner is the skill that reconstructs φ best. No
//! trained gate — routing is a property of the skills themselves.
//!
//! The decision layer is the debugged cortiq-router recipe (the task-routing
//! service, `cortiq-bot/cortiq-router/src/router.rs`): raw squared
//! reconstruction error per skill, a temperature-calibrated softmax over
//! −error for the confidence, and a NOVELTY ENSEMBLE of three independent
//! OOD signals — the winner's error as a z-score against its own training
//! shell, the leader margin, and the calibrated confidence — thresholded by
//! θ that was set to the (1−fpr) quantile of in-scope held-out scores.
//! Files without the calibration fall back to the normalized error E with a
//! fixed threshold (the pre-calibration behaviour, unchanged).
//!
//! Router policy v2 (`header.router`, bit ROUTER_V2, spec §9.4) sits on top
//! of the same recipe: the BACKBONE is a class with its own descriptor, the
//! calibration is fitted over skills + backbone, and a skill runs only when
//! [`decide_backbone_gated`] says so — everything else, including a stale
//! calibration, runs the backbone. v2 skill records are never routed by
//! the legacy argmin ([`error_rows`] skips them).

use crate::pipeline::Pipeline;
use base64::Engine as _;
use cortiq_core::format::{CmfHeader, RoutingCalibration, SelectionDescriptor, SkillRecord};
use cortiq_core::knowledge::{BACKBONE_CLASS_ID, METRIC_MSE_UNIT, parse_hex64};
use cortiq_core::quant::{f16_to_f32, f32_to_f16};
use cortiq_core::{CmfModel, PhiSpec, RouterPolicy, hash64};

/// Ensemble weights and margin sharpness (cortiq-router constants).
pub const NOVELTY_W_ENERGY: f32 = 0.5;
pub const NOVELTY_W_MARGIN: f32 = 0.25;
pub const NOVELTY_W_CONF: f32 = 0.25;
pub const NOVELTY_MARGIN_K: f32 = 8.0;

#[derive(Debug, Clone)]
pub struct SkillRoute {
    pub id: String,
    /// Normalized reconstruction error E = ‖r − BBᵀr‖²/‖φ‖² ∈ [0, 1]; lower = closer.
    pub error: f32,
    /// Raw squared reconstruction error (the calibrated recipe's quantity).
    pub raw_error: f32,
    /// Calibrated probability (temperature softmax over −raw_error) — 0 when
    /// the file carries no calibration.
    pub probability: f32,
}

/// The full routing decision for one prompt.
#[derive(Debug, Clone)]
pub struct Routing {
    /// best-first
    pub scores: Vec<SkillRoute>,
    /// winner's calibrated confidence (0 without calibration)
    pub confidence: f32,
    /// leader margin in `1/(1+err)` units
    pub margin: f32,
    /// novelty ensemble score ∈ [0,1] (NaN without calibration)
    pub novelty: f32,
    /// OOD verdict: calibrated θ when present, else E_min > `fallback_tau`
    pub is_novel: bool,
    pub calibrated: bool,
}

impl Routing {
    pub fn winner(&self) -> Option<&SkillRoute> {
        self.scores.first()
    }
}

pub fn decode_f16(b64: &str) -> Option<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    Some(
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - mx).exp();
        s += *x;
    }
    for x in v.iter_mut() {
        *x /= s.max(1e-30);
    }
}

/// Raw squared reconstruction error of φ against a (mean, basis rows) subspace.
pub fn recon_error(phi: &[f32], mean: &[f32], basis: &[f32], rank: usize) -> f32 {
    let hidden = phi.len();
    let r: Vec<f32> = phi.iter().zip(mean).map(|(p, m)| p - m).collect();
    let rr: f32 = r.iter().map(|v| v * v).sum();
    let mut proj = 0f32;
    for k in 0..rank {
        let row = &basis[k * hidden..(k + 1) * hidden];
        let c: f32 = row.iter().zip(&r).map(|(b, v)| b * v).sum();
        proj += c * c;
    }
    (rr - proj).max(0.0)
}

/// Decision from per-skill (id, raw_error, err_mean, err_std, ‖φ‖²) rows —
/// the pure recipe, shared by `route_full` and the file-level calibration.
pub fn decide(
    rows: &[ErrorRow],
    calib: Option<&cortiq_core::format::RoutingCalibration>,
    fallback_tau: f32,
) -> Routing {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&a, &b| rows[a].1.total_cmp(&rows[b].1));
    let mut scores: Vec<SkillRoute> = idx
        .iter()
        .map(|&i| SkillRoute {
            id: rows[i].0.clone(),
            error: rows[i].1 / rows[i].4.max(1e-12),
            raw_error: rows[i].1,
            probability: 0.0,
        })
        .collect();
    if scores.is_empty() {
        return Routing {
            scores,
            confidence: 0.0,
            margin: 0.0,
            novelty: 1.0,
            is_novel: true,
            calibrated: calib.is_some(),
        };
    }
    let (Some(c), Some(&top)) = (calib, idx.first()) else {
        let e_min = scores[0].error;
        return Routing {
            scores,
            confidence: 0.0,
            margin: 0.0,
            novelty: f32::NAN,
            is_novel: e_min > fallback_tau,
            calibrated: false,
        };
    };
    // confidence: temperature softmax over −raw_error
    let mut logits: Vec<f32> = idx
        .iter()
        .map(|&i| -rows[i].1 / c.temperature.max(1e-3))
        .collect();
    softmax(&mut logits);
    for (s, p) in scores.iter_mut().zip(&logits) {
        s.probability = *p;
    }
    let confidence = logits[0];
    // margin in 1/(1+err) units
    let inv = |e: f32| 1.0 / (1.0 + e);
    let margin = if idx.len() > 1 {
        inv(rows[idx[0]].1) - inv(rows[idx[1]].1)
    } else {
        inv(rows[idx[0]].1)
    };
    // energy: winner z-score against its training shell
    let (em, es) = (
        rows[top].2.unwrap_or(0.0),
        rows[top].3.unwrap_or(1.0).max(1e-4),
    );
    let z = (rows[top].1 - em) / es;
    let novelty = NOVELTY_W_ENERGY * sigmoid(z)
        + NOVELTY_W_MARGIN / (1.0 + margin * NOVELTY_MARGIN_K)
        + NOVELTY_W_CONF * (1.0 - confidence);
    Routing {
        scores,
        confidence,
        margin,
        novelty,
        is_novel: novelty > c.novelty_theta,
        calibrated: true,
    }
}

/// Per-skill error rows for a φ (skills with malformed descriptors skipped).
pub fn error_rows(
    model: &CmfModel,
    phi_of_layer: &mut dyn FnMut(usize) -> Vec<f32>,
) -> Vec<ErrorRow> {
    let hidden = model.arch().hidden_size;
    let mut rows = Vec::new();
    for skill in &model.header.skills {
        // v2 records route only through the backbone-gated policy — never
        // through argmin-over-skills, where a single skill wins every
        // prompt.
        if skill.is_v2() {
            continue;
        }
        let Some(sel) = &skill.selection else {
            continue;
        };
        let unit = match sel.metric.as_str() {
            "mse" => false,
            "mse_unit" => true,
            m => {
                tracing::warn!("skill '{}': unknown metric '{}'", skill.id, m);
                continue;
            }
        };
        let mut phi = phi_of_layer(sel.phi_layer);
        if unit {
            let n = phi.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in phi.iter_mut() {
                *x /= n;
            }
        }
        let (Some(mean), Some(basis)) = (decode_f16(&sel.mean), decode_f16(&sel.basis)) else {
            tracing::error!("skill '{}': malformed selection payload", skill.id);
            continue;
        };
        if mean.len() != hidden || basis.len() != sel.rank * hidden || phi.len() != hidden {
            tracing::error!("skill '{}': selection dims mismatch", skill.id);
            continue;
        }
        let e = recon_error(&phi, &mean, &basis, sel.rank);
        let pp: f32 = phi.iter().map(|v| v * v).sum();
        rows.push((skill.id.clone(), e, sel.err_mean, sel.err_std, pp));
    }
    rows
}

/// Full decision for a prompt.
pub fn route_full(
    model: &CmfModel,
    pipeline: &mut Pipeline,
    ids: &[u32],
    fallback_tau: f32,
) -> Routing {
    let mut phi_cache: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut phi_of = |layer: usize| -> Vec<f32> {
        if let Some((_, p)) = phi_cache.iter().find(|(l, _)| *l == layer) {
            return p.clone();
        }
        let p = pipeline.probe_phi(ids, layer);
        phi_cache.push((layer, p.clone()));
        p
    };
    let rows = error_rows(model, &mut phi_of);
    decide(&rows, model.header.routing.as_ref(), fallback_tau)
}

/// Score every routable skill; sorted best-first (compatibility API).
pub fn route(model: &CmfModel, pipeline: &mut Pipeline, ids: &[u32]) -> Vec<SkillRoute> {
    route_full(model, pipeline, ids, 0.30).scores
}

/// Held-out in-scope φ samples of every skill (from the descriptors), as
/// (skill index, φ). Empty when no skill carries them.
pub fn holdout_phis(model: &CmfModel) -> Vec<(usize, Vec<f32>)> {
    let hidden = model.arch().hidden_size;
    let mut out = Vec::new();
    for (si, skill) in model.header.skills.iter().enumerate() {
        let Some(sel) = &skill.selection else {
            continue;
        };
        let (Some(h), Some(n)) = (sel.holdout.as_ref(), sel.holdout_n) else {
            continue;
        };
        let Some(v) = decode_f16(h) else { continue };
        if v.len() != n * hidden {
            continue;
        }
        for i in 0..n {
            out.push((si, v[i * hidden..(i + 1) * hidden].to_vec()));
        }
    }
    out
}

/// Fit the file-level calibration from the skills' held-out φ samples: the
/// temperature by NLL of the true skill under softmax(−err/T) over a
/// geometric grid, then θ as the (1−fpr) quantile of in-scope novelty
/// scores. Every skill's φ must be at the SAME phi_layer (mixed layers are
/// scored per skill; the samples of a skill are compared against every
/// descriptor's own layer only when equal — otherwise skipped).
pub fn calibrate(
    model: &CmfModel,
    target_fpr: f32,
) -> Option<cortiq_core::format::RoutingCalibration> {
    let samples = holdout_phis(model);
    if samples.is_empty() {
        return None;
    }
    // per sample: rows over all skills — φ is a per-layer quantity, so only
    // skills sharing the sample's phi_layer are comparable
    let skills = &model.header.skills;
    let mut per_sample: Vec<(Vec<ErrorRow>, usize)> = Vec::new();
    for (si, phi) in &samples {
        let layer = skills[*si]
            .selection
            .as_ref()
            .map(|s| s.phi_layer)
            .unwrap_or(0);
        let mut phi_of =
            |l: usize| -> Vec<f32> { if l == layer { phi.clone() } else { Vec::new() } };
        let rows = error_rows(model, &mut phi_of);
        let Some(pos) = rows.iter().position(|r| r.0 == skills[*si].id) else {
            continue;
        };
        per_sample.push((rows, pos));
    }
    if per_sample.is_empty() {
        return None;
    }
    // temperature: geometric grid, minimize NLL of the true skill
    let mut best_t = 1.0f32;
    let mut best_nll = f32::INFINITY;
    let mut t = 1e-3f32;
    // errors are raw squared residuals of hidden states — the scale spans
    // orders of magnitude across models, hence the wide grid
    while t <= 1e6 {
        let mut nll = 0.0f32;
        for (rows, pos) in &per_sample {
            let mut logits: Vec<f32> = rows.iter().map(|r| -r.1 / t).collect();
            softmax(&mut logits);
            nll -= logits[*pos].max(1e-9).ln();
        }
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
        t *= 1.15;
    }
    let mut cal = cortiq_core::format::RoutingCalibration {
        temperature: best_t,
        novelty_theta: 0.5,
        samples: per_sample.len(),
        target_fpr,
    };
    // θ: (1−fpr) quantile of the in-scope novelty scores
    let mut nov: Vec<f32> = per_sample
        .iter()
        .map(|(rows, _)| decide(rows, Some(&cal), 1.0).novelty)
        .filter(|v| v.is_finite())
        .collect();
    nov.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if !nov.is_empty() {
        let q = (1.0 - target_fpr).clamp(0.0, 1.0);
        let idx = (((nov.len() - 1) as f32) * q).round() as usize;
        cal.novelty_theta = (nov[idx.min(nov.len() - 1)] + 1e-4).min(0.999);
    }
    Some(cal)
}

// ───────────────────────── router policy v2 ─────────────────────────

/// One φ-error row: (class id, raw squared error, err_mean, err_std, ‖φ‖²)
/// — the tuple [`decide`] takes.
pub type ErrorRow = (String, f32, Option<f32>, Option<f32>, f32);

/// What a request runs on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteTarget {
    Backbone,
    Skill(String),
}

/// A router-v2 decision: the target, the full scored routing (backbone row
/// included, id [`BACKBONE_CLASS_ID`]) and a human-readable reason.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub target: RouteTarget,
    pub routing: Routing,
    pub reason: String,
}

impl RouteDecision {
    /// The chosen skill id, `None` for the backbone.
    pub fn skill(&self) -> Option<&str> {
        match &self.target {
            RouteTarget::Skill(s) => Some(s),
            RouteTarget::Backbone => None,
        }
    }

    /// A decision nobody computed: the caller pinned the target
    /// (`--skill <id>`, `--skill none`, `--route backbone`, or a file
    /// without routing). No scores, novelty NaN.
    pub fn forced(target: RouteTarget, reason: impl Into<String>) -> Self {
        Self {
            target,
            routing: Routing {
                scores: Vec::new(),
                confidence: 0.0,
                margin: 0.0,
                novelty: f32::NAN,
                is_novel: false,
                calibrated: false,
            },
            reason: reason.into(),
        }
    }

    /// `"backbone"` or the skill id.
    pub fn target_label(&self) -> &str {
        self.skill().unwrap_or("backbone")
    }

    /// Unit error E of the backbone class (router-v2 decisions only).
    pub fn e_base(&self) -> Option<f32> {
        self.routing
            .scores
            .iter()
            .find(|s| s.id == BACKBONE_CLASS_ID)
            .map(|s| s.error)
    }

    /// Unit error E of the nearest SKILL class (the best non-backbone
    /// row; for a legacy file the argmin winner).
    pub fn e_skill(&self) -> Option<f32> {
        self.routing
            .scores
            .iter()
            .find(|s| s.id != BACKBONE_CLASS_ID)
            .map(|s| s.error)
    }

    /// The nearest skill's id (whether or not it won).
    pub fn nearest_skill(&self) -> Option<&str> {
        self.routing
            .scores
            .iter()
            .find(|s| s.id != BACKBONE_CLASS_ID)
            .map(|s| s.id.as_str())
    }

    /// One human line: `route: <target> | novelty … | E_base … | E_skill …
    /// | <reason>` (the stderr / server-log form).
    pub fn describe(&self) -> String {
        let f = |v: Option<f32>| match v {
            Some(x) if x.is_finite() => format!("{x:.4}"),
            _ => "—".to_string(),
        };
        let nov = if self.routing.novelty.is_finite() {
            format!("{:.3}", self.routing.novelty)
        } else {
            "—".to_string()
        };
        format!(
            "route: {} | novelty {nov} | E_base {} | E_skill {} | {}",
            self.target_label(),
            f(self.e_base()),
            f(self.e_skill()),
            self.reason
        )
    }

    /// The decision as the tools and `serve` report it:
    /// `{target, novelty, e_base, e_skill, reason}` (non-finite → null).
    pub fn summary_json(&self) -> serde_json::Value {
        let num = |v: Option<f32>| -> serde_json::Value {
            match v {
                Some(x) if x.is_finite() => serde_json::json!(x),
                _ => serde_json::Value::Null,
            }
        };
        serde_json::json!({
            "target": self.target_label(),
            "novelty": num(Some(self.routing.novelty)),
            "e_base": num(self.e_base()),
            "e_skill": num(self.e_skill()),
            "reason": self.reason,
        })
    }
}

/// Knobs of [`route_request_with`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RouteOptions {
    /// Debug / gate measurement: every non-retired calibration class may
    /// win — `quarantine` (fresh bake), `stale_regate` (re-gate after a
    /// requant) and an `active` record whose gate is not measured yet —
    /// not only the auto-routable ones. Never on in production: such a
    /// skill is not allowed to auto-route.
    pub include_quarantine: bool,
}

/// May the decision PICK this skill class under `opts`? Auto-routable
/// (v2, `active`, gate `measured`), or — gate measurement only
/// ([`RouteOptions::include_quarantine`]) — any other non-retired v2
/// class. Every calibration class is SCORED either way (see
/// [`policy_rows`]); this only decides whether its win runs the skill.
pub fn is_routable(s: &SkillRecord, opts: RouteOptions) -> bool {
    s.is_auto_routable()
        || (opts.include_quarantine
            && s.is_v2()
            && s.selection.is_some()
            && s.status.as_deref() != Some("retired"))
}

/// The skills [`route_policy_with`] can pick under `opts` (calibration
/// classes that are [`is_routable`]), in id order.
pub fn routable_skills<'a>(header: &'a CmfHeader, opts: RouteOptions) -> Vec<&'a SkillRecord> {
    calibration_classes(header)
        .into_iter()
        .filter(|s| is_routable(s, opts))
        .collect()
}

/// Why no skill can win a router-v2 decision on this file under `opts`
/// (`None` = at least one can): no policy, no calibration, a stale
/// `skills_hash`, or no routable class. A gate measured while this is
/// `Some` is vacuous — every prompt runs the backbone by construction.
pub fn no_candidate_reason(header: &CmfHeader, opts: RouteOptions) -> Option<String> {
    let Some(policy) = &header.router else {
        return Some("the file declares no router policy (ROUTER_V2)".into());
    };
    if header.routing.is_none() {
        return Some("the router is not calibrated (header.routing absent)".into());
    }
    let h = skills_hash(header);
    if parse_hex64(&policy.skills_hash) != Some(h) {
        return Some(format!(
            "skills_hash mismatch (router {}, file {h:016x}): the calibration is stale",
            policy.skills_hash
        ));
    }
    if routable_skills(header, opts).is_empty() {
        let classes: Vec<String> = calibration_classes(header)
            .iter()
            .map(|s| format!("{}={}", s.id, s.status.as_deref().unwrap_or("?")))
            .collect();
        return Some(format!(
            "no routable skill class (classes {classes:?}; include_quarantine {})",
            opts.include_quarantine
        ));
    }
    None
}

// ───────────────────────── prompt contract ─────────────────────────

/// The ChatML frame Embryo skills are trained under: `<|im_start|>role\n
/// text<|im_end|>\n` per turn, the answer after `<|im_start|>assistant\n`.
pub const PROMPT_CONTRACT_CMF_IM_V1: &str = "cmf-im-v1";
const IM_USER: &str = "<|im_start|>user\n";
const IM_ASSISTANT: &str = "<|im_end|>\n<|im_start|>assistant\n";

/// One user turn rendered under cmf-im-v1 — the frame `dump-logits`,
/// `probe-utility` and the φ calibration use.
pub fn render_cmf_im_v1(user_text: &str) -> String {
    format!("{IM_USER}{user_text}{IM_ASSISTANT}")
}

/// The user text of a prompt that is EXACTLY one cmf-im-v1 user turn
/// plus the generation prompt (no system turn, no history, no template
/// markers inside the text); `None` otherwise.
pub fn cmf_im_v1_single_user_turn(prompt: &str) -> Option<&str> {
    let q = prompt.strip_prefix(IM_USER)?.strip_suffix(IM_ASSISTANT)?;
    (!q.contains("<|im_start|>") && !q.contains("<|im_end|>")).then_some(q)
}

/// How the prompt a lane will generate from is framed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptFrame {
    /// Raw completion text (no chat template).
    Raw,
    /// The cmf-im-v1 ChatML frame (a template or the ChatML fallback
    /// that renders `<|im_start|>role\n…<|im_end|>`).
    CmfImV1,
    /// Some other chat template.
    Other,
}

impl PromptFrame {
    fn label(self) -> &'static str {
        match self {
            Self::Raw => "a raw completion prompt",
            Self::CmfImV1 => "cmf-im-v1",
            Self::Other => "another chat template",
        }
    }
}

/// The frame a tokenizer's chat rendering produces: its template when it
/// carries one (cmf-im-v1 iff it spells `<|im_start|>`), else the ChatML
/// fallback — cmf-im-v1 only when the vocabulary has both markers.
pub fn chat_frame(tok: &crate::tokenizer::Tokenizer) -> PromptFrame {
    match &tok.chat_template {
        Some(t) if t.contains("<|im_start|>") && t.contains("<|im_end|>") => PromptFrame::CmfImV1,
        Some(_) => PromptFrame::Other,
        None if tok.im_start_id.is_some() && tok.im_end_id.is_some() => PromptFrame::CmfImV1,
        None => PromptFrame::Other,
    }
}

/// The chosen skill's `prompt_contract` against the frame the caller
/// generates with — the one check `run`, `explain` and `serve` share.
/// Returns the final decision and whether the caller must RENDER the
/// user text under cmf-im-v1 ([`render_cmf_im_v1`]) for the skill lane.
///
/// * backbone, a skill without a contract, or a frame that satisfies it:
///   unchanged, no rendering;
/// * a cmf-im-v1 skill on a raw prompt with `can_render`: the skill runs
///   on the rendered turn (the frame it was trained and gated under);
/// * anything else: the BACKBONE runs, reason "prompt contract … not
///   satisfied" — a skill never continues a context it was not trained on.
pub fn enforce_prompt_contract(
    header: &CmfHeader,
    decision: RouteDecision,
    frame: PromptFrame,
    can_render: bool,
) -> (RouteDecision, bool) {
    let Some(id) = decision.skill() else {
        return (decision, false);
    };
    let Some(contract) = header
        .skills
        .iter()
        .find(|s| s.id == id)
        .and_then(|s| s.prompt_contract.clone())
    else {
        return (decision, false);
    };
    let known = contract == PROMPT_CONTRACT_CMF_IM_V1;
    if known && frame == PromptFrame::CmfImV1 {
        return (decision, false);
    }
    if known && frame == PromptFrame::Raw && can_render {
        let mut d = decision;
        d.reason = format!(
            "{}; prompt rendered as {PROMPT_CONTRACT_CMF_IM_V1} (the skill's contract)",
            d.reason
        );
        return (d, true);
    }
    let reason = format!(
        "prompt contract '{contract}' of skill '{id}' not satisfied ({}): the backbone runs \
         [decision was: {}]",
        frame.label(),
        decision.reason
    );
    (
        RouteDecision {
            target: RouteTarget::Backbone,
            routing: decision.routing,
            reason,
        },
        false,
    )
}

/// Does the file declare router policy v2 (`header.router`, bit
/// ROUTER_V2)? Then only [`decide_backbone_gated`] may pick a skill.
pub fn is_router_v2(model: &CmfModel) -> bool {
    model.header.router.is_some()
}

/// The per-request decision `run`, `explain`, `serve` and the probes
/// share (spec §9.4). `backbone` must be a pipeline WITHOUT a skill
/// overlay; its sequence state is reset (φ is a fresh prefill).
///
/// * ROUTER_V2 file: `ids = phi.prefix_ids ++ encode_plain(user_text) ++
///   phi.suffix_ids` (no added-token matching inside the user text), φ =
///   [`Pipeline::probe_phi_span`] over the user-text span at `phi.layer`,
///   rows = backbone + EVERY calibration class (the set the calibration
///   was fitted on); a skill runs only when the winner is routable
///   ([`route_policy_with`]).
/// * legacy file: recon-argmin over the v1 skills on the encoded prompt
///   (unchanged), except that a NOVEL input runs the backbone — the
///   argmin winner used to run whatever the novelty verdict said.
pub fn route_request(model: &CmfModel, backbone: &mut Pipeline, user_text: &str) -> RouteDecision {
    route_request_with(model, backbone, user_text, RouteOptions::default())
}

/// [`route_request`] with [`RouteOptions`].
pub fn route_request_with(
    model: &CmfModel,
    backbone: &mut Pipeline,
    user_text: &str,
    opts: RouteOptions,
) -> RouteDecision {
    // Router v2: q is tokenized as plain text — a literal `<|im_end|>` in
    // the user's message stays bytes, as in the trainer's `Bpe::encode`
    // the descriptors and the calibration were built with (R7). Legacy
    // files keep the historical `encode` (their descriptors were fitted
    // on it).
    let q_ids = if model.header.router.is_some() {
        backbone.tokenizer.encode_plain(user_text)
    } else {
        backbone.tokenizer.encode(user_text)
    };
    route_request_ids(model, backbone, &q_ids, opts)
}

/// [`route_request_with`] on an already encoded user text.
pub fn route_request_ids(
    model: &CmfModel,
    backbone: &mut Pipeline,
    q_ids: &[u32],
    opts: RouteOptions,
) -> RouteDecision {
    match &model.header.router {
        Some(policy) => {
            let (ids, span) = phi_span_ids(&policy.phi, q_ids);
            let phi = backbone.probe_phi_span(&ids, policy.phi.layer, span);
            route_policy_with(&model.header, &phi, opts)
        }
        None => route_legacy(model, backbone, q_ids),
    }
}

/// Legacy files (no ROUTER_V2): the recon-argmin winner, unless the input
/// is novel (calibrated novelty > θ, or E_min > `CMF_OOD_TAU`, default
/// 0.30, without a calibration) — then the backbone.
fn route_legacy(model: &CmfModel, backbone: &mut Pipeline, ids: &[u32]) -> RouteDecision {
    let tau = std::env::var("CMF_OOD_TAU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.30);
    let routing = route_full(model, backbone, ids, tau);
    let Some(w) = routing.winner().cloned() else {
        return RouteDecision {
            target: RouteTarget::Backbone,
            routing,
            reason: "no routable skill in this container".into(),
        };
    };
    if routing.is_novel {
        let reason = if routing.calibrated {
            format!(
                "novel input (novelty {:.3} over θ): nearest '{}' not taken",
                routing.novelty, w.id
            )
        } else {
            format!(
                "novel input (E_min {:.4} > τ {tau}, uncalibrated file): nearest '{}' not taken",
                w.error, w.id
            )
        };
        return RouteDecision {
            target: RouteTarget::Backbone,
            routing,
            reason,
        };
    }
    let reason = format!(
        "legacy recon-argmin: '{}' (E {:.4}), in scope",
        w.id, w.error
    );
    RouteDecision {
        target: RouteTarget::Skill(w.id),
        routing,
        reason,
    }
}

/// The skill classes a router-v2 calibration covers: v2 records with a
/// selection descriptor, not retired, sorted by id. Status changes other
/// than retirement (quarantine → active, stale_regate) do not change the
/// class set, so they do not invalidate the calibration.
pub fn calibration_classes(header: &CmfHeader) -> Vec<&SkillRecord> {
    let mut v: Vec<&SkillRecord> = header
        .skills
        .iter()
        .filter(|s| s.is_v2() && s.selection.is_some() && s.status.as_deref() != Some("retired"))
        .collect();
    v.sort_by(|a, b| a.id.as_bytes().cmp(b.id.as_bytes()));
    v
}

/// hash64 binding a calibration to the descriptors it was fitted on:
/// `b"cmf-skills-v2\0"`, then the backbone descriptor (`router.base`, a
/// single 0 byte when there is no router), then every
/// [`calibration_classes`] descriptor in id order. Per descriptor:
/// `id 0 metric 0 phi_layer(u64 LE) rank(u64 LE) mean 0 basis 0` +
/// err_mean and err_std as (tag u8, f32 bits LE) + `0x1e`. The holdout
/// samples are not part of it (they feed the fit, not the decision).
/// `router.skills_hash` must equal this or the backbone runs.
pub fn skills_hash(header: &CmfHeader) -> u64 {
    fn push(buf: &mut Vec<u8>, id: &str, d: &SelectionDescriptor) {
        buf.extend_from_slice(id.as_bytes());
        buf.push(0);
        buf.extend_from_slice(d.metric.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&(d.phi_layer as u64).to_le_bytes());
        buf.extend_from_slice(&(d.rank as u64).to_le_bytes());
        buf.extend_from_slice(d.mean.as_bytes());
        buf.push(0);
        buf.extend_from_slice(d.basis.as_bytes());
        buf.push(0);
        for x in [d.err_mean, d.err_std] {
            match x {
                Some(v) => {
                    buf.push(1);
                    buf.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                None => buf.push(0),
            }
        }
        buf.push(0x1e);
    }
    let mut buf = b"cmf-skills-v2\0".to_vec();
    match &header.router {
        Some(r) => push(&mut buf, BACKBONE_CLASS_ID, &r.base),
        None => buf.push(0),
    }
    for s in calibration_classes(header) {
        push(&mut buf, &s.id, s.selection.as_ref().expect("filtered"));
    }
    hash64(&buf)
}

/// Canonical φ input for a user message (spec §9.4): `prefix_ids ++ q_ids
/// ++ suffix_ids`, and the positions of `q_ids` — the span φ is pooled
/// over. `q_ids` is `Tokenizer::encode_plain(q)` exactly (no BOS, no
/// added-token matching — the trainer's `Bpe::encode`).
pub fn phi_span_ids(policy: &PhiSpec, q_ids: &[u32]) -> (Vec<u32>, std::ops::Range<usize>) {
    let mut ids =
        Vec::with_capacity(policy.prefix_ids.len() + q_ids.len() + policy.suffix_ids.len());
    ids.extend_from_slice(&policy.prefix_ids);
    let start = ids.len();
    ids.extend_from_slice(q_ids);
    let end = ids.len();
    ids.extend_from_slice(&policy.suffix_ids);
    (ids, start..end)
}

/// `span_mean` + `unit`: mean of the per-position hidden rows
/// (`hiddens` = n × `hidden`, the residual stream AFTER `phi.layer`) over
/// `span`, unit-normalized. An empty span gives the zero vector, which the
/// decision treats as degenerate (backbone).
pub fn pool_span_unit(hiddens: &[f32], hidden: usize, span: std::ops::Range<usize>) -> Vec<f32> {
    let rows = hiddens.len() / hidden.max(1);
    let (a, b) = (span.start.min(rows), span.end.min(rows));
    let mut phi = vec![0.0f32; hidden];
    if b <= a {
        return phi;
    }
    for r in a..b {
        for (p, x) in phi.iter_mut().zip(&hiddens[r * hidden..(r + 1) * hidden]) {
            *p += x;
        }
    }
    let n = phi.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        phi.iter_mut().for_each(|x| *x /= n);
    }
    phi
}

// ───────────────────────── descriptor fitting ─────────────────────────

/// Size of the held-out tail [`fit_descriptor`] keeps out of the fit (the
/// LAST `holdout_count(n)` samples, in input order): 20 %, at least 1,
/// and never so many that fewer than 2 train samples remain once n ≥ 3
/// (the trainer's `holdout_count`).
pub fn holdout_count(n: usize) -> usize {
    (n / 5).clamp(1, n.saturating_sub(2).max(1))
}

/// f16 LE base64 of `v` — the on-disk form of the descriptor vectors
/// (`SelectionDescriptor.mean` / `basis` / `holdout`); [`decode_f16`] is
/// its inverse.
pub fn encode_f16(v: &[f32]) -> String {
    let bytes: Vec<u8> = v
        .iter()
        .flat_map(|x| f32_to_f16(*x).to_le_bytes())
        .collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn f16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| f16_to_f32(f32_to_f16(*x))).collect()
}

/// Deterministic standard-normal values (LCG + Box–Muller): the start of
/// the subspace iteration and the replacement of a collapsed direction.
fn gauss_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u1, u2) = (next().max(1e-12), next());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = 2.0 * std::f64::consts::PI * u2;
        out.push((r * t.cos()) as f32);
        out.push((r * t.sin()) as f32);
    }
    out.truncate(n);
    out
}

/// Top-`k` principal directions (orthonormal rows, `k × h`) of the
/// CENTERED samples by subspace (block power) iteration on the implicit
/// covariance Σ ccᵀ — no `h × h` matrix, `O(iters · n · k · h)`. The
/// iterate is re-orthonormalized (Gram–Schmidt) every step; a direction
/// that collapses to zero (a rank-deficient cloud) is replaced by a fresh
/// random one orthogonal to the rest, so the rows stay orthonormal.
/// Deterministic for a given `seed`.
pub fn top_principal_rows(
    centered: &[Vec<f32>],
    h: usize,
    k: usize,
    iters: usize,
    seed: u64,
) -> Vec<f32> {
    let mut q = gauss_vec(seed.wrapping_add(777), k * h);
    let mut spare = seed.wrapping_add(1);
    let mut orth = |q: &mut [f32]| {
        for i in 0..k {
            loop {
                for j in 0..i {
                    let dot: f32 = (0..h).map(|t| q[i * h + t] * q[j * h + t]).sum();
                    for t in 0..h {
                        q[i * h + t] -= dot * q[j * h + t];
                    }
                }
                let nrm: f32 = (0..h)
                    .map(|t| q[i * h + t] * q[i * h + t])
                    .sum::<f32>()
                    .sqrt();
                if nrm > 1e-12 {
                    for t in 0..h {
                        q[i * h + t] /= nrm;
                    }
                    break;
                }
                let fresh = gauss_vec(spare, h);
                spare = spare.wrapping_add(1);
                q[i * h..(i + 1) * h].copy_from_slice(&fresh);
            }
        }
    };
    orth(&mut q);
    let mut tmp = vec![0f32; k * h];
    let mut coef = vec![0f32; k];
    for _ in 0..iters {
        tmp.fill(0.0);
        for c in centered {
            for (i, ci) in coef.iter_mut().enumerate() {
                *ci = q[i * h..(i + 1) * h]
                    .iter()
                    .zip(c)
                    .map(|(a, b)| a * b)
                    .sum();
            }
            for (i, &ci) in coef.iter().enumerate() {
                if ci != 0.0 {
                    for (t, x) in tmp[i * h..(i + 1) * h].iter_mut().zip(c) {
                        *t += ci * x;
                    }
                }
            }
        }
        q.copy_from_slice(&tmp);
        orth(&mut q);
    }
    q
}

/// Fit a router-v2 descriptor (`mse_unit`) from RAW φ samples — the
/// trainer's recipe (`cortiq-embryo` `skill::fit_selection`) for the
/// runtime tools (`cortiq route-fit`): every φ is unit-normalized; the
/// LAST [`holdout_count`] samples are held out and never enter the fit;
/// mean + top-`rank` PCA rows ([`top_principal_rows`], `rank` clamped to
/// `train − 1` and ≥ 1) of the train part; `err_mean` / `err_std` of the
/// train reconstruction errors against the descriptor AS STORED (f16
/// rounded — the numbers the decision sees); the held-out unit φ carried
/// in `holdout` for [`calibrate_v2`]. `None` for fewer than 2 samples or
/// empty / ragged vectors.
pub fn fit_descriptor(
    phis_raw: &[Vec<f32>],
    phi_layer: usize,
    rank: usize,
) -> Option<SelectionDescriptor> {
    let n = phis_raw.len();
    if n < 2 {
        return None;
    }
    let h = phis_raw[0].len();
    if h == 0 || phis_raw.iter().any(|p| p.len() != h) {
        return None;
    }
    let phis: Vec<Vec<f32>> = phis_raw.iter().map(|p| unit_copy(p)).collect();
    let n_hold = holdout_count(n);
    let (train, hold) = phis.split_at(n - n_hold);
    let nt = train.len();
    let mut mean = vec![0f32; h];
    for p in train {
        for (m, v) in mean.iter_mut().zip(p) {
            *m += v / nt as f32;
        }
    }
    let centered: Vec<Vec<f32>> = train
        .iter()
        .map(|p| p.iter().zip(&mean).map(|(v, m)| v - m).collect())
        .collect();
    let rank = rank.min(nt.saturating_sub(1)).max(1);
    let basis = top_principal_rows(&centered, h, rank, 120, 99);
    let (mq, bq) = (f16_round(&mean), f16_round(&basis));
    let errs: Vec<f32> = train.iter().map(|p| recon_error(p, &mq, &bq, rank)).collect();
    let em = errs.iter().sum::<f32>() / nt as f32;
    let es = (errs.iter().map(|e| (e - em).powi(2)).sum::<f32>() / nt as f32)
        .sqrt()
        .max(1e-4);
    let hold_flat: Vec<f32> = hold.concat();
    Some(SelectionDescriptor {
        metric: METRIC_MSE_UNIT.into(),
        phi_layer,
        mean: encode_f16(&mean),
        basis: encode_f16(&basis),
        rank,
        err_mean: Some(em),
        err_std: Some(es),
        holdout: Some(encode_f16(&hold_flat)),
        holdout_n: Some(hold.len()),
    })
}

/// The cmf-im-v1 φ frame (spec §9.4) of a tokenizer: `prefix_ids =
/// encode("<|im_start|>user\n")`, `suffix_ids =
/// encode("<|im_end|>\n<|im_start|>assistant\n")` with the added tokens —
/// what the trainer's `phi_spec` renders (`Bpe::encode_with_specials`)
/// and checks the runtime against. Refused when the tokenizer has no
/// `<|im_start|>` / `<|im_end|>` added token (the frame would tokenize as
/// text and never match a chat prompt).
pub fn cmf_im_v1_phi_spec(tok: &crate::tokenizer::Tokenizer, layer: usize) -> Result<PhiSpec, String> {
    for s in ["<|im_start|>", "<|im_end|>"] {
        if tok.encode(s).len() != 1 {
            return Err(format!(
                "the tokenizer has no added token {s}: the cmf-im-v1 frame cannot be rendered"
            ));
        }
    }
    Ok(PhiSpec {
        layer,
        pool: "span_mean".into(),
        norm: "unit".into(),
        prefix_ids: tok.encode(IM_USER),
        suffix_ids: tok.encode(IM_ASSISTANT),
    })
}

fn unit_copy(phi: &[f32]) -> Vec<f32> {
    let n = phi.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        phi.iter().map(|x| x / n).collect()
    } else {
        phi.to_vec()
    }
}

/// Row of a unit φ against one `mse_unit` descriptor (None if malformed).
pub fn descriptor_row(
    id: &str,
    sel: &SelectionDescriptor,
    phi_unit: &[f32],
    hidden: usize,
) -> Option<ErrorRow> {
    if sel.metric != METRIC_MSE_UNIT {
        return None;
    }
    let (mean, basis) = (decode_f16(&sel.mean)?, decode_f16(&sel.basis)?);
    if mean.len() != hidden || basis.len() != sel.rank * hidden || phi_unit.len() != hidden {
        return None;
    }
    let e = recon_error(phi_unit, &mean, &basis, sel.rank);
    let pp: f32 = phi_unit.iter().map(|v| v * v).sum();
    Some((id.to_string(), e, sel.err_mean, sel.err_std, pp))
}

/// The backbone row and the rows of EVERY calibration class
/// ([`calibration_classes`]: v2, with a descriptor, not retired — the set
/// [`calibrate_v2`] fits T and θ over and [`skills_hash`] binds) for φ at
/// `router.phi.layer` (normalized here). Whether a class may WIN is the
/// decision's business ([`is_routable`]): dropping a quarantined class
/// from the rows would move the argmin — its in-scope prompt would go to
/// the nearest ACTIVE skill — and sharpen the softmax over fewer classes
/// (lower novelty than calibrated). Without a router: (None, []).
pub fn policy_rows(header: &CmfHeader, phi: &[f32]) -> (Option<ErrorRow>, Vec<ErrorRow>) {
    let Some(r) = &header.router else {
        return (None, Vec::new());
    };
    let hidden = header.arch.hidden_size;
    let phi = unit_copy(phi);
    let base = descriptor_row(BACKBONE_CLASS_ID, &r.base, &phi, hidden);
    let skills = calibration_classes(header)
        .into_iter()
        .filter_map(|s| {
            let sel = s.selection.as_ref()?;
            if sel.phi_layer != r.phi.layer {
                return None;
            }
            descriptor_row(&s.id, sel, &phi, hidden)
        })
        .collect();
    (base, skills)
}

/// Policy `backbone_gated` (spec §9.4) with every skill row a candidate
/// (the calibration's own view: [`calibrate_v2`] measures what the file
/// would do were every class active). See [`decide_backbone_gated_with`].
pub fn decide_backbone_gated(
    base: Option<&ErrorRow>,
    skills: &[ErrorRow],
    calibration: Option<&RoutingCalibration>,
    policy: Option<&RouterPolicy>,
    file_skills_hash: u64,
) -> RouteDecision {
    decide_backbone_gated_with(
        base,
        skills,
        &|_| true,
        calibration,
        policy,
        file_skills_hash,
    )
}

/// Policy `backbone_gated` (spec §9.4) — the one decision `run`, `explain`,
/// `serve` and the probes share. `skills` are the rows of EVERY
/// calibration class (the argmin and the softmax run over the same class
/// set the calibration was fitted on); `routable(id)` says which of them
/// may run. A skill wins iff: a policy is declared, its `skills_hash`
/// equals `file_skills_hash` (the calibration is not stale), a calibration
/// and a backbone row exist, some class is routable, the calibrated winner
/// is a skill AND routable, the input is not novel, and `E_skill + margin
/// < E_base` in unit error. Every other case runs the BACKBONE
/// (fail-closed) — a prompt nearest to a quarantined class runs the
/// backbone, never the next-nearest active skill. The negated comparisons
/// are deliberate: a NaN error or norm fails closed.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub fn decide_backbone_gated_with(
    base: Option<&ErrorRow>,
    skills: &[ErrorRow],
    routable: &dyn Fn(&str) -> bool,
    calibration: Option<&RoutingCalibration>,
    policy: Option<&RouterPolicy>,
    file_skills_hash: u64,
) -> RouteDecision {
    let mut rows: Vec<ErrorRow> = Vec::with_capacity(skills.len() + 1);
    rows.extend(base.cloned());
    rows.extend(skills.iter().cloned());
    let backbone = |routing: Routing, reason: String| RouteDecision {
        target: RouteTarget::Backbone,
        routing,
        reason,
    };
    let Some(policy) = policy else {
        return backbone(
            decide(&rows, None, 1.0),
            "no router policy (ROUTER_V2 absent): the backbone runs".into(),
        );
    };
    if parse_hex64(&policy.skills_hash) != Some(file_skills_hash) {
        return backbone(
            decide(&rows, None, 1.0),
            format!(
                "skills_hash mismatch (router {}, file {file_skills_hash:016x}): the calibration \
                 is stale — recalibrate",
                policy.skills_hash
            ),
        );
    }
    let Some(cal) = calibration else {
        return backbone(decide(&rows, None, 1.0), "router not calibrated".into());
    };
    let Some(base) = base else {
        return backbone(
            decide(&rows, Some(cal), 1.0),
            "no backbone descriptor row".into(),
        );
    };
    if !(base.4 > 1e-12) {
        return backbone(
            decide(&rows, Some(cal), 1.0),
            "degenerate φ (empty span)".into(),
        );
    }
    let routing = decide(&rows, Some(cal), 1.0);
    if !skills.iter().any(|r| routable(&r.0)) {
        return backbone(
            routing,
            "no routable skill (v2, active, measured gate)".into(),
        );
    }
    let e_base = base.1 / base.4.max(1e-12);
    let Some(winner) = routing.winner().cloned() else {
        return backbone(routing, "no scored class".into());
    };
    if winner.id == BACKBONE_CLASS_ID {
        return backbone(
            routing,
            format!("the backbone is the nearest class (E_base {e_base:.4})"),
        );
    }
    if !routable(&winner.id) {
        return backbone(
            routing,
            format!(
                "nearest class '{}' is not routable (not active with a measured gate): the \
                 backbone runs",
                winner.id
            ),
        );
    }
    if routing.is_novel {
        let reason = format!(
            "novel input (novelty {:.3} > θ {:.3})",
            routing.novelty, cal.novelty_theta
        );
        return backbone(routing, reason);
    }
    let e_skill = winner.error;
    if !(e_skill + policy.margin < e_base) {
        return backbone(
            routing,
            format!(
                "margin not beaten: E_skill {e_skill:.4} + margin {:.4} ≥ E_base {e_base:.4}",
                policy.margin
            ),
        );
    }
    let reason = format!(
        "skill '{}': E {e_skill:.4} + margin {:.4} < E_base {e_base:.4}, confidence {:.3}, \
         novelty {:.3} ≤ θ {:.3}",
        winner.id, policy.margin, routing.confidence, routing.novelty, cal.novelty_theta
    );
    RouteDecision {
        target: RouteTarget::Skill(winner.id),
        routing,
        reason,
    }
}

/// [`policy_rows`] + [`decide_backbone_gated`] with the file's own
/// calibration and [`skills_hash`]: the decision for a canonical φ.
pub fn route_policy(header: &CmfHeader, phi: &[f32]) -> RouteDecision {
    route_policy_with(header, phi, RouteOptions::default())
}

/// [`route_policy`] with [`RouteOptions`]: rows over every calibration
/// class, a win runs the skill only when it [`is_routable`] under `opts`.
pub fn route_policy_with(header: &CmfHeader, phi: &[f32], opts: RouteOptions) -> RouteDecision {
    let (base, skills) = policy_rows(header, phi);
    let routable = |id: &str| {
        header
            .skills
            .iter()
            .find(|s| s.id == id)
            .is_some_and(|s| is_routable(s, opts))
    };
    decide_backbone_gated_with(
        base.as_ref(),
        &skills,
        &routable,
        header.routing.as_ref(),
        header.router.as_ref(),
        skills_hash(header),
    )
}

/// Upper end of the two-sided Clopper–Pearson (exact binomial) interval
/// at `confidence` (0.95 → the 97.5 % Beta quantile) for `k` events in `n`
/// trials. `n == 0` → 1.
pub fn clopper_pearson_upper(k: usize, n: usize, confidence: f64) -> f64 {
    if n == 0 || k >= n {
        return 1.0;
    }
    let q = 1.0 - (1.0 - confidence) / 2.0;
    let (a, b) = ((k + 1) as f64, (n - k) as f64);
    if k == 0 {
        // closed form: I_x(1, n) = 1 − (1−x)^n
        return 1.0 - (1.0 - q).powf(1.0 / n as f64);
    }
    // bisection on the regularized incomplete beta
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if reg_inc_beta(a, b, mid) < q {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn ln_gamma(x: f64) -> f64 {
    // Lanczos (g = 7, n = 9)
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut acc = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        acc += c / (x + i as f64);
    }
    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

/// Regularized incomplete beta I_x(a, b) (continued fraction, Lentz).
fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let cf = |a: f64, b: f64, x: f64| -> f64 {
        let tiny = 1e-300;
        let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        if d.abs() < tiny {
            d = tiny;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..400 {
            let m = m as f64;
            let m2 = 2.0 * m;
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = 1.0 + aa / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            h *= d * c;
            let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if d.abs() < tiny {
                d = tiny;
            }
            c = 1.0 + aa / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * cf(b, a, 1.0 - x) / b
    }
}

/// Calibration v2 (spec §9.4), pure over the header: classes = the
/// backbone (`router.base`, its holdout = general prompts) + every
/// [`calibration_classes`] skill (its holdout = in-scope prompts). The
/// temperature minimizes the NLL of the true class over ALL classes
/// including the backbone; θ is the (1−`target_fpr`) quantile of novelty
/// on the in-scope skill holdouts. `measured` reports, on these same
/// holdouts and under the full [`decide_backbone_gated`] decision with every
/// calibrated skill as a candidate, the in-scope recall and the backbone
/// false-accept rate with its Clopper–Pearson 95 % upper bound. These are
/// calibration-set numbers; the gate measures a DISJOINT general-eval set.
/// Write the result with `header.routing = cal` and `router.skills_hash =
/// hex(skills_hash(header))`.
pub fn calibrate_v2(
    header: &CmfHeader,
    target_fpr: f32,
) -> Result<(RoutingCalibration, serde_json::Value), String> {
    let policy = header
        .router
        .as_ref()
        .ok_or("calibrate_v2: the file declares no router policy (header.router)")?;
    let hidden = header.arch.hidden_size;
    let classes = calibration_classes(header);
    if classes.is_empty() {
        return Err("calibrate_v2: no v2 skill with a selection descriptor".into());
    }
    // (id, descriptor) — index 0 is the backbone.
    let mut descs: Vec<(&str, &SelectionDescriptor)> = vec![(BACKBONE_CLASS_ID, &policy.base)];
    for s in &classes {
        let sel = s.selection.as_ref().expect("filtered");
        if sel.metric != METRIC_MSE_UNIT || sel.phi_layer != policy.phi.layer {
            return Err(format!(
                "calibrate_v2: skill '{}' descriptor is {}@{}, the policy needs {METRIC_MSE_UNIT}@{}",
                s.id, sel.metric, sel.phi_layer, policy.phi.layer
            ));
        }
        descs.push((&s.id, sel));
    }
    let holdout = |d: &SelectionDescriptor| -> Vec<Vec<f32>> {
        let (Some(h), Some(n)) = (d.holdout.as_ref(), d.holdout_n) else {
            return Vec::new();
        };
        match decode_f16(h) {
            Some(v) if v.len() == n * hidden => (0..n)
                .map(|i| unit_copy(&v[i * hidden..(i + 1) * hidden]))
                .collect(),
            _ => Vec::new(),
        }
    };
    // samples: (true class, rows over every class)
    let mut samples: Vec<(usize, Vec<ErrorRow>)> = Vec::new();
    for (ci, (_, d)) in descs.iter().enumerate() {
        for phi in holdout(d) {
            let rows: Option<Vec<ErrorRow>> = descs
                .iter()
                .map(|(id, dd)| descriptor_row(id, dd, &phi, hidden))
                .collect();
            let rows = rows.ok_or("calibrate_v2: a descriptor is malformed (dims / base64)")?;
            samples.push((ci, rows));
        }
    }
    let n_general = samples.iter().filter(|(c, _)| *c == 0).count();
    let n_in = samples.len() - n_general;
    if n_general == 0 {
        return Err("calibrate_v2: router.base carries no holdout (general prompts)".into());
    }
    if n_in == 0 {
        return Err("calibrate_v2: no skill carries an in-scope holdout".into());
    }

    // Temperature: NLL of the true class over all classes (geometric grid).
    let mut best_t = 1.0f32;
    let mut best_nll = f32::INFINITY;
    let mut t = 1e-3f32;
    while t <= 1e6 {
        let mut nll = 0.0f32;
        for (ci, rows) in &samples {
            let mut logits: Vec<f32> = rows.iter().map(|r| -r.1 / t).collect();
            softmax(&mut logits);
            nll -= logits[*ci].max(1e-9).ln();
        }
        if nll < best_nll {
            best_nll = nll;
            best_t = t;
        }
        t *= 1.15;
    }
    let mut cal = RoutingCalibration {
        temperature: best_t,
        novelty_theta: 0.5,
        samples: samples.len(),
        target_fpr,
    };
    // θ: (1−fpr) quantile of in-scope novelty.
    let mut nov: Vec<f32> = samples
        .iter()
        .filter(|(c, _)| *c > 0)
        .map(|(_, rows)| decide(rows, Some(&cal), 1.0).novelty)
        .filter(|v| v.is_finite())
        .collect();
    nov.sort_by(|a, b| a.total_cmp(b));
    if !nov.is_empty() {
        let q = (1.0 - target_fpr).clamp(0.0, 1.0);
        let idx = (((nov.len() - 1) as f32) * q).round() as usize;
        cal.novelty_theta = (nov[idx.min(nov.len() - 1)] + 1e-4).min(0.999);
    }

    // Measured under the full decision, every calibrated skill a candidate.
    let hash = skills_hash(header);
    let mut pol = policy.clone();
    pol.skills_hash = format!("{hash:016x}");
    let (mut fa, mut hit) = (0usize, 0usize);
    let mut per: Vec<(usize, usize)> = vec![(0, 0); descs.len()];
    for (ci, rows) in &samples {
        let d = decide_backbone_gated(Some(&rows[0]), &rows[1..], Some(&cal), Some(&pol), hash);
        if *ci == 0 {
            fa += usize::from(d.skill().is_some());
        } else {
            per[*ci].0 += 1;
            if d.skill() == Some(descs[*ci].0) {
                hit += 1;
                per[*ci].1 += 1;
            }
        }
    }
    let per_skill: serde_json::Map<String, serde_json::Value> = descs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, (id, _))| {
            let (n, h) = per[i];
            (
                id.to_string(),
                serde_json::json!({"n": n, "recall": if n > 0 { h as f64 / n as f64 } else { f64::NAN }}),
            )
        })
        .collect();
    let measured = serde_json::json!({
        "set": "calibration",
        "n_in": n_in,
        "n_general": n_general,
        "in_scope_recall": hit as f64 / n_in as f64,
        "false_accept": fa as f64 / n_general as f64,
        "false_accept_upper95": clopper_pearson_upper(fa, n_general, 0.95),
        "temperature": cal.temperature,
        "novelty_theta": cal.novelty_theta,
        "target_fpr": target_fpr,
        "margin": policy.margin,
        "skills_hash": format!("{hash:016x}"),
        "per_skill": per_skill,
    });
    Ok((cal, measured))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cortiq_core::knowledge::skill_kind;

    const HID: usize = 16;

    #[test]
    fn holdout_count_is_a_fifth_at_least_one_and_leaves_train_samples() {
        for (n, want) in [(1, 1), (2, 1), (3, 1), (5, 1), (6, 1), (10, 2), (50, 10), (4000, 800)] {
            assert_eq!(holdout_count(n), want, "n = {n}");
        }
    }

    #[test]
    fn encode_f16_round_trips_through_decode() {
        let v: Vec<f32> = (0..HID).map(|i| (i as f32 - 7.5) * 0.125).collect();
        assert_eq!(decode_f16(&encode_f16(&v)).unwrap(), v);
    }

    /// A cloud spread along one direction (plus a little noise): the
    /// rank-1 basis is that direction, the rows are orthonormal at rank
    /// 2, the held-out tail is 20 % and never enters the mean, the error
    /// statistics are finite, and too few samples give None.
    #[test]
    fn fit_descriptor_recovers_the_dominant_direction() {
        let h = 8;
        let base: Vec<f32> = (0..h).map(|i| 1.0 + 0.05 * i as f32).collect();
        let dir: Vec<f32> = (0..h).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let noise = gauss_vec(5, 40 * h);
        let samples: Vec<Vec<f32>> = (0..40)
            .map(|s| {
                let a = (s as f32 / 39.0 - 0.5) * 2.0;
                (0..h)
                    .map(|i| base[i] + a * dir[i] + 0.02 * noise[s * h + i])
                    .collect()
            })
            .collect();
        let d = fit_descriptor(&samples, 3, 1).expect("fit");
        assert_eq!(d.metric, METRIC_MSE_UNIT);
        assert_eq!(d.phi_layer, 3);
        assert_eq!(d.rank, 1);
        assert_eq!(d.holdout_n, Some(8));
        let basis = decode_f16(&d.basis).unwrap();
        assert_eq!(basis.len(), h);
        // The centered unit cloud varies along the unit-normalized dir
        // component orthogonal to the mean: the found row is aligned with
        // the unit-projected `dir` (sign free).
        let mean = decode_f16(&d.mean).unwrap();
        let unit = |v: &[f32]| unit_copy(v);
        let mut proj = vec![0f32; h];
        let (hi, lo) = (unit(&samples[39]), unit(&samples[0]));
        for i in 0..h {
            proj[i] = hi[i] - lo[i];
        }
        let proj = unit(&proj);
        let cos: f32 = basis.iter().zip(&proj).map(|(a, b)| a * b).sum();
        assert!(cos.abs() > 0.95, "cos {cos}");
        assert!(d.err_mean.unwrap().is_finite() && d.err_std.unwrap() >= 1e-4);
        assert_eq!(mean.len(), h);
        let hold = decode_f16(d.holdout.as_ref().unwrap()).unwrap();
        assert_eq!(hold.len(), 8 * h);
        // The held-out rows are the last 8 samples, unit-normalized.
        let last = unit(&samples[39]);
        for (a, b) in hold[7 * h..].iter().zip(&last) {
            assert!((a - b).abs() < 2e-3, "{a} vs {b}");
        }
        // rank 2: orthonormal rows.
        let d2 = fit_descriptor(&samples, 3, 2).unwrap();
        let b2 = decode_f16(&d2.basis).unwrap();
        let dot: f32 = b2[..h].iter().zip(&b2[h..]).map(|(a, b)| a * b).sum();
        let n1: f32 = b2[..h].iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!(dot.abs() < 1e-2 && (n1 - 1.0).abs() < 1e-2, "dot {dot} n1 {n1}");
        // rank is clamped to train − 1 (3 samples → train 2 → rank 1).
        assert_eq!(fit_descriptor(&samples[..3], 0, 16).unwrap().rank, 1);
        assert!(fit_descriptor(&samples[..1], 0, 1).is_none());
        assert!(fit_descriptor(&[vec![1.0], vec![1.0, 2.0]], 0, 1).is_none());
        // Deterministic.
        assert_eq!(fit_descriptor(&samples, 3, 2).unwrap().basis, d2.basis);
    }

    fn f16_b64(v: &[f32]) -> String {
        let bytes: Vec<u8> = v
            .iter()
            .flat_map(|x| cortiq_core::quant::f32_to_f16(*x).to_le_bytes())
            .collect();
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    fn header() -> CmfHeader {
        serde_json::from_value(serde_json::json!({
            "version": 2,
            "quant_type": "F32",
            "arch": {
                "arch_name": "synthetic", "hidden_size": HID, "intermediate_size": 32,
                "num_layers": 8, "num_attention_heads": 2, "num_kv_heads": 1, "head_dim": 8,
                "vocab_size": 16, "layer_types": vec!["FullAttention"; 8],
                "rms_norm_eps": 1e-6, "max_position_embeddings": 64
            }
        }))
        .unwrap()
    }

    /// Deterministic xorshift noise in [−1, 1).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// A unit sample of class `axis`: e_axis + σ·noise.
    fn sample(rng: &mut Rng, axis: usize, sigma: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..HID).map(|_| sigma * rng.next()).collect();
        v[axis] += 1.0;
        unit_copy(&v)
    }

    /// Descriptor fitted on `train` samples of class `axis`: mean = unit
    /// centroid, rank-1 basis on a noise axis, error stats of the train set.
    fn descriptor(
        train: &[Vec<f32>],
        holdout: &[Vec<f32>],
        noise_axis: usize,
    ) -> SelectionDescriptor {
        let mut mean = vec![0.0f32; HID];
        for s in train {
            for (m, x) in mean.iter_mut().zip(s) {
                *m += x / train.len() as f32;
            }
        }
        let mut basis = vec![0.0f32; HID];
        basis[noise_axis] = 1.0;
        let q = |v: &[f32]| -> Vec<f32> {
            v.iter()
                .map(|x| f16_to_f32(cortiq_core::quant::f32_to_f16(*x)))
                .collect()
        };
        let (mq, bq) = (q(&mean), q(&basis));
        let errs: Vec<f32> = train.iter().map(|s| recon_error(s, &mq, &bq, 1)).collect();
        let em = errs.iter().sum::<f32>() / errs.len() as f32;
        let es = (errs.iter().map(|e| (e - em).powi(2)).sum::<f32>() / errs.len() as f32).sqrt();
        SelectionDescriptor {
            metric: "mse_unit".into(),
            phi_layer: 2,
            mean: f16_b64(&mean),
            basis: f16_b64(&basis),
            rank: 1,
            err_mean: Some(em),
            err_std: Some(es),
            holdout: Some(f16_b64(&holdout.concat())),
            holdout_n: Some(holdout.len()),
        }
    }

    fn policy(base: SelectionDescriptor) -> RouterPolicy {
        RouterPolicy {
            version: 2,
            policy: "backbone_gated".into(),
            granularity: "request".into(),
            phi: PhiSpec {
                layer: 2,
                pool: "span_mean".into(),
                norm: "unit".into(),
                prefix_ids: vec![1, 7, 8],
                suffix_ids: vec![2, 1, 9],
            },
            base,
            margin: 0.05,
            skills_hash: "0".into(),
            measured: None,
        }
    }

    fn skill(id: &str, sel: SelectionDescriptor, status: &str) -> SkillRecord {
        SkillRecord {
            id: id.into(),
            layers: vec![5],
            selection: Some(sel),
            kind: Some(skill_kind::FFN_REPLACE.into()),
            status: Some(status.into()),
            gate: Some(serde_json::json!({"status": "measured"})),
            ..Default::default()
        }
    }

    fn row(id: &str, e: f32) -> ErrorRow {
        (id.into(), e, Some(0.05), Some(0.05), 1.0)
    }

    fn truth_policy() -> RouterPolicy {
        let mut p = policy(SelectionDescriptor {
            metric: "mse_unit".into(),
            phi_layer: 2,
            mean: String::new(),
            basis: String::new(),
            rank: 0,
            err_mean: None,
            err_std: None,
            holdout: None,
            holdout_n: None,
        });
        p.skills_hash = "00000000000000ab".into();
        p
    }

    #[test]
    fn backbone_gated_truth_table() {
        let pol = truth_policy();
        let cal = RoutingCalibration {
            temperature: 0.1,
            novelty_theta: 0.99,
            samples: 10,
            target_fpr: 0.05,
        };
        let (base, near) = (row(BACKBONE_CLASS_ID, 0.9), row("herbs", 0.05));
        let go = |b: Option<&ErrorRow>,
                  s: &[ErrorRow],
                  c: Option<&RoutingCalibration>,
                  p: Option<&RouterPolicy>,
                  h: u64| { decide_backbone_gated(b, s, c, p, h) };
        let std_skills = std::slice::from_ref(&near);

        // The skill wins only when everything holds.
        let d = go(Some(&base), std_skills, Some(&cal), Some(&pol), 0xab);
        assert_eq!(d.target, RouteTarget::Skill("herbs".into()), "{}", d.reason);

        let cases: Vec<(&str, RouteDecision, &str)> = vec![
            (
                "no policy",
                go(Some(&base), std_skills, Some(&cal), None, 0xab),
                "no router policy",
            ),
            (
                "hash mismatch",
                go(Some(&base), std_skills, Some(&cal), Some(&pol), 0xac),
                "skills_hash mismatch",
            ),
            (
                "no calibration",
                go(Some(&base), std_skills, None, Some(&pol), 0xab),
                "not calibrated",
            ),
            (
                "no base row",
                go(None, std_skills, Some(&cal), Some(&pol), 0xab),
                "no backbone descriptor",
            ),
            (
                "no skills",
                go(Some(&base), &[], Some(&cal), Some(&pol), 0xab),
                "no routable skill",
            ),
            (
                "winner is the backbone",
                go(
                    Some(&row(BACKBONE_CLASS_ID, 0.01)),
                    std_skills,
                    Some(&cal),
                    Some(&pol),
                    0xab,
                ),
                "backbone is the nearest",
            ),
            (
                "novel",
                go(
                    Some(&base),
                    std_skills,
                    Some(&RoutingCalibration {
                        novelty_theta: 0.0,
                        ..cal.clone()
                    }),
                    Some(&pol),
                    0xab,
                ),
                "novel input",
            ),
            (
                "margin not beaten",
                go(
                    Some(&row(BACKBONE_CLASS_ID, 0.32)),
                    &[row("herbs", 0.30)],
                    Some(&cal),
                    Some(&pol),
                    0xab,
                ),
                "margin not beaten",
            ),
            (
                "degenerate φ",
                go(
                    Some(&(BACKBONE_CLASS_ID.into(), 0.9, None, None, 0.0)),
                    std_skills,
                    Some(&cal),
                    Some(&pol),
                    0xab,
                ),
                "degenerate",
            ),
        ];
        for (name, d, needle) in cases {
            assert_eq!(
                d.target,
                RouteTarget::Backbone,
                "case '{name}' routed to a skill"
            );
            assert!(
                d.reason.contains(needle),
                "case '{name}': reason '{}'",
                d.reason
            );
        }
    }

    /// A unit sample around Σ w·e_axis + σ·noise.
    fn sample_mix(rng: &mut Rng, axes: &[(usize, f32)], sigma: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..HID).map(|_| sigma * rng.next()).collect();
        for &(a, w) in axes {
            v[a] += w;
        }
        unit_copy(&v)
    }

    /// R1: the runtime decision scores EVERY calibration class, the set
    /// `calibrate_v2` fitted on. `mushrooms` (quarantine) lies inside the
    /// affine subspace of `herbs` (active): its in-scope prompt is nearest
    /// to `mushrooms` → the BACKBONE runs. With the quarantined class
    /// dropped from the rows (the pre-fix `policy_rows`), the same prompt
    /// went to `herbs`.
    #[test]
    fn a_prompt_nearest_to_a_quarantined_class_runs_the_backbone() {
        let mut rng = Rng(0x5eed_1234_abcd_0001);
        let draw = |rng: &mut Rng, axes: &[(usize, f32)], n: usize| -> Vec<Vec<f32>> {
            (0..n).map(|_| sample_mix(rng, axes, 0.02)).collect()
        };
        let (gen_ax, herb_ax, mush_ax) = (vec![(0, 1.0)], vec![(1, 1.0)], vec![(1, 1.0), (4, 0.8)]);
        let (gt, gh) = (draw(&mut rng, &gen_ax, 200), draw(&mut rng, &gen_ax, 100));
        let (ht, hh) = (draw(&mut rng, &herb_ax, 200), draw(&mut rng, &herb_ax, 100));
        let (mt, mh) = (draw(&mut rng, &mush_ax, 200), draw(&mut rng, &mush_ax, 100));
        let mut h = header();
        h.router = Some(policy(descriptor(&gt, &gh, 3)));
        // herbs: mean e1, basis e4 — mushrooms (e1 + 0.8·e4) lie close to
        // that line, nearer to their own class.
        h.skills.push(skill("herbs", descriptor(&ht, &hh, 4), "active"));
        h.skills
            .push(skill("mushrooms", descriptor(&mt, &mh, 5), "quarantine"));
        let (mut cal, _) = calibrate_v2(&h, 0.05).unwrap();
        // Isolate the argmin: novelty never vetoes in this test.
        cal.novelty_theta = 0.999;
        h.routing = Some(cal.clone());
        let hash = skills_hash(&h);
        h.router.as_mut().unwrap().skills_hash = format!("{hash:016x}");

        let (mut to_backbone, mut old_to_herbs) = (0usize, 0usize);
        let n = 100;
        for _ in 0..n {
            let q = sample_mix(&mut rng, &mush_ax, 0.02);
            let d = route_policy(&h, &q);
            assert_eq!(
                d.nearest_skill(),
                Some("mushrooms"),
                "the quarantined class is scored: {}",
                d.reason
            );
            if d.target == RouteTarget::Backbone {
                to_backbone += 1;
                assert!(d.reason.contains("not routable"), "{}", d.reason);
            }
            // Pre-fix rows: backbone + auto-routable skills only.
            let (base, rows) = policy_rows(&h, &q);
            let herbs_only: Vec<ErrorRow> = rows.into_iter().filter(|r| r.0 == "herbs").collect();
            let old = decide_backbone_gated(base.as_ref(), &herbs_only, Some(&cal), h.router.as_ref(), hash);
            old_to_herbs += usize::from(old.skill() == Some("herbs"));
            // The gate-measurement flag lets the quarantined class win.
            let dq = route_policy_with(&h, &q, RouteOptions { include_quarantine: true });
            assert_eq!(dq.skill(), Some("mushrooms"), "{}", dq.reason);
        }
        assert_eq!(to_backbone, n, "a mushroom prompt ran a skill");
        assert!(
            old_to_herbs * 10 >= n * 9,
            "the pre-fix decision did not misroute ({old_to_herbs}/{n}) — the regression is untested"
        );
        // herbs keeps its own prompts.
        for _ in 0..50 {
            let d = route_policy(&h, &sample_mix(&mut rng, &herb_ax, 0.02));
            assert_eq!(d.skill(), Some("herbs"), "{}", d.reason);
        }
        // stale_regate is scored too, and runnable only under the flag.
        h.skills[1].status = Some("stale_regate".into());
        let q = sample_mix(&mut rng, &mush_ax, 0.02);
        assert_eq!(route_policy(&h, &q).target, RouteTarget::Backbone);
        assert_eq!(
            route_policy_with(&h, &q, RouteOptions { include_quarantine: true }).skill(),
            Some("mushrooms")
        );
        assert!(no_candidate_reason(&h, RouteOptions::default()).is_none());
        h.skills[0].status = Some("quarantine".into());
        assert!(
            no_candidate_reason(&h, RouteOptions::default())
                .is_some_and(|r| r.contains("no routable skill class"))
        );
        assert!(no_candidate_reason(&h, RouteOptions { include_quarantine: true }).is_none());
    }

    #[test]
    fn prompt_contract_is_enforced_in_one_place() {
        let mut h = header();
        h.skills.push(SkillRecord {
            prompt_contract: Some(PROMPT_CONTRACT_CMF_IM_V1.into()),
            ..skill("herbs", truth_policy().base, "active")
        });
        h.skills.push(SkillRecord {
            prompt_contract: Some("other-v9".into()),
            ..skill("odd", truth_policy().base, "active")
        });
        h.skills.push(skill("free", truth_policy().base, "active"));
        let pick = |id: &str| RouteDecision::forced(RouteTarget::Skill(id.into()), "t");
        // cmf-im-v1 frame satisfies it; a raw prompt is rendered when allowed.
        let (d, r) = enforce_prompt_contract(&h, pick("herbs"), PromptFrame::CmfImV1, false);
        assert_eq!((d.skill(), r), (Some("herbs"), false));
        let (d, r) = enforce_prompt_contract(&h, pick("herbs"), PromptFrame::Raw, true);
        assert_eq!((d.skill(), r), (Some("herbs"), true));
        // Not renderable here, another template, an unknown contract → backbone.
        for (id, frame, can) in [
            ("herbs", PromptFrame::Raw, false),
            ("herbs", PromptFrame::Other, true),
            ("odd", PromptFrame::CmfImV1, true),
        ] {
            let (d, r) = enforce_prompt_contract(&h, pick(id), frame, can);
            assert_eq!(d.target, RouteTarget::Backbone, "{id} {frame:?}");
            assert!(!r && d.reason.contains("not satisfied"), "{}", d.reason);
        }
        // No contract, or the backbone: untouched.
        let (d, r) = enforce_prompt_contract(&h, pick("free"), PromptFrame::Raw, false);
        assert_eq!((d.skill(), r), (Some("free"), false));
        let bb = RouteDecision::forced(RouteTarget::Backbone, "t");
        assert_eq!(enforce_prompt_contract(&h, bb, PromptFrame::Other, false).0.skill(), None);
        // The single-turn form and its inverse.
        let p = render_cmf_im_v1("Что лечит зверобой?");
        assert_eq!(p, "<|im_start|>user\nЧто лечит зверобой?<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(cmf_im_v1_single_user_turn(&p), Some("Что лечит зверобой?"));
        assert_eq!(cmf_im_v1_single_user_turn("Что лечит зверобой?"), None);
        let two = format!("{}{}", render_cmf_im_v1("a"), "b<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(cmf_im_v1_single_user_turn(&two), None, "history is not one turn");
        assert_eq!(
            cmf_im_v1_single_user_turn(
                "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"
            ),
            None
        );
    }

    #[test]
    fn skills_hash_tracks_descriptors_not_status() {
        let mut rng = Rng(7);
        let tr: Vec<Vec<f32>> = (0..20).map(|_| sample(&mut rng, 1, 0.1)).collect();
        let mut h = header();
        h.router = Some(policy(descriptor(&tr, &[], 3)));
        h.skills
            .push(skill("herbs", descriptor(&tr, &[], 4), "quarantine"));
        let h0 = skills_hash(&h);
        h.skills[0].status = Some("active".into());
        h.skills[0].gate = Some(serde_json::json!({"status": "measured", "x": 1}));
        h.skills[0].selection.as_mut().unwrap().holdout = Some("AAAA".into());
        assert_eq!(
            skills_hash(&h),
            h0,
            "status/gate/holdout do not invalidate the calibration"
        );
        // A v1 record is not a class.
        h.skills.push(SkillRecord {
            id: "legacy".into(),
            selection: Some(descriptor(&tr, &[], 5)),
            ..Default::default()
        });
        assert_eq!(skills_hash(&h), h0);
        // A changed descriptor, a new class, a retirement, a new base: stale.
        let mut h2 = h.clone();
        h2.skills[0].selection.as_mut().unwrap().err_mean = Some(0.123);
        assert_ne!(skills_hash(&h2), h0);
        let mut h3 = h.clone();
        h3.skills
            .push(skill("more", descriptor(&tr, &[], 6), "quarantine"));
        assert_ne!(skills_hash(&h3), h0);
        let mut h4 = h.clone();
        h4.skills[0].status = Some("retired".into());
        assert_ne!(skills_hash(&h4), h0);
        let mut h5 = h.clone();
        h5.router.as_mut().unwrap().base.rank = 0;
        assert_ne!(skills_hash(&h5), h0);
    }

    #[test]
    fn phi_span_is_the_user_text_only() {
        let p = policy(descriptor(&[vec![1.0; HID]], &[], 0)).phi;
        let (ids, span) = phi_span_ids(&p, &[40, 41, 42, 43]);
        assert_eq!(ids, vec![1, 7, 8, 40, 41, 42, 43, 2, 1, 9]);
        assert_eq!(span, 3..7);
        assert_eq!(&ids[span.clone()], &[40, 41, 42, 43]);
        // span_mean + unit over exactly those rows
        let hiddens: Vec<f32> = (0..ids.len())
            .flat_map(|r| {
                (0..HID).map(move |c| {
                    if (3..7).contains(&r) {
                        (c == 0) as u8 as f32
                    } else {
                        100.0
                    }
                })
            })
            .collect();
        let phi = pool_span_unit(&hiddens, HID, span);
        assert!((phi[0] - 1.0).abs() < 1e-6 && phi[1..].iter().all(|x| *x == 0.0));
        assert!(
            pool_span_unit(&hiddens, HID, 5..5)
                .iter()
                .all(|x| *x == 0.0)
        );
    }

    #[test]
    fn clopper_pearson_matches_reference_values() {
        // exact two-sided 95 % intervals (standard tables)
        assert!((clopper_pearson_upper(0, 10, 0.95) - 0.308_5).abs() < 1e-3);
        assert!((clopper_pearson_upper(5, 10, 0.95) - 0.812_9).abs() < 1e-3);
        assert!((clopper_pearson_upper(1, 100, 0.95) - 0.054_5).abs() < 1e-3);
        let u = clopper_pearson_upper(0, 500, 0.95);
        assert!((u - 0.007_351).abs() < 1e-5, "{u}");
        assert_eq!(clopper_pearson_upper(3, 3, 0.95), 1.0);
    }

    /// Two separable classes (backbone around e0, herbs around e1): the
    /// fitted policy accepts no general sample and recalls the in-scope
    /// ones, and the file-level decision agrees on fresh samples.
    #[test]
    fn calibrate_v2_separates_backbone_and_skill() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let draw = |rng: &mut Rng, axis: usize, n: usize| -> Vec<Vec<f32>> {
            (0..n).map(|_| sample(rng, axis, 0.1)).collect()
        };
        let (gt, gh) = (draw(&mut rng, 0, 200), draw(&mut rng, 0, 120));
        let (st, sh) = (draw(&mut rng, 1, 200), draw(&mut rng, 1, 120));
        let mut h = header();
        h.router = Some(policy(descriptor(&gt, &gh, 3)));
        h.skills
            .push(skill("herbs", descriptor(&st, &sh, 4), "active"));

        let (cal, measured) = calibrate_v2(&h, 0.01).unwrap();
        assert_eq!(measured["false_accept"], 0.0, "{measured}");
        assert!(
            measured["in_scope_recall"].as_f64().unwrap() >= 0.98,
            "{measured}"
        );
        assert!(
            measured["false_accept_upper95"].as_f64().unwrap() < 0.031,
            "{measured}"
        );
        assert_eq!(measured["n_general"], 120);
        assert_eq!(measured["n_in"], 120);

        // Publish the calibration the way a tool would, then route fresh φ.
        h.routing = Some(cal);
        let hash = skills_hash(&h);
        h.router.as_mut().unwrap().skills_hash = format!("{hash:016x}");
        let (mut fa, mut hit) = (0, 0);
        for _ in 0..200 {
            fa += usize::from(
                route_policy(&h, &sample(&mut rng, 0, 0.1))
                    .skill()
                    .is_some(),
            );
            hit +=
                usize::from(route_policy(&h, &sample(&mut rng, 1, 0.1)).skill() == Some("herbs"));
        }
        assert_eq!(fa, 0);
        assert!(hit >= 190, "recall {hit}/200");
        // An off-manifold input is not captured by the skill.
        let d = route_policy(&h, &sample(&mut rng, 9, 0.1));
        assert_eq!(d.target, RouteTarget::Backbone, "{}", d.reason);

        // A quarantined skill is never auto-routed; a stale hash routes nothing.
        let mut q = h.clone();
        q.skills[0].status = Some("quarantine".into());
        assert_eq!(
            route_policy(&q, &sample(&mut rng, 1, 0.1)).target,
            RouteTarget::Backbone
        );
        let mut stale = h.clone();
        stale
            .skills
            .push(skill("new", descriptor(&st, &[], 5), "active"));
        let d = route_policy(&stale, &sample(&mut rng, 1, 0.1));
        assert_eq!(d.target, RouteTarget::Backbone);
        assert!(d.reason.contains("stale"), "{}", d.reason);
    }
}
