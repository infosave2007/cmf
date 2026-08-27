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

use crate::pipeline::Pipeline;
use base64::Engine as _;
use cortiq_core::CmfModel;
use cortiq_core::quant::f16_to_f32;

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
    rows: &[(String, f32, Option<f32>, Option<f32>, f32)],
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
) -> Vec<(String, f32, Option<f32>, Option<f32>, f32)> {
    let hidden = model.arch().hidden_size;
    let mut rows = Vec::new();
    for skill in &model.header.skills {
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
    let mut per_sample: Vec<(Vec<(String, f32, Option<f32>, Option<f32>, f32)>, usize)> =
        Vec::new();
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
