//! Recon-argmin skill routing (spec §9, P1 signal-consistency): the
//! container's selection descriptors define per-skill affine subspaces
//! over φ(x); the winner is the skill that reconstructs φ best. No
//! trained gate — routing is a property of the skills themselves.

use crate::pipeline::Pipeline;
use base64::Engine as _;
use cortiq_core::quant::f16_to_f32;
use cortiq_core::CmfModel;

#[derive(Debug, Clone)]
pub struct SkillRoute {
    pub id: String,
    /// Normalized reconstruction error E ∈ [0, 1]; lower = closer.
    pub error: f32,
}

fn decode_f16(b64: &str) -> Option<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    Some(
        bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
    )
}

/// Score every routable skill; sorted best-first. Empty when the file
/// carries no selection descriptors.
pub fn route(model: &CmfModel, pipeline: &mut Pipeline, ids: &[u32]) -> Vec<SkillRoute> {
    let hidden = model.arch().hidden_size;
    let mut phi_cache: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut out = Vec::new();

    for skill in &model.header.skills {
        let Some(sel) = &skill.selection else { continue };
        if sel.metric != "mse" {
            tracing::warn!("skill '{}': unknown metric '{}'", skill.id, sel.metric);
            continue;
        }
        let phi = match phi_cache.iter().find(|(l, _)| *l == sel.phi_layer) {
            Some((_, p)) => p.clone(),
            None => {
                let p = pipeline.probe_phi(ids, sel.phi_layer);
                phi_cache.push((sel.phi_layer, p.clone()));
                p
            }
        };
        let (Some(mean), Some(basis)) = (decode_f16(&sel.mean), decode_f16(&sel.basis)) else {
            tracing::error!("skill '{}': malformed selection payload", skill.id);
            continue;
        };
        if mean.len() != hidden || basis.len() != sel.rank * hidden {
            tracing::error!("skill '{}': selection dims mismatch", skill.id);
            continue;
        }
        // r = φ − mean;  E = ‖r − B·Bᵀr‖² / ‖φ‖²  (B rows orthonormal).
        // Normalizing by ‖φ‖ (not ‖r‖!) keeps the distance-to-mean
        // signal — the whole point of the affine subspace.
        let r: Vec<f32> = phi.iter().zip(&mean).map(|(p, m)| p - m).collect();
        let rr: f32 = r.iter().map(|v| v * v).sum();
        let pp: f32 = phi.iter().map(|v| v * v).sum();
        let mut proj = 0f32;
        for k in 0..sel.rank {
            let row = &basis[k * hidden..(k + 1) * hidden];
            let c: f32 = row.iter().zip(&r).map(|(b, v)| b * v).sum();
            proj += c * c;
        }
        let e = (rr - proj).max(0.0) / pp.max(1e-12);
        out.push(SkillRoute {
            id: skill.id.clone(),
            error: e,
        });
    }
    out.sort_by(|a, b| a.error.total_cmp(&b.error));
    out
}
