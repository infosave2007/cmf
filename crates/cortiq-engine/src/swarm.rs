//! Dynamic per-token skill routing with hysteresis (spec §9 made
//! runtime; VMF-2026 experiment №2).
//!
//! The recon-argmin error E(skill) is computed against the rolling φ
//! (EMA of the router layer's hidden state — on-policy, fireball-style).
//! Switching uses TWO thresholds — a first-order-transition analogue of
//! the VMF condensation potential V(𝒲,T) = D(T²−T₀²)𝒲² − E·T·𝒲³ +
//! (λ/4)𝒲⁴, whose cubic term opens a barrier between the "off" (φ far
//! from any skill) and "on" (φ inside a skill's subspace) minima:
//!   - activate a skill only when its E drops below `e_on` (nucleation);
//!   - abandon the active skill only when its E rises above `e_off`
//!     (> e_on), or a rival beats it by more than `margin`.
//! The barrier `e_off − e_on` is exactly what suppresses thrashing at
//! domain boundaries (the very effect a single threshold cannot give).

use cortiq_core::quant::f16_to_f32;
use cortiq_core::SelectionDescriptor;
use base64::Engine as _;

/// One routable skill's precomputed subspace (decoded once).
pub struct RoutableSkill {
    /// Index into model.header.skills (pipeline.set_active_skill).
    pub idx: usize,
    pub id: String,
    pub phi_layer: usize,
    mean: Vec<f32>,
    basis: Vec<f32>,
    rank: usize,
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

impl RoutableSkill {
    pub fn from_descriptor(
        idx: usize,
        id: String,
        sel: &SelectionDescriptor,
        hidden: usize,
    ) -> Option<Self> {
        if sel.metric != "mse" {
            return None;
        }
        let mean = decode_f16(&sel.mean)?;
        let basis = decode_f16(&sel.basis)?;
        if mean.len() != hidden || basis.len() != sel.rank * hidden {
            return None;
        }
        Some(Self {
            idx,
            id,
            phi_layer: sel.phi_layer,
            mean,
            basis,
            rank: sel.rank,
        })
    }

    /// Normalized reconstruction error E = ‖r − BBᵀr‖²/‖φ‖²,
    /// r = φ − mean (identical math to router::route).
    pub fn error(&self, phi: &[f32]) -> f32 {
        let hidden = self.mean.len();
        if phi.len() != hidden {
            return f32::INFINITY;
        }
        let r: Vec<f32> = phi.iter().zip(&self.mean).map(|(p, m)| p - m).collect();
        let rr: f32 = r.iter().map(|v| v * v).sum();
        let pp: f32 = phi.iter().map(|v| v * v).sum();
        let mut proj = 0f32;
        for k in 0..self.rank {
            let row = &self.basis[k * hidden..(k + 1) * hidden];
            let c: f32 = row.iter().zip(&r).map(|(b, v)| b * v).sum();
            proj += c * c;
        }
        (rr - proj).max(0.0) / pp.max(1e-12)
    }
}

/// Hysteresis controller for dynamic routing.
pub struct DynRouter {
    pub skills: Vec<RoutableSkill>,
    /// Nucleation threshold: activate below this E.
    pub e_on: f32,
    /// Abandon threshold: drop the active skill above this E (> e_on).
    pub e_off: f32,
    /// A rival must beat the active skill by this margin to steal it.
    pub margin: f32,
    /// Re-route every `period` tokens (dispatch amortization; 1 = every).
    pub period: usize,
    /// Currently active skill index (model.header.skills), None = base.
    active: Option<usize>,
    tick: usize,
    /// Switch log for demo/telemetry: (token#, from_id, to_id).
    pub switches: Vec<(usize, Option<String>, Option<String>)>,
    /// Min recon error E at the last evaluation tick (telemetry): low E =
    /// high coherence with a skill subspace. INFINITY before any eval.
    last_best_e: f32,
}

impl DynRouter {
    pub fn new(skills: Vec<RoutableSkill>) -> Self {
        let e_on = std::env::var("CMF_ROUTE_EON").ok().and_then(|v| v.parse().ok()).unwrap_or(0.62);
        let e_off = std::env::var("CMF_ROUTE_EOFF").ok().and_then(|v| v.parse().ok()).unwrap_or(0.74);
        let margin = std::env::var("CMF_ROUTE_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.03);
        let period = std::env::var("CMF_ROUTE_PERIOD").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).max(1);
        Self {
            skills,
            e_on,
            e_off,
            margin,
            period,
            active: None,
            tick: 0,
            switches: Vec::new(),
            last_best_e: f32::INFINITY,
        }
    }

    /// The single phi_layer to capture (skills share it in the swarm;
    /// if they differ, the first is used and a warning is the caller's).
    pub fn phi_layer(&self) -> Option<usize> {
        self.skills.first().map(|s| s.phi_layer)
    }

    /// Decide the active skill for the next window given the current φ.
    /// Returns Some(new_active) when a switch is warranted (caller calls
    /// pipeline.set_active_skill), else None (unchanged). `token_no` is
    /// only for the switch log.
    pub fn step(&mut self, phi: &[f32], token_no: usize) -> Option<Option<usize>> {
        self.tick += 1;
        if self.tick % self.period != 0 || phi.is_empty() || self.skills.is_empty() {
            return None;
        }
        // Score all skills.
        let mut best_idx = None;
        let mut best_e = f32::INFINITY;
        let mut active_e = f32::INFINITY;
        for s in &self.skills {
            let e = s.error(phi);
            if Some(s.idx) == self.active {
                active_e = e;
            }
            if e < best_e {
                best_e = e;
                best_idx = Some(s.idx);
            }
        }

        self.last_best_e = best_e; // telemetry: coherence at this eval

        let next = decide(
            self.active, active_e, best_idx, best_e, self.e_on, self.e_off, self.margin,
        );

        if next != self.active {
            let from = self.active.and_then(|i| self.skills.iter().find(|s| s.idx == i)).map(|s| s.id.clone());
            let to = next.and_then(|i| self.skills.iter().find(|s| s.idx == i)).map(|s| s.id.clone());
            self.switches.push((token_no, from, to));
            self.active = next;
            return Some(next);
        }
        None
    }

    pub fn active(&self) -> Option<usize> {
        self.active
    }

    /// Id of the currently active skill (telemetry), None = backbone.
    pub fn active_id(&self) -> Option<String> {
        self.active
            .and_then(|i| self.skills.iter().find(|s| s.idx == i))
            .map(|s| s.id.clone())
    }

    /// Min recon error E at the last evaluation (telemetry coherence).
    pub fn last_best_e(&self) -> f32 {
        self.last_best_e
    }

    /// Reset per-generation state (active=backbone, empty log, tick 0) so
    /// the router matches a freshly-reset pipeline overlay.
    pub fn reset(&mut self) {
        self.active = None;
        self.tick = 0;
        self.switches.clear();
        self.last_best_e = f32::INFINITY;
    }
}

/// Pure hysteresis decision (first-order transition analogue): given the
/// current active skill, its error, and the best rival, return the next
/// active. Two thresholds e_on < e_off open the anti-thrash barrier.
#[allow(clippy::too_many_arguments)]
pub fn decide(
    active: Option<usize>,
    active_e: f32,
    best_idx: Option<usize>,
    best_e: f32,
    e_on: f32,
    e_off: f32,
    margin: f32,
) -> Option<usize> {
    match active {
        // Nucleation: activate the best only if it clears e_on.
        None => {
            if best_e < e_on {
                best_idx
            } else {
                None
            }
        }
        Some(cur) => {
            if active_e > e_off {
                // Melted: re-nucleate, else fall back to backbone.
                if best_e < e_on {
                    best_idx
                } else {
                    None
                }
            } else if best_idx != Some(cur) && best_e + margin < active_e {
                // Rival decisively better while active still holds.
                best_idx
            } else {
                Some(cur)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::decide;

    #[test]
    fn hysteresis_barrier_suppresses_thrashing() {
        let (e_on, e_off, m) = (0.60, 0.75, 0.03);

        // From backbone: does NOT activate in the barrier band [e_on,e_off).
        assert_eq!(decide(None, f32::INFINITY, Some(0), 0.70, e_on, e_off, m), None);
        // From backbone: activates below e_on (nucleation).
        assert_eq!(decide(None, f32::INFINITY, Some(0), 0.55, e_on, e_off, m), Some(0));

        // Active skill 0 at E=0.70 (in the band) STAYS — this is the whole
        // point: a single threshold at 0.62 would have flip-flopped here.
        assert_eq!(decide(Some(0), 0.70, Some(1), 0.68, e_on, e_off, m), Some(0));
        // Active melts above e_off → re-nucleate to the qualifying rival.
        assert_eq!(decide(Some(0), 0.80, Some(1), 0.55, e_on, e_off, m), Some(1));
        // Active melts but no rival clears e_on → back to backbone.
        assert_eq!(decide(Some(0), 0.80, Some(1), 0.70, e_on, e_off, m), None);
        // Rival must beat active by `margin`, not merely be lower.
        assert_eq!(decide(Some(0), 0.70, Some(1), 0.69, e_on, e_off, m), Some(0));
        assert_eq!(decide(Some(0), 0.70, Some(1), 0.66, e_on, e_off, m), Some(1));
    }
}
