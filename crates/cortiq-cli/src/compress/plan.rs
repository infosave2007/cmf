//! S2: how wide each layer's FFN stays.
//!
//! The target is a total number of kept FFN neurons. `uniform` gives every
//! compressible layer the same width; `curve` removes neurons, one `align`
//! step at a time, from whichever layer's relative refit error grows least
//! for that step (the curves come from S1). Protected layers keep their
//! full width. Widths stay multiples of `align` and at least `min_width` of
//! the original.

use anyhow::{Result, bail, ensure};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct LayerPlan {
    pub layer: usize,
    pub width: usize,
    pub keep: usize,
    pub protected: bool,
    /// Relative output error after the refit at this width (train curve).
    pub cost: f64,
    /// Train tokens per kept neuron (refit sample size, F2).
    pub tokens_per_unit: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Plan {
    pub hidden: usize,
    pub layers: Vec<LayerPlan>,
    pub ffn_params_before: u64,
    pub ffn_params_after: u64,
    pub params_before: u64,
    pub params_after: u64,
    pub warnings: Vec<String>,
}

/// How the target was given.
#[derive(Clone, Debug)]
pub enum Target {
    /// Fraction of all FFN neurons kept, or an absolute mean width.
    Ffn(f64),
    FfnWidth(usize),
    /// Total parameters (without the MTP draft) after compression.
    Params(u64),
    /// Fraction of the source parameters.
    Ratio(f64),
}

/// `12B`, `900M`, `1.5e9` → parameters.
pub fn parse_size(s: &str) -> Result<u64> {
    let t = s.trim();
    let (num, mul) = match t.chars().last() {
        Some('B' | 'b') => (&t[..t.len() - 1], 1e9),
        Some('M' | 'm') => (&t[..t.len() - 1], 1e6),
        Some('K' | 'k') => (&t[..t.len() - 1], 1e3),
        _ => (t, 1.0),
    };
    let v: f64 = num.parse().map_err(|_| anyhow::anyhow!("--size {s:?}: expected e.g. 12B, 900M"))?;
    ensure!(v > 0.0, "--size must be positive");
    Ok((v * mul) as u64)
}

pub struct PlanInput<'a> {
    pub hidden: usize,
    /// Original width per layer.
    pub widths: &'a [usize],
    /// Refit curves per layer (`curves[l][k]`, length width+1); None = no
    /// data (uniform only).
    pub curves: Option<&'a [Vec<f64>]>,
    pub protect: &'a [bool],
    pub align: usize,
    pub min_width: f64,
    pub uniform: bool,
    /// Parameters outside the trunk FFNs (embeddings, attention, norms…).
    pub other_params: u64,
    pub train_tokens: usize,
}

pub fn allocate(inp: &PlanInput, target: &Target) -> Result<Plan> {
    let h = inp.hidden;
    let nl = inp.widths.len();
    let per_neuron = 3 * h as u64;
    let ffn_before: u64 = inp.widths.iter().map(|&w| w as u64 * per_neuron).sum();
    let total_neurons: usize = inp.widths.iter().sum();
    let want_neurons: usize = match *target {
        Target::Ffn(f) => {
            ensure!(f > 0.0 && f <= 1.0, "--ffn fraction must be in (0, 1]");
            (f * total_neurons as f64).round() as usize
        }
        Target::FfnWidth(w) => w * nl,
        Target::Params(p) => {
            ensure!(p > inp.other_params, "--size {p} is below the parameters outside the FFNs ({})", inp.other_params);
            ((p - inp.other_params) / per_neuron) as usize
        }
        Target::Ratio(r) => {
            ensure!(r > 0.0 && r <= 1.0, "--ratio must be in (0, 1]");
            let p = ((inp.other_params + ffn_before) as f64 * r) as u64;
            ensure!(p > inp.other_params, "--ratio {r} leaves nothing for the FFNs (non-FFN parameters alone are {})", inp.other_params);
            ((p - inp.other_params) / per_neuron) as usize
        }
    };
    let align = inp.align.max(1);
    let floor = |w: usize| -> usize {
        let m = ((w as f64 * inp.min_width).ceil() as usize).div_ceil(align) * align;
        m.clamp(align.min(w), w)
    };
    let mut keep: Vec<usize> = inp.widths.to_vec();
    let fixed: usize = (0..nl).filter(|&l| inp.protect[l]).map(|l| inp.widths[l]).sum();
    let min_total: usize = (0..nl).map(|l| if inp.protect[l] { inp.widths[l] } else { floor(inp.widths[l]) }).sum();
    if want_neurons < min_total {
        bail!(
            "target needs {want_neurons} FFN neurons but protected layers and the --min-width floor already keep {min_total}; \
             lower --min-width, protect fewer layers, or aim higher"
        );
    }
    if want_neurons < total_neurons {
        let free: Vec<usize> = (0..nl).filter(|&l| !inp.protect[l]).collect();
        ensure!(!free.is_empty(), "every layer is protected — nothing to compress");
        match (inp.uniform, inp.curves) {
            (false, Some(curves)) => {
                let mut cur: usize = keep.iter().sum();
                while cur > want_neurons {
                    // cheapest next step: smallest error growth per neuron removed
                    let mut best: Option<(f64, usize, usize)> = None;
                    for &l in &free {
                        let k = keep[l];
                        let step = if k % align != 0 { k % align } else { align };
                        if k < step || k - step < floor(inp.widths[l]) {
                            continue;
                        }
                        let c = &curves[l];
                        let growth = (c[k - step] - c[k]) / step as f64;
                        if best.is_none_or(|b| growth < b.0) {
                            best = Some((growth, l, step));
                        }
                    }
                    let Some((_, l, step)) = best else { break };
                    keep[l] -= step;
                    cur -= step;
                }
            }
            _ => {
                // the same aligned width for every free layer
                let budget = want_neurons - fixed;
                let n_free = free.len();
                for &l in &free {
                    let w = ((budget / n_free) / align * align).clamp(floor(inp.widths[l]), inp.widths[l]);
                    keep[l] = w;
                }
            }
        }
    }
    let ffn_after: u64 = keep.iter().map(|&k| k as u64 * per_neuron).sum();
    let mut warnings = Vec::new();
    let layers: Vec<LayerPlan> = (0..nl)
        .map(|l| {
            let cost = inp.curves.map_or(f64::NAN, |c| c[l][keep[l]]);
            let tpu = inp.train_tokens as f64 / keep[l].max(1) as f64;
            LayerPlan { layer: l, width: inp.widths[l], keep: keep[l], protected: inp.protect[l], cost, tokens_per_unit: tpu }
        })
        .collect();
    let kept_frac = keep.iter().sum::<usize>() as f64 / total_neurons as f64;
    if kept_frac < 0.75 {
        warnings.push(format!(
            "{:.0}% of the FFN neurons are kept: below ~75% a narrowed model needs healing (measured: 27B at 75% width +16% ppl unhealed)",
            kept_frac * 100.0
        ));
    }
    Ok(Plan {
        hidden: h,
        layers,
        ffn_params_before: ffn_before,
        ffn_params_after: ffn_after,
        params_before: inp.other_params + ffn_before,
        params_after: inp.other_params + ffn_after,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(w: usize, steep: f64) -> Vec<f64> {
        (0..=w).map(|k| ((w - k) as f64 / w as f64).powf(steep)).collect()
    }

    #[test]
    fn curve_allocation_hits_budget_and_respects_protection() {
        let widths = vec![512; 4];
        let curves = vec![curve(512, 1.0), curve(512, 3.0), curve(512, 3.0), curve(512, 1.0)];
        let protect = vec![true, false, false, false];
        let p = allocate(
            &PlanInput {
                hidden: 64,
                widths: &widths,
                curves: Some(&curves),
                protect: &protect,
                align: 64,
                min_width: 0.25,
                uniform: false,
                other_params: 0,
                train_tokens: 10_000,
            },
            &Target::Ffn(0.75),
        )
        .unwrap();
        let ks: Vec<usize> = p.layers.iter().map(|l| l.keep).collect();
        assert_eq!(ks[0], 512);
        assert_eq!(ks.iter().sum::<usize>(), 1536);
        assert!(ks.iter().all(|k| k % 64 == 0 && *k >= 128));
        // the steep (cheap-to-cut) layers give up more than the linear one
        assert!(ks[1] < ks[3] && ks[2] < ks[3], "{ks:?}");
    }

    #[test]
    fn size_parsing() {
        assert_eq!(parse_size("12B").unwrap(), 12_000_000_000);
        assert_eq!(parse_size("900M").unwrap(), 900_000_000);
        assert!(parse_size("x").is_err());
    }
}
