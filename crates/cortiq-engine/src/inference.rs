//! CPU forward-pass primitives: RMS-norm, SiLU, sparse SwiGLU FFN.

use crate::pool::Pool;
use cortiq_core::types::NormStyle;

/// SiLU activation function.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// RMS normalization with explicit weight semantics.
///
/// - `NormStyle::Qwen`  (Llama family): `x̂ · w`
/// - `NormStyle::Gemma`:                `x̂ · (1 + w)`
///
/// Applying the wrong style corrupts every normalization in the
/// forward pass — the style comes from the model arch, never guessed.
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f64, style: NormStyle) -> Vec<f32> {
    let n = input.len();
    let mean_sq: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n as f64;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt() as f32;
    match style {
        NormStyle::Qwen => input
            .iter()
            .zip(weight)
            .map(|(&x, &w)| x * inv_rms * w)
            .collect(),
        NormStyle::Gemma => input
            .iter()
            .zip(weight)
            .map(|(&x, &w)| x * inv_rms * (1.0 + w))
            .collect(),
    }
}

/// Sparse SwiGLU FFN: compute only `active_indices` neurons.
///
/// `out = down_projᵀ[·, active] · (silu(gate_proj[active, ·]·h) ⊙ up_proj[active, ·]·h)`
///
/// With a full index list this is bit-identical to the dense path —
/// masking is an execution schedule, not an approximation.
pub fn sparse_ffn_forward(
    hidden_states: &[f32],
    gate_proj_full: &[f32], // [intermediate_size, hidden_size]
    up_proj_full: &[f32],   // [intermediate_size, hidden_size]
    down_proj_full: &[f32], // [hidden_size, intermediate_size]
    hidden_size: usize,
    intermediate_size: usize,
    active_indices: &[u16],
    pool: Option<&Pool>,
) -> Vec<f32> {
    let n_active = active_indices.len();
    if n_active == 0 {
        return vec![0.0; hidden_states.len()];
    }

    let seq_len = hidden_states.len() / hidden_size;
    let mut output = vec![0.0f32; seq_len * hidden_size];

    for s in 0..seq_len {
        let h = &hidden_states[s * hidden_size..(s + 1) * hidden_size];

        // Gather active rows of gate/up, fuse silu(gate)·up. Each active
        // neuron is independent → row-parallel, bit-identical to serial.
        let mut act = vec![0.0f32; n_active];
        let neuron_act = |ai: usize| -> f32 {
            let idx = active_indices[ai];
            let row = idx as usize * hidden_size;
            if row + hidden_size > gate_proj_full.len() {
                return 0.0;
            }
            let mut gate_sum = 0.0f32;
            let mut up_sum = 0.0f32;
            for k in 0..hidden_size {
                gate_sum += gate_proj_full[row + k] * h[k];
                up_sum += up_proj_full[row + k] * h[k];
            }
            silu(gate_sum) * up_sum
        };
        match pool {
            Some(pool) if n_active >= 256 => {
                let act_ptr = SendMut(act.as_mut_ptr());
                pool.run(&move |widx, n| {
                    let chunk = n_active.div_ceil(n);
                    let start = widx * chunk;
                    let end = (start + chunk).min(n_active);
                    for ai in start..end {
                        // SAFETY: disjoint index ranges per worker.
                        unsafe { *act_ptr.at(ai) = neuron_act(ai) };
                    }
                });
            }
            _ => {
                for (ai, dst) in act.iter_mut().enumerate() {
                    *dst = neuron_act(ai);
                }
            }
        }

        // Scatter through the active columns of down_proj.
        let out = &mut output[s * hidden_size..(s + 1) * hidden_size];
        for (ai, &idx) in active_indices.iter().enumerate() {
            let val = act[ai];
            if val.abs() < 1e-12 {
                continue;
            }
            for k in 0..hidden_size {
                out[k] += down_proj_full[k * intermediate_size + idx as usize] * val;
            }
        }
    }

    output
}

/// Fused two-position sparse FFN: gate/up/down weight rows are streamed
/// from memory ONCE for both positions. Bit-identical to two single
/// calls (per-position accumulation order is unchanged).
#[allow(clippy::too_many_arguments)]
pub fn sparse_ffn_forward_pair(
    h1: &[f32],
    h2: &[f32],
    gate_proj_full: &[f32],
    up_proj_full: &[f32],
    down_proj_full: &[f32],
    hidden_size: usize,
    intermediate_size: usize,
    active_indices: &[u16],
    pool: Option<&Pool>,
) -> (Vec<f32>, Vec<f32>) {
    let n_active = active_indices.len();
    let mut out1 = vec![0.0f32; hidden_size];
    let mut out2 = vec![0.0f32; hidden_size];
    if n_active == 0 {
        return (out1, out2);
    }

    let mut act1 = vec![0.0f32; n_active];
    let mut act2 = vec![0.0f32; n_active];
    let neuron_pair = |ai: usize| -> (f32, f32) {
        let idx = active_indices[ai];
        let row = idx as usize * hidden_size;
        if row + hidden_size > gate_proj_full.len() {
            return (0.0, 0.0);
        }
        let (mut g1, mut u1, mut g2, mut u2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for k in 0..hidden_size {
            let gw = gate_proj_full[row + k];
            let uw = up_proj_full[row + k];
            g1 += gw * h1[k];
            g2 += gw * h2[k];
            u1 += uw * h1[k];
            u2 += uw * h2[k];
        }
        (silu(g1) * u1, silu(g2) * u2)
    };
    match pool {
        Some(pool) if n_active >= 256 => {
            let a1 = SendMut(act1.as_mut_ptr());
            let a2 = SendMut(act2.as_mut_ptr());
            pool.run(&move |widx, n| {
                let chunk = n_active.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(n_active);
                for ai in start..end {
                    let (v1, v2) = neuron_pair(ai);
                    // SAFETY: disjoint index ranges per worker.
                    unsafe {
                        *a1.at(ai) = v1;
                        *a2.at(ai) = v2;
                    }
                }
            });
        }
        _ => {
            for ai in 0..n_active {
                let (v1, v2) = neuron_pair(ai);
                act1[ai] = v1;
                act2[ai] = v2;
            }
        }
    }

    // Down scatter: one pass over the active columns for both positions.
    for (ai, &idx) in active_indices.iter().enumerate() {
        let (v1, v2) = (act1[ai], act2[ai]);
        if v1.abs() < 1e-12 && v2.abs() < 1e-12 {
            continue;
        }
        for k in 0..hidden_size {
            let dw = down_proj_full[k * intermediate_size + idx as usize];
            out1[k] += dw * v1;
            out2[k] += dw * v2;
        }
    }
    (out1, out2)
}

#[derive(Clone, Copy)]
struct SendMut(*mut f32);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}

impl SendMut {
    /// See pool::SendMut::at — captures the Sync wrapper, not the field.
    #[inline]
    fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-4);
    }

    #[test]
    fn rms_norm_qwen_multiplies_by_w() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4]; // identity weight in Qwen semantics
        let out = rms_norm(&input, &weight, 1e-6, NormStyle::Qwen);
        let rms = (30.0_f64 / 4.0).sqrt() as f32;
        assert!((out[0] - 1.0 / rms).abs() < 1e-4);

        // w = 2 doubles the output — x̂·w, not x̂·(1+w).
        let out2 = rms_norm(&input, &vec![2.0; 4], 1e-6, NormStyle::Qwen);
        assert!((out2[0] - 2.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn rms_norm_gemma_adds_one() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![0.0; 4]; // identity weight in Gemma semantics
        let out = rms_norm(&input, &weight, 1e-6, NormStyle::Gemma);
        let rms = (30.0_f64 / 4.0).sqrt() as f32;
        assert!((out[0] - 1.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_sparse_ffn_full_active() {
        let hidden = 4;
        let inter = 8;
        let h = vec![1.0f32; hidden];
        let gate = vec![0.1f32; inter * hidden];
        let up = vec![0.1f32; inter * hidden];
        let down = vec![0.1f32; hidden * inter];
        let active: Vec<u16> = (0..inter as u16).collect();

        let out = sparse_ffn_forward(&h, &gate, &up, &down, hidden, inter, &active, None);
        assert_eq!(out.len(), hidden);
        assert!(out.iter().all(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn test_sparse_ffn_half_active() {
        let hidden = 4;
        let inter = 8;
        let h = vec![1.0f32; hidden];
        let gate = vec![0.1f32; inter * hidden];
        let up = vec![0.1f32; inter * hidden];
        let down = vec![0.1f32; hidden * inter];

        let full: Vec<u16> = (0..inter as u16).collect();
        let half: Vec<u16> = (0..inter as u16 / 2).collect();

        let out_full = sparse_ffn_forward(&h, &gate, &up, &down, hidden, inter, &full, None);
        let out_half = sparse_ffn_forward(&h, &gate, &up, &down, hidden, inter, &half, None);

        let mag_full: f32 = out_full.iter().map(|v| v.abs()).sum();
        let mag_half: f32 = out_half.iter().map(|v| v.abs()).sum();
        assert!(mag_half < mag_full);
        assert!(mag_half > 0.0);
    }
}
