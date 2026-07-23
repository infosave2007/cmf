//! Calibration-time Hessian capture for GPTQ / the holographic transfer.
//!
//! While `begin()` is active, every `QTensor::matmat` (the batched prefill
//! path) folds its input activations into a per-tensor second-moment
//! `H = Σ_t x_t·x_tᵀ` and per-channel `Σ x²` (for the activation RMS field
//! of the two-field outlier score). One global hook covers every linear in
//! the model — attention projections, FFN, experts — with no per-layer
//! wiring. Zero cost when off (one relaxed atomic load). Single-threaded
//! accumulation from the caller's thread, so it never races the parallel
//! matmat that follows.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static ON: AtomicBool = AtomicBool::new(false);
/// Accumulate the full `H = X·Xᵀ` (needed only for the GPTQ fold). Off ⇒
/// diagonal-only (`Σx²`), which is all the ternary + per-row correction
/// path needs — and the only thing that fits for a 12B (full H would be
/// ~100 GB across the model).
static FULL_H: AtomicBool = AtomicBool::new(true);
static REG: Mutex<Option<HashMap<String, HessianAcc>>> = Mutex::new(None);

/// Accumulated input statistics of one linear layer over the calibration set.
pub struct HessianAcc {
    pub cols: usize,
    /// `H = Σ_t x_t·x_tᵀ`, dense `[cols·cols]`, f64.
    pub h: Vec<f64>,
    /// `Σ_t x_t²` per input channel `[cols]` — RMS = sqrt(sumsq / count).
    pub sumsq: Vec<f64>,
    /// Token-position samples folded in.
    pub count: usize,
}

impl HessianAcc {
    /// Per-input-channel activation RMS.
    pub fn rms(&self) -> Vec<f32> {
        let n = self.count.max(1) as f64;
        self.sumsq.iter().map(|&s| (s / n).sqrt() as f32).collect()
    }
}

/// Start capturing (clears any prior registry). `full_h` = accumulate the
/// dense Hessian (fold path); false = diagonal-only (ternary + correction).
pub fn begin(full_h: bool) {
    *REG.lock().unwrap() = Some(HashMap::new());
    FULL_H.store(full_h, Ordering::SeqCst);
    ON.store(true, Ordering::SeqCst);
}

/// Stop capturing and take the accumulated Hessians.
pub fn end() -> HashMap<String, HessianAcc> {
    ON.store(false, Ordering::SeqCst);
    REG.lock().unwrap().take().unwrap_or_default()
}

#[inline]
pub fn capturing() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Fold a batch of `b` input vectors (`xs = [b·cols]`) into `name`'s Hessian.
pub fn accumulate(name: &str, xs: &[f32], b: usize, cols: usize) {
    let mut guard = REG.lock().unwrap();
    let Some(map) = guard.as_mut() else {
        return;
    };
    let full = FULL_H.load(Ordering::Relaxed);
    let acc = map.entry(name.to_string()).or_insert_with(|| HessianAcc {
        cols,
        h: if full {
            vec![0.0; cols * cols]
        } else {
            Vec::new()
        },
        sumsq: vec![0.0; cols],
        count: 0,
    });
    if acc.cols != cols {
        return; // shape mismatch — skip defensively
    }
    for bi in 0..b {
        let x = &xs[bi * cols..(bi + 1) * cols];
        for i in 0..cols {
            let xi = x[i] as f64;
            if xi == 0.0 {
                continue;
            }
            acc.sumsq[i] += xi * xi;
            if full {
                let hrow = &mut acc.h[i * cols..i * cols + cols];
                for (j, &xj) in x.iter().enumerate() {
                    hrow[j] += xi * xj as f64;
                }
            }
        }
        acc.count += 1;
    }
}
