//! The two quantities a Looped Transformer breaks in the DTG-MA bake,
//! pinned by closed form rather than by a three-hour run on a stand.
//!
//! Both defects below were found by baking a real 4 B model and reading
//! perplexity an hour later. Both are arithmetic, and both are decided
//! in microseconds here. That is the whole point: a bake that starts
//! wrong cannot be diagnosed from its own output, because every later
//! number is a compensation for the first one.
//!
//! Patent 2 (application 19/452,464) fixes what these tests assert: the
//! backbone weights are FROZEN and the mask is what learns, so step zero
//! must BE the backbone — at any loop depth — and one training step must
//! mean one token's worth of movement, not one per visit.

use cortiq_engine::skillbake::{mask_init_logit, mask_step_scale};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The gate's EFFECTIVE start — what the stack multiplies by over a
/// whole token — must not depend on how many times the loop runs.
///
/// The old code hardcoded logit 2.0. One visit gives σ = 0.881, which
/// the recipe was tuned around; two visits compound it to 0.776, and on
/// Nanbeige 4.2 (22 layers x 2) that took held-out perplexity from a
/// baseline of 4.187 to 278.371 by step 30 with nothing yet pruned. The
/// mask had not learned anything wrong — it had simply turned the model
/// down before training began.
#[test]
fn the_effective_start_is_the_same_at_every_loop_depth() {
    let want = sigmoid(2.0);
    for loops in 1..=6 {
        let m0 = mask_init_logit(loops);
        let effective = sigmoid(m0).powi(loops as i32);
        let rel = (effective - want).abs() / want;
        println!(
            "loops={loops}: m0={m0:.4} σ={:.4} effective={effective:.6} (want {want:.6})",
            sigmoid(m0)
        );
        assert!(
            rel < 1e-4,
            "loops={loops}: effective start {effective} is not σ(2.0)={want}"
        );
    }
}

/// One loop must reproduce the old constant exactly, or every model that
/// is not looped silently changes behaviour.
#[test]
fn a_single_loop_reproduces_the_validated_constant() {
    let m0 = mask_init_logit(1);
    assert!(
        (m0 - 2.0).abs() < 1e-5,
        "an unlooped model must still start at 2.0, got {m0}"
    );
    assert_eq!(mask_step_scale(1), 1.0, "an unlooped step must be unscaled");
}

/// The start must stay inside the sigmoid's live range.
///
/// The first attempt at this fix pushed the effective start to 0.999 —
/// exactly the backbone, and immovable: the update carries
/// σ'(m) = σ(1−σ), which is 5e-4 there against 0.105 at 2.0, a
/// two-hundred-fold smaller step. Measured on the same model: 60 steps,
/// 0% pruned, held-out perplexity equal to the baseline to three digits.
/// An initialisation that cannot learn is not an improvement, so the
/// gradient it leaves behind is an invariant too.
#[test]
fn the_start_leaves_a_usable_gradient() {
    for loops in 1..=4 {
        let s = sigmoid(mask_init_logit(loops));
        let dsigma = s * (1.0 - s);
        println!("loops={loops}: σ={s:.4} σ'={dsigma:.5}");
        assert!(
            dsigma > 0.02,
            "loops={loops}: σ'={dsigma:.5} is too flat to train (identity trap)"
        );
    }
}

/// A step must mean the same thing at any loop depth.
///
/// The backward accumulates every visit of a physical layer into the
/// same mask gradient, so at `loops` visits an unscaled Adam step is
/// `loops` times the tuned one. Left unscaled, neurons cross τ inside
/// the first evaluation window and each one is then missing from both
/// passes — which is why fixing only the start halved the damage
/// (278.4 → 134.9) instead of removing it.
#[test]
fn the_mask_step_is_normalised_by_the_visit_count() {
    for loops in 1..=8 {
        let scale = mask_step_scale(loops);
        assert!(
            (scale * loops as f64 - 1.0).abs() < 1e-12,
            "loops={loops}: scale {scale} does not undo the per-visit accumulation"
        );
    }
    // Degenerate input must not divide by zero.
    assert_eq!(mask_step_scale(0), 1.0, "loops=0 must be treated as 1");
}

/// Monotonicity, as a guard against a future edit that swaps a power for
/// a product: more visits mean each visit must open wider, never less.
#[test]
fn deeper_loops_open_the_per_visit_gate_wider() {
    let mut prev = 0f32;
    for loops in 1..=6 {
        let s = sigmoid(mask_init_logit(loops));
        assert!(
            s > prev,
            "loops={loops}: per-visit gate {s} did not open past {prev}"
        );
        assert!(s < 1.0, "loops={loops}: gate saturated at {s}");
        prev = s;
    }
}
