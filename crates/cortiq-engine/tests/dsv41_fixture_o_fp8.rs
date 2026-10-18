//! `CMF_DSV41_O_FP8=1` on the V4.1 fixture: the CPU attention tail
//! quantizes the attention output to FP8 before the grouped `wo_a`
//! (reference commit 4dd75a8b), and the logits follow the reference with
//! `act_quant(o)` instead of the one without it.
//!
//!     cargo test -p cortiq-engine --test dsv41_fixture_o_fp8 -- --nocapture
//!
//! Its own test binary: the switch is read once per process.

// The fixture's `json!` config nests past the default macro depth.
#![recursion_limit = "256"]

#[path = "common/dsv41_fixture.rs"]
mod dsv41_fixture;

use cortiq_engine::dsv41::{self, Dsv41Cfg, Dsv41State};
use cortiq_engine::qtensor::float_activations_scope;
use dsv41_fixture::*;

#[test]
fn o_fp8_follows_reference_act_quant() {
    // SAFETY: the only test in this binary, before any V4.1 code runs.
    // CMF_GPU=0: a workspace build carries the gpu feature; keep the MoE on
    // the host as well so the comparison is the strict CPU path.
    unsafe {
        std::env::set_var("CMF_DSV41_O_FP8", "1");
        std::env::set_var("CMF_GPU", "0");
    }
    let fx = Fixture::build(RELEASE_COMPRESSED_VOCAB);
    let model = fx.open("ofp8");
    let source = model.arch().deepseek_v41.clone().expect("source config");
    let (cfg, n_layers) = Dsv41Cfg::from_source(&source, model.arch()).expect("parse cfg");
    let got = float_activations_scope(|| {
        let (globals, layers, hash) =
            dsv41::load(&model, &cfg, n_layers, fx.token_map()).expect("load layers");
        let mut st = Dsv41State::new(&cfg, hash);
        TOKENS
            .iter()
            .enumerate()
            .map(|(p, &t)| {
                let mut logits = Vec::new();
                dsv41::forward_token(&globals, &layers, &cfg, &mut st, t, p, None, &mut logits);
                logits
            })
            .collect::<Vec<_>>()
    });
    let with = reference_logits(
        &fx,
        &TOKENS,
        RefOpts {
            o_fp8: true,
            ..RefOpts::cortiq()
        },
    );
    let without = reference_logits(&fx, &TOKENS, RefOpts::cortiq());
    let exact = got
        .iter()
        .zip(&with)
        .filter(|(g, w)| worst_rel(std::slice::from_ref(*g), std::slice::from_ref(*w)).0 < 1e-5)
        .count();
    let (worst, at) = worst_rel(&got, &with);
    let (apart, _) = worst_rel(&got, &without);
    println!(
        "dsv41 fixture O_FP8: {exact}/{} positions exact against act_quant(o) reference, \
         worst {worst:.3e} at {at}; {apart:.3e} away from the reference without it",
        TOKENS.len()
    );
    assert!(exact * 3 >= TOKENS.len() * 2, "{exact}/{} exact", TOKENS.len());
    assert!(worst < 0.15, "O_FP8 forward: {worst:.3e} at {at}");
    assert!(apart > 1e-4, "act_quant(o) did not change the logits");
}
