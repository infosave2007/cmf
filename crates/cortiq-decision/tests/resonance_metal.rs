//! Hardware contracts shared with Vulkan; run explicitly on Apple Silicon.
#![cfg(target_os = "macos")]
use cortiq_decision::bert::EncoderDevice;
#[path = "support/resonance_gpu.rs"]
mod contracts;

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn metal_reconstruction_matches_reference_with_rank_and_dimension_tails() {
    contracts::reconstruction_parity(EncoderDevice::Metal);
}
