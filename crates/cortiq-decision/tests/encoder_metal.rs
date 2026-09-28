//! Hardware contracts shared with Vulkan; run explicitly on Apple Silicon.
#![cfg(target_os = "macos")]
use cortiq_decision::bert::EncoderDevice;
#[path = "support/encoder_gpu.rs"]
mod contracts;

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn metal_matches_cpu_and_serializes_concurrent_requests() {
    contracts::encoder_parity_concurrency(EncoderDevice::Metal);
}

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn metal_handles_linear_tails_and_unfused_attention_heads() {
    contracts::encoder_tails_heads(EncoderDevice::Metal);
}
