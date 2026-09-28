//! Execute these contracts on a real, explicitly selected Vulkan GPU.
#![cfg(feature = "vulkan")]
use cortiq_decision::bert::EncoderDevice;
#[path = "support/encoder_gpu.rs"]
mod encoder_gpu;
#[path = "support/pipeline_gpu.rs"]
mod pipeline_gpu;
#[path = "support/resonance_gpu.rs"]
mod resonance_gpu;

#[test]
#[ignore = "requires a hardware Vulkan adapter"]
fn vulkan_encoder_parity_concurrency() {
    encoder_gpu::encoder_parity_concurrency(EncoderDevice::Vulkan);
}

#[test]
#[ignore = "requires a hardware Vulkan adapter"]
fn vulkan_encoder_tails_heads() {
    encoder_gpu::encoder_tails_heads(EncoderDevice::Vulkan);
}

#[test]
#[ignore = "requires a hardware Vulkan adapter"]
fn vulkan_reconstruction_parity() {
    resonance_gpu::reconstruction_parity(EncoderDevice::Vulkan);
}

#[test]
#[ignore = "requires a hardware Vulkan adapter"]
fn vulkan_joint_pipeline_contract() {
    pipeline_gpu::joint_pipeline_contract(EncoderDevice::Vulkan);
}
