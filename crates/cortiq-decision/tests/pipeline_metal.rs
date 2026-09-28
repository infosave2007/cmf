//! Hardware contracts shared with Vulkan; run explicitly on Apple Silicon.
#![cfg(target_os = "macos")]
use cortiq_decision::bert::EncoderDevice;
#[path = "support/pipeline_gpu.rs"]
mod contracts;

#[test]
#[ignore = "requires an Apple Silicon Metal device"]
fn joint_pipeline_matches_cpu_handles_mixed_devices_and_lock_order() {
    contracts::joint_pipeline_contract(EncoderDevice::Metal);
}
