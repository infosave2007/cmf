//! What a dependent dispatch costs on Metal: N dispatches in ONE command
//! buffer, each reading the previous one's output, in a serial encoder
//! (implicit hazards), in a concurrent encoder with a buffer barrier between
//! every pair, and in a concurrent encoder without barriers (independent
//! work, the floor). The kernel is tiny, so the time per dispatch is the
//! launch + drain overhead a decode token pays per stage.
//!
//! cargo run --release -p cortiq-engine --example metal_barrier_bench [-- N TGS]

#[cfg(target_os = "macos")]
fn main() {
    use metal::*;
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let n = *args.first().unwrap_or(&300);
    let tgs = *args.get(1).unwrap_or(&14) as u64;
    let dev = Device::system_default().expect("no Metal device");
    let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void step(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] = x[i] * 0.999f + 1.0f;
}"#;
    let lib = dev
        .new_library_with_source(src, &CompileOptions::new())
        .expect("compile");
    let pso = dev
        .new_compute_pipeline_state_with_function(&lib.get_function("step", None).unwrap())
        .unwrap();
    let q = dev.new_command_queue();
    let buf = dev.new_buffer(tgs * 256 * 4, MTLResourceOptions::StorageModeShared);
    let barrier = |enc: &ComputeCommandEncoderRef| {
        use metal::objc::{msg_send, sel, sel_impl};
        let scope: u64 = 1;
        let _: () = unsafe { msg_send![enc, memoryBarrierWithScope: scope] };
    };
    let run = |mode: u8| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..7 {
            let cmd = q.new_command_buffer();
            let enc = if mode == 0 {
                cmd.new_compute_command_encoder()
            } else {
                cmd.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
            };
            enc.set_compute_pipeline_state(&pso);
            enc.set_buffer(0, Some(&buf), 0);
            for k in 0..n {
                enc.dispatch_thread_groups(MTLSize::new(tgs, 1, 1), MTLSize::new(256, 1, 1));
                if mode == 1 && k + 1 < n {
                    barrier(enc);
                }
            }
            enc.end_encoding();
            let t0 = std::time::Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            let gpu = unsafe {
                use metal::foreign_types::ForeignTypeRef;
                use metal::objc::{msg_send, sel, sel_impl};
                let p = cmd.as_ptr();
                let s: f64 = msg_send![p, GPUStartTime];
                let e: f64 = msg_send![p, GPUEndTime];
                (e - s) * 1e6
            };
            let wall = t0.elapsed().as_secs_f64() * 1e6;
            best = best.min(if gpu > 0.0 { gpu } else { wall });
        }
        best / n as f64
    };
    println!(
        "{n} dispatches of {tgs} threadgroups x 256 threads, GPU time per dispatch (best of 7):"
    );
    println!("  serial encoder (implicit hazards):  {:6.2} us", run(0));
    println!("  concurrent + barrier every pair:    {:6.2} us", run(1));
    println!("  concurrent, no barriers (floor):    {:6.2} us", run(2));
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("metal_barrier_bench runs on macOS only");
}
