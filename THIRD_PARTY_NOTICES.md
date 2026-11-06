# Third-party notices

## Splash

`crates/cortiq-engine/src/gpu_metal/ane.rs` and `crates/cortiq-engine/src/gpu_metal/ane_msl.metal` are
derived from [Splash](https://github.com/incoai/splash) by Inco AI: `runtime/ane/Program.mm`,
`runtime/ops/AneFfn.cpp` and `runtime/metal/kernels/prefill/ane_ffn.metal`. Splash is under the Apache
License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0

What changed: the client and the program text are written in Rust. The weights are decoded from `q4tp` on the CPU,
with no Hadamard rotation and fp16 activations, and the Neural Engine takes all of a layer's channels.
