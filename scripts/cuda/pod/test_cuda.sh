#!/bin/bash
# The CUDA backend's unit tests on the pod's card (kernel parity against the
# host codecs included), on the iteration profile.
root=${CUDA_RUN_ROOT:-/root/cuda}
. "$HOME/.cargo/env"
cd "$root/src" || exit 1
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true
cargo test --release -p cortiq-engine --features cuda --lib --target-dir target-fast gpu_cuda -- --nocapture 2>&1 |
  grep -E "^error|error\[|^test |test result|panicked|cuda:|assert|off by|differs" | tail -30
echo TEST_CUDA_DONE
