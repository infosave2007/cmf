#!/bin/bash
# Build cortiq with the CUDA backend on the pod.
#   build_cuda.sh fast     iteration build: no LTO, 16 codegen units, its own target dir
#   build_cuda.sh release  the release profile, what the measurements run
set -u
mode=${1:-fast}
root=${CUDA_RUN_ROOT:-/root/cuda}
. "$HOME/.cargo/env"
cd "$root/src" || exit 1
mkdir -p "$root/bin"
if [ "$mode" = fast ]; then
  export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true
  cargo build --release -p cortiq-cli --features cuda --target-dir target-fast 2>&1 | grep -E "^error|^warning: unused|gpu_cuda|Finished|could not compile" | tail -40
  cp target-fast/release/cortiq "$root/bin/cortiq-cuda-fast.new" && mv "$root/bin/cortiq-cuda-fast.new" "$root/bin/cortiq-cuda-fast" && echo BUILD_FAST_DONE
else
  cargo build --release -p cortiq-cli --features cuda 2>&1 | grep -E "^error|gpu_cuda|Finished|could not compile" | tail -40
  cp target/release/cortiq "$root/bin/cortiq-cuda.new" && mv "$root/bin/cortiq-cuda.new" "$root/bin/cortiq-cuda" && echo BUILD_RELEASE_DONE
fi
