#!/bin/bash
# Build cortiq (wgpu + CUDA) on the pod.
#   build.sh fast     iteration build: no LTO, 16 codegen units, its own target dir
#   build.sh release  the release profile, what the measurements run
set -u
mode=${1:-fast}
root=${H3I_ROOT:-/root/h3i}
. "$HOME/.cargo/env"
cd "$root/srcrs" || exit 1
mkdir -p "$root/bin"
if [ "$mode" = fast ]; then
  export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true
  cargo build --release -p cortiq-cli --features cuda --target-dir target-fast 2>&1 | grep -E "^error|Finished|could not compile" -A12 | tail -60
  [ "${PIPESTATUS[0]}" = 0 ] || { echo BUILD_FAST_FAILED; exit 1; }
  cp target-fast/release/cortiq "$root/bin/cortiq-fast.new" && mv "$root/bin/cortiq-fast.new" "$root/bin/cortiq-fast" && echo BUILD_FAST_DONE
else
  cargo build --release -p cortiq-cli --features cuda 2>&1 | grep -E "^error|Finished|could not compile" -A12 | tail -60
  [ "${PIPESTATUS[0]}" = 0 ] || { echo BUILD_RELEASE_FAILED; exit 1; }
  cp target/release/cortiq "$root/bin/cortiq.new" && mv "$root/bin/cortiq.new" "$root/bin/cortiq" && echo BUILD_RELEASE_DONE
fi
