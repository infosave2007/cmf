#!/bin/bash
# The closing sequence on a pod: the release build, both backends measured
# with it (text to image at three sizes, edits at two canvases), then the
# unit tests of the chains and clippy. One log each under logs/.
#   final_gpu.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-fin}
cd "$root" || exit 1
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1
export BIN=$root/bin/cortiq
bash scripts/baseline.sh ${p} > logs/${p}_baseline.log 2>&1
cp out/${p}_cuda_1024.png out/base_cuda_1024.png
cp out/${p}_cuda_2048.png out/base_cuda_2048.png
rm -f out/base_edit_cuda.png
bash scripts/vk_check.sh ${p}v > logs/${p}_vk_check.log 2>&1
bash scripts/vk_wide.sh ${p}w > logs/${p}_vk_wide.log 2>&1
unset BIN
. "$HOME/.cargo/env"
cd srcrs
cargo test -p cortiq-engine --lib --features cuda -- h3 2>&1 | grep -E "^test |test result|error" | tail -40 > "$root/logs/${p}_tests.log"
cargo clippy --workspace --all-targets 2>&1 | grep -E "^(warning|error)" | sort | uniq -c | tail -20 > "$root/logs/${p}_clippy.log"; echo "clippy rc=${PIPESTATUS[0]}" >> "$root/logs/${p}_clippy.log"
cargo clippy --workspace --all-targets --features cuda 2>&1 | grep -E "^(warning|error)" | sort | uniq -c | tail -20 > "$root/logs/${p}_clippy_cuda.log"; echo "clippy rc=${PIPESTATUS[0]}" >> "$root/logs/${p}_clippy_cuda.log"
echo FINAL_GPU_DONE > "$root/logs/${p}_done"
