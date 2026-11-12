#!/bin/bash
# The closing sequence for LTX-2.5 on a pod: the release build, the reference
# clip (384 x 256, 49 frames) on CUDA with the prompt encoded and from the
# cache, twice to check the seed reproduces, within an 8 GB ceiling, at
# 768 x 512, then over Vulkan; the unit tests and clippy last.
#   ltx_final.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-lfin}
cd "$root" || exit 1
py=$root/venv/bin/python
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; exit 1; }
export BIN=$root/bin/cortiq
EXTRA=--no-context-cache bash scripts/ltx.sh ${p}_cuda_enc cuda
bash scripts/ltx.sh ${p}_cuda cuda
bash scripts/ltx.sh ${p}_cuda2 cuda > /dev/null
echo "cuda twice: $($py scripts/cmp_y4m.py out/${p}_cuda.y4m out/${p}_cuda2.y4m)"
VRAM_MB=8000 bash scripts/ltx.sh ${p}_cuda_8000 cuda
echo "cuda 8000 vs none: $($py scripts/cmp_y4m.py out/${p}_cuda_8000.y4m out/${p}_cuda.y4m)"
bash scripts/ltx.sh ${p}_cuda_wide cuda 768 512 49
EXTRA=--no-context-cache bash scripts/ltx.sh ${p}_vk_enc vulkan
bash scripts/ltx.sh ${p}_vk vulkan
bash scripts/ltx.sh ${p}_vk2 vulkan > /dev/null
echo "vulkan twice: $($py scripts/cmp_y4m.py out/${p}_vk.y4m out/${p}_vk2.y4m)"
echo "vulkan vs cuda: $($py scripts/cmp_y4m.py out/${p}_vk.y4m out/${p}_cuda.y4m)"
VRAM_MB=8000 bash scripts/ltx.sh ${p}_vk_8000 vulkan
echo "vulkan 8000 vs none: $($py scripts/cmp_y4m.py out/${p}_vk_8000.y4m out/${p}_vk.y4m)"
bash scripts/ltx.sh ${p}_vk_wide vulkan 768 512 49
unset BIN
. "$HOME/.cargo/env"
cd srcrs
cargo test -p cortiq-engine --lib --features cuda -- h3 ltx conv1d 2>&1 | grep -E "^test |test result|error" | tail -30 > "$root/logs/${p}_tests.log"
cargo clippy --locked --workspace --all-targets 2>&1 | grep -E "^(warning|error)" | sort | uniq -c | tail -20 > "$root/logs/${p}_clippy.log"; echo "clippy rc=${PIPESTATUS[0]}" >> "$root/logs/${p}_clippy.log"
cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -E "^(warning|error)" | sort | uniq -c | tail -20 > "$root/logs/${p}_clippy_cuda.log"; echo "clippy rc=${PIPESTATUS[0]}" >> "$root/logs/${p}_clippy_cuda.log"
cd "$root"
tail -3 logs/${p}_tests.log; cat logs/${p}_clippy.log logs/${p}_clippy_cuda.log
echo LTX_FINAL_DONE
