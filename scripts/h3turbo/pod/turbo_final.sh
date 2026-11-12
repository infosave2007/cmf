#!/bin/bash
# The closing sequence for MiniMax-H3-Turbo on a pod: the release build,
# then both backends measured with it — the reference clip (512 x 288, 39
# frames; the second Vulkan run has the pipeline cache warm), the ceilings,
# a five-second clip at 768 x 448 — then the unit tests and clippy.
#   turbo_final.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-tfin}
cd "$root" || exit 1
py=$root/venv/bin/python
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; exit 1; }
export BIN=$root/bin/cortiq
bash scripts/anim.sh ${p}_vk0 vulkan
bash scripts/anim.sh ${p}_vk vulkan
bash scripts/anim.sh ${p}_cuda cuda
echo "vulkan vs cuda: $($py scripts/cmp_frames.py out/${p}_vk.frames out/${p}_cuda.frames out/${p}_vk.wav out/${p}_cuda.wav | tr '\n' ' ')"
echo "vulkan twice:   $($py scripts/cmp_frames.py out/${p}_vk.frames out/${p}_vk0.frames out/${p}_vk.wav out/${p}_vk0.wav | tr '\n' ' ')"
for mb in 8000 6000; do
  VRAM_MB=$mb bash scripts/anim.sh ${p}_vk_$mb vulkan
  echo "vulkan $mb vs none: $($py scripts/cmp_frames.py out/${p}_vk_$mb.frames out/${p}_vk.frames out/${p}_vk_$mb.wav out/${p}_vk.wav | tr '\n' ' ')"
done
for mb in 8000 4000; do
  VRAM_MB=$mb bash scripts/anim.sh ${p}_cuda_$mb cuda
done
bash scripts/anim.sh ${p}_cuda_wide cuda 768 448 124
bash scripts/anim.sh ${p}_vk_wide vulkan 768 448 124
echo "wide vulkan vs cuda: $($py scripts/cmp_frames.py out/${p}_vk_wide.frames out/${p}_cuda_wide.frames | tr '\n' ' ')"
unset BIN
. "$HOME/.cargo/env"
cd srcrs
cargo test -p cortiq-engine --lib --features cuda -- h3 2>&1 | grep -E "^test |test result|error" | tail -30 > "$root/logs/${p}_tests.log"
cargo clippy --locked --workspace --all-targets 2>&1 | grep -E "^(warning|error)" | sort | uniq -c | tail -20 > "$root/logs/${p}_clippy.log"; echo "clippy rc=${PIPESTATUS[0]}" >> "$root/logs/${p}_clippy.log"
cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -E "^(warning|error)" | sort | uniq -c | tail -20 > "$root/logs/${p}_clippy_cuda.log"; echo "clippy rc=${PIPESTATUS[0]}" >> "$root/logs/${p}_clippy_cuda.log"
cd "$root"
tail -3 logs/${p}_tests.log; cat logs/${p}_clippy.log logs/${p}_clippy_cuda.log
echo TURBO_FINAL_DONE
