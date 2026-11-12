#!/bin/bash
# The closing checks after the ceiling: the release build, Vulkan under
# ceilings, a 16 GB card's budget (13800 MB) on text to image and an edit,
# an edit under 8 GB, then the unit tests and clippy.
#   final_gpu2.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-fin2}
cd "$root" || exit 1
bash scripts/build.sh release > logs/${p}_build_release.log 2>&1 || exit 1
export BIN=$root/bin/cortiq
bash scripts/vk_ceil.sh ${p}c 12000 8000 6000 4000 > logs/${p}_vk_ceil.log 2>&1
line() {
  echo "$1 $(grep -E "image\(s\), " out/$1.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$1.vram_trace | tail -1) MiB"
  grep -E "references:|declined|panicked" out/$1.log | grep -v "tile batches" | cut -c1-200 | head -3
}
{
  VRAM_MB=13800 bash scripts/gen.sh ${p}_16g_2048 vulkan 2048 2048 > /dev/null; line ${p}_16g_2048
  VRAM_MB=13800 bash scripts/gen.sh ${p}_16g_4096 vulkan 4096 4096 > /dev/null; line ${p}_16g_4096
  export IMAGE=$root/out/base_cuda_1024.png PROMPT=$root/prompts/edit_min_cafe.txt
  VRAM_MB=13800 bash scripts/gen.sh ${p}_16g_edit2048 vulkan 2048 2048 > /dev/null; line ${p}_16g_edit2048
  "$root/venv/bin/python" scripts/cmp_png.py out/finw_vulkan_edit2048.png out/${p}_16g_edit2048.png
  VRAM_MB=8000 MIN_AREA=1024 bash scripts/gen.sh ${p}_8g_edit1024 vulkan 1024 1024 > /dev/null; line ${p}_8g_edit1024
  "$root/venv/bin/python" scripts/cmp_png.py out/finv_edit_1024.png out/${p}_8g_edit1024.png
  unset IMAGE PROMPT
} > logs/${p}_cards.log 2>&1
unset BIN
. "$HOME/.cargo/env"
cd srcrs
cargo test -p cortiq-engine --lib --features cuda -- h3 2>&1 | grep -E "^test |test result|error" | tail -40 > "$root/logs/${p}_tests.log"
cargo clippy --workspace --all-targets 2>&1 > "$root/logs/${p}_clippy_full.log"; echo "clippy rc=$?" >> "$root/logs/${p}_clippy_full.log"
cargo clippy --workspace --all-targets --features cuda 2>&1 > "$root/logs/${p}_clippy_cuda_full.log"; echo "clippy rc=$?" >> "$root/logs/${p}_clippy_cuda_full.log"
echo FINAL_GPU2_DONE > "$root/logs/${p}_done"
