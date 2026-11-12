#!/bin/bash
# The last check of a tree on a pod: the iteration build, a Vulkan image
# without a ceiling and under 6 GB compared with a kept picture, the unit
# tests of the chains, clippy with and without CUDA.
#   last_check.sh KEPT_1024.png
set -u
root=${H3I_ROOT:-/root/h3i}
kept=$1
cd "$root" || exit 1
bash scripts/build.sh fast | tail -2 || exit 1
MIN_AREA=1024 bash scripts/gen.sh last_1024 vulkan 1024 1024 | tail -1
"$root/venv/bin/python" scripts/cmp_png.py "$kept" out/last_1024.png
VRAM_MB=6000 MIN_AREA=1024 bash scripts/gen.sh last_1024_6g vulkan 1024 1024 | tail -1
"$root/venv/bin/python" scripts/cmp_png.py "$kept" out/last_1024_6g.png
grep -E "declined|panicked" out/last_1024.log out/last_1024_6g.log | grep -v "tile batches" | head -4
. "$HOME/.cargo/env"
cd srcrs
cargo test -p cortiq-engine --lib --features cuda -- h3 2>&1 | grep -E "test result|FAILED|panicked|^error" | tail -5
for f in "" "--features cuda"; do
  cargo clippy --workspace --all-targets $f > "$root/logs/last_clippy.txt" 2>&1; echo "clippy [$f] rc=$? warnings in the chain files: $(grep -cE '^\s+--> crates/cortiq-engine/src/(gpu_wgpu/h3|gpu_wgpu/h3_conv|qwen3vis|gpu\.rs)' "$root/logs/last_clippy.txt")"
done
echo LAST_CHECK_DONE
