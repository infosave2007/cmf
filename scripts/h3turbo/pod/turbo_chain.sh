#!/bin/bash
# MiniMax-H3-Turbo with the resident chain against the paths it replaces, on
# the reference clip (512 x 288, 39 frames, 4 steps).
#   turbo_chain.sh [TAGPREFIX]
# 1. Vulkan per-operation with the release binary (the chains of the prompt
#    encoder and the VAE off: what 0.8.74 ran);
# 2. the fast build of the current sources;
# 3. Vulkan and CUDA with it, frames and sound compared with the runs before.
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-tc}
cd "$root" || exit 1
py=$root/venv/bin/python
if [ ! -f out/tb0_vk.sum ] || ! grep -q "rc=0" out/tb0_vk.sum; then
  BIN=$root/bin/cortiq CMF_H3_WGPU=0 bash scripts/anim.sh tb0_vk vulkan
fi
bash scripts/build.sh fast || exit 1
bash scripts/anim.sh ${p}_vk vulkan
bash scripts/anim.sh ${p}_cuda cuda
echo "== vulkan chain vs vulkan per-op"; $py scripts/cmp_frames.py out/${p}_vk.frames out/tb0_vk.frames out/${p}_vk.wav out/tb0_vk.wav
echo "== cuda chain vs cuda per-op";     $py scripts/cmp_frames.py out/${p}_cuda.frames out/tb_cuda.frames out/${p}_cuda.wav out/tb_cuda.wav
echo "== vulkan chain vs cuda chain";    $py scripts/cmp_frames.py out/${p}_vk.frames out/${p}_cuda.frames out/${p}_vk.wav out/${p}_cuda.wav
echo TURBO_CHAIN_DONE
