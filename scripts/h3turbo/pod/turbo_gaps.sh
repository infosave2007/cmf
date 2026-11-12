#!/bin/bash
# What is left after the chain, on the reference clip (512 x 288, 39 frames):
# the f16 ranges of the video stack on both backends, the memory ceiling,
# and one forward against the host's f32 arithmetic.
#   turbo_gaps.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-tg}
cd "$root" || exit 1
py=$root/venv/bin/python
mkdir -p out/dump
# 1. ranges: the largest stored half per site on Vulkan, clamps on CUDA
CMF_H3_WGPU_AMAX=1 bash scripts/anim.sh ${p}_vk_amax vulkan
grep -i "amax\|site\|attn-in\|mlp-mid" out/${p}_vk_amax.log | tail -12
CMF_H3_SAT=1 bash scripts/anim.sh ${p}_cuda_sat cuda
grep -i "sat" out/${p}_cuda_sat.log | tail -6
# 2. the ceiling: 8 GB and 4 GB against the unbounded runs
bash scripts/anim.sh ${p}_vk vulkan
bash scripts/anim.sh ${p}_cuda cuda
for mb in 8000 4000; do
  VRAM_MB=$mb bash scripts/anim.sh ${p}_vk_$mb vulkan
  echo "vulkan $mb vs none: $($py scripts/cmp_frames.py out/${p}_vk_$mb.frames out/${p}_vk.frames out/${p}_vk_$mb.wav out/${p}_vk.wav | tr '\n' ' ')"
  VRAM_MB=$mb bash scripts/anim.sh ${p}_cuda_$mb cuda
  echo "cuda $mb vs none: $($py scripts/cmp_frames.py out/${p}_cuda_$mb.frames out/${p}_cuda.frames out/${p}_cuda_$mb.wav out/${p}_cuda.wav | tr '\n' ' ')"
done
# 3. one forward against the host (f32 activations): step 0 of a one-step run
CMF_ANIM_DUMP=$root/out/dump/vk bash scripts/anim.sh ${p}_vk_1 vulkan 512 288 39 1
CMF_ANIM_DUMP=$root/out/dump/cuda bash scripts/anim.sh ${p}_cuda_1 cuda 512 288 39 1
CMF_MMH3_GPU=0 CMF_SDOT=0 CMF_ANIM_DUMP=$root/out/dump/cpu bash scripts/anim.sh ${p}_cpu_1 cpu 512 288 39 1
for s in v a; do
  echo "step 0 $s  vulkan chain / host: $($py scripts/cmp_f32.py out/dump/vk.${s}0.f32 out/dump/cpu.${s}0.f32)"
  echo "step 0 $s  cuda chain / host:   $($py scripts/cmp_f32.py out/dump/cuda.${s}0.f32 out/dump/cpu.${s}0.f32)"
done
rm -rf out/dump out/${p}_*.frames
echo TURBO_GAPS_DONE
