#!/bin/bash
# One forward of the video transformer compared between the resident chain
# and the per-operation path on each backend (CMF_ANIM_DUMP: the velocities
# of every sampling step), on the reference clip. The prompt encoder and the
# decoders run the same way in both arms, so step 0 differs by the
# transformer alone; the later steps show how the sampler amplifies it.
#   turbo_fwd.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-tf}
cd "$root" || exit 1
py=$root/venv/bin/python
bash scripts/build.sh fast || exit 1
mkdir -p out/dump
for be in vulkan cuda; do
  CMF_ANIM_DUMP=$root/out/dump/${p}_${be}_chain bash scripts/anim.sh ${p}_${be}_chain $be
  CMF_MMH3_CHAIN=0 CMF_ANIM_DUMP=$root/out/dump/${p}_${be}_perop bash scripts/anim.sh ${p}_${be}_perop $be
done
for i in 0 1 2 3; do
  for s in v a; do
    echo "step $i $s  vulkan chain/per-op: $($py scripts/cmp_f32.py out/dump/${p}_vulkan_chain.$s$i.f32 out/dump/${p}_vulkan_perop.$s$i.f32)"
    echo "step $i $s  cuda   chain/per-op: $($py scripts/cmp_f32.py out/dump/${p}_cuda_chain.$s$i.f32 out/dump/${p}_cuda_perop.$s$i.f32)"
    echo "step $i $s  chain vulkan/cuda:   $($py scripts/cmp_f32.py out/dump/${p}_vulkan_chain.$s$i.f32 out/dump/${p}_cuda_chain.$s$i.f32)"
    echo "step $i $s  per-op vulkan/cuda:  $($py scripts/cmp_f32.py out/dump/${p}_vulkan_perop.$s$i.f32 out/dump/${p}_cuda_perop.$s$i.f32)"
  done
done
rm -rf out/dump
echo TURBO_FWD_DONE
