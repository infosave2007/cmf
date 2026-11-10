#!/bin/bash
# The CUDA path against the Vulkan one on the same prompts: a free greedy
# run (do the ids agree?) and a run teacher-forced along the Vulkan ids
# (logits and top-1 at every step).
#   chain_cuda_check.sh <binary> [q4tp|q8_2f ...] [-- ENV=VALUE ...]
root=${CUDA_RUN_ROOT:-/root/cuda}
bin=${1:-$root/bin/cortiq-cuda}; shift
quants=(); envs=()
while [ $# -gt 0 ]; do
  case "$1" in --) shift; envs=("$@"); break;; *) quants+=("$1"); shift;; esac
done
[ ${#quants[@]} -eq 0 ] && quants=(q4tp q8_2f)
suffix=${CHECK_SUFFIX:-}
for q in "${quants[@]}"; do
  m=$root/models/spark-x2.6-4b-$q-turbo.cmf
  $root/scripts/ref_outputs.sh cuda_${q}${suffix} $bin $m CMF_GPU=cuda "${envs[@]}"
  $root/scripts/ref_outputs.sh cuda_${q}${suffix}_forced $bin $m CMF_GPU=cuda FORCE=vk_$q "${envs[@]}"
  $root/venv/bin/python $root/scripts/compare_refs.py vk_$q cuda_${q}${suffix}
  $root/venv/bin/python $root/scripts/compare_refs.py vk_$q cuda_${q}${suffix}_forced
done
echo CHAIN_CUDA_CHECK_DONE
