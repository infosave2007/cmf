#!/bin/bash
# The table: CUDA against Vulkan on the pod's card, the two paths taking
# turns at every prompt length so that drift on a rented box lands on both.
#   chain_final.sh [reps]          (FINAL_PREFIX=<tag> names the runs, `fin` by default)
# Per repeat and prompt length (1K / 4K / 16K), q4tp: Vulkan with exact
# attention (CMF_O1=off — what the CUDA path runs), CUDA as it ships, CUDA
# with CMF_CUDA_PREC=exact; q8_2f: Vulkan exact, CUDA. At 16K also the turbo
# file as Vulkan ships it (far field past the seal).
root=${CUDA_RUN_ROOT:-/root/cuda}
b=$root/scripts/bench_one.sh
vk=$root/bin/cortiq-vk
cu=$root/bin/cortiq-cuda
q4=$root/models/spark-x2.6-4b-q4tp-turbo.cmf
q8=$root/models/spark-x2.6-4b-q8_2f-turbo.cmf
reps=${1:-3}
p=${FINAL_PREFIX:-fin}
for rep in $(seq 1 "$reps"); do
  for ctx in 1000 4000 16000; do
    $b ${p}_vk_q4_${ctx}_r$rep $vk $q4 $ctx CMF_O1=off
    $b ${p}_cu_q4_${ctx}_r$rep $cu $q4 $ctx CMF_GPU=cuda
    $b ${p}_cux_q4_${ctx}_r$rep $cu $q4 $ctx CMF_GPU=cuda CMF_CUDA_PREC=exact
    $b ${p}_vk_q8_${ctx}_r$rep $vk $q8 $ctx CMF_O1=off
    $b ${p}_cu_q8_${ctx}_r$rep $cu $q8 $ctx CMF_GPU=cuda
  done
  $b ${p}_vkturbo_q4_16000_r$rep $vk $q4 16000
done
echo CHAIN_FINAL_DONE
