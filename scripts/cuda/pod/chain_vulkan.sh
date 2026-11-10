#!/bin/bash
# Vulkan (wgpu) baseline on the pod's card: prompt 1K / 4K / 16K, both files.
# 16K twice for the turbo file: as shipped (far field past the seal) and with
# CMF_O1=off (exact attention, what the CUDA path runs).
root=${CUDA_RUN_ROOT:-/root/cuda}
b=$root/scripts/bench_one.sh
vk=$root/bin/cortiq-vk
q4=$root/models/spark-x2.6-4b-q4tp-turbo.cmf
q8=$root/models/spark-x2.6-4b-q8_2f-turbo.cmf
for rep in 1 2; do
  for ctx in 1000 4000 16000; do
    $b vk_q4_exact_${ctx}_r$rep $vk $q4 $ctx CMF_O1=off
  done
  $b vk_q4_turbo_16000_r$rep $vk $q4 16000
done
for ctx in 1000 4000 16000; do
  $b vk_q8_exact_${ctx}_r1 $vk $q8 $ctx CMF_O1=off
done
echo CHAIN_VULKAN_DONE
