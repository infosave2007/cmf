#!/bin/bash
# Reference outputs of the existing paths: Vulkan (as shipped, exact
# attention), Vulkan with the f32 kernels (CMF_COOP=0), and the CPU
# teacher-forced along the Vulkan ids (the noise floor between paths).
root=${CUDA_RUN_ROOT:-/root/cuda}
r=$root/scripts/ref_outputs.sh
vk=$root/bin/cortiq-vk
for q in q4tp q8_2f; do
  m=$root/models/spark-x2.6-4b-$q-turbo.cmf
  $r vk_$q $vk $m CMF_O1=off
  $r vk_${q}_coop0 $vk $m CMF_O1=off CMF_COOP=0 FORCE=vk_$q
  $r cpu_${q}_forced $vk $m CMF_GPU=0 FORCE=vk_$q
done
echo CHAIN_REFS_DONE
