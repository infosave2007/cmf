#!/bin/bash
# Which path is right where two of them disagree: the host with float
# activations (CMF_GPU=0 CMF_SDOT=0) as the reference, every path
# teacher-forced along the Vulkan ids on the short prompts (the long ones
# take the float host many minutes each).
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
mkdir -p prompts_short
for p in p1_short_en p2_ru p3_code p5_quotes; do cp prompts/$p.txt prompts_short/; done
export PROMPTS_DIR=$root/prompts_short
r=./scripts/ref_outputs.sh
for q in q4tp q8_2f; do
  m=models/spark-x2.6-4b-$q-turbo.cmf
  $r cudax_${q}_forced bin/cortiq-cuda $m CMF_GPU=cuda CMF_CUDA_PREC=exact FORCE=vk_$q
  $r cudaf_${q}_forced bin/cortiq-cuda $m CMF_GPU=cuda FORCE=vk_$q
  $r truth_${q} bin/cortiq-vk $m CMF_GPU=0 CMF_SDOT=0 FORCE=vk_$q
done
echo CHAIN_TRUTH_DONE
