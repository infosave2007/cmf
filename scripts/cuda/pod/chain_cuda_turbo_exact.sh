#!/bin/bash
# The far field with both prompts read in f32 (wgpu with CMF_COOP=0, CUDA
# with CMF_CUDA_PREC=exact): the seals then start from the same rows, and
# what is left between the two paths is the far-field token kernels.
#   chain_cuda_turbo_exact.sh <binary>
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
export PROMPTS_DIR=$root/prompts_long
r=./scripts/ref_outputs.sh
for q in q4tp q8_2f; do
  m=models/spark-x2.6-4b-$q-turbo.cmf
  $r wholevk0_$q "$bin" $m CMF_O1_WHOLE=1 CMF_COOP=0 CMF_MAX_SEQ=131072
  $r turbocux_${q}_forced "$bin" $m CMF_GPU=cuda CMF_CUDA_PREC=exact CMF_MAX_SEQ=131072 FORCE=wholevk0_$q
  ./venv/bin/python ./scripts/compare_refs.py wholevk0_$q turbocux_${q}_forced
done
echo CHAIN_CUDA_TURBO_EXACT_DONE
