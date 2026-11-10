#!/bin/bash
# The far-field token kernels against the host's (`nystrom.rs`) from one and
# the same seal: the prompt on the CUDA batch graph both times, then the
# decode on the card, and on the host with float activations
# (CMF_GPU_WGPU_GRAPH=prefill CMF_SDOT=0), forced along the same ids.
#   chain_cuda_turbo_host.sh <binary> <ids tag prefix> [q4tp|q8_2f ...]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
ids=${2:-wholevk}
shift 2
[ $# -eq 0 ] && set -- q4tp q8_2f
export PROMPTS_DIR=$root/prompts_long
r=./scripts/ref_outputs.sh
for q in "$@"; do
  m=models/spark-x2.6-4b-$q-turbo.cmf
  $r hostdec_$q "$bin" $m CMF_GPU=cuda CMF_GPU_WGPU_GRAPH=prefill CMF_SDOT=0 CMF_MAX_SEQ=131072 FORCE=${ids}_$q
  $r devdec_$q "$bin" $m CMF_GPU=cuda CMF_MAX_SEQ=131072 FORCE=${ids}_$q
  ./venv/bin/python ./scripts/compare_refs.py hostdec_$q devdec_$q
done
echo CHAIN_CUDA_TURBO_HOST_DONE
