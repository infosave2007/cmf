#!/bin/bash
# The CUDA path on the pod's card: prompt 1K / 4K / 16K.
#   chain_cuda.sh <binary> <label> [q4tp|q8_2f ...] [-- ENV=VALUE ...]
root=${CUDA_RUN_ROOT:-/root/cuda}
bin=$1; label=$2; shift 2
quants=(); envs=()
while [ $# -gt 0 ]; do
  case "$1" in --) shift; envs=("$@"); break;; *) quants+=("$1"); shift;; esac
done
[ ${#quants[@]} -eq 0 ] && quants=(q4tp)
reps=${REPS:-1}
for q in "${quants[@]}"; do
  m=$root/models/spark-x2.6-4b-$q-turbo.cmf
  for rep in $(seq 1 "$reps"); do
    for ctx in ${CTXS:-1000 4000 16000}; do
      $root/scripts/bench_one.sh cu_${q}_${label}_${ctx}_r$rep "$bin" "$m" "$ctx" CMF_GPU=cuda "${envs[@]}"
    done
  done
done
echo CHAIN_CUDA_DONE $label
