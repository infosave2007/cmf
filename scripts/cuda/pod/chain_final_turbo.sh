#!/bin/bash
# The turbo table: at every prompt length past the seal the wgpu far field
# (as shipped: sealed at 8 192 rows), the CUDA far field (the prompt read
# whole, one seal) and CUDA with exact attention take turns. Then one long
# decode a path for what the card holds while it decodes.
#   chain_final_turbo.sh <binary> [reps]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
reps=${2:-3}
b=./scripts/bench_one.sh
q4=models/spark-x2.6-4b-q4tp-turbo.cmf
q8=models/spark-x2.6-4b-q8_2f-turbo.cmf
p=${FINAL_PREFIX:-ft}
for rep in $(seq 1 "$reps"); do
  for ctx in 16000 32000 64000; do
    $b ${p}_vkt_q4_${ctx}_r$rep "$bin" $q4 $ctx CMF_MAX_SEQ=131072
    $b ${p}_cut_q4_${ctx}_r$rep "$bin" $q4 $ctx CMF_GPU=cuda CMF_MAX_SEQ=131072
    $b ${p}_cux_q4_${ctx}_r$rep "$bin" $q4 $ctx CMF_GPU=cuda CMF_O1=off CMF_MAX_SEQ=131072
  done
  for ctx in 16000 32000; do
    $b ${p}_cut_q8_${ctx}_r$rep "$bin" $q8 $ctx CMF_GPU=cuda CMF_MAX_SEQ=131072
  done
done
for ctx in 16000 32000 64000; do
  $b ${p}_vktL_q4_${ctx} "$bin" $q4 $ctx CMF_MAX_SEQ=131072 BENCH_TOKENS=500
  $b ${p}_cutL_q4_${ctx} "$bin" $q4 $ctx CMF_GPU=cuda CMF_MAX_SEQ=131072 BENCH_TOKENS=500
  $b ${p}_cuxL_q4_${ctx} "$bin" $q4 $ctx CMF_GPU=cuda CMF_O1=off CMF_MAX_SEQ=131072 BENCH_TOKENS=500
done
echo CHAIN_FINAL_TURBO_DONE
