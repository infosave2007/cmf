#!/bin/bash
# The turbo file as the wgpu path runs it (the far field past the seal of
# 8 192 rows): the table's rows at 16K / 32K / 64K and greedy outputs on the
# long prompts. What the CUDA far field is measured against.
#   chain_turbo_refs.sh [binary] [reps]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
vk=${1:-bin/cortiq-vk}
reps=${2:-3}
q4=models/spark-x2.6-4b-q4tp-turbo.cmf
q8=models/spark-x2.6-4b-q8_2f-turbo.cmf
b=./scripts/bench_one.sh
for rep in $(seq 1 "$reps"); do
  for ctx in 16000 32000 64000; do
    $b vkt_q4_${ctx}_r$rep "$vk" $q4 $ctx CMF_MAX_SEQ=131072
  done
done
for ctx in 16000 32000; do
  $b vkt_q8_${ctx}_r1 "$vk" $q8 $ctx CMF_MAX_SEQ=131072
done
mkdir -p prompts_turbo
cp prompts_long/p6_long_16k.txt prompts_turbo/
cat prompts_long/p6_long_16k.txt prompts_long/p6_long_16k.txt > prompts_turbo/p7_long_40k.txt
PROMPTS_DIR=$root/prompts_turbo ./scripts/ref_outputs.sh vkturbo_q4tp "$vk" $q4 CMF_MAX_SEQ=131072
echo CHAIN_TURBO_REFS_DONE
