#!/bin/bash
# The wgpu path with the first token's head on the token graph
# (`graph_head_after_prompt`): free greedy runs on every prompt against the
# CUDA path, the first-token logits against the float host, and what the
# change does to the first token's time and the card's memory.
#   chain_vkfix.sh <binary with the fix>
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
for q in q4tp q8_2f; do
  m=models/spark-x2.6-4b-$q-turbo.cmf
  ./scripts/ref_outputs.sh vkfix_${q}_all "$bin" $m CMF_O1=off
  PROMPTS_DIR=$root/prompts_long ./scripts/ref_outputs.sh vkfix_${q}_all "$bin" $m CMF_O1=off
  ./venv/bin/python ./scripts/compare_refs.py cuda_${q}_final vkfix_${q}_all
done
b=./scripts/bench_one.sh
for rep in 1 2; do
  for q in q8 q4; do
    f=q8_2f; [ $q = q4 ] && f=q4tp
    m=models/spark-x2.6-4b-$f-turbo.cmf
    for ctx in 1000 16000; do
      $b vkold_${q}_${ctx}_r$rep bin/cortiq-vk $m $ctx CMF_O1=off
      $b vkfix_${q}_${ctx}_r$rep "$bin" $m $ctx CMF_O1=off
    done
  done
done
echo CHAIN_VKFIX_DONE
