#!/bin/bash
# The release binary of this branch, both of its GPU paths: free greedy runs
# on every prompt (the wgpu path and the CUDA path must give the same ids),
# the CUDA path against the one measured before, the wgpu path forced along
# the reference ids against the float host (the first token's head now runs
# on the token graph), and one pass of the table's rows on both paths.
#   chain_release_check.sh [binary]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
r=./scripts/ref_outputs.sh
py=./venv/bin/python
mkdir -p prompts_short
for p in p1_short_en p2_ru p3_code p5_quotes; do cp prompts/$p.txt prompts_short/; done
for q in q4tp q8_2f; do
  m=models/spark-x2.6-4b-$q-turbo.cmf
  for d in prompts prompts_long; do
    PROMPTS_DIR=$root/$d $r rel_cuda_$q "$bin" $m CMF_GPU=cuda
    PROMPTS_DIR=$root/$d $r rel_vk_$q "$bin" $m CMF_O1=off
  done
  $py ./scripts/compare_refs.py rel_cuda_$q rel_vk_$q
  $py ./scripts/compare_refs.py cuda_${q}_final rel_cuda_$q
  PROMPTS_DIR=$root/prompts_short $r vkhead_$q "$bin" $m CMF_O1=off FORCE=vk_$q
done
$py ./scripts/compare_truth.py
./scripts/chat_check.sh | head -3
./scripts/serve_check.sh | tail -5
b=./scripts/bench_one.sh
for q in q4 q8; do
  f=q4tp; [ $q = q8 ] && f=q8_2f
  m=models/spark-x2.6-4b-$f-turbo.cmf
  for ctx in 1000 4000 16000; do
    $b rel_cu_${q}_${ctx}_r1 "$bin" $m $ctx CMF_GPU=cuda
    $b rel_vk_${q}_${ctx}_r1 "$bin" $m $ctx CMF_O1=off
  done
done
echo CHAIN_RELEASE_CHECK_DONE
