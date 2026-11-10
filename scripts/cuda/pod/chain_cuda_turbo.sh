#!/bin/bash
# The far field on the CUDA path: a prompt past the seal is read whole on
# the card, sealed once, and decoded by the far-field token kernels.
# Checked against the wgpu path taken the same way (CMF_O1_WHOLE=1): greedy
# ids of a free run and the logits of a run forced along the wgpu ids.
#   chain_cuda_turbo.sh <binary> [check|bench|all] [reps]
root=${CUDA_RUN_ROOT:-/root/cuda}
cd "$root" || exit 1
bin=${1:-bin/cortiq-cuda}
what=${2:-all}
reps=${3:-1}
py=./venv/bin/python
r=./scripts/ref_outputs.sh
b=./scripts/bench_one.sh
mkdir -p prompts_turbo
[ -f prompts_turbo/p6_long_16k.txt ] || cp prompts_long/p6_long_16k.txt prompts_turbo/
[ -f prompts_turbo/p7_long_40k.txt ] || cat prompts_long/p6_long_16k.txt prompts_long/p6_long_16k.txt > prompts_turbo/p7_long_40k.txt
export PROMPTS_DIR=$root/prompts_turbo
if [ "$what" != bench ]; then
  for q in q4tp q8_2f; do
    m=models/spark-x2.6-4b-$q-turbo.cmf
    $r turbocu_$q "$bin" $m CMF_GPU=cuda CMF_MAX_SEQ=131072 CMF_CUDA_DEBUG=1 RUST_LOG=info
    grep -h -E "far field|o1[ :]|declin|refus|exact rows|sealed" out/ref/turbocu_$q/*.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-220 | sort | uniq -c | sort -rn | head -24
    $r wholevk_$q "$bin" $m CMF_O1_WHOLE=1 CMF_MAX_SEQ=131072
    $r turbocu_${q}_forced "$bin" $m CMF_GPU=cuda CMF_MAX_SEQ=131072 FORCE=wholevk_$q
    $py ./scripts/compare_refs.py wholevk_$q turbocu_$q
    $py ./scripts/compare_refs.py wholevk_$q turbocu_${q}_forced
  done
fi
if [ "$what" != check ]; then
  for rep in $(seq 1 "$reps"); do
    for ctx in 16000 32000 64000; do
      $b cut_q4_${ctx}_r$rep "$bin" models/spark-x2.6-4b-q4tp-turbo.cmf $ctx CMF_GPU=cuda CMF_MAX_SEQ=131072
    done
  done
  for ctx in 16000 32000; do
    $b cut_q8_${ctx}_r1 "$bin" models/spark-x2.6-4b-q8_2f-turbo.cmf $ctx CMF_GPU=cuda CMF_MAX_SEQ=131072
  done
fi
echo CHAIN_CUDA_TURBO_DONE
