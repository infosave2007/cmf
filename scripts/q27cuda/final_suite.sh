#!/bin/bash
# The final CUDA numbers for docs/CUDA.md: the ten prompts with drafts, four
# without, and the bench at 4 000 / 16 000 / 32 000 tokens, release build.
#   final_suite.sh <bin> <model>
set -u
bin=$1; model=$2
cd /root/cuda
r=src/scripts/q27cuda/run_prompts.sh
$r fin-spec  "$bin" "$model" 512 chat0 chat1 chat2 chat3 chat4 chat5 chat6 chat7 copy0 edit0 -- CMF_GPU=cuda
$r fin-plain "$bin" "$model" 512 chat0 chat3 copy0 edit0 -- CMF_GPU=cuda CMF_GRAPH_SPEC=0
for ctx in 4000 16000 32000; do
  XDG_RUNTIME_DIR=/tmp CMF_GPU=cuda CMF_GPU_PROBE=0 "$bin" bench "$model" --ctx $ctx --tokens 64 --core \
    > out/fin-bench$ctx.txt 2>&1
done
echo FINAL_DONE > out/final.done
