#!/bin/bash
# The chat push's final numbers, release build: the ten prompts with drafts
# (chat0 twice: the first run pays the PTX cache), four without, the server
# A/B and the three-turn chat.
#   final_suite2.sh <bin>
set -u
bin=$1
cd /root/cuda
m=/root/models/qwen38-27b-q4tp.cmf
r=src/scripts/q27cuda/run_prompts.sh
$r fin2-spec "$bin" $m 512 chat0 chat0 chat1 chat2 chat3 chat4 chat5 chat6 chat7 copy0 edit0 -- CMF_GPU=cuda
$r fin2-plain "$bin" $m 512 chat0 chat3 copy0 edit0 -- CMF_GPU=cuda CMF_GRAPH_SPEC=0
CORTIQ_BIN="$bin" bash src/scripts/q27cuda/serve_ab.sh fin2spec X=1 > /dev/null 2>&1
CORTIQ_BIN="$bin" bash src/scripts/q27cuda/serve_ab.sh fin2plain CMF_GRAPH_SPEC=0 > /dev/null 2>&1
bash src/scripts/q27cuda/serve_chat.sh fin2 "$bin" X=1 > /dev/null 2>&1
echo FINAL2_DONE > out/final2.done
