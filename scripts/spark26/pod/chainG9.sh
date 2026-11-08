#!/bin/bash
# chainG9.sh: after chainG8 and the cortiq.xt7 build (the defaults: a sliding stack's prompt through
# the batched graph with 512-row submits where the tiled attend exists): the default route against
# the chunked walk, the coop GEMMs and the coop GEMMs without the f16 plane at 4096, 16000 and 32000
# with 64 decode tokens (the decode-after-graph question), the turbo file's card rows at 16000, 32000
# and 64000 (default against walk, 60 tokens as the card's table), greedy-text parity on 9 000 and
# 60 000 characters. Output logs/gab_xt7.txt, logs/gab_xt7t.txt, logs/parity_xt7_*.out, chainG9.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG8.done ] && [ -f logs/build_cortiq.xt7.done ]; do sleep 5; done
[ -x cortiq.xt7 ] || { echo "no cortiq.xt7"; touch logs/chainG9.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; T=out/spark-x2.6-4b-q4tp-turbo.cmf; B=./cortiq.xt7
BIN=$B TOKENS=64 VARIANTS="default: walk:CMF_DENSE_GRAPH_PREFILL=0 coop:CMF_BATCH_COOP=1 coopnof16:CMF_BATCH_COOP=1,CMF_BATCH_F16=0" bash quiet.sh bash graph_ab.sh xt7 4096 16000 32000
BIN=$B F=$T TOKENS=60 VARIANTS="default: walk:CMF_DENSE_GRAPH_PREFILL=0 coop:CMF_BATCH_COOP=1" bash quiet.sh bash graph_ab.sh xt7t 16000 32000 64000
BIN=$B bash graph_parity.sh $F 9000 xt7_9k > logs/parity_xt7_9k.out 2>&1
BIN=$B bash graph_parity.sh $F 60000 xt7_60k > logs/parity_xt7_60k.out 2>&1
touch logs/chainG9.done
