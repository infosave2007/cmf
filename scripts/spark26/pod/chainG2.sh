#!/bin/bash
# chainG2.sh: once chainG1 is done and cortiq.xt (the query-tiled attend) is built, the
# walk/graph/coop512 A/B at 4096 and 16000 tokens on a quiet card with that binary, then
# greedy-text parity on 9 000- and 60 000-character prompts. Output logs/gab_xt.txt,
# logs/parity_xt*.out, chainG2.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG1.done ] && [ -f logs/build_cortiq.xt.done ]; do sleep 5; done
[ -x cortiq.xt ] || { echo "no cortiq.xt"; touch logs/chainG2.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=./cortiq.xt VARIANTS="walk:CMF_DENSE_GRAPH_PREFILL=0 graph:CMF_DENSE_GRAPH_PREFILL=1 coop512:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512" bash quiet.sh bash graph_ab.sh xt 4096 16000
BIN=./cortiq.xt bash graph_parity.sh $F 9000 xt9k > logs/parity_xt9k.out 2>&1
BIN=./cortiq.xt bash graph_parity.sh $F 60000 xt60k > logs/parity_xt60k.out 2>&1
touch logs/chainG2.done
