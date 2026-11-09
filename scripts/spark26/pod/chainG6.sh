#!/bin/bash
# chainG6.sh: after chainG5 and the cortiq.xt4 build (tiled attend on the matrix units):
# graph and coop512 at 4096 and 16000 on a quiet card, the stage profile at 16000, greedy-text
# parity on 9 000 and 60 000 characters, and WikiText-2 perplexity (64 x 512, second half
# scored) of the walk against the graph path. Output logs/gab_xt4.txt, logs/g6_prof_16000.log,
# logs/parity_xt4_*.out, logs/g6_ppl.txt, chainG6.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG5.done ] && [ -f logs/build_cortiq.xt4.done ]; do sleep 5; done
[ -x cortiq.xt4 ] || { echo "no cortiq.xt4"; touch logs/chainG6.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt4
BIN=$B VARIANTS="graph:CMF_DENSE_GRAPH_PREFILL=1 coop512:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512" bash quiet.sh bash graph_ab.sh xt4 4096 16000
bash quiet.sh bash -c "CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g6_prof_16000.log 2>&1"
BIN=$B bash graph_parity.sh $F 9000 xt4_9k > logs/parity_xt4_9k.out 2>&1
BIN=$B bash graph_parity.sh $F 60000 xt4_60k > logs/parity_xt4_60k.out 2>&1
: > logs/g6_ppl.txt
for v in walk:CMF_DENSE_GRAPH_PREFILL=0 graph:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512; do
  name=${v%%:*}; envs=${v#*:}
  s=$(date +%s)
  env ${envs//,/ } $B ppl $F -f data/wiki.test.raw --windows 64 --window-len 512 --window-stride 512 --window-score-from 256 > logs/g6_ppl_$name.log 2>&1
  echo "$name | $(( $(date +%s) - s )) s | $(grep -iE "perplex|ppl" logs/g6_ppl_$name.log | tail -n 2 | tr '\n' ' ' | cut -c1-200)" >> logs/g6_ppl.txt
done
touch logs/chainG6.done
