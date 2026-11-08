#!/bin/bash
# chainG5.sh: after chainG4 and the cortiq.xt3 build (tiled attend with named registers in
# both phases): graph and coop512 at 4096 and 16000 on a quiet card, the stage profile at
# 16000, greedy-text parity on 9 000 and 60 000 characters.
# Output logs/gab_xt3.txt, logs/g5_prof_16000.log, logs/parity_xt3_*.out, chainG5.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG4.done ] && [ -f logs/build_cortiq.xt3.done ]; do sleep 5; done
[ -x cortiq.xt3 ] || { echo "no cortiq.xt3"; touch logs/chainG5.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=./cortiq.xt3 VARIANTS="graph:CMF_DENSE_GRAPH_PREFILL=1 coop512:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512" bash quiet.sh bash graph_ab.sh xt3 4096 16000
bash quiet.sh bash -c "CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 ./cortiq.xt3 bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g5_prof_16000.log 2>&1"
BIN=./cortiq.xt3 bash graph_parity.sh $F 9000 xt3_9k > logs/parity_xt3_9k.out 2>&1
BIN=./cortiq.xt3 bash graph_parity.sh $F 60000 xt3_60k > logs/parity_xt3_60k.out 2>&1
touch logs/chainG5.done
