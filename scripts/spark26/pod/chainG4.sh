#!/bin/bash
# chainG4.sh: after chainG3 and the cortiq.xt2 build (tiled attend, 8x4 register tile in the
# value pass, tiles as the fastest dispatch axis): graph and coop512 at 4096 and 16000 on a
# quiet card, the GPU stage profile at 16000, then greedy-text parity on 9 000 characters.
# Output logs/gab_xt2.txt, logs/g4_prof_16000.log, logs/parity_xt2_9k.out, chainG4.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG3.done ] && [ -f logs/build_cortiq.xt2.done ]; do sleep 5; done
[ -x cortiq.xt2 ] || { echo "no cortiq.xt2"; touch logs/chainG4.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=./cortiq.xt2 VARIANTS="graph:CMF_DENSE_GRAPH_PREFILL=1 coop512:CMF_DENSE_GRAPH_PREFILL=1,CMF_BATCH_COOP=1,CMF_BATCH_K=512" bash quiet.sh bash graph_ab.sh xt2 4096 16000
bash quiet.sh bash -c "CMF_DENSE_GRAPH_PREFILL=1 CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 ./cortiq.xt2 bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g4_prof_16000.log 2>&1"
BIN=./cortiq.xt2 bash graph_parity.sh $F 9000 xt2_9k > logs/parity_xt2_9k.out 2>&1
touch logs/chainG4.done
