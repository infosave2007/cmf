#!/bin/bash
# chainG11.sh: after chainG10 and the cortiq.xt9 build (the f16-plane GEMM with a 128-row batch
# tile, `CMF_BATCH_MM_TILE=128`): the 128-row tile against the 64-row one at 4096 and 16000 on a
# quiet card, the stage profile at 16000 with the 128-row tile, greedy-text parity on 9 000
# characters with it. Output logs/gab_xt9.txt, logs/g11_prof_16000.log, logs/parity_xt9_9k.out,
# chainG11.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG10.done ] && [ -f logs/build_cortiq.xt9.done ]; do sleep 5; done
[ -x cortiq.xt9 ] || { echo "no cortiq.xt9"; touch logs/chainG11.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt9
BIN=$B TOKENS=8 VARIANTS="m128:CMF_BATCH_COOP=1,CMF_BATCH_MM_TILE=128 m64:CMF_BATCH_COOP=1" bash quiet.sh bash graph_ab.sh xt9 4096 16000
bash quiet.sh bash -c "CMF_BATCH_COOP=1 CMF_BATCH_MM_TILE=128 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g11_prof_16000.log 2>&1"
CMF_BATCH_MM_TILE=128 BIN=$B bash graph_parity.sh $F 9000 xt9_9k > logs/parity_xt9_9k.out 2>&1
touch logs/chainG11.done
