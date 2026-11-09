#!/bin/bash
# chainG12.sh: after chainG11 and the cortiq.xt10 build (pipelined in-kernel-dequant coop GEMM,
# `CMF_BATCH_MM=v2`): the grouped-walk coop GEMM against its pipelined twin at 4096 and 16000 on a
# quiet card, the stage profile of the twin at 16000, greedy-text parity on 9 000 characters with the
# twin. Output logs/gab_xt10.txt, logs/g12_prof_16000.log, logs/parity_xt10_9k.out, chainG12.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG11.done ] && [ -f logs/build_cortiq.xt10.done ]; do sleep 5; done
[ -x cortiq.xt10 ] || { echo "no cortiq.xt10"; touch logs/chainG12.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt10
C="CMF_BATCH_COOP=1,CMF_BATCH_K=512,CMF_BATCH_F16=0"
BIN=$B VARIANTS="coop:$C coopv2:$C,CMF_BATCH_MM=v2 coopv2b:$C,CMF_BATCH_MM=v2" bash quiet.sh bash graph_ab.sh xt10 4096 16000
bash quiet.sh bash -c "CMF_BATCH_COOP=1 CMF_BATCH_K=512 CMF_BATCH_F16=0 CMF_BATCH_MM=v2 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g12_prof_16000.log 2>&1"
BIN=$B CMF_BATCH_MM=v2 CMF_BATCH_F16=0 bash graph_parity.sh $F 9000 xt10_9k > logs/parity_xt10_9k.out 2>&1
touch logs/chainG12.done
