#!/bin/bash
# chainG10.sh: after chainG9 and the cortiq.xt8 build (the cooperative GEMMs walk their tiles in
# groups of 8 batch tiles for L2 reuse of the weight tile): coop GEMMs with and without the f16
# plane and the scalar default at 4096, 16000 and 32000 on a quiet card, the stage profile at 16000
# with the coop GEMMs, greedy-text parity on 9 000 characters. Output logs/gab_xt8.txt,
# logs/g10_prof_16000.log, logs/parity_xt8_9k.out, chainG10.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/chainG9.done ] && [ -f logs/build_cortiq.xt8.done ]; do sleep 5; done
[ -x cortiq.xt8 ] || { echo "no cortiq.xt8"; touch logs/chainG10.done; exit 1; }
F=hf/cmf/spark-x2.6-4b-q4tp.cmf; B=./cortiq.xt8
BIN=$B TOKENS=8 VARIANTS="coop:CMF_BATCH_COOP=1 coopnof16:CMF_BATCH_COOP=1,CMF_BATCH_F16=0 default:" bash quiet.sh bash graph_ab.sh xt8 4096 16000 32000
bash quiet.sh bash -c "CMF_BATCH_COOP=1 CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/g10_prof_16000.log 2>&1"
BIN=$B bash graph_parity.sh $F 9000 xt8_9k > logs/parity_xt8_9k.out 2>&1
touch logs/chainG10.done
