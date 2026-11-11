#!/bin/bash
# The Qwen3.8-27B-based Turbo files on CUDA, a quick check: a DFlash sidecar
# for each (converted for its trunk, router at the 3090's prices), then per
# model a greedy answer to the page's question and to a code prompt, plain
# against drafts against DFlash (the texts must match), and one planted fact
# in 60 000 characters (the far field).
#   turbo_models_check.sh <bin>
set -u
bin=$1
cd /root/cuda
P=1.12,1.15,1.17,1.19,1.21,1.22,1.24,0.19,0.03
for m in arete-27b-turbo orcacyber-nano-1.0-turbo; do
  f=/root/models/$m.cmf
  [ -f /root/models/$m.dflash.cmf ] || $bin convert --model /root/models/dflash2 --output $f --dflash-sidecar --quant q4tp > out/conv_$m.log 2>&1
  $bin dflash-route-fit $f /root/models/dfrows/rows --edge 1.05 --prices $P 2>&1 | grep -v INFO | tail -1
  d=out/tm_$m; mkdir -p $d
  echo "What happened in Tiananmen Square in June 1989? Answer in four sentences." > $d/q.txt
  for arm in plain spec dflash; do
    e="CMF_GRAPH_SPEC=0"; [ $arm = spec ] && e="X=1"; [ $arm = dflash ] && e="CMF_DFLASH=1"
    for p in q chat0; do
      pf=$d/q.txt; [ $p = chat0 ] && pf=src/scripts/q27cuda/prompts/chat0.txt
      env $e CMF_GPU=cuda CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp timeout 900 $bin run $f --prompt-file $pf --greedy --no-think -n 256 > $d/$arm-$p.out 2> $d/$arm-$p.log
      echo "$m $arm $p rc=$? $(grep -h "tokens · first" $d/$arm-$p.out | cut -c1-90) text $(grep -v -E "tokens · first token|^\[" $d/$arm-$p.out | md5sum | cut -c1-8)"
    done
  done
  sed -n "/^Prompt/,\$p" $d/plain-q.out | head -12
  PASSKEY_FILE=$d/pk.txt python3 src/scripts/spark26/passkey_file.py $bin $f data/wiki.test.raw 60000 2 CMF_GPU=cuda CMF_GPU_PROBE=0 2>&1 | tail -3
done
echo TM_DONE
