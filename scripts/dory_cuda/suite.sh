#!/bin/bash
# Dory on CUDA against the wgpu (Vulkan) path on the same card, release build:
# the 12 real prompts (`scripts/dory/suite.py`), `cortiq bench --core` after
# prompts of 1 000 to 64 000 tokens, wikitext-2 perplexity and planted facts.
#   suite.sh <bin> <out-dir>
set -u
bin=$1; o=$2
cd /root/cuda
mkdir -p $o
M=/root/models/dory
w=$o/wiki.txt; [ -f $w ] || head -c 400000 data/wiki.test.raw > $w
for f in dory-turbo dory-q4tp-v2; do
  for arm in cuda vk; do
    e="CMF_GPU=cuda"; [ $arm = vk ] && e="X=1"
    ( cd $o && env $e CMF_GPU_PROBE=0 python3 /root/cuda/src/scripts/dory/suite.py /root/cuda/$bin $M/$f.cmf $f-$arm ) >> $o/real.txt 2>&1
  done
done
for f in dory-turbo dory-q4tp-v2; do
  for arm in cuda vk; do
    e="CMF_GPU=cuda"; [ $arm = vk ] && e="X=1"
    for c in 1000 4000 16000 32000 64000; do
      [ $f = dory-q4tp-v2 ] && [ $c = 64000 ] && continue
      echo "== $f $arm ctx $c" >> $o/bench.txt
      env $e CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 timeout 1500 $bin bench $M/$f.cmf --core --tokens 60 --ctx $c 2>&1 | grep -E "Prompt|Decode|Memory|GPU:" >> $o/bench.txt
    done
  done
done
for f in dory-turbo dory-q4tp-v2; do
  env CMF_GPU=cuda CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp timeout 1200 $bin ppl $M/$f.cmf -f $w --windows 12 --window-len 512 2>&1 | grep -E "PPL" | sed "s/^/$f cuda /" >> $o/ppl.txt
done
for ch in 60000 100000; do
  PASSKEY_FILE=$o/pk_$ch.txt timeout 3000 python3 src/scripts/spark26/passkey_file.py $bin $M/dory-turbo.cmf data/wiki.test.raw $ch 8 CMF_GPU=cuda CMF_GPU_PROBE=0 CMF_MAX_SEQ=70000 >> $o/passkey.txt 2>&1
done
echo SUITE_DONE > $o/done
