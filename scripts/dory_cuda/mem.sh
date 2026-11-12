#!/bin/bash
# Dory's peak VRAM over `cortiq bench --core` (CUDA and Vulkan, the turbo file
# and the standard one) and Vulkan's perplexity on the suite's text.
#   mem.sh <bin> <out-dir>
set -u
bin=$1; o=$2
cd /root/cuda
mkdir -p $o
M=/root/models/dory
B=src/scripts/cuda/pod/bench_one.sh
for f in dory-turbo dory-q4tp-v2; do
  for c in 1000 4000 16000 32000 64000; do
    [ $f = dory-q4tp-v2 ] && [ $c = 64000 ] && continue
    CMF_MAX_SEQ=70000 bash $B dk_${f}_cuda_$c $bin $M/$f.cmf $c CMF_GPU=cuda >> $o/mem.txt 2>&1
    [ $f = dory-turbo ] && CMF_MAX_SEQ=70000 bash $B dk_${f}_vk_$c $bin $M/$f.cmf $c X=1 >> $o/mem.txt 2>&1
  done
done
for f in dory-turbo dory-q4tp-v2; do
  env CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp timeout 1800 $bin ppl $M/$f.cmf -f $o/wiki.txt --windows 12 --window-len 512 2>&1 | grep -E "PPL" | sed "s/^/$f vk /" >> $o/ppl.txt
done
echo MEM_DONE > $o/mem.done
