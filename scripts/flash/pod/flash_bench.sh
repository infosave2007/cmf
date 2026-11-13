#!/bin/bash
# Build and run the flash bench for head widths 128 / 64 (and the causal mask).
set -u
cd /root/h3i/flash || exit 1
exec 9>/root/h3i/gpu.lock; flock 9
for cfg in "128 0" "64 0" "128 1" "256 0"; do
  set -- $cfg
  /usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -DHD=$1 -DCAUSAL=$2 flash_bench.cu -o fb_$1_$2 2>&1 | grep -v "warning\|^$" | head -20
  echo "causal=$2: $(./fb_$1_$2 ${N:-36736} ${P:-16})"
done
echo FLASH_BENCH_DONE
