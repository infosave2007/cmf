#!/bin/bash
set -u
cd /root/h3i/flash || exit 1
exec 9>/root/h3i/gpu.lock; flock 9
/usr/local/cuda/bin/nvcc -O3 -arch=sm_86 conv_bench.cu -lcublas -o cb 2>&1 | grep -v "warning\|^$" | head -20
./cb
echo CONV_BENCH_DONE
