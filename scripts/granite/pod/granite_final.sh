#!/bin/bash
# Granite-4.2-30B release measurement: release build, 64-token bench on CUDA
# and Vulkan (two runs each), the greedy texts compared.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
f=cmf/granite-4.2-30b-q4tp.cmf
bash scripts/build.sh release > logs/grf_build.log 2>&1 || { echo BUILD_FAILED; tail -30 logs/grf_build.log; exit 1; }
exec 9>"$root/gpu.lock"; flock 9
for be in cuda 1; do
  for i in 1 2; do
    CMF_GPU=$be timeout 900 bin/cortiq bench $f --tokens 64 --json > out/grf_bench_${be}_$i.log 2>&1
    echo "$be run $i: $(grep -o '"decode_tok_s_steady": *[0-9.]*' out/grf_bench_${be}_$i.log | head -1) $(grep -o '"prefill_tok_s": *[0-9.]*' out/grf_bench_${be}_$i.log | head -1) $(grep -o '"ttft_s": *[0-9.]*' out/grf_bench_${be}_$i.log | head -1)"
  done
  CMF_GPU=$be timeout 900 bin/cortiq run $f --prompt "Write a Python function that checks whether a number is prime." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/grf_run_$be.txt
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cmp -s out/grf_run_cuda.txt out/grf_run_1.txt && echo "GENERATED TEXT IDENTICAL" || echo "GENERATED TEXT DIFFERS"
echo GRANITE_FINAL_DONE
