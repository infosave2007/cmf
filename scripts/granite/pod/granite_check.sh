#!/bin/bash
# Granite-4.2-30B on CUDA against Vulkan: decode speed (64 tokens, bench) and
# the greedy text of one prompt on each (the same tokens = the same model).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bin=${BIN:-cortiq-fast}
f=${MODEL:-cmf/granite-4.2-30b-q4tp.cmf}
tag=${1:-gr}
if [ "$bin" = cortiq-fast ]; then
  bash scripts/build.sh fast > logs/${tag}_build.log 2>&1 || { echo BUILD_FAILED; tail -40 logs/${tag}_build.log; exit 1; }
fi
exec 9>"$root/gpu.lock"; flock 9
for be in cuda 1; do
  CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 900 bin/$bin bench $f --tokens 64 --json > out/${tag}_bench_$be.log 2>&1
  echo "== $be bench: $(grep -o '"decode_tok_s_steady": *[0-9.]*' out/${tag}_bench_$be.log | head -1) $(grep -o '"prefill_tok_s": *[0-9.]*' out/${tag}_bench_$be.log | head -1)"
  grep -iE "declin|error|panick" out/${tag}_bench_$be.log | sort -u | head -4 | cut -c1-200
  CMF_GPU=$be timeout 900 bin/$bin run $f --prompt "Write a Python function that checks whether a number is prime." -n 48 --greedy > out/${tag}_run_$be.txt 2> out/${tag}_run_$be.log
  echo "-- $be text: $(tr '\n' ' ' < out/${tag}_run_$be.txt | cut -c1-300)"
done
cmp -s out/${tag}_run_cuda.txt out/${tag}_run_1.txt && echo "TEXT IDENTICAL" || echo "TEXT DIFFERS"
echo GRANITE_DONE
