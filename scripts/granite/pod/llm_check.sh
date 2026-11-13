#!/bin/bash
# An LLM on CUDA against Vulkan: 64-token decode bench and the greedy text of
# one prompt on each. llm_check.sh MODEL TAG [PROMPT]
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
f=$1
tag=$2
prompt=${3:-Translate into French: The weather is nice today and we are going for a walk in the park.}
bin=${BIN:-cortiq-fast}
exec 9>"$root/gpu.lock"; flock 9
for be in cuda 1; do
  CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 900 bin/$bin bench $f --tokens 64 --json > out/${tag}_bench_$be.log 2>&1
  echo "== $be: $(grep -o '"decode_tok_s_steady": *[0-9.]*' out/${tag}_bench_$be.log | head -1) $(grep -o '"prefill_tok_s": *[0-9.]*' out/${tag}_bench_$be.log | head -1)"
  grep -ohE "cuda [a-z ]*declined: [^\"]*" out/${tag}_bench_$be.log | sort -u | head -3 | cut -c1-200
  CMF_GPU=$be timeout 900 bin/$bin run $f --prompt "$prompt" -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/${tag}_run_$be.txt
done
cmp -s out/${tag}_run_cuda.txt out/${tag}_run_1.txt && echo "GENERATED TEXT IDENTICAL" || { echo "GENERATED TEXT DIFFERS"; tail -3 out/${tag}_run_cuda.txt; tail -3 out/${tag}_run_1.txt; }
echo LLM_CHECK_DONE
