#!/bin/bash
# A routed-FFN model's long prompt on CUDA (16-row chunks through the small
# path) against Vulkan: bench --ctx 1024 and 4096, and the greedy text of the
# short prompt (fast build unless BIN=cortiq). moe_prefill.sh REPO FILE
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
repo=$1; file=$2
bin=${BIN:-cortiq-fast}
if [ "$bin" = cortiq-fast ]; then
  bash scripts/build.sh fast > logs/mp_build.log 2>&1 || { echo BUILD_FAILED; tail -30 logs/mp_build.log; exit 1; }
fi
[ -s cmf/$file ] || venv/bin/hf download "$repo" "$file" --local-dir cmf > logs/mp_dl.log 2>&1
tag=$(basename "$file" .cmf)_mp
BIN=$bin bash scripts/llm_check.sh cmf/$file $tag "Translate into French: The weather is nice today and we are going for a walk in the park."
for ctx in 1024 4096; do
  for be in cuda 1; do
    echo "ctx $ctx $be: $(flock gpu.lock env CMF_GPU=$be timeout 1200 bin/$bin bench cmf/$file --tokens 32 --ctx $ctx --json 2>/dev/null | grep -oE '(decode_tok_s_steady|prefill_tok_s|ttft_s).: *[0-9.]*' | tr '\n' ' ')"
  done
done
echo MOE_PREFILL_DONE
