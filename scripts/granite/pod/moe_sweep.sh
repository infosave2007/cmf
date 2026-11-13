#!/bin/bash
# The routed-FFN models on the CUDA hybrid arm, release build: per line
# "repo file prompt-kind" download, llm_check (CUDA vs Vulkan), two more bench
# runs each, delete the file. Results in logs/moe_sweep.out.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/moes_build.log 2>&1 || { echo BUILD_FAILED; exit 1; }
code="Write a Python function that checks whether a number is prime."
tr_fr="Translate into French: The weather is nice today and we are going for a walk in the park."
while read -r repo file kind; do
  [ -z "$repo" ] && continue
  tag=$(basename "$file" .cmf)
  [ -s cmf/$file ] || venv/bin/hf download "$repo" "$file" --local-dir cmf > logs/${tag}_dl.log 2>&1
  [ -s cmf/$file ] || { echo "== $tag download failed"; continue; }
  p=$code; [ "$kind" = tr ] && p=$tr_fr
  echo "#### $tag"
  BIN=cortiq bash scripts/llm_check.sh cmf/$file $tag "$p"
  for i in 1 2; do
    for be in cuda 1; do
      echo "$be bench $i: $(flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq bench cmf/$file --tokens 64 --json 2>/dev/null | grep -oE '(decode_tok_s_steady|prefill_tok_s|ttft_s).: *[0-9.]*' | tr '\n' ' ')"
    done
  done
  rm -f cmf/$file
done <<LIST
infosave/Qwen3.6-35B-A3Bcmf qwen36-35b-a3b-q4tp.cmf code
infosave/Ornith-1.5-cmf ornith-1.5-35b-a3b-q4tp.cmf code
infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp-v2.cmf code
infosave/KAT-Coder-V2.5-CMF KAT-Coder-V2.5-CMF.cmf code
infosave/Hy-MT2-cmf hy-mt2-30b-a3b-q4tp.cmf tr
LIST
echo MOE_SWEEP_DONE
