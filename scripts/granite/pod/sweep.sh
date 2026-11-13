#!/bin/bash
# Release build, then per line "repo file" of LIST (stdin): download if
# missing, llm_check (CUDA vs Vulkan) plus two bench runs a backend, and the
# CPU's greedy text against both; the file deleted afterwards when DEL=1.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/sweep_build.log 2>&1 || { echo BUILD_FAILED; exit 1; }
p="Write a Python function that checks whether a number is prime."
while read -r repo file; do
  [ -z "$repo" ] && continue
  tag=$(basename "$file" .cmf)_s
  [ -s cmf/$file ] || venv/bin/hf download "$repo" "$file" --local-dir cmf > logs/${tag}_dl.log 2>&1
  [ -s cmf/$file ] || { echo "== $tag download failed"; continue; }
  echo "#### $tag"
  BIN=cortiq bash scripts/llm_check.sh cmf/$file $tag "$p" | grep -v '^Prompt\|^$' | cut -c1-200 | head -6
  CMF_GPU=0 timeout 900 bin/cortiq run cmf/$file --prompt "$p" -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/${tag}_cpu.txt
  for be in cuda 1; do cmp -s out/${tag}_cpu.txt out/${tag}_run_$be.txt && echo "$be text == CPU" || echo "$be text != CPU"; done
  for i in 1 2; do
    for be in cuda 1; do
      echo "$be bench $i: $(flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq bench cmf/$file --tokens 64 ${BENCH_ARGS:-} --json 2>/dev/null | grep -oE '(decode_tok_s_steady|prefill_tok_s).: *[0-9.]*' | tr '\n' ' ')"
    done
  done
  [ "${DEL:-0}" = 1 ] && rm -f cmf/$file
done
echo SWEEP_DONE
