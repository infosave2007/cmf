#!/bin/bash
# Greedy answers to the prompt set, one process per prompt; the stats lines
# land in out/<tag>/<prompt>.log, a summary in out/<tag>/summary.txt.
#   run_prompts.sh <tag> <bin> <model> <max_tokens> <prompt names...> [-- ENV=VALUE ...]
set -u
tag=$1; bin=$2; model=$3; ntok=$4; shift 4
names=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do names+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
for kv in "$@"; do export "$kv"; done
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
root=/root/cuda; out=$root/out/$tag; mkdir -p "$out"
: > "$out/summary.txt"
for n in "${names[@]}"; do
  timeout 1800 "$bin" run "$model" --prompt-file "$root/src/scripts/q27cuda/prompts/$n.txt" --greedy --no-think -n "$ntok" \
    > "$out/$n.out" 2> "$out/$n.log"
  rc=$?
  st=$(grep -h -E 'tokens · first token' "$out/$n.out" "$out/$n.log" | tail -1)
  tr=$(grep -h -E 'speculation trial' "$out/$n.log" | tail -1)
  h=$(grep -v -E 'tokens · first token|^\[' "$out/$n.out" | md5sum | cut -c1-8)
  echo "$n rc=$rc $st | $tr | text $h" | tee -a "$out/summary.txt"
done
echo "RUN_DONE $tag" >> "$out/summary.txt"
