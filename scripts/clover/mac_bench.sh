#!/bin/bash
# mac_bench.sh <tag> [ENV=VAL ...]: prose and code prompts on one model, one at a time,
# each with a time limit; prints the speed lines.
B=${BIN:-$HOME/dev/cmf-clover-runs/target-fast/release/cortiq}
R=$HOME/dev/cmf-clover-runs/mac-2026-11-01
M=${MODEL:-$R/hf/clover-35b-a3b-preview-q4tp-v2.cmf}
tag=$1; shift
mkdir -p "$R/out"
for p in prose code; do
  case $p in
    prose) P="Write a short essay about the history of the Roman Empire and why it fell.";;
    code)  P="Write a Python function that merges two sorted lists, then write unit tests for it.";;
  esac
  o=$R/out/$tag-$p.txt
  env "$@" perl -e 'alarm shift; exec @ARGV' "${TMO:-400}" "$B" run "$M" --prompt "$P" --no-think --max-tokens "${MAXT:-256}" > "$o" 2>&1
  echo "== $tag $p rc=$?: $(grep -E '^\[[0-9]+ tokens' "$o" | cut -c1-120)"
  grep -E '^\[(self-speculation|block diffusion|Metal)' "$o" | cut -c1-240
done
