#!/bin/bash
# Greedy outputs of one backend for every prompt, with the ids and every
# decode step's logits kept (CMF_GEN_IDS_OUT, CMF_LOGIT_DUMP_ALL).
#   ref_outputs.sh <tag> <binary> <model.cmf> [FORCE=<other tag>] [ENV=VALUE ...]
# PROMPTS_DIR=<dir> (environment): the prompts to run instead of prompts/.
# FORCE=<tag>: teacher-force this backend along <tag>'s generated ids
# (CMF_FORCE_IDS), so logits compare step by step past a near-tie.
set -u
tag=$1; bin=$2; model=$3; shift 3
root=${CUDA_RUN_ROOT:-/root/cuda}
n=${REF_TOKENS:-96}
force=""
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0
for kv in "$@"; do
  case "$kv" in FORCE=*) force=${kv#FORCE=};; *) export "$kv";; esac
done
out=$root/out/ref/$tag; mkdir -p "$out"
for p in "${PROMPTS_DIR:-$root/prompts}"/*.txt; do
  name=$(basename "$p" .txt)
  rm -rf "$out/$name.logits"; mkdir -p "$out/$name.logits"
  unset CMF_FORCE_IDS
  if [ -n "$force" ]; then
    export CMF_FORCE_IDS=$root/out/ref/$force/$name.gen
    [ -f "$CMF_FORCE_IDS" ] || { echo "$tag/$name: no $CMF_FORCE_IDS"; continue; }
  fi
  t0=$(date +%s.%N)
  CMF_GEN_IDS_OUT=$out/$name.ids CMF_LOGIT_DUMP_ALL=$out/$name.logits \
    "$bin" run "$model" --prompt-file "$p" --greedy --no-think -n "$n" > "$out/$name.txt" 2> "$out/$name.log"
  rc=$?
  steps=$(ls "$out/$name.logits" | wc -l)
  # the generated ids alone: the last <steps> of the sequence
  python3 - "$out/$name.ids" "$steps" "$out/$name.gen" <<'PY'
import sys
ids = open(sys.argv[1]).read().split()
steps = int(sys.argv[2])
open(sys.argv[3], "w").write(" ".join(ids[len(ids) - steps:]) if steps else "")
PY
  echo "$tag/$name: rc=$rc steps=$steps $(printf '%.1f' "$(echo "$(date +%s.%N) - $t0" | bc)") s"
done
echo "REF_DONE $tag"
