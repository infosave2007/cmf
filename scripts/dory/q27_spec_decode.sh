#!/bin/bash
# Decode with drafts against decode without, one run at a time with a pause before each (a fanless
# Mac throttles): q27_spec_decode.sh CORTIQ MODEL PROMPT_DIR LOG PAUSE TOKENS [ENV=VAL ...]
# The extra ENV goes to the run with drafts. A pair's texts have to be the same.
cortiq=$1; model=$2; d=$3; L=$4; pause=$5; n=$6; shift 6
one() { # NAME FILE ENV...
  local name=$1 f=$2; shift 2
  sleep "$pause"
  env "$@" RUST_LOG=info "$cortiq" run "$model" --prompt-file "$f" --greedy --no-think --max-tokens "$n" > "$L.out" 2> "$L.err"
  echo "$name: $(grep -E '^\[' "$L.out" | head -1 | cut -c1-110) | text $(grep -vE '^\[' "$L.out" | md5 -q | cut -c1-8)" >> "$L"
  sed -e 's/\x1b\[[0-9;]*m//g' "$L.err" | grep -iE "panicked|ERROR|speculation" | sed -e 's/^.*pipeline: //' | tail -2 | cut -c1-200 >> "$L"
}
for p in prose ru code quote edit; do
  one "$p plain" "$d/p_$p.txt" CMF_GRAPH_SPEC=0
  one "$p drafts" "$d/p_$p.txt" CMF_GRAPH_SPEC=1 "$@"
done
echo DONE >> "$L"
