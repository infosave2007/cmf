#!/bin/bash
# Two draft heads on a Mac, one run at a time with a pause before each (a fanless Mac throttles):
# q27_mtp_ab.sh CORTIQ OLD_MODEL NEW_MODEL PROMPT_DIR LOG PAUSE TOKENS NAME... [-- ENV=VAL ...]
# Per prompt p_NAME.txt: OLD without drafts, OLD with drafts, NEW with drafts. The three texts have to be the same.
# The extra ENV goes to the runs with drafts.
cortiq=$1; old=$2; new=$3; d=$4; L=$5; pause=$6; n=$7; shift 7
names=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do names+=("$1"); shift; done; [ "$1" = "--" ] && shift
one() { # LABEL MODEL FILE ENV...
  local label=$1 model=$2 f=$3; shift 3
  sleep "$pause"
  env "$@" RUST_LOG=info "$cortiq" run "$model" --prompt-file "$f" --greedy --no-think --max-tokens "$n" > "$L.out" 2> "$L.err"
  echo "$label: $(grep -E '^\[' "$L.out" | head -1 | cut -c1-110) | text $(grep -vE '^\[|^Loading model' "$L.out" | md5 -q | cut -c1-8)" >> "$L"
  sed -e 's/\x1b\[[0-9;]*m//g' "$L.err" | grep -iE "panicked|ERROR|speculation|draft depth" | sed -e 's/^.*pipeline: //' | tail -2 | cut -c1-200 >> "$L"
}
for p in "${names[@]}"; do
  one "$p old plain" "$old" "$d/p_$p.txt" CMF_GRAPH_SPEC=0
  one "$p old drafts" "$old" "$d/p_$p.txt" CMF_GRAPH_SPEC=1 "$@"
  one "$p new drafts" "$new" "$d/p_$p.txt" CMF_GRAPH_SPEC=1 "$@"
done
echo DONE >> "$L"
