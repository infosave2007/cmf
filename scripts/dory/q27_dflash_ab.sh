#!/bin/bash
# The DFlash drafter against the model's own MTP block on a Mac, one run at a time with a pause
# before each (a fanless Mac throttles):
# q27_dflash_ab.sh CORTIQ MODEL PROMPT_DIR LOG PAUSE TOKENS NAME... [-- ARM...]
# MODEL has a `.dflash.cmf` beside it. Per prompt p_NAME.txt the arms run in order; the default ones:
#   mtp     the MTP chain alone (no sidecar loaded)
#   dflash  every round the DFlash block
#   auto    the round goes to the drafter that holds more
# also: plain (no drafts), k1..k7 (the MTP chain pinned at that depth, every round taken).
# The texts of a prompt have to be the same in every arm on native Metal; on wgpu a batched verify
# is not the token graph bit for bit and a near tie can turn.
cortiq=$1; model=$2; d=$3; L=$4; pause=$5; n=$6; shift 6
names=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do names+=("$1"); shift; done; [ "$1" = "--" ] && shift
arms=("$@"); [ ${#arms[@]} -eq 0 ] && arms=(mtp dflash auto)
sum8() { if command -v md5 > /dev/null; then md5 -q | cut -c1-8; else md5sum | cut -c1-8; fi; }
one() { # LABEL FILE ENV...
  local label=$1 f=$2; shift 2
  sleep "$pause"
  env "$@" CMF_GRAPH_SPEC_TIME=1 RUST_LOG=info "$cortiq" run "$model" --prompt-file "$f" --greedy --no-think --max-tokens "$n" > "$L.out" 2> "$L.err"
  echo "$label: $(grep -E '^\[' "$L.out" | head -1 | cut -c1-110) | text $(grep -vE '^\[|^Loading model' "$L.out" | sum8)" >> "$L"
  grep "spec-round:" "$L.err" | awk '{d+=$3; v+=$8; c+=$13; a+=$(NF-8); n++} END {if (n) printf "   rounds %d: draft %.1f ms, verify %.1f ms, commit %.1f ms\n", n, d/n, v/n, c/n}' >> "$L"
  echo "   drafter turns: $(grep -c 'spec-drafter: the DFlash' "$L.err") to DFlash, $(grep -c 'spec-drafter: the MTP' "$L.err") to MTP; $(grep -E '^\[MTP draft' "$L.out" | head -1)" >> "$L"
  sed -e 's/\x1b\[[0-9;]*m//g' "$L.err" | grep -iE "panicked|ERROR" | tail -2 | cut -c1-200 >> "$L"
}
for p in "${names[@]}"; do
  for arm in "${arms[@]}"; do
    case $arm in
      plain)  one "$p plain"  "$d/p_$p.txt" CMF_GRAPH_SPEC=0 ;;
      mtp)    one "$p mtp"    "$d/p_$p.txt" CMF_GRAPH_SPEC=1 ;;
      dflash) one "$p dflash" "$d/p_$p.txt" CMF_GRAPH_SPEC=1 CMF_DFLASH=1 CMF_DFLASH_PICK=dflash ;;
      auto)   one "$p auto"   "$d/p_$p.txt" CMF_GRAPH_SPEC=1 CMF_DFLASH=1 ;;
      k[1-7]) one "$p $arm"   "$d/p_$p.txt" CMF_GRAPH_SPEC=1 CMF_GRAPH_SPEC_K=${arm#k} CMF_GRAPH_SPEC_PIN=1 ;;
    esac
  done
done
echo DONE >> "$L"
