#!/bin/bash
# Method order on Qwen3.5-0.8B at a 4-bit budget, 75 % FFN, equal file size:
#   A = DTG-MA -> NVG : compress --select dtgma (masks, fold, heal), NVG-encode the narrowed model, metric training
#   B = NVG -> DTG-MA : NVG-encode the source, nvg-train --mask-keep 0.75 (gates, cut, metric)
# B's FFN tensors are encoded at BPW/1.074: cutting 25 % of the rows of a binary low-rank factor
# removes ~19.5 % of its bytes, not 25 %, so equal bits per weight would leave B larger.
# Both are scored on the same held-out task records by compress S6 (--compare).
set -euo pipefail
ROOT=${ROOT:-/root/ord}
BIN=${BIN:-/root/q27/cortiq}
HF=${HF:-$ROOT/hf/Qwen3.5-0.8B}
DATA=${DATA:-$ROOT/td08.tdata}
BPW=${BPW:-4.0}
TOK=${TOK:-262144}
cd "$ROOT"
J=$ROOT/JOURNAL.md
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$J"; }
run() { local name=$1; shift; local t0=$SECONDS; log "start $name"; "$@" > "$name.log" 2>&1 || { rc=$?; [ $rc -eq 5 ] || { log "FAILED $name ($rc)"; exit $rc; }; }; log "end $name ($((SECONDS - t0)) s)"; }
cmp_args=()
[ -f a-final.cmf ] && cmp_args+=(--compare "$ROOT/a-final.cmf")
[ -f b-final.cmf ] && cmp_args+=(--compare "$ROOT/b-final.cmf")

[ -f a.cmf ] || run A1-dtgma "$BIN" compress "$HF" "$DATA" "$ROOT/a.cmf" --ffn 0.75 --select dtgma --work-dir "$ROOT/a.work"
SRC=$ROOT/a.work/S0.prepare/src-f16.cmf
[ -f a-nvg.cmf ] || run A2-nvg "$BIN" convert --model "$ROOT/a.cmf" --quant blr --bpw "$BPW" --blr-method nvg --blr-paths 2 --calib task-train.txt --output "$ROOT/a-nvg.cmf"
[ -f a-final.cmf ] || run A3-metric "$BIN" nvg-train "$ROOT/a-nvg.cmf" --teacher "$SRC" --calib task-train.txt --tokens "$TOK" --epochs 2 --output "$ROOT/a-final.cmf"

if [ ! -f b-nvg.cmf ]; then
  nl=$(python3 -I -c "import json,sys; print(json.load(open('$HF/config.json')).get('text_config', json.load(open('$HF/config.json')))['num_hidden_layers'])")
  bf=$(python3 -I -c "print(round($BPW / 1.074, 4))")
  : > b-plan.txt
  for ((l = 0; l < nl; l++)); do
    for t in gate_proj up_proj down_proj; do echo "tensor model.layers.$l.mlp.$t.weight $bf" >> b-plan.txt; done
  done
  run B1-nvg "$BIN" convert --model "$SRC" --quant blr --bpw "$BPW" --blr-plan b-plan.txt --blr-method nvg --blr-paths 2 --calib task-train.txt --output "$ROOT/b-nvg.cmf"
fi
[ -f b-final.cmf ] || run B2-mask-metric "$BIN" nvg-train "$ROOT/b-nvg.cmf" --teacher "$SRC" --calib task-train.txt --mask-keep 0.75 --tokens "$TOK" --epochs 2 --output "$ROOT/b-final.cmf"

ls -la a.cmf a-nvg.cmf a-final.cmf b-nvg.cmf b-final.cmf | tee -a "$J"
run S6-compare "$BIN" compress "$HF" "$DATA" "$ROOT/a.cmf" --ffn 0.75 --select dtgma --work-dir "$ROOT/a.work" --compare "$ROOT/a-final.cmf" --compare "$ROOT/b-final.cmf"
sed -n '/^| /p' "$ROOT/a.work/S6.eval/report.md" | tee -a "$J"
