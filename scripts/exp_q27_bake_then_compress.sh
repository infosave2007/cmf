#!/bin/bash
# Owner 29.10: the coding skill baked into Qwen3.8-27B first (DTG-MA soft gates on the deep half, small
# constant L1, held-out best, Soft-Fold into down_proj), then the usual compress to 14B — against the
# clean run's 14B (compared in S6). Teacher data: the 27B's own answers to the 483 prompts.
set -u
ROOT=${ROOT:-/root/q27b}
B=${BIN:-/root/q27c/cortiq46}
SRC=${SRC:-/root/q27c/p14B.work/S0.prepare/src-f16.cmf}
DATA=${DATA:-/root/q27/q27-code.tdata}
REF=${REF:-/root/q27c/p14B.rejected.cmf}
FROM=${FROM:-32}
STEPS=${STEPS:-300}
mkdir -p "$ROOT" && cd "$ROOT"
export XDG_RUNTIME_DIR=/tmp
J=$ROOT/JOURNAL.md
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$J"; }
run() { local name=$1; shift; local t0=$SECONDS; log "start $name"; "$@" > "$name.log" 2>&1; local rc=$?; log "end $name (exit $rc, $((SECONDS - t0)) s)"; return $rc; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 >> "$ROOT/gpu.csv" &
mon=$!
[ -f baked27.cmf ] || run bake "$B" dtgma-bake "$SRC" "$DATA" --output "$ROOT/baked27.cmf" --from-layer "$FROM" \
  --steps "$STEPS" --eval-every 50 --max-batches "$STEPS" || { kill $mon; exit 1; }
grep -E "S2 \(device\)|\[A\]|dtgma bake" bake.log | tee -a "$J"
run compress "$B" compress "$ROOT/baked27.cmf" "$DATA" "$ROOT/p14B.cmf" q4tp --size 14B --min-width 0.2 \
  --illum-tokens 200000 --keep-grams off --heal fcd --heal-loss ce --heal-epochs 2 --write-on-fail \
  --work-dir "$ROOT/p14B.work" --compare "$REF"
kill $mon 2> /dev/null
grep -E "^\[S[0-9]\.[a-z]+\] (done|cached)" "$ROOT/p14B.work/JOURNAL.md" | tee -a "$J"
sed -n '/^| /p' "$ROOT/p14B.work/S6.eval/report.md" | tee -a "$J"
echo Q27 BAKE DONE | tee -a "$J"
