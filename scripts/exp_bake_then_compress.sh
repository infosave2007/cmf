#!/bin/bash
# Bake the coding skill into Qwen3.5-0.8B first (DTG-MA mask trained to its held-out best at a small constant
# L1, Soft-Fold, full-layer FCD of the last 4 layers), then compress as usual — against compressing directly.
# Teacher: Qwen3.8-27B answers to the 483 coding prompts, re-tokenized for the 0.8B. Same final size (A's
# parameter count), q4tp; B's S6 scores A's file beside its own (--compare).
set -u
ROOT=${ROOT:-/root/ord}
B=${BIN:-/root/q27c/cortiq48}
D=${DATA:-$ROOT/q08t27.tdata}
HF=${HF:-$ROOT/hf/Qwen3.5-0.8B}
STEPS_A=${STEPS_A:-1600}
# held-out records that fit the 4096-token chunk (67 of the 88 val records do)
HELD=${HELD:-60}
# train records that fit it (677 of 789); the bake demands exactly this many
CALIB=${CALIB:-670}
cd "$ROOT"
export XDG_RUNTIME_DIR=/tmp
J=$ROOT/JOURNAL-bake.md
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$J"; }
run() { local name=$1; shift; local t0=$SECONDS; log "start $name"; "$@" > "$name.log" 2>&1; local rc=$?; log "end $name (exit $rc, $((SECONDS - t0)) s)"; }

[ -f t27-train.txt ] || python3 -I tdata2text.py "$D" train t27-train.txt
[ -f t27-val.txt ] || python3 -I tdata2text.py "$D" val t27-val.txt

run A-compress "$B" compress "$HF" "$D" "$ROOT/bA.cmf" q4tp --ffn 0.75 --heal fcd --heal-loss ce --heal-epochs 3 \
  --work-dir "$ROOT/bw-A" --write-on-fail
sed -n '/^| /p' bw-A/S6.eval/report.md | tee -a "$J"
A=$ROOT/bA.cmf; [ -f "$A" ] || A=$ROOT/bA.rejected.cmf
P=$(python3 -I -c "import json; print(json.load(open('$ROOT/bw-A/S2.plan/done.json'))['params_after'])")
log "A: $P params"

# the device bake (`dtgma-bake`): `skill bake` is the host engine and ran 43 min here without reaching
# its first evaluation; the same recipe as the 27B run (gates on the deeper half, 300 steps)
run B1-bake "$B" dtgma-bake bw-A/S0.prepare/src-f16.cmf "$D" --output "$ROOT/baked.cmf" --from-layer "${FROM:-12}" \
  --steps "${STEPS:-300}" --eval-every 50
grep -E "S2 \(device\)|\[A\]|dtgma bake|rror" B1-bake.log | tail -20 | tee -a "$J"

run B2-compress "$B" compress "$ROOT/baked.cmf" "$D" "$ROOT/bB.cmf" q4tp --size "$P" --heal fcd --heal-loss ce \
  --heal-epochs 3 --work-dir "$ROOT/bw-B" --write-on-fail --compare "$A"
sed -n '/^| /p' bw-B/S6.eval/report.md | tee -a "$J"
echo BAKE EXP DONE | tee -a "$J"
