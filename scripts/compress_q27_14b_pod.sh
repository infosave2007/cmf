#!/bin/bash
# Qwen3.8-27B -> 14B for coding with `cortiq compress`, on one RTX PRO 6000 (96 GB) pod.
#
#   check   environment: GPU, Vulkan, cgroup RAM/CPU, local disk
#   gate    S0+S1 once, then fold WITHOUT heal at every size in $SIZES (S2, S3, S6 each);
#           one illumination serves them all (Grams kept on disk)
#   heal    heal + evaluate the chosen size ($POINT), then q4tp
#   clean   delete the Grams and the f16 copies of the sizes not healed
#
# Inputs (local disk, never /workspace):
#   $HF     the bf16 checkpoint (hf download Qwen/Qwen3.8-27B)
#   $DATA   teacher data: the 27B's own no-think answers via the API, tokenized
#           with the 27B tokenizer (q27-code.tdata)
#   $BIN    cortiq built from this branch
# Every stage is cached in its work dir: a rerun continues where it stopped.
# The same script dry-runs on a Mac with a small model:
#   ROOT=… HF=…/Qwen3.5-0.8B DATA=….tdata BIN=… SIZES="0.70B 0.65B 0.60B" DISK_GB=20 \
#     scripts/compress_q27_14b_pod.sh gate
set -euo pipefail
ROOT=${ROOT:-/root/q27}
HF=${HF:-$ROOT/hf/Qwen3.8-27B}
DATA=${DATA:-$ROOT/q27-code.tdata}
BIN=${BIN:-$ROOT/cortiq}
SIZES=${SIZES:-"20B 17B 14B"}
POINT=${POINT:-14B}
HEAL_EPOCHS=${HEAL_EPOCHS:-2}
# illumination tokens: 200k = 19 per kept neuron at the 20B point (0.8B: 20 per neuron = all 166k);
# the default 20 × 17408 = 348k costs ~1.7× the forward for nothing. Empty = the default.
ILLUM_TOKENS=${ILLUM_TOKENS-200000}
# S0 f16+q8 copies 82 + Grams 155 + three folds (f16+q8: 60/51/42) + heal 42 + q4tp 8;
# the HF checkpoint (56) sits outside: rent a pod with >= 600 GB of local disk
DISK_GB=${DISK_GB:-480}
J=$ROOT/JOURNAL.md
phase=${1:-check}
read -r -a sizes <<< "$SIZES"
base=p${sizes[0]}

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$J"; }
need_disk() { # GB
  local free
  free=$(df -Pk "$ROOT" | awk 'NR == 2 { print int($4 / 1048576) }')
  [ "$free" -ge "$1" ] || { log "STOP: $free GB free on $ROOT, need $1"; exit 3; }
}
# one stage chain; every size but the first shares the first one's S0/S1
compress() { # size extra...
  local size=$1 name=p$1; shift
  local w=$ROOT/$name.work
  if [ "$name" != "$base" ] && [ ! -e "$w/S1.illuminate" ]; then
    mkdir -p "$w"
    ln -sfn "$ROOT/$base.work/S0.prepare" "$w/S0.prepare"
    ln -sfn "$ROOT/$base.work/S1.illuminate" "$w/S1.illuminate"
  fi
  log "start $name $*"
  local t0=$SECONDS rc
  set +e
  "$BIN" compress "$HF" "$DATA" "$ROOT/$name.cmf" --size "$size" --work-dir "$w" \
    --min-width 0.2 --keep-grams on ${ILLUM_TOKENS:+--illum-tokens "$ILLUM_TOKENS"} "$@" > "$ROOT/$name.log" 2>&1
  rc=$?
  set -e
  grep -E "^S[0-6]:|PASS|FAIL|rror" "$ROOT/$name.log" | tail -12 | tee -a "$J"
  log "end $name ($((SECONDS - t0)) s) exit $rc"
  [ $rc -eq 0 ] || [ $rc -eq 5 ] # 5 = quality gates failed (the report is still written)
}

mkdir -p "$ROOT"
case $phase in
check)
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
  "$BIN" gpu 2>&1 | head -5
  echo "cgroup RAM: $(( $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 0) / 1073741824 )) GB"
  echo "cgroup CPU: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo -)"
  df -h "$ROOT"
  ls "$HF" | head -5
  ls "$DATA"
  ;;
gate)
  need_disk "$DISK_GB"
  for s in "${sizes[@]}"; do compress "$s" --heal none; done
  for s in "${sizes[@]}"; do
    sed -n '/^| /p' "$ROOT/p$s.work/S6.eval/report.md" | head -6 | sed "s/^/p$s: /" | tee -a "$J"
  done
  log "gate done — predicted healed NLL = source + 0.5·(fold − source) (0.8B and 2B measured 0.47–0.48)"
  ;;
heal)
  need_disk "${HEAL_DISK_GB:-60}"
  # the files stay even when a gate fails (*.rejected.cmf): they are what gets analysed
  compress "$POINT" --heal fcd --heal-epochs "$HEAL_EPOCHS" --write-on-fail
  compress "$POINT" q4tp --heal fcd --heal-epochs "$HEAL_EPOCHS" --write-on-fail
  ;;
clean)
  rm -rf "$ROOT/$base.work/S1.illuminate/grams"
  for s in "${sizes[@]}"; do
    [ "$s" = "$POINT" ] || rm -f "$ROOT/p$s.work/S3.fold/fold-f16.cmf"
  done
  df -h "$ROOT"
  ;;
*) echo "phase: check | gate | heal | clean"; exit 2 ;;
esac
