#!/usr/bin/env bash
# Fine-tune an Embryo checkpoint on the card tasks as a chain of SFT stages
# (the trainer caps a stage at 500 steps), exporting and answering the R set
# at the listed steps. Grading (grade.py, needs Docker) is a separate step:
# a 340M genome leaves no memory for the Docker VM on a 24 GB Mac.
#
#   ft_chain.sh NAME START_CKPT BATCH LR "EVAL_STEPS" LAST_STEP
#
# Environment: STAGE0 (run root), EMBRYO (trainer), CORTIQ (engine),
# O1 (the Embryo run directory with the replay and raw-val shards),
# SET (prefix of the SFT shards in $STAGE0/ft: SET-train.sft, SET-dev.sft,
# SET-final.sft) [cards], SEQ (their window) [512]. With an empty EVAL_STEPS
# nothing is answered here (a model of the lookup contract goes through
# run_lookup.py).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
B="${STAGE0:-$HOME/embryo-data/hermes-core/stage0}"
E="${EMBRYO:-$REPO/target/release/cortiq-embryo}"
C="${CORTIQ:-$REPO/target/release/cortiq}"
O="${O1:-$HOME/embryo-data/o1}"
PY="$B/venv/bin/python"
TOK="$O/sft-t27/gen1.tokenizer.json"
NAME="$1"; START="$2"; BATCH="$3"; LR="$4"; EVALS=" $5 "; LAST="$6"
SET="${SET:-cards}"; SEQ="${SEQ:-512}"
cd "$B/ft"
log() { echo "$(date +%T) $*" | tee -a "$NAME.status"; }

prev=0; prev_ck="$START"
while [ "$prev" -lt "$LAST" ]; do
  if [ "$prev" -eq 0 ]; then next=100; extra="--fresh-moments"
  elif [ "$prev" -eq 100 ]; then next=500; extra="--allow-extended --sampler-skip $prev --skip-eval-before"
  else next=$((prev + 500)); extra="--allow-extended --sampler-skip $prev --skip-eval-before"; fi
  [ "$next" -gt "$LAST" ] && next="$LAST"
  out="$NAME-$next.ckpt"
  # shellcheck disable=SC2086
  "$E" sft --ckpt "$prev_ck" --shard "$SET-train.sft" --val "$SET-dev.sft" --final-val "$SET-final.sft" \
    --replay "$O/chat-sft/replay-16m.u16" --raw-val "$O/sft-t27/rawval-tails-2m.u16" --out "$out" \
    --batch "$BATCH" --seq "$SEQ" --steps "$next" --val-batches 32 --raw-val-batches 8 --lr "$LR" --dropless $extra \
    > "$NAME-stage-$next.log" 2>&1
  [ -f "$out" ] || { log "stage $next FAILED, see $NAME-stage-$next.log"; exit 1; }
  rec="$(grep 'SFT receipt' "$NAME-stage-$next.log" | tail -1)"
  ms="$(grep -o 'gpu_ms=[0-9.]*' "$NAME-stage-$next.log" | tail -20 | cut -d= -f2 | awk '{s+=$1} END {printf "%.0f", s/NR}')"
  log "step $next: dev NLL $(echo "$rec" | grep -o 'response_after=[0-9.]*' | cut -d= -f2), final $(echo "$rec" | grep -o 'Some([0-9.]*' | cut -d'(' -f2), raw LM $(echo "$rec" | grep -o 'raw_after=[0-9.]*' | cut -d= -f2), $ms ms/step"
  [ "$prev_ck" != "$START" ] && rm -f "$prev_ck"
  if [[ "$EVALS" == *" $next "* ]]; then
    "$E" export --ckpt "$out" --tokenizer "$TOK" --out "$NAME-$next.cmf" --dtype f16 --chat > /dev/null 2>&1
    "$PY" -I "$HERE/run_cortiq.py" --cortiq "$C" --model "$NAME-$next.cmf" --tasks "$B/run/r.jsonl" \
      --out "$B/run/$NAME-$next.outputs.jsonl" >> "$NAME.status" 2>&1
    rm -f "$NAME-$next.cmf"
    log "step $next: answered R"
  fi
  prev="$next"; prev_ck="$out"
done
log "done: $prev_ck"
