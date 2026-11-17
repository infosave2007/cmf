#!/usr/bin/env bash
# One `birth` run of the Hermes core campaign on the pod, described by a
# config file (pass1.conf, base1.conf): where it starts, how far it goes,
# which shards with which weights, which held-out sets.
#
#   train.sh CONF       (detached: setsid nohup ./train.sh base1.conf < /dev/null > /dev/null 2>&1 &)
#
# The config is a bash file that sets
#   NAME     run name: run/NAME.log, run/NAME.ckpt, run/NAME.start, run/NAME.done | run/NAME.failed
#   START    the checkpoint the run begins from
#   ORIGIN   the step of START (0 for a checkpoint without optimizer moments): the schedule and
#            the stages count from it
#   STEPS    the last step, absolute
#   SHARDS   (file:weight ...)  under /root/hc/data
#   VAL      the main held-out shard;  VALS  (name=file ...)  the per-domain ones
# and optionally STAGE [2000] (steps between held-out losses and checkpoints), WARMUP [300],
# LR [1e-4], SEED [17], BATCH [8].
#
# A run that dies or is stopped is resumed from its last checkpoint on the
# same schedule (--schedule-from ORIGIN), up to three times per launch.
# Every attempt takes its own sampler seed (SEED + the attempts already in
# the log): a resumed sampler starts its streams anew, and with one seed it
# would read the run's first windows again. `stage_sync.py` turns each
# stage into a report and a backup on the Hub.
set -u
source "$1"
: "${NAME:?}" "${START:?}" "${STEPS:?}" "${VAL:?}"
ORIGIN="${ORIGIN:-0}"; STAGE="${STAGE:-2000}"; WARMUP="${WARMUP:-300}"; LR="${LR:-1e-4}"; SEED="${SEED:-17}"; BATCH="${BATCH:-8}"
cd /root/hc/data
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
E=/root/hc/cmf/target/release/cortiq-embryo
R=/root/hc/run
[ -f "$R/$NAME.start" ] || date -u +%s > "$R/$NAME.start"
vals=()
for v in "${VALS[@]}"; do vals+=(--val-extra "$v"); done
common=(--shard "${SHARDS[@]}" --val "$VAL" "${vals[@]}"
  --batch "$BATCH" --seq 1024 --steps "$STEPS" --warmup "$WARMUP" --lr "$LR" --lr-restart
  --eval-every "$STAGE" --save-every "$STAGE"
  --carry --carry-reset-every 16 --carry-eot 32760 --dropless --out "$R/$NAME.ckpt")
for attempt in 1 2 3 4; do
  if [ -f "$R/$NAME.ckpt" ]; then src="$R/$NAME.ckpt"; extra=(--schedule-from "$ORIGIN"); else src="$START"; extra=(); fi
  prior=$(grep -c "^== attempt" "$R/$NAME.log" 2>/dev/null)
  seed=$((SEED + ${prior:-0}))
  echo "== attempt $attempt, resume $src, seed $seed, $(date -u +%FT%TZ)" >> "$R/$NAME.log"
  "$E" birth --resume "$src" "${common[@]}" "${extra[@]}" --seed "$seed" >> "$R/$NAME.log" 2>&1
  code=$?
  if [ "$code" -eq 0 ] && grep -q "saved .* (step $STEPS)" "$R/$NAME.log"; then
    echo done > "$R/$NAME.done"
    exit 0
  fi
  echo "== exit $code, $(date -u +%FT%TZ)" >> "$R/$NAME.log"
  sleep 20
done
echo failed > "$R/$NAME.failed"
exit 1
