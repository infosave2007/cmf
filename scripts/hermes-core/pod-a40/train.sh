#!/usr/bin/env bash
# One pass of the grown 340M genome over the text corpus on the pod
# (Hermes sources, OS command documentation, Stack Exchange: 493.3 M tokens
# = 60 200 steps of B8 x T1024). A stage is 2000 steps: held-out losses per
# domain and a checkpoint; `stage_sync.py` turns each stage into a report
# and a backup on the Hub.
#
#   train.sh            (detached: setsid nohup ./train.sh < /dev/null > /dev/null 2>&1 &)
#
# A run that dies is resumed from its last checkpoint on the same schedule
# (--schedule-from 0), up to three times; the sampler then starts its
# streams again. Files: run/pass1.log, run/pass1.ckpt, run/pass1.start,
# run/pass1.done or run/pass1.failed.
set -u
cd /root/hc/data
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
E=/root/hc/cmf/target/release/cortiq-embryo
R=/root/hc/run
NAME="${NAME:-pass1}"
STEPS="${STEPS:-60200}"
STAGE="${STAGE:-2000}"
date -u +%s > "$R/$NAME.start"
common=(--shard se.u16:0.884 docs.u16:0.076 hermes_code.u16:0.040 --val val.u16
  --val-extra hermes=val_hermes.u16 --val-extra docs=val_docs.u16 --val-extra se=val_se.u16
  --batch 8 --seq 1024 --steps "$STEPS" --warmup 300 --lr 1e-4 --lr-restart
  --eval-every "$STAGE" --save-every "$STAGE"
  --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 --out "$R/$NAME.ckpt")
for attempt in 1 2 3 4; do
  if [ -f "$R/$NAME.ckpt" ]; then src="$R/$NAME.ckpt"; extra=(--schedule-from 0); else src=grown340.ckpt; extra=(); fi
  echo "== attempt $attempt, resume $src, $(date -u +%FT%TZ)" >> "$R/$NAME.log"
  "$E" birth --resume "$src" "${common[@]}" "${extra[@]}" >> "$R/$NAME.log" 2>&1
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
