#!/usr/bin/env bash
# When the previous run of the campaign is done and its uploader has sent the last stage,
# measure the held-out losses of its checkpoint on the next run's sets, start the next run
# and become its uploader. HF_TOKEN comes in the environment (read from stdin by the caller).
#
#   IFS= read -r HF_TOKEN; export HF_TOKEN; setsid nohup ./chain.sh pass1 base1.conf "<report title>" &
set -u
PREV="$1"; CONF="$2"; TITLE="$3"
cd /root/hc
source "$CONF"
R=/root/hc/run
for i in $(seq 1 4000); do
  [ -f "$R/$PREV.done" ] && break
  [ -f "$R/$PREV.failed" ] && { echo "$(date -u +%T) $PREV failed: nothing started" >> "$R/chain.log"; exit 1; }
  sleep 15
done
[ -f "$R/$PREV.done" ] || exit 1
for i in $(seq 1 80); do
  ps -eo args | grep -q "stage_syn[c].*--name $PREV " || break
  sleep 15
done
echo "$(date -u +%T) $PREV done; base losses of $START on the sets of $NAME" >> "$R/chain.log"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
vals=(); for v in "${VALS[@]}"; do vals+=(--val-extra "$v"); done
( cd /root/hc/data && /root/hc/cmf/target/release/cortiq-embryo birth --resume "$START" --shard "${SHARDS[@]}" --val "$VAL" "${vals[@]}" \
    --batch "${BATCH:-8}" --seq 1024 --steps $((ORIGIN + 1)) --warmup 1000 --lr 1e-4 --lr-restart --eval-every 1 --save-every 100000 \
    --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 5 --out "$R/$NAME.base.ckpt" > "$R/$NAME.base.log" 2>&1 )
rm -f "$R/$NAME.base.ckpt"
echo "$(date -u +%T) starting $NAME" >> "$R/chain.log"
setsid nohup ./train2.sh "$CONF" < /dev/null > /dev/null 2>&1 &
exec ./uploader.sh "$CONF" "$TITLE"
