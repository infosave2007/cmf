#!/usr/bin/env bash
# Swap the trainer binary at a stage boundary: stop after `saved … (step N)`,
# prove the new binary leaves the same checkpoint bits as the old one over
# 8 steps with two subspace refreshes, resume the run from its checkpoint.
#   swap.sh STEP
set -u
STEP="$1"
R=/root/hc/run; L=$R/pass1.log; O=/root/hc/swap.log
NEW=/root/hc/cmf/target/release/cortiq-embryo; OLD=/root/hc/cortiq-embryo.old
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
say() { echo "$(date -u +%T) $*" >> $O; }
for i in $(seq 1 360); do grep -q "saved .* (step $STEP)" $L && break; sleep 5; done
grep -q "saved .* (step $STEP)" $L || { say "deadline: step $STEP was not saved, nothing done"; exit 1; }
TS=$(ps -eo pid,args | awk '$2=="bash" && $3=="./train.sh" {print $1}')
BP=$(pgrep -x cortiq-embryo)
say "stopping train.sh $TS and birth $BP after step $STEP"
[ -n "$TS" ] && kill $TS
[ -n "$BP" ] && kill $BP
for i in $(seq 1 60); do pgrep -x cortiq-embryo > /dev/null || break; sleep 1; done
cd /root/hc/data
gate() { # binary name
  local t0=$(date +%s)
  "$1" birth --resume grown340.ckpt --shard se.u16:0.884 docs.u16:0.076 hermes_code.u16:0.040 --val val.u16 \
    --batch 8 --seq 1024 --steps 8 --warmup 4 --lr 1e-4 --lr-restart --eval-every 100000 --save-every 100000 --pca-every 4 \
    --carry --carry-reset-every 16 --carry-eot 32760 --dropless --seed 17 --out $R/$2.ckpt > $R/$2.log 2>&1
  local code=$?
  local h=$(sha256sum $R/$2.ckpt 2>/dev/null | cut -c1-32)
  say "gate $2: exit $code, $(($(date +%s) - t0)) s, sha256 $h"
  rm -f $R/$2.ckpt
  echo "$h"
}
HO=$(gate $OLD gate-old)
HN=$(gate $NEW gate-new)
if [ -n "$HO" ] && [ "$HO" = "$HN" ]; then
  say "same bits: the run continues on the new binary"
else
  say "DIFFERENT BITS ($HO vs $HN): the old binary is put back"
  cp $OLD $NEW
fi
cp /root/hc/train.new.sh /root/hc/train.sh && chmod +x /root/hc/train.sh
cd /root/hc && setsid nohup ./train.sh < /dev/null > /dev/null 2>&1 &
say "train.sh relaunched"
