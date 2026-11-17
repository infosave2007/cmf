#!/usr/bin/env bash
# Change the mix of a running base at a stage boundary: after `saved … (step N)`
# stop the run, install the new config and resume from that checkpoint. The
# schedule goes on (the runner resumes with --schedule-from ORIGIN) and the new
# attempt takes its own sampler seed. If the new mix does not reach its first
# logged step, the old config is put back and the run resumes on it.
#
#   remix.sh STEP NEW_CONF      (detached: setsid nohup ./remix.sh 176200 base1.new.conf < /dev/null > /dev/null 2>&1 &)
#
# Progress goes to /root/hc/remix.log.
set -u
STEP="$1"; NEW="$2"
cd /root/hc
source "$NEW"
R=/root/hc/run; L=$R/$NAME.log; O=/root/hc/remix.log; CONF=/root/hc/$NAME.conf; RUNNER=./train2.sh
say() { echo "$(date -u +%T) $*" >> $O; }
for f in "${SHARDS[@]}"; do [ -s "data/${f%%:*}" ] || { say "missing shard ${f%%:*}: nothing done"; exit 1; }; done
for v in "$VAL" "${VALS[@]#*=}"; do [ -s "data/$v" ] || { say "missing held-out shard $v: nothing done"; exit 1; }; done
for i in $(seq 1 1440); do grep -q "saved .* (step $STEP)" $L && break; sleep 5; done
grep -q "saved .* (step $STEP)" $L || { say "deadline: step $STEP was not saved, nothing done"; exit 1; }
stop() {
  local ts bp
  ts=$(ps -eo pid,args | awk -v r="$RUNNER" '$2=="bash" && $3==r {print $1}')
  bp=$(pgrep -x cortiq-embryo)
  say "stopping the runner $ts and birth $bp"
  [ -n "$ts" ] && kill $ts
  sleep 1
  [ -n "$ts" ] && kill -0 $ts 2> /dev/null && kill -9 $ts  # a runner left alive would start the old mix again
  [ -n "$bp" ] && kill $bp
  for i in $(seq 1 60); do pgrep -x cortiq-embryo > /dev/null || return 0; sleep 1; done
  return 1
}
start() { setsid nohup $RUNNER "$CONF" < /dev/null > /dev/null 2>&1 & }
stop || { say "birth did not stop: nothing changed"; exit 1; }
before=$(grep -c "^step" $L)
cp "$CONF" "$CONF.before-$STEP" && cp "$NEW" "$CONF"
start
say "relaunched with $NEW after step $STEP"
for i in $(seq 1 60); do
  [ "$(grep -c "^step" $L)" -gt "$before" ] && { say "the new mix runs: $(grep "^step" $L | tail -1 | cut -c1-90)"; exit 0; }
  sleep 5
done
say "no step of the new mix in 5 minutes: the old config is put back"
stop
rm -f "$R/$NAME.failed"
cp "$CONF.before-$STEP" "$CONF"
start
exit 1
