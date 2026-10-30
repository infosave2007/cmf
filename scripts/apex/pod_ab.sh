#!/bin/bash
# Interleaved A/B of decode knobs on one CMF: every round runs each config
# once (bench --core, 256 tokens, MTP off unless the config says otherwise)
# and prints the steady tok/s with the cgroup's CPU-throttling delta of
# that run. Read the medians across rounds, not single runs.
#
# usage: pod_ab.sh MODEL.cmf ROUNDS "ENV=V ..." ["ENV=V ..." ...]
#        (use "-" for the plain default config)
set -u
M=$1; R=$2; shift 2
B=${CORTIQ:-/root/apex/bin/cortiq}
export XDG_RUNTIME_DIR=/tmp
ST=/sys/fs/cgroup/cpu/cpu.stat
for r in $(seq 1 "$R"); do
  for cfg in "$@"; do
    envs=(CMF_MTP=0)
    [ "$cfg" != "-" ] && envs+=($cfg)
    t0=$(awk '/^throttled_time/{print $2}' $ST); n0=$(awk '/^nr_throttled/{print $2}' $ST)
    tps=$(env "${envs[@]}" "$B" bench "$M" --core --ignore-eos --tokens 256 2>&1 \
      | grep -a -o "Decode:.*steady" | grep -o "[0-9.]* tok/s")
    t1=$(awk '/^throttled_time/{print $2}' $ST); n1=$(awk '/^nr_throttled/{print $2}' $ST)
    printf "round %s | %-40s | %s | throttled %s periods %.0f ms\n" "$r" "$cfg" "$tps" \
      "$((n1 - n0))" "$(echo "($t1 - $t0) / 1000000" | bc -l)"
  done
done
echo AB_DONE
