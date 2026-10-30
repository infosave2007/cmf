#!/bin/bash
# Interleaved A/B of prompt-ingest knobs: bench --ctx N (synthetic prompt),
# MTP off, prints the batched prefill rate per run.
#
# usage: pod_ab_prefill.sh MODEL.cmf CTX ROUNDS "ENV=V ..." [...]   ("-" = default)
set -u
M=$1; CTX=$2; R=$3; shift 3
B=${CORTIQ:-/root/apex/bin/cortiq}
export XDG_RUNTIME_DIR=/tmp
for r in $(seq 1 "$R"); do
  for cfg in "$@"; do
    envs=(CMF_MTP=0)
    [ "$cfg" != "-" ] && envs+=($cfg)
    out=$(env "${envs[@]}" "$B" bench "$M" --core --ignore-eos --tokens 16 --ctx "$CTX" 2>&1)
    pf=$(echo "$out" | grep -a -o "prefill [0-9.]* tok/s" | head -1)
    dc=$(echo "$out" | grep -a -o "Decode:.*steady" | grep -o "[0-9.]* tok/s")
    printf "round %s | %-28s | %s | decode %s\n" "$r" "$cfg" "$pf" "$dc"
  done
done
echo AB_DONE
