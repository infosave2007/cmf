#!/bin/bash
# Interleaved decode A/B across binaries and env: each config is
# "BINARY [ENV=V ...]"; bench --core, 256 tokens, MTP off.
#
# usage: pod_ab_bins.sh MODEL.cmf ROUNDS "BIN [ENV=V ...]" [...]
set -u
M=$1; R=$2; shift 2
export XDG_RUNTIME_DIR=/tmp
for r in $(seq 1 "$R"); do
  for cfg in "$@"; do
    read -r bin envs <<< "$cfg"
    tps=$(env CMF_MTP=0 $envs "$bin" bench "$M" --core --ignore-eos --tokens 256 2>&1 \
      | grep -a -o "Decode:.*steady" | grep -o "[0-9.]* tok/s")
    printf "round %s | %-55s | %s\n" "$r" "$cfg" "$tps"
  done
done
echo AB_DONE
