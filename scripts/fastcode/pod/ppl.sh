#!/bin/bash
# ppl.sh NAME FILE TEXT TOKENS [ENV=V ...] [ppl args]: cortiq ppl -> logs/ppl_NAME.log, one summary line
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp RUST_LOG=${RUST_LOG:-info}
BIN=${BIN:-/root/fc/cortiq.m}
n=$1; m=$2; f=$3; t=$4; shift 4
envs=(); extra=()
for a in "$@"; do case "$a" in *=*) envs+=("$a");; *) extra+=("$a");; esac; done
t0=$(date +%s)
env "${envs[@]}" $BIN ppl $m -f $f --tokens $t "${extra[@]}" > logs/ppl_$n.log 2>&1
echo "$n: $(grep -a "PPL" logs/ppl_$n.log | tail -1 | cut -c1-140) | $(( $(date +%s) - t0 )) s | $(grep -a -c -i "panic\|error" logs/ppl_$n.log) err | $(grep -a -o "device path off[^,]*" logs/ppl_$n.log | head -1)"
echo PPL_DONE
