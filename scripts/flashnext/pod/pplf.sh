#!/bin/bash
# pplf.sh NAME TEXTFILE TOKENS [ENV=VALUE ...] [--ppl-args ...]: perplexity of $FILE on TEXTFILE -> logs/pplf_NAME.log
set -uo pipefail
cd /root/rip && . fnx/env.sh
n=$1; f=$2; t=$3; shift 3
envs=(); extra=()
for a in "$@"; do case "$a" in *=*) envs+=("$a");; *) extra+=("$a");; esac; done
t0=$(date +%s)
env "${envs[@]}" $BIN ppl $FILE -f $f --tokens $t "${extra[@]}" > logs/pplf_$n.log 2>&1
echo "$n: $(grep -a "PPL" logs/pplf_$n.log | tail -1 | cut -c1-120) | $(( $(date +%s) - t0 )) s | $(grep -a -c -i "declin\|refus\|panic\|error" logs/pplf_$n.log) decline/error lines | $(grep -a -o "sealed [0-9]* rows[^,]*" logs/pplf_$n.log | tail -1)"
echo PPL_DONE
