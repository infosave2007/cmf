#!/bin/bash
# bench.sh NAME CTX [ENV=VALUE ...]: cortiq bench --core --ctx CTX --tokens 60 of $FILE + $PROFILE -> logs/bench_NAME.log
set -uo pipefail
cd /root/rip && . fnx/env.sh
n=$1; c=$2; shift 2
t0=$(date +%s)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > logs/bench_$n.mem &
mp=$!
env CMF_QWEN_PROFILE=$PROFILE "$@" $BIN bench $FILE --core --ctx $c --tokens 60 > logs/bench_$n.log 2>&1
kill $mp 2>/dev/null
echo "$n ctx=$c: $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bench_$n.log | grep -a -i "prefill\|decode\|tok/s\|first" | tr "\n" " " | cut -c1-400) | peak $(sort -n logs/bench_$n.mem | tail -1) MiB | $(( $(date +%s) - t0 )) s | $(grep -a -o "sealed [0-9]* rows[^,]*" logs/bench_$n.log | tail -1)"
echo BENCH_DONE
