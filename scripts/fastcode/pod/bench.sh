#!/bin/bash
# bench.sh NAME FILE CTX [ENV=V ...]: cortiq bench --core --ctx CTX --tokens 128 -> logs/bench_NAME.log + one summary line
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp RUST_LOG=${RUST_LOG:-info}
BIN=${BIN:-/root/fc/cortiq.f7}
n=$1; m=$2; c=$3; shift 3
t0=$(date +%s)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > logs/bench_$n.mem &
mp=$!
env "$@" $BIN bench $m --core --ignore-eos --ctx $c --tokens ${TOK:-128} > logs/bench_$n.log 2>&1
kill $mp 2>/dev/null
echo "$n ctx=$c: $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bench_$n.log | grep -a -i -E "prefill|decode|tok/s|accept" | grep -v INFO | tr "\n" " " | cut -c1-420) | peak $(sort -n logs/bench_$n.mem | tail -1) MiB | $(( $(date +%s) - t0 )) s | $(grep -a -c -i "panic\|device path off" logs/bench_$n.log) off/panic"
