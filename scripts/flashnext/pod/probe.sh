#!/bin/bash
# probe.sh [TOKENS]: the routed experts hot-neuron statistic of $FILE over data/probe.txt (code, then wiki) -> hot/experts.stat
set -uo pipefail
cd /root/rip && . fnx/env.sh
t0=$(date +%s)
$BIN hot-probe --experts $FILE data/probe.txt hot/experts.stat --tokens ${1:-8192} --window 2048 > logs/probe.log 2>&1
sed -e "s/\x1b\[[0-9;]*m//g" logs/probe.log | grep -a -i "hot-probe\|reached\|error\|panic" | cut -c1-220 | tail -4
echo "$(( $(date +%s) - t0 )) s"
echo PROBE_DONE
