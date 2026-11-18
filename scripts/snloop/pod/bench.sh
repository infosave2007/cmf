#!/bin/bash
# bench.sh BIN FILE TAG BACKEND CTX...: `cortiq bench --core --ignore-eos --tokens T --ctx N` on cuda|vulkan,
# with the peak card memory (nvidia-smi, MiB) and peak host RSS (VmHWM, kB) sampled while it runs.
# BENCH_TOKENS (default 128), BENCH_ENV extra env.
set -u
B=$1; F=$2; T=$3; BK=$4; shift 4
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=${CMF_MAX_SEQ:-270000}
[ $BK = cuda ] && export CMF_GPU=cuda
O=/root/sn/logs/bench_$T.txt; : > $O
for n in "$@"; do
  L=/root/sn/logs/bench_$T.$n.log
  env ${BENCH_ENV:-} $B bench $F --core --ignore-eos --tokens ${BENCH_TOKENS:-128} --ctx $n > $L 2>&1 & pid=$!
  p=0; h=0
  while kill -0 $pid 2>/dev/null; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$p" ] && p=$v
    w=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w
    sleep 0.25
  done
  echo "$n | vram_mib=$p | rss_kb=$h | $(grep -E -i 'prefill|first token|decode' $L | tr -s ' ' | tr '\n' ';')" | tee -a $O
done
echo BENCH_DONE >> $O
