#!/bin/bash
# turbobench.sh FILE TAG CTX...: bench --core --ignore-eos --tokens 60 --ctx N; peak VRAM (nvidia-smi, MiB)
# and peak host RSS (VmHWM, kB) sampled while it runs.
cd /root/cherrypit; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
F=$1; T=$2; shift 2; O=logs/tb_$T.txt; : > $O
for n in "$@"; do
  ./cortiq bench $F --core --ignore-eos --tokens 60 --ctx $n > logs/tb_$T.$n.log 2>&1 & pid=$!
  p=0; h=0
  while kill -0 $pid 2>/dev/null; do
    v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$v" -gt "$p" ] && p=$v
    w=$(awk "/VmHWM/{print \$2}" /proc/$pid/status 2>/dev/null); [ -n "$w" ] && [ "$w" -gt "$h" ] && h=$w
    sleep 0.5
  done
  echo "$n | vram_mib=$p | rss_kb=$h | $(grep -E "Decode" logs/tb_$T.$n.log | tr -s " ")" | tee -a $O
done
echo TB_DONE >> $O
