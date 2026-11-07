#!/bin/bash
# Speed of one CMF file on this Mac: bench.sh FILE TAG
#   decode: bench --core --ignore-eos, 3 runs (Metal, then CPU with CMF_GPU=0)
#   prompt: bench --ctx 2048 (prefill tok/s at 2k, decode at that depth), peak RSS via time -l
set -u
FILE=$1; TAG=$2
B=${BIN:-$HOME/dev/cherrypit/bin/cortiq}
OUT=$HOME/dev/cherrypit/out/bench; mkdir -p "$OUT"
gpu() { ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
one() { # label env... -- args...
  local label=$1; shift
  local log="$OUT/$TAG.$label.log"
  echo "== $label (gpu busy before: $(gpu)%, load $(uptime | awk -F'load averages: ' '{print $2}'))" | tee -a "$OUT/$TAG.summary"
  /usr/bin/time -l env "$@" > "$log" 2>&1
  grep -E "tok/s|prefill|decode|TTFT|maximum resident|real" "$log" | grep -v INFO | tee -a "$OUT/$TAG.summary"
}
for i in 1 2 3; do one metal$i "$B" bench "$FILE" --core --ignore-eos --tokens 128; done
one metal_ctx2k "$B" bench "$FILE" --core --ignore-eos --ctx 2048 --tokens 64
for i in 1 2 3; do one cpu$i CMF_GPU=0 "$B" bench "$FILE" --core --ignore-eos --tokens 48; done
one cpu_ctx2k CMF_GPU=0 "$B" bench "$FILE" --core --ignore-eos --ctx 2048 --tokens 16
