#!/bin/bash
# tinyfar.sh [BIN]: the small model's far field (a seal after 600 rows, 800 rows scored token by token)
# on the host, on the device, and on the device without the staged kernels. The three PPL(o1) lines
# should agree to the last digits; the exact line is the same pass without a far field.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
B=${1:-./cortiq}
run() { # name, env...
  local n=$1; shift
  env "$@" $B ppl tiny-q4tp.cmf -f data/wiki.test.raw --tokens 1400 --o1 all --o1-m 8 --o1-window 64 --o1-sink 4 --o1-prefill 600 > logs/tinyfar_$n.log 2>&1
  echo "$n: $(grep -a 'PPL(' logs/tinyfar_$n.log | tr '\n' ' ' | cut -c1-200) | $(grep -a -c -i 'declin\|refus' logs/tinyfar_$n.log) decline lines"
}
run host CMF_GPU=0
run dev CMF_O1_GPU=1
run dev_nostages CMF_O1_GPU=1 CMF_O1_STAGES=0
