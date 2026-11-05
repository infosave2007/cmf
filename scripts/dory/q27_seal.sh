#!/bin/bash
# The seal position against the VRAM ceiling: q27_seal.sh BIN LOG SEAL...
cd /root/q
export XDG_RUNTIME_DIR=/tmp
B=$1; L=$2; shift 2
: > $L
V=src/scripts/dory/vram_peak.sh
for S in "$@"; do
  T=m/q27t$S.cmf
  $B far-turbo m/qwen38-27b-q4tp.cmf $T --exact-weights keep27.bin --seal $S 2>&1 | tail -1 | cut -c1-120 | tee -a $L
  for c in 1000 4000 12000 32000; do
    CMF_O1_PREFIX_MAX=$S $V "seal $S ctx $c" $L /root/q/sp -- $B bench $T --core --tokens 60 --ignore-eos --ctx $c
  done
  CMF_O1_PREFIX_MAX=$S CMF_O1_SPILL=/root/q/sp $V "seal $S ctx 32000 spill" $L /root/q/sp -- $B bench $T --core --tokens 60 --ignore-eos --ctx 32000
done
echo DONE >> $L
