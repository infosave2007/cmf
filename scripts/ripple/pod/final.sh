#!/bin/bash
# final.sh LOG FILE CTX...: the card's table on a quiet card. For every context the turbo file as it
# is and the same file with the far field off (CMF_O1=off: exact attention, the standard file's
# behaviour): peak device memory over the run, prompt and decode speed. MODES="turbo" runs one.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000
L=$1; F=$2; shift 2
V=scripts/vram_peak.sh
for c in "$@"; do
  for m in ${MODES:-standard turbo}; do
    if [ "$m" = standard ]; then
      bash $V "standard $c" $L /root/rip/sp -- env CMF_O1=off ./cortiq bench $F --core --tokens 60 --ignore-eos --ctx $c
    else
      bash $V "turbo $c" $L /root/rip/sp -- ./cortiq bench $F --core --tokens 60 --ignore-eos --ctx $c
    fi
  done
done
echo FINAL_DONE >> $L
