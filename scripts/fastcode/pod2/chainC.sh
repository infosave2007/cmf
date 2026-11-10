#!/bin/bash
# chainC.sh: where a resident prompt frame's time goes — fcq2 on the whole card, one stage left out at a time
# (CMF_Q4_SKIP: the output is wrong, the time of the rest stays), prompt tok/s at 4K and 16K.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 720); do grep -q CHAING_DONE $J && break; sleep 10; done
B=${BIN:-/root/fc/cortiq.s4}
for s in none experts route gdn qsa hc ple; do
  while read -r l; do log "skip=$s $l"; done < <(SRV_ENV="CMF_Q4_SKIP=$s" BIN=$B timeout 1200 ./tier.sh prof_$s cortiq 32 4000,16000 out/fcq2.cmf out/fcq2.profile 2>&1 | grep -a -E "ctx |FAILED")
done
log CHAINC_DONE
