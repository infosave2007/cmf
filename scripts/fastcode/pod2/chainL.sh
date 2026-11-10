#!/bin/bash
# chainL.sh: a q4tp cut that fits a 32 GB card whole — 160 experts a layer from fcaq256 by its code+task profile
# (moe-defrag of a cut composes the source numbers); speed on the whole card, HumanEval and the aquarium.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
B=/root/fc/cortiq.best
rm -f fn/qwen38-flash-next-q2tp-v2.cmf
log "chainL disk: $(df -h /root | tail -1 | awk '{print $4}') free"
log "defrag fcaq160: $(timeout 1800 $B moe-defrag out/fcaq256.cmf --stats out/fcaq256.profile --keep 160 --output out/fcaq160.cmf 2>&1 | tail -3 | tr '\n' ' ')"
[ -f out/fcaq160.cmf ] || { log CHAINL_DONE; exit 1; }
log "fcaq160 size $(stat -c %s out/fcaq160.cmf)"
while read -r l; do log "a160 $l"; done < <(QUALITY=1 SPEEDTEST=speedtest2.py BIN=$B timeout 7200 ./tierq.sh l_a160_c32 cortiq 32 1000,4000,16000,32000 out/fcaq160.cmf out/fcaq160.profile 2>&1)
log CHAINL_DONE
