#!/bin/bash
# chainI.sh (cortiq.best): the published q4tp cut fcaq256 on 32 and 24 GB cards (it needs 41 GB whole): streamed
# prompt, host tier, exact decode; speed, and HumanEval + aquarium on the 32 GB card.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1440); do grep -q CHAINH_DONE $J && break; sleep 10; done
B=/root/fc/cortiq.best
while read -r l; do log "aq32 $l"; done < <(QUALITY=1 SRV_ENV="CMF_QWEN_HTIER_MB=auto CMF_QWEN_PROF=1" SPEEDTEST=speedtest2.py BIN=$B timeout 7200 ./tierq.sh i_aq_c32 cortiq 32 1000,4000,16000 out/fcaq256.cmf out/fcaq256.profile 2>&1)
while read -r l; do log "aq24 $l"; done < <(SRV_ENV="CMF_QWEN_HTIER_MB=auto" SPEEDTEST=speedtest2.py BIN=$B timeout 3600 ./tierq.sh i_aq_c24 cortiq 24 1000,4000,16000 out/fcaq256.cmf out/fcaq256.profile 2>&1)
log CHAINI_DONE
