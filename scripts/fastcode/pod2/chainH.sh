#!/bin/bash
# chainH.sh (cortiq.best): HumanEval + aquarium of fcq2 on the whole card (D2's first quality run had no pyarrow),
# then generation on the whole card by MTP draft length (default k=3).
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1440); do grep -q CHAINK_DONE $J && break; sleep 10; done
B=/root/fc/cortiq.best
while read -r l; do log "qh $l"; done < <(QUALITY=1 SPEEDTEST=speedtest2.py BIN=$B timeout 5400 ./tierq.sh qh_c32 cortiq 32 4000 out/fcq2.cmf out/fcq2.profile 2>&1)
for v in "k2:CMF_QWEN_MTP_K=2" "k4:CMF_QWEN_MTP_K=4" "plain:CMF_QWEN_MTP=0"; do
  name=${v%%:*}; envs=${v#*:}
  while read -r l; do log "mtp_$name $l"; done < <(SRV_ENV="$envs" SPEEDTEST=speedtest2.py BIN=$B timeout 1800 ./tierq.sh h_${name}_c32 cortiq 32 4000,16000 out/fcq2.cmf out/fcq2.profile 2>&1)
done
log CHAINH_DONE
