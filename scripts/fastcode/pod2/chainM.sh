#!/bin/bash
# chainM.sh (cortiq.best, warmed servers): the q4tp cut fcaq256 — the file with Strata Coder's quality — by card size.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
rm -f out/fcaq160.cmf
for c in 32 24 16 12; do
  while read -r l; do log "m_aq$c $l"; done < <(SPEEDTEST=speedtest2.py BIN=/root/fc/cortiq.best WARM=90 timeout 3600 ./tierw.sh m_aq_c$c cortiq $c 1000,4000,16000 out/fcaq256.cmf out/fcaq256.profile 2>&1)
done
log CHAINM_DONE
