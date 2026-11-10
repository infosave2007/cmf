#!/bin/bash
# chainK.sh (cortiq.s7, warmed servers): exact decode with the request's arena refilled by its patent-1 pack (no mask),
# and the pinned staging ring (on 12 GB it beat the host tier for the prompt), alone and with the packs.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1440); do grep -q CHAINJ_DONE $J && break; sleep 10; done
run() { local name=$1; shift; local envs=$1; shift
  while read -r l; do log "$name $l"; done < <(SRV_ENV="$envs" SPEEDTEST=speedtest2.py BIN=/root/fc/cortiq.s7 WARM=90 timeout 3600 ./tierw.sh "$@" 2>&1); }
run k_packs24 "CMF_QWEN_PACKS=/root/fc/packs CMF_QWEN_PACK_LOG=1" k_packs_c24 cortiq 24 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
run k_stage16 "CMF_QWEN_HTIER_MB=0 CMF_QWEN_STAGE_MB=768 CMF_QWEN_PROF=1" k_stage_c16 cortiq 16 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
run k_stage12 "CMF_QWEN_HTIER_MB=0 CMF_QWEN_STAGE_MB=768 CMF_QWEN_PROF=1" k_stage_c12 cortiq 12 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
run k_stpk12 "CMF_QWEN_HTIER_MB=0 CMF_QWEN_STAGE_MB=768 CMF_QWEN_PACKS=/root/fc/packs CMF_QWEN_PACK_LOG=1" k_stpk_c12 cortiq 12 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
log CHAINK_DONE
