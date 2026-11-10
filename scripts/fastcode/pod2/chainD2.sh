#!/bin/bash
# chainD2.sh (cortiq.s4): fcq2 quality on the whole card; the narrow bundle (masked decode: the prompt's own routing,
# then the patent-1 packs) at 24 and 16 GB; exact decode with fast misses (host tier = pinned RAM + DMA, cold experts
# on the CPU, the pinned staging ring) at 16 and 12 GB; quality of the 24 GB packs mode.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1080); do grep -q CHAING3_DONE $J && grep -q BUILD_DONE logs/build_s4.log && break; sleep 10; done
B=/root/fc/cortiq.best
run() { local name=$1; shift; local envs=$1; shift
  while read -r l; do log "$name $l"; done < <(SRV_ENV="$envs" SPEEDTEST=speedtest2.py BIN=$B timeout 5400 ./tierq.sh "$@" 2>&1); }
P=out/fcq2.profile
QUALITY=1 run q "" q2_c32 cortiq 32 1000,4000,16000 out/fcq2.cmf $P
for c in 24 16; do
  run mask "CMF_QWEN_HTIER_MB=auto CMF_QWEN_MASK=1 CMF_QWEN_PACK_LOG=1" d_mask_c$c cortiq $c 1000,4000,16000 out/fcq2.cmf $P
  run packs "CMF_QWEN_HTIER_MB=auto CMF_QWEN_MASK=1 CMF_QWEN_PACKS=/root/fc/packs CMF_QWEN_PACK_LOG=1" d_packs_c$c cortiq $c 1000,4000,16000 out/fcq2.cmf $P
done
for c in 16 12; do
  run htier "CMF_QWEN_HTIER_MB=auto CMF_QWEN_PROF=1" d_htier_c$c cortiq $c 1000,4000,16000 out/fcq2.cmf $P
  run cpu "CMF_QWEN_HTIER_MB=auto CMF_QWEN_COLD_DEVICE=0" d_cpu_c$c cortiq $c 1000,4000,16000 out/fcq2.cmf $P
done
run stage "CMF_QWEN_STAGE_MB=768" d_stage_c12 cortiq 12 1000,4000,16000 out/fcq2.cmf $P
QUALITY=1 run q24 "CMF_QWEN_HTIER_MB=auto CMF_QWEN_MASK=1 CMF_QWEN_PACKS=/root/fc/packs" q2_packs_c24 cortiq 24 4000 out/fcq2.cmf $P
log CHAIND2_DONE
