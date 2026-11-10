#!/bin/bash
# chainJ.sh: cortiq.s7 = s6 + the host tier on by default below a whole card, 4096-token stream chunks on 16 GB cards,
# the request's repack through the host tier in parallel. 24 GB packs + masked decode (TTFT at 1K), 16 and 12 GB with
# no extra env (the defaults a user gets). cortiq.best -> s7 when its 3K check runs.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1440); do grep -q CHAIND2_DONE $J && grep -q BUILD_DONE logs/build_s7.log && break; sleep 10; done
env CMF_QWEN_PROFILE=out/fcq2.profile CMF_QWEN_TOPLOG=1 timeout 900 ./cortiq.s7 run out/fcq2.cmf --prompt-file data/p_stream.txt -n 48 --temperature 0 --no-think > logs/chk_s7.txt 2> logs/chk_s7.err && ln -sf cortiq.s7 cortiq.best
log "chk s7 $(grep -a -m1 toplog logs/chk_s7.err | cut -c1-110) best -> $(readlink cortiq.best)"
run() { local name=$1; shift; local envs=$1; shift
  while read -r l; do log "$name $l"; done < <(SRV_ENV="$envs" SPEEDTEST=speedtest2.py BIN=/root/fc/cortiq.s7 WARM=${WARM:-} timeout 3600 ./tierw.sh "$@" 2>&1); }
run j_packs "CMF_QWEN_MASK=1 CMF_QWEN_PACKS=/root/fc/packs CMF_QWEN_PACK_LOG=1" j_packs_c24 cortiq 24 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
run j_def16 "CMF_QWEN_PROF=1" j_def_c16 cortiq 16 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
run j_def12 "CMF_QWEN_PROF=1" j_def_c12 cortiq 12 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
WARM=90 run j_warm16 "CMF_QWEN_PROF=1" j_warm_c16 cortiq 16 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
WARM=90 run j_warm12 "CMF_QWEN_PROF=1" j_warm_c12 cortiq 12 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
WARM=90 run j_warm24 "CMF_QWEN_MASK=1 CMF_QWEN_PACKS=/root/fc/packs CMF_QWEN_PACK_LOG=1" j_warm_c24 cortiq 24 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
log CHAINJ_DONE
