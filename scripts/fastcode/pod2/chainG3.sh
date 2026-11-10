#!/bin/bash
# chainG3.sh: q8_2f projections of prompt frames on the matrix units (cortiq.s6) — greedy text and top logits of the
# 3K prompt against s4, prompt/generation speed on the whole card and streamed at 16 GB. cortiq.best -> s6 if it ran.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1080); do grep -q CHAING2_DONE $J && grep -q BUILD_DONE logs/build_s6.log && break; sleep 10; done
chk() { local name=$1 b=$2; shift 2
  env CMF_QWEN_PROFILE=out/fcq2.profile CMF_QWEN_TOPLOG=1 "$@" timeout 900 $b run out/fcq2.cmf --prompt-file data/p_stream.txt -n 48 --temperature 0 --no-think > logs/chk_$name.txt 2> logs/chk_$name.err
  local rc=$?
  log "chk $name rc=$rc $(grep -a -m1 toplog logs/chk_$name.err | cut -c1-110) | $(grep -a -v -E '^(Loading|Ready|Prompt|\[)' logs/chk_$name.txt | tr '\n' ' ' | tail -c 260)"
  return $rc
}
if chk s6res /root/fc/cortiq.s6; then ln -sf cortiq.s6 cortiq.best; else ln -sf cortiq.s4 cortiq.best; fi
log "cortiq.best -> $(readlink cortiq.best)"
run() { local name=$1; shift; local envs=$1; shift
  while read -r l; do log "$name $l"; done < <(SRV_ENV="$envs" SPEEDTEST=speedtest2.py timeout 3600 ./tierq.sh "$@" 2>&1); }
BIN=/root/fc/cortiq.s6 run s6 "" g3_s6_c32 cortiq 32 1000,4000,16000,32000 out/fcq2.cmf out/fcq2.profile
BIN=/root/fc/cortiq.s6 run s6h "CMF_QWEN_HTIER_MB=auto" g3_s6h_c16 cortiq 16 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
log CHAING3_DONE
