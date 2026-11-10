#!/bin/bash
# chainG.sh: 512-row prompt frames (cortiq.s5b) against 256 (cortiq.s4), fcq2 on the whole card: greedy text and top
# logits of the 3K prompt, then prompt/generation speed; s5 streamed on a 16 GB card with the host tier.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1080); do grep -q CHAINC_DONE $J && grep -q BUILD_DONE logs/build_s5b.log && break; sleep 10; done
chk() { local name=$1 b=$2; shift 2
  env CMF_QWEN_PROFILE=out/fcq2.profile CMF_QWEN_TOPLOG=1 "$@" timeout 900 $b run out/fcq2.cmf --prompt-file data/p_stream.txt -n 48 --temperature 0 --no-think > logs/chk_$name.txt 2> logs/chk_$name.err
  log "chk $name rc=$? $(grep -a -m1 toplog logs/chk_$name.err | cut -c1-110) | $(grep -a -v -E '^(Loading|Ready|Prompt|\[)' logs/chk_$name.txt | tr '\n' ' ' | tail -c 260)"
}
chk g2_s4res /root/fc/cortiq.s4
chk s5bres /root/fc/cortiq.s5b

run() { local name=$1; shift; local envs=$1; shift
  while read -r l; do log "$name $l"; done < <(SRV_ENV="$envs" SPEEDTEST=speedtest2.py timeout 3600 ./tierq.sh "$@" 2>&1); }

BIN=/root/fc/cortiq.s5b run s5 "" g2_s5_c32 cortiq 32 1000,4000,16000,32000 out/fcq2.cmf out/fcq2.profile
BIN=/root/fc/cortiq.s5b run s5h "CMF_QWEN_HTIER_MB=auto" g2_s5h_c16 cortiq 16 1000,4000,16000 out/fcq2.cmf out/fcq2.profile
log CHAING2_DONE
