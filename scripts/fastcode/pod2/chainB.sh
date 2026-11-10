#!/bin/bash
# chainB.sh: the streamed prompt (cortiq.s1). Greedy text and the last row's top logits of a ~3K-token prompt:
# resident (every expert on the card), the same file streamed (CMF_QWEN_RESIDENT=0), the old eight-token frames,
# and streamed at a 12 GB card's budget. Then speed by card size (speedtest2: a longer answer) for fcq2 and Strata.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 360); do grep -q CHAINA_DONE $J && grep -q BUILD_DONE logs/build_s1.log && break; sleep 10; done
log "chainB start: $(grep -E 'cortiq|PATCH|error' logs/build_s1.log | tail -2 | tr '\n' ' ')"
B=/root/fc/cortiq.s1
[ -x $B ] || { log "CHAINB no s1 binary"; log CHAINB_DONE; exit 1; }
{ head -c 11000 data/eval_code.txt; printf '\n\nExplain in two sentences what the code above does, then write one Python helper that it is missing.\n'; } > data/p_stream.txt
chk() { local name=$1; shift
  local t0=$(date +%s.%N)
  env CMF_QWEN_PROFILE=out/fcq2.profile CMF_QWEN_TOPLOG=1 CMF_QWEN_PROF_STREAM=1 "$@" timeout 900 $B run out/fcq2.cmf --prompt-file data/p_stream.txt -n 48 --temperature 0 --no-think > logs/chk_$name.txt 2> logs/chk_$name.err
  log "chk $name rc=$? $(python3 -c "print(round($(date +%s.%N)-$t0,1))") s md5 $(md5sum < logs/chk_$name.txt | cut -c1-10) $(grep -a -m1 toplog logs/chk_$name.err | cut -c1-110) | $(head -c 160 logs/chk_$name.txt | tr '\n' ' ')"
}
chk res
chk stream CMF_QWEN_RESIDENT=0
chk old CMF_QWEN_RESIDENT=0 CMF_QWEN_STREAM=0
chk st12 CMF_GPU_VRAM_MB=9728
chk st12old CMF_GPU_VRAM_MB=9728 CMF_QWEN_STREAM=0
run() { while read -r l; do log "$l"; done < <(SPEEDTEST=speedtest2.py BIN=$B timeout 3600 ./tier.sh "$@" 2>&1); }
run s1q2_c32 cortiq 32 1000,4000,16000,32000 out/fcq2.cmf out/fcq2.profile
for c in 24 16 12; do run s1q2_c$c cortiq $c 1000,4000,16000 out/fcq2.cmf out/fcq2.profile; done
for c in 32 24 16 12; do run st2_c$c strata $c 1000,4000,16000,32000; done
log CHAINB_DONE
