#!/bin/bash
# hotfc2.sh: the hot file's method on the task-aware q4tp cut fcaq256, with the cut's slots read from the bf16
# checkpoint by their source numbers (cortiq.f26: hot-bake --defrag-stats; hotfc.sh's bake would have read
# checkpoint expert e into cut slot e). Waits for the running hot-probe, then the 28 busiest experts of every layer
# and the shared expert whole at q8_2f from bf16, then ppl, server speed, HumanEval and the aquarium.
# HF_TOKEN arrives on stdin, never on disk.
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
read -r HF_TOKEN; export HF_TOKEN
for i in $(seq 1 360); do grep -q BUILD_DONE logs/build_f26.log 2>/dev/null && break; sleep 10; done
[ -x cortiq.f26 ] || { log "HOTFC2 no f26 build: $(tail -3 logs/build_f26.log | tr '\n' ' ')"; log HOTFC2_DONE; exit 1; }
# the probe (cortiq.f24, PID from hotfc.sh) writes hot/fcaq_experts.stat when it ends; deadline 3 h
for i in $(seq 1 1080); do pgrep -f "[h]ot-probe out/fcaq256" > /dev/null || break; sleep 10; done
[ -s hot/fcaq_experts.stat ] || { log "HOTFC2 no stat"; log HOTFC2_DONE; exit 1; }
log "hot-probe fcaq256 done: hot/fcaq_experts.stat $(stat -c %s hot/fcaq_experts.stat) B"
B=/root/fc/cortiq.f26
rm -f fn/qwen38-flash-next-q4tp.cmf
log "hotfc2 disk: $(df -h /root | tail -1)"
log "hot-bake: $(nice -n 10 $B hot-bake out/fcaq256.cmf Qwen/Qwen3.8-Flash-Next hot/fcaq_experts.stat out/fcaqhot.cmf --experts --whole-experts 28 --shared-q8 --bf16-down --defrag-stats hot/code_aq.profile --scratch /root/fc/bakesrc 2>&1 | grep -v "^\s*$" | tail -4 | tr '\n' ' ')"
rm -rf /root/fc/bakesrc
[ -f out/fcaqhot.cmf ] || { log HOTFC2_DONE; exit 1; }
cp out/fcaq256.profile out/fcaqhot.profile
log "fcaqhot size $(stat -c %s out/fcaqhot.cmf)"
P="CMF_QWEN_PROFILE=out/fcaqhot.profile"
log "$(BIN=$B ./ppl.sh fcaqhot_rust out/fcaqhot.cmf data/expert_store_v089.rs 1536 $P | head -1)"
log "$(BIN=$B ./ppl.sh fcaqhot_fresh out/fcaqhot.cmf data/fresh_code.txt 8192 $P | head -1)"
BIN=$B ./serve_tests.sh fin_fcaqhot cortiq out/fcaqhot.cmf out/fcaqhot.profile > logs/fin_fcaqhot.log 2>&1
while read -r l; do log "fcaqhot $l"; done < <(grep -a -E "server up|ctx [0-9]+:|HUMANEVAL|AQUARIUM|peak VRAM|FAILED" logs/fin_fcaqhot.log)
log "fcaqhot aquarium importmap=$(grep -c importmap res/fin_fcaqhot_aquarium.html 2>/dev/null)"
log HOTFC2_DONE
