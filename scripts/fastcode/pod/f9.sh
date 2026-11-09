#!/bin/bash
# f9.sh: stop chain6 (its fc256 server ran f7 and fell to the host at 16384), build f9 (prompt frames of 128 on
# resident cut models, hybrid-fixes' GDN scan, sealed QSA past the index cap), check it, measure it.
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for p in $(pgrep -f "chain6[.]sh") $(pgrep -f "serve_tests[.]sh") $(pgrep -f "speedtest[.]py"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "cortiq[.]f7 serve"); do kill -- -$p 2>/dev/null; kill $p 2>/dev/null; done
sleep 5; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill $p 2>/dev/null; done
log "STOP chain6: the fc256 server test started on f7 (before the f8 swap) and fell to the host path at 16384 tokens"
./build.sh f9 > logs/build_f9.log 2>&1
grep -q BUILD_DONE logs/build_f9.log || { log "BUILD f9 FAILED $(grep -a -E "error|FAILED" logs/build_f9.log | head -3 | tr '\n' ' ')"; exit 1; }
export BIN=/root/fc/cortiq.f9
P="CMF_QWEN_PROFILE=out/fc256.profile"
# the same greedy text with 8-token and 128-token prompt frames
head -c 9000 data/fresh_code.txt > data/p_9k.txt
for CH in 8 128; do
  env $P CMF_QWEN_PREFILL_CHUNK=$CH CMF_QWEN_MTP=0 timeout 600 $BIN run out/fc256.cmf --prompt-file data/p_9k.txt --max-tokens 64 --greedy --no-think > logs/eq_ch$CH.log 2>&1
  sed -e "s/\x1b\[[0-9;]*m//g" logs/eq_ch$CH.log | grep -a -v -E "^[0-9]{4}-|INFO|WARN" | tail -n +1 > logs/eq_ch$CH.txt
done
log "f9 greedy text, 8- vs 128-token prompt frames: $(cmp -s logs/eq_ch8.txt logs/eq_ch128.txt && echo IDENTICAL || echo "DIFFER ($(diff logs/eq_ch8.txt logs/eq_ch128.txt | wc -l) diff lines)"); first token $(grep -a -o "first token [0-9.]* s" logs/eq_ch8.log | tail -1) vs $(grep -a -o "first token [0-9.]* s" logs/eq_ch128.log | tail -1)"
log "$(./ppl.sh f9_fc256_es out/fc256.cmf data/expert_store_v089.rs 1536 $P | head -1)"
for C in 1000 4000 16000; do log "$(./bench.sh f9_fc256_c$C out/fc256.cmf $C $P)"; done
log "$(./bench.sh f9_fc256_c32000_o1 out/fc256.cmf 32000 $P CMF_QWEN_O1=32,1800,4,12288)"
log F9_DONE
