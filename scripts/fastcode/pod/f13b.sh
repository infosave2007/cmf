#!/bin/bash
# f13b.sh: f13 (prompt frames of 256, tiles opt-in) — text 256 vs 128 frames, ppl, speed, MTP depth, stage sweep
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
P="CMF_QWEN_PROFILE=out/fc256.profile"
for CH in 256 128; do
  env $P CMF_QWEN_PREFILL_CHUNK=$CH CMF_QWEN_MTP=0 timeout 900 ./cortiq.f13 run out/fc256.cmf --prompt-file data/p_9k.txt --max-tokens 64 --greedy --no-think > logs/eq13_$CH.log 2>&1
  sed -e "s/\x1b\[[0-9;]*m//g" logs/eq13_$CH.log | grep -a -v -E "^[0-9]{4}-|INFO|WARN|^\[64 tokens" > logs/eq13_$CH.txt
done
log "f13 greedy text, 256- vs 128-token frames: $(cmp -s logs/eq13_256.txt logs/eq13_128.txt && echo IDENTICAL || echo DIFFER) | vs f11: $(cmp -s logs/eq13_128.txt logs/eq11.txt && echo same || echo differs) | $(grep -a -c -i panic logs/eq13_256.log) panics"
export BIN=/root/fc/cortiq.f13
for C in 1000 4000 16000; do log "$(./bench.sh f13_c$C out/fc256.cmf $C $P)"; done
log "$(TOK=64 ./bench.sh f13_c4000_plain out/fc256.cmf 4000 $P CMF_QWEN_MTP=0)"
for K in 2 4 5; do log "$(./bench.sh f13_k${K}_c4000 out/fc256.cmf 4000 $P CMF_QWEN_MTP_K=$K)"; done
./sweep.sh /root/fc/cortiq.f13 s13
log F13_DONE
