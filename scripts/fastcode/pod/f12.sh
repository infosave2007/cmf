#!/bin/bash
# f12.sh: expert tiles for prompt frames + lean readback; correctness against the tile-less path, speed, stage sweep
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
./build.sh f12 > logs/build_f12.log 2>&1
grep -q BUILD_DONE logs/build_f12.log || { log "BUILD f12 FAILED $(grep -a -E "^error|FAILED" -A3 logs/build_f12.log | head -5 | tr '\n' ' ')"; exit 1; }
P="CMF_QWEN_PROFILE=out/fc256.profile"
for M in tiles notiles; do
  E=""; [ $M = notiles ] && E="CMF_QWEN_TILES=0"
  env $P $E CMF_QWEN_MTP=0 timeout 900 ./cortiq.f12 run out/fc256.cmf --prompt-file data/p_9k.txt --max-tokens 64 --greedy --no-think > logs/eq12_$M.log 2>&1
  sed -e "s/\x1b\[[0-9;]*m//g" logs/eq12_$M.log | grep -a -v -E "^[0-9]{4}-|INFO|WARN|^\[64 tokens" > logs/eq12_$M.txt
done
log "f12 greedy text, expert tiles vs per-token kernels: $(cmp -s logs/eq12_tiles.txt logs/eq12_notiles.txt && echo IDENTICAL || echo "DIFFER: $(diff logs/eq12_tiles.txt logs/eq12_notiles.txt | head -6 | tr '\n' ' ' | cut -c1-400)") | vs f11: $(cmp -s logs/eq12_notiles.txt logs/eq11.txt && echo same || echo differs) | $(grep -a -c -i panic logs/eq12_tiles.log) panics"
log "$(./ppl.sh f12_es out/fc256.cmf data/expert_store_v089.rs 1536 $P | head -1)"
for C in 1000 4000 16000; do log "$(BIN=/root/fc/cortiq.f12 ./bench.sh f12_c$C out/fc256.cmf $C $P)"; done
log "$(BIN=/root/fc/cortiq.f12 ./bench.sh f12_c4000_notiles out/fc256.cmf 4000 $P CMF_QWEN_TILES=0)"
log "$(BIN=/root/fc/cortiq.f12 TOK=64 ./bench.sh f12_c4000_plain out/fc256.cmf 4000 $P CMF_QWEN_MTP=0)"
log "$(BIN=/root/fc/cortiq.f12 TOK=64 ./bench.sh f12_c4000_plain_nolean out/fc256.cmf 4000 $P CMF_QWEN_MTP=0 CMF_QWEN_LEAN=0)"
./sweep.sh /root/fc/cortiq.f12 s12
log F12_DONE
