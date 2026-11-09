#!/bin/bash
# prof13.sh (after f13): GPU-timestamp stage profile of decode (per-layer frames so every layer is marked), and
# the indexer A/B (old ranking top-k vs radix) on decode at 4k and 16k
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 120); do grep -q F13_DONE $J && break; sleep 10; done
P="CMF_QWEN_PROFILE=out/fc256.profile"
env $P CMF_GPU_TS=1 CMF_QWEN_TS=1 CMF_QWEN_PROF=1 CMF_QWEN_RESIDENT=0 CMF_QWEN_MTP=0 timeout 600 ./cortiq.f13 run out/fc256.cmf --prompt-file data/p_9k.txt --max-tokens 40 --greedy --no-think > logs/prof13.log 2>&1
log "prof13 decode stages: $(sed -e "s/\x1b\[[0-9;]*m//g" logs/prof13.log | grep -a -o "pending=.*" | tail -3 | head -1 | cut -c1-600)"
for C in 4000 16000; do
  log "$(BIN=/root/fc/cortiq.f13 TOK=64 ./bench.sh f13_topk_old_c$C out/fc256.cmf $C $P CMF_QWEN_MTP=0 CMF_QWEN_TOPK_R=0)"
  log "$(BIN=/root/fc/cortiq.f13 TOK=64 ./bench.sh f13_topk_new_c$C out/fc256.cmf $C $P CMF_QWEN_MTP=0)"
done
log PROF13_DONE
