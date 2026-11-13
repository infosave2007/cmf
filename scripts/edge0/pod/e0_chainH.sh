#!/bin/bash
# b18: full card both models (two runs 1K-32K, HumanEval on the 145B), then the emulated 32/24/16/12 GB cards
log() { echo "[$(date +%T)] $*"; }
cd /root/e0
/root/e0_build.sh b18 > logs/b18.log 2>&1; cat logs/b18.log
cp /root/e0/cortiq.b18 /root/e0/cortiq.cur
log "145 cuda full"; HE=1 /root/e0_bench.sh c145h out/edge0-145b.cmf cuda /root/e0/cortiq.b18 2>&1 | tail -14
nvidia-smi --query-gpu=memory.used --format=csv,noheader
log "120 cuda full"; /root/e0_bench.sh c120h out/edge0-120b.cmf cuda /root/e0/cortiq.b18 2>&1 | tail -12
for GB in 32 24 16 12; do
  for M in 145 120; do
    log "$M cuda ${GB} GB"
    BIN=/root/e0/cortiq.b18 ./e0_tier.sh c${M}_${GB} out/edge0-${M}b.cmf $GB cortiq 1024,4096,16384 1 CMF_QWEN_PROF=2 2>&1 | grep -v "^$" | tail -9
    grep -h "latent experts on the card\|MTP head stays off\|staying on the host" logs/tier_c${M}_${GB}.serve.log | head -3 | cut -c1-200
    grep -h "latent bank" logs/tier_c${M}_${GB}.serve.log | tail -1 | cut -c1-260
  done
done
log CHAIN-H-DONE
