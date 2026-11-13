#!/bin/bash
# b21: the emulated 32/24/16/12 GB cards on CUDA (both models, 1K/4K/16K, one run), HumanEval on the 24 GB card,
# then the same cards on Vulkan (the banks that do not fit in system memory)
log() { echo "[$(date +%T)] $*"; }
cd /root/e0
/root/e0_build.sh b21 > logs/b21.log 2>&1; cat logs/b21.log
cp /root/e0/cortiq.b21 /root/e0/cortiq.cur
export BIN=/root/e0/cortiq.b21
for GB in 32 24 16 12; do
  for M in 145 120; do
    log "$M cuda ${GB} GB"
    ./e0_tier.sh c${M}_${GB} out/edge0-${M}b.cmf $GB cortiq 1024,4096,16384 1 CMF_QWEN_PROF=2 2>&1 | grep "ctx\|RSS\|did not\|MiB$"
    grep -h "latent experts on the card\|MTP head stays off\|staying on the host\|room for" logs/tier_c${M}_${GB}.serve.log | head -3 | cut -c1-200
    grep -h "latent bank" logs/tier_c${M}_${GB}.serve.log | tail -1 | cut -c1-200
  done
done
log "humaneval 145 24 GB"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
./vramfill $((24 * 1024 - 700)) > logs/he24.fill.log 2>&1 &
FP=$!; sleep 8
env CMF_SERVE_SLOTS=1 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 $BIN serve out/edge0-145b.cmf --host 127.0.0.1 --port 8093 > logs/he24.serve.log 2>&1 &
SP=$!
for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8093/v1/models >/dev/null 2>&1 && break; sleep 2; done
/root/venv/bin/python tools/humaneval.py http://127.0.0.1:8093 edge0 logs/he24.jsonl 164 2>&1 | tail -1
kill $SP; wait $SP 2>/dev/null; kill $FP; wait $FP 2>/dev/null; sleep 3
for GB in 24 32 16 12; do
  for M in 145 120; do
    log "$M vulkan ${GB} GB"
    ./e0_tier.sh v${M}_${GB} out/edge0-${M}b.cmf $GB vulkan 1024,4096 1 2>&1 | grep "ctx\|RSS\|did not\|MiB$\|latent experts"
    grep -h "device path\|refused\|do not fit\|OOM\|out of memory" logs/tier_v${M}_${GB}.serve.log | head -3 | cut -c1-200
  done
done
log CHAIN-J-DONE
