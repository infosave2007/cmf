#!/bin/bash
# b20; our own routing profile of both models (full card, the calibration prompts); what it covers
log() { echo "[$(date +%T)] $*"; }
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
/root/e0_build.sh b20 > logs/b20.log 2>&1; cat logs/b20.log
for M in 145 120; do
  log "profile $M"
  rm -f out/own-$M.profile
  env CMF_SERVE_SLOTS=1 CMF_GPU=cuda CMF_QWEN_LAT_PROFILE_SAVE=/root/e0/out/own-$M.profile /root/e0/cortiq.b20 serve out/edge0-${M}b.cmf --host 127.0.0.1 --port 8099 > logs/prof_own$M.serve.log 2>&1 &
  SP=$!
  for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8099/v1/models >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  /root/venv/bin/python tools/profile_feed.py http://127.0.0.1:8099 edge0 40 2>&1 | tail -1
  kill $SP; wait $SP 2>/dev/null; sleep 2
  ls -la out/own-$M.profile
  /root/venv/bin/python tools/profcov.py out/own-$M.profile src/e145n/profile.bin
done
log profiles-done
# the 24 GB card on the 145B with our profile: chunk sizes, the VNNI host kernel on and off
cp out/own-145.profile out/edge0-145b.profile; cp out/own-120.profile out/edge0-120b.profile
for V in "c1024:" "c2048:CMF_QWEN_CUDA_CHUNK=2048" "c4096:CMF_QWEN_CUDA_CHUNK=4096" "novnni:CMF_NVFP4_V512=0"; do
  T=${V%%:*}; E=${V#*:}
  log "145 24 GB $T"
  BIN=/root/e0/cortiq.b20 ./e0_tier.sh x145_24_$T out/edge0-145b.cmf 24 cortiq 1024,4096,16384 1 CMF_QWEN_PROF=2 $E 2>&1 | grep "ctx\|RSS\|did not"
  grep -h "latent experts on the card" logs/tier_x145_24_$T.serve.log | head -1 | cut -c1-160
  grep -h "latent bank" logs/tier_x145_24_$T.serve.log | tail -2 | cut -c1-260
done
log CHAIN-I2-DONE
