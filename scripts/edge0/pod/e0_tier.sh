#!/bin/bash
# e0_tier.sh TAG MODEL TIER_GB ENGINE(cortiq|vulkan|strata) [CTXS] [RUNS] [KEY=VAL...] (cortiq = its CUDA backend): an emulated card of TIER_GB (vramfill
# holds the rest of the card, so only TIER_GB*1024-700 MiB are free when the engine starts), a server of ENGINE on it,
# speedtest over CTXS (default 1024,4096,16384) RUNS times (default 2); logs/tier_TAG.*
TAG=$1; M=$2; GB=$3; ENG=$4; CTXS=${5:-1024,4096,16384}; RUNS=${6:-2}; shift 6 2>/dev/null || shift $#
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
FREE=$((GB * 1024 - 700))
./vramfill $FREE > logs/tier_$TAG.fill.log 2>&1 &
FP=$!
for i in $(seq 1 60); do grep -q holding logs/tier_$TAG.fill.log && break; sleep 1; done
cat logs/tier_$TAG.fill.log
PORT=8096
if [ "$ENG" = cortiq ]; then
  env CMF_SERVE_SLOTS=1 CMF_GPU=cuda CMF_GPU_VRAM_MB=$((GB * 1024)) "$@" ${BIN:-/root/e0/cortiq.cur} serve $M --host 127.0.0.1 --port $PORT > logs/tier_$TAG.serve.log 2>&1 &
  SP=$!; NAME=edge0
elif [ "$ENG" = vulkan ]; then
  # the weight budget a card of this size gets by itself: its memory less the reserve of cortiq's Vulkan backend
  RES=$(( GB >= 32 ? 4096 : (GB >= 24 ? 3072 : 2560) ))
  env CMF_SERVE_SLOTS=1 CMF_GPU_VRAM_MB=$((GB * 1024 - RES)) "$@" ${BIN:-/root/e0/cortiq.cur} serve $M --host 127.0.0.1 --port $PORT > logs/tier_$TAG.serve.log 2>&1 &
  SP=$!; NAME=edge0
else
  # Strata's own launcher from the model folder (its model.json), on another port
  (cd $M && env EDGE0=/root/e0/strata/edge0 PORT=$PORT LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 "$@" bash ./start.sh > /root/e0/logs/tier_$TAG.serve.log 2>&1) &
  SP=$!; NAME=
fi
for i in $(seq 1 900); do curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
if ! curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null; then echo "server did not come up"; tail -20 logs/tier_$TAG.serve.log; kill $FP; exit 1; fi
[ -z "$NAME" ] && NAME=$(curl -s http://127.0.0.1:$PORT/v1/models | /root/venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "server up after $((i*2)) s"
grep -h "latent experts on the card\|latent experts:\|VRAM free with everything\|GPU cache (" logs/tier_$TAG.serve.log $M/serve.log 2>/dev/null | tail -3 | cut -c1-200
for r in $(seq 1 $RUNS); do
  /root/venv/bin/python tools/speedtest.py http://127.0.0.1:$PORT $NAME logs/tier_${TAG}_r$r.json $CTXS 2>&1 | tail -4
done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
for p in $(pgrep -P $SP) $(pgrep -f "strata --serve") $SP; do ps -o rss=,args= -p $p 2>/dev/null | awk '{printf "%.1f GB RSS %s\n", $1/1048576, $2}'; done
# the engine and its children by PID, then the filler
for p in $(pgrep -P $SP); do for q in $(pgrep -P $p); do kill $q 2>/dev/null; done; kill $p 2>/dev/null; done
kill $SP 2>/dev/null; wait $SP 2>/dev/null
kill $FP; wait $FP 2>/dev/null
sleep 3
echo TIER-DONE
