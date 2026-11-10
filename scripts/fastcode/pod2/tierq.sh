#!/bin/bash
# tierq.sh NAME KIND (QUALITY=1: HumanEval and the aquarium after the speed) CARD_GIB CTXS [FILE PROFILE]: one server at a card size, speed by context through the client
# that measured Strata. cortiq: CMF_GPU_VRAM_MB = the auto budget of a CARD_GIB card (card minus clamp(card/8,
# 2.5, 4) GiB); strata: a VRAM hog leaves CARD_GIB free (its expert cache sizes itself from free VRAM).
# CARD_GIB 32 = the whole 5090 (no cap, no hog). Peak VRAM sampled every second (the hog is subtracted).
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
n=$1; kind=$2; card=$3; ctxs=$4
PY=/root/fc/venv/bin/python
hp=""; hogmib=0
if [ $kind = strata ] && [ $card != 32 ]; then
  /root/fc/strata/control/python/bin/python3.12 -I hog.py $card > logs/hog_$n.log 2>&1 &
  hp=$!
  for i in $(seq 1 60); do grep -q HOG_READY logs/hog_$n.log && break; sleep 1; done
  hogmib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
fi
if [ $kind = strata ]; then
  port=8090
  setsid bash -c "cd /root/fc/strata/control && exec /root/fc/strata/control/python/bin/python3.12 serve/server.py --engine strata --config /root/.local/share/strata/install/strata-coder-iq1_m.json --port $port" > logs/srv_$n.log 2>&1 &
else
  port=8091
  cap=""
  if [ $card != 32 ]; then cap="CMF_GPU_VRAM_MB=$(python3 -c "c=$card; r=min(max(c/8,2.5),4.0); print(int((c-r)*1024))")"; fi
  setsid env CMF_QWEN_PROFILE=$6 $cap ${SRV_ENV:-} ${BIN:-/root/fc/cortiq.v1} serve $5 --host 127.0.0.1 --port $port > logs/srv_$n.log 2>&1 &
fi
sp=$!
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > logs/srv_$n.mem &
mp=$!
ok=0
for i in $(seq 1 360); do curl -s -m 5 http://127.0.0.1:$port/v1/models > /dev/null && { ok=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
if [ $ok = 1 ]; then
  echo "$n server up after $((i*5)) s ${cap:-}"
  timeout 3000 $PY -I ${SPEEDTEST:-speedtest.py} http://127.0.0.1:$port $n res/${n}_speed.json $ctxs 2>&1 | grep -a "ctx "
  if [ -n "${QUALITY:-}" ]; then
    timeout 3600 $PY -I humaneval.py http://127.0.0.1:$port $n res/${n}_humaneval.jsonl 164 2>&1 | tail -1
    timeout 1800 $PY -I aquarium.py http://127.0.0.1:$port $n res/${n}_aquarium.html 2>&1 | tail -1
    echo "$n aquarium importmap=$(grep -c importmap res/${n}_aquarium.html 2>/dev/null)"
  fi
else
  echo "$n SERVER_FAILED"; tail -5 logs/srv_$n.log
fi
kill -- -$sp 2>/dev/null; sleep 8; kill -9 -- -$sp 2>/dev/null
for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -le $([ -n "$hp" ] && echo 1 || echo 0) ] && break; sleep 2; done
kill $mp 2>/dev/null
[ -n "$hp" ] && kill $hp 2>/dev/null
echo "$n peak VRAM $(( $(sort -n logs/srv_$n.mem | tail -1) - hogmib )) MiB (card $card GiB, hog $hogmib MiB)"
sleep 5
