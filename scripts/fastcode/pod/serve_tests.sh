#!/bin/bash
# serve_tests.sh NAME KIND [FILE PROFILE]: start a server (KIND = cortiq | strata), run speed ctx 1k/4k/16k/32k,
# HumanEval (164) and the aquarium prompt through the same client, stop it. Results in res/NAME_*.
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
n=$1; kind=$2; mkdir -p res
PY=/root/fc/venv/bin/python
if [ $kind = strata ]; then
  port=8090
  setsid bash -c "cd /root/fc/strata/control && exec /root/fc/strata/control/python/bin/python3.12 serve/server.py --engine strata --config /root/.local/share/strata/install/strata-coder-iq1_m.json --port $port" > logs/srv_$n.log 2>&1 &
else
  port=8091
  setsid env CMF_QWEN_PROFILE=$4 ${SRV_ENV:-} ${BIN:-/root/fc/cortiq.f7} serve $3 --host 127.0.0.1 --port $port > logs/srv_$n.log 2>&1 &
fi
sp=$!
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > logs/srv_$n.mem &
mp=$!
ok=0
for i in $(seq 1 360); do curl -s -m 5 http://127.0.0.1:$port/v1/models > /dev/null && { ok=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
if [ $ok = 1 ]; then
  echo "$n server up after $((i*5)) s"
  $PY -I speedtest.py http://127.0.0.1:$port $n res/${n}_speed.json 1000,4000,16000,32000 2>&1 | tail -5
  $PY -I humaneval.py http://127.0.0.1:$port $n res/${n}_humaneval.jsonl 164 2>&1 | tail -1
  $PY -I aquarium.py http://127.0.0.1:$port $n res/${n}_aquarium.html 2>&1 | tail -1
else
  echo "$n SERVER_FAILED"; tail -20 logs/srv_$n.log
fi
kill -- -$sp 2>/dev/null; sleep 8; kill -9 -- -$sp 2>/dev/null
# the card must be empty before the next arm
for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 2; done
kill $mp 2>/dev/null
echo "$n peak VRAM $(sort -n logs/srv_$n.mem | tail -1) MiB"
sleep 10
