#!/bin/bash
# aq_multi.sh: the owner's aquarium prompt, several honest attempts per model under the same settings
# (thinking greedy; no thinking T=0.7 seeds 1-3; thinking T=0.6 seeds 1-2), pages kept for a browser check
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 720); do grep -q CHAIN5_DONE $J && break; sleep 20; done
PY=/root/fc/venv/bin/python
run_set() { # NAME PORT
  local n=$1 port=$2
  $PY -I aquarium.py http://127.0.0.1:$port $n res/aq_${n}_think.html '{"temperature": 0, "max_tokens": 30000, "enable_thinking": true, "chat_template_kwargs": {"enable_thinking": true}}' 2>&1 | tail -1 | sed "s/^/$n think: /" >> $J
  for sd in 1 2 3; do $PY -I aquarium.py http://127.0.0.1:$port $n res/aq_${n}_t07s$sd.html "{\"temperature\": 0.7, \"top_p\": 0.95, \"seed\": $sd}" 2>&1 | tail -1 | sed "s/^/$n t0.7 seed $sd: /" >> $J; done
  for sd in 1 2; do $PY -I aquarium.py http://127.0.0.1:$port $n res/aq_${n}_think_t06s$sd.html "{\"temperature\": 0.6, \"top_p\": 0.95, \"seed\": $sd, \"max_tokens\": 30000, \"enable_thinking\": true, \"chat_template_kwargs\": {\"enable_thinking\": true}}" 2>&1 | tail -1 | sed "s/^/$n think t0.6 seed $sd: /" >> $J; done
}
# Fast Code on f11
setsid env CMF_QWEN_PROFILE=out/fc256.profile /root/fc/cortiq.f11 serve out/fc256.cmf --host 127.0.0.1 --port 8091 > logs/srv_aq_fc256.log 2>&1 &
sp=$!
for i in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:8091/v1/models > /dev/null && break; sleep 5; done
run_set fc256 8091
kill -- -$sp 2>/dev/null; sleep 8; kill -9 -- -$sp 2>/dev/null
for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] && break; sleep 2; done
# Strata engine
setsid bash -c "cd /root/fc/strata/control && exec /root/fc/strata/control/python/bin/python3.12 serve/server.py --engine strata --config /root/.local/share/strata/install/strata-coder-iq1_m.json --port 8090" > logs/srv_aq_strata.log 2>&1 &
sp=$!
for i in $(seq 1 120); do curl -s -m 5 http://127.0.0.1:8090/v1/models > /dev/null && break; sleep 5; done
run_set strata 8090
kill -- -$sp 2>/dev/null; sleep 8; kill -9 -- -$sp 2>/dev/null
log AQ_MULTI_DONE
