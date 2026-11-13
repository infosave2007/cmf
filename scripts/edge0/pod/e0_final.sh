#!/bin/bash
# after the 120B Vulkan run on 24 GB: stop the rest of the Vulkan matrix, then HumanEval of the 120B on the full card
log() { echo "[$(date +%T)] $*"; }
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
until grep -q "145 vulkan 32 GB" logs/chainJ.log; do sleep 5; done
for p in $(ps -eo pid,args | grep -v grep | grep -E "e0_chainJ.sh|e0_tier.sh" | awk "{print \$1}"); do kill $p; done
sleep 1
for p in $(ps -eo pid,args | grep -v grep | grep -E "cortiq.b21 serve|/vramfill |speedtest.py" | awk "{print \$1}"); do kill $p; done
sleep 6
log "humaneval 120 full"
env CMF_SERVE_SLOTS=1 CMF_GPU=cuda /root/e0/cortiq.b21 serve out/edge0-120b.cmf --host 127.0.0.1 --port 8092 > logs/he120.serve.log 2>&1 &
SP=$!
for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8092/v1/models >/dev/null 2>&1 && break; sleep 2; done
/root/venv/bin/python tools/humaneval.py http://127.0.0.1:8092 edge0 logs/he120.jsonl 164 2>&1 | tail -1
kill $SP; wait $SP 2>/dev/null
log FINAL-DONE
