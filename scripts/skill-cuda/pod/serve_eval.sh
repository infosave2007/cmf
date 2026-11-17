#!/bin/bash
# serve_eval.sh TAG MODEL BACKEND [SKILL] [N]: one server, the test set, then the server goes (by PID)
set -u
tag=$1; m=$2; be=$3; sk=${4:-}; n=${5:-100000}
B=${B:-/root/h3i/bin/cortiq-skill}
port=$((18000 + RANDOM % 1000))
cd /root/h3i/skill/emo
args=(); [ -n "$sk" ] && args=(--skill "$sk")
env CMF_GPU=$be XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 ${EXTRA_ENV:-} $B serve $m -p $port --host 127.0.0.1 "${args[@]}" > logs_serve_$tag.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:$port/v1/models && break; sleep 1; done
timeout 3600 /root/h3i/venv/bin/python eval.py $port $tag $n
kill $pid; wait $pid 2>/dev/null
