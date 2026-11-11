#!/bin/bash
# serve_chat.sh TAG BIN [ENV=VAL ...] — `cortiq serve` on CUDA, the three-turn
# chat of serve_chat.py over edit0 (a 10 KB file), then stop by PID.
tag=$1; bin=$2; shift 2
cd /root/cuda
d=out/chat_$tag; rm -rf $d; mkdir -p $d
env "$@" CMF_GPU=cuda CMF_GPU_PROBE=0 CMF_PREFILL_PROF=1 setsid $bin serve /root/models/qwen38-27b-q4tp.cmf \
  --port 8090 --host 127.0.0.1 > $d/server.log 2>&1 < /dev/null &
pid=$!
for i in $(seq 1 90); do curl -s -m 2 127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 2; done
python3 src/scripts/q27cuda/serve_chat.py 8090 src/scripts/q27cuda/prompts/edit0.txt $d
kill $pid; sleep 2
sed 's/\x1b\[[0-9;]*m//g' $d/server.log | grep -E "eval time|kv-reuse|ERROR|WARN|panick" > $d/evals.txt
cat $d/evals.txt
