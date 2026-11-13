#!/bin/bash
# e0_bench.sh TAG MODEL BACKEND(cuda|vulkan) [BIN]: start a server, speedtest at 1K/4K/16K/32K (two runs),
# HumanEval when HE=1; logs/bench_TAG.*
TAG=$1; M=$2; BK=$3; BIN=${4:-/root/e0/cortiq.cur}
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
ENVS=(CMF_SERVE_SLOTS=1)
[ "$BK" = cuda ] && ENVS+=(CMF_GPU=cuda)
PORT=8090
env "${ENVS[@]}" $BIN serve $M --host 127.0.0.1 --port $PORT > logs/bench_$TAG.serve.log 2>&1 &
SP=$!
for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
if ! curl -s -m 2 http://127.0.0.1:$PORT/v1/models >/dev/null; then echo "server did not come up"; tail -20 logs/bench_$TAG.serve.log; exit 1; fi
echo "server up after $((i*2)) s"
for r in 1 2; do
  /root/venv/bin/python tools/speedtest.py http://127.0.0.1:$PORT edge0 logs/bench_${TAG}_r$r.json 1024,4096,16384,32768 2>&1 | tail -4
done
if [ "${HE:-0}" = 1 ]; then
  /root/venv/bin/python tools/humaneval.py http://127.0.0.1:$PORT edge0 logs/bench_${TAG}_he.jsonl 164 2>&1 | tail -3
fi
kill $SP; wait $SP 2>/dev/null
echo BENCH-DONE
