#!/bin/bash
# serve_eval.sh TAG MODEL [serve args...]: a cortiq server on MODEL, engine_eval.py against it, then the server goes (by PID)
set -u
tag=$1; m=$2; shift 2
B=${B:-cortiq}
PY=${PY:-python3}
here=$(cd "$(dirname "$0")" && pwd)
port=$((18100 + RANDOM % 800))
$B serve $m -p $port --host 127.0.0.1 "$@" > logs_serve_$tag.log 2>&1 &
pid=$!
for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:$port/v1/models && break; sleep 1; done
timeout 3000 $PY $here/engine_eval.py $port $tag
kill $pid; wait $pid 2>/dev/null
