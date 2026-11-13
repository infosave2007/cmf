#!/bin/bash
# e0_srv.sh BIN MODEL CTXS TAG "ENV..." [TAG "ENV..."]...: a full-card server per variant, one speedtest pass each
BIN=$1; M=$2; CTXS=$3; shift 3
cd /root/e0
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH XDG_RUNTIME_DIR=/tmp
while [ $# -ge 2 ]; do
  TAG=$1; ENVS=$2; shift 2
  env CMF_SERVE_SLOTS=1 CMF_GPU=cuda $ENVS $BIN serve $M --host 127.0.0.1 --port 8098 > logs/srv_$TAG.log 2>&1 &
  SP=$!
  for i in $(seq 1 600); do curl -s -m 2 http://127.0.0.1:8098/v1/models >/dev/null 2>&1 && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  echo "== $TAG [$ENVS]"
  /root/venv/bin/python tools/speedtest.py http://127.0.0.1:8098 edge0 logs/srv_$TAG.json $CTXS 2>&1 | grep ctx
  kill $SP; wait $SP 2>/dev/null; sleep 2
done
echo SRV-DONE
