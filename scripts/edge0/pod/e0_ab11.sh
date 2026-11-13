#!/bin/bash
# build the current /root/e0/cmf/crates as BIN, then quick 4K probes: e0_ab11.sh BIN probe-spec...
log() { echo "[$(date +%T)] $*"; }
BIN=$1; shift
. /root/.cargo/env
cd /root/e0/cmf && cargo build --release -p cortiq-cli --features cuda > /root/e0/logs/build_$(basename $BIN).log 2>&1; log "build rc=$?"
grep -E "^error" -A6 /root/e0/logs/build_$(basename $BIN).log | head -20
cp /root/e0/cmf/target/release/cortiq $BIN
cd /root/e0
export XDG_RUNTIME_DIR=/tmp LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
quick() { # tag env...
  local tag=$1; shift
  env CMF_SERVE_SLOTS=1 "$@" $BIN serve out/edge0-145b.cmf --host 127.0.0.1 --port 8091 > logs/ab_$tag.serve.log 2>&1 &
  local sp=$!
  for i in $(seq 1 300); do curl -s -m 2 http://127.0.0.1:8091/v1/models >/dev/null 2>&1 && break; sleep 2; done
  /root/venv/bin/python tools/speedtest.py http://127.0.0.1:8091 edge0 logs/ab_$tag.json 4096 2>&1 | tail -1 | sed "s/^/$tag: /"
  kill $sp; wait $sp 2>/dev/null
}
for spec in "$@"; do
  tag=${spec%%:*}; envs=${spec#*:}; [ "$envs" = "$spec" ] && envs=""
  quick $tag $envs
done
log AB-DONE
