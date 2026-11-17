#!/bin/bash
# #151 check: build the win-sysmem branch (src151, fast, CUDA) and ask a
# running `serve` for /v1/cortiq/status on wgpu and on CUDA.
set -u
cd /root/h3i || exit 1
. "$HOME/.cargo/env"
export XDG_RUNTIME_DIR=/tmp
(cd src151 && CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli --features cuda --target-dir target-fast 2>&1 | grep -E "^error|Finished" -A8 | tail -20)
cp src151/target-fast/release/cortiq bin/cortiq-151 || { echo ST151_BUILD_FAILED; exit 1; }
mkdir -p cmf/dl logs/st151
f=cmf/dl/lfm3-1.7b-q4tp.cmf
[ -s $f ] || timeout 1800 venv/bin/hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q4tp.cmf --local-dir cmf/dl > logs/st151/dl.log 2>&1
for be in 1 cuda; do
  CMF_GPU=$be timeout 300 bin/cortiq-151 serve $f --port 8093 > logs/st151/serve_$be.log 2>&1 &
  sp=$!
  for i in $(seq 1 120); do curl -s -o /dev/null http://127.0.0.1:8093/v1/models && break; sleep 1; done
  echo "== CMF_GPU=$be before a request:"; curl -s http://127.0.0.1:8093/v1/cortiq/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('execution_mode'), d.get('performance',{}).get('vram_used_mb'))"
  curl -s -o /dev/null -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' http://127.0.0.1:8093/v1/chat/completions
  echo "== CMF_GPU=$be after a request:"; curl -s http://127.0.0.1:8093/v1/cortiq/status | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('execution_mode'), d.get('performance',{}).get('vram_used_mb'))"
  kill $sp; wait $sp 2>/dev/null
done
echo ST151_DONE
