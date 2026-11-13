#!/bin/bash
# after chain A: build b6 (synced to crates.b6), smoke both backends, benchmark both models on both backends
log() { echo "[$(date +%T)] $*"; }
while ! grep -q CHAIN-A-DONE /root/e0/logs/chainA.log; do sleep 20; done
. /root/.cargo/env
cd /root/e0/cmf && rm -rf crates.prev && mv crates crates.prev && mv crates.b6 crates
cargo build --release -p cortiq-cli --features cuda > /root/e0/logs/build6.log 2>&1; log "build rc=$?"
grep -E "^error" -A6 /root/e0/logs/build6.log | head -20
cp /root/e0/cmf/target/release/cortiq /root/e0/cortiq.b6
cd /root/e0
export XDG_RUNTIME_DIR=/tmp LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
P2="Write a Rust function that parses a semantic version string like 1.2.3-beta+build into a struct, with unit tests. Code only."
CMF_GPU=cuda timeout 1800 /root/e0/cortiq.b6 run out/edge0-145b.cmf --prompt "$P2" --greedy --no-think -n 400 > logs/b6_cuda.log 2>&1; log "cuda rc=$?"
timeout 1800 /root/e0/cortiq.b6 run out/edge0-145b.cmf --prompt "$P2" --greedy --no-think -n 400 > logs/b6_vk.log 2>&1; log "vk rc=$?"
grep -a "tok/s\|MTP draft" logs/b6_cuda.log logs/b6_vk.log
B=/root/e0/cortiq.b6
log "145 cuda"; /root/e0_bench.sh c145b6 out/edge0-145b.cmf cuda $B
log "145 vulkan"; /root/e0_bench.sh v145b6 out/edge0-145b.cmf vulkan $B
log "120 cuda"; /root/e0_bench.sh c120b6 out/edge0-120b.cmf cuda $B
log "120 vulkan"; /root/e0_bench.sh v120b6 out/edge0-120b.cmf vulkan $B
log CHAIN-B-DONE
