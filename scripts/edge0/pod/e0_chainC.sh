#!/bin/bash
# Vulkan benchmarks on b6 (grouped prompt kernels), then build b7 (CUDA U split by group), CUDA smoke and benchmarks
log() { echo "[$(date +%T)] $*"; }
cd /root/e0
log "145 vulkan b6"; /root/e0_bench.sh v145b6 out/edge0-145b.cmf vulkan /root/e0/cortiq.b6
log "120 vulkan b6"; /root/e0_bench.sh v120b6 out/edge0-120b.cmf vulkan /root/e0/cortiq.b6
. /root/.cargo/env
cd /root/e0/cmf && rm -rf crates.prev6 && mv crates crates.prev6 && mv crates.b7 crates
cargo build --release -p cortiq-cli --features cuda > /root/e0/logs/build7.log 2>&1; log "build rc=$?"
grep -E "^error" -A6 /root/e0/logs/build7.log | head -20
cp /root/e0/cmf/target/release/cortiq /root/e0/cortiq.b7
cd /root/e0
export XDG_RUNTIME_DIR=/tmp LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
P2="Write a Rust function that parses a semantic version string like 1.2.3-beta+build into a struct, with unit tests. Code only."
CMF_GPU=cuda timeout 1800 /root/e0/cortiq.b7 run out/edge0-145b.cmf --prompt "$P2" --greedy --no-think -n 400 > logs/b7_cuda.log 2>&1; log "cuda rc=$?"
grep -a "tok/s\|MTP draft" logs/b7_cuda.log
log "145 cuda b7"; /root/e0_bench.sh c145b7 out/edge0-145b.cmf cuda /root/e0/cortiq.b7
log "120 cuda b7"; /root/e0_bench.sh c120b7 out/edge0-120b.cmf cuda /root/e0/cortiq.b7
log CHAIN-C-DONE
