#!/bin/bash
# baseline benchmarks with the b5 binary: both models, both backends; HumanEval on CUDA
cd /root/e0
B=/root/e0/cortiq.b5
log() { echo "[$(date +%T)] $*"; }
log "145 cuda"; HE=1 /root/e0_bench.sh c145 out/edge0-145b.cmf cuda $B
log "145 vulkan"; /root/e0_bench.sh v145 out/edge0-145b.cmf vulkan $B
log "120 cuda"; HE=1 /root/e0_bench.sh c120 out/edge0-120b.cmf cuda $B
log "120 vulkan"; /root/e0_bench.sh v120 out/edge0-120b.cmf vulkan $B
log CHAIN-A-DONE
