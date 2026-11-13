#!/bin/bash
# final benchmarks with b10: both models, both backends (speedtest 1K/4K/16K/32K, two runs)
log() { echo "[$(date +%T)] $*"; }
cd /root/e0
B=/root/e0/cortiq.b10
log "145 cuda"; /root/e0_bench.sh c145f out/edge0-145b.cmf cuda $B
log "145 vulkan"; /root/e0_bench.sh v145f out/edge0-145b.cmf vulkan $B
log "120 cuda"; /root/e0_bench.sh c120f out/edge0-120b.cmf cuda $B
log "120 vulkan"; /root/e0_bench.sh v120f out/edge0-120b.cmf vulkan $B
log CHAIN-D-DONE
