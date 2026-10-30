#!/bin/bash
# q4tp-v2: GPTQ for every routed and shared expert of Apex-35B (cortiq
# quantize-moe, codec q4tp) against 262 144 calibration tokens — wikitext-2
# train 0.4, UltraChat rendered in the checkpoint's template 0.4, Rust
# sources 0.2 — the recipe of the Clover q4tp-v2 file on the same trunk.
# The pod's cgroup holds 50 GB: 20 layers a capture pass (~21 GB of
# activations) instead of the default 24.
#
# usage: pod_gptq.sh [CORTIQ]
set -eu
B=${1:-/root/apex/bin/cortiq}
export XDG_RUNTIME_DIR=/tmp
cd /root/apex
"$B" quantize-moe out/apex-35b-v1.0-q4tp.cmf --hf /root/models/apex \
  --calib data/wiki.train.raw=0.4 --calib data/chat.txt=0.4 --calib data/code.txt=0.2 \
  --tokens 262144 --window 2048 --lambda 0.01 --codec q4tp \
  --layers-per-pass 20 --threads 8 --scratch /root/apex/moe-scratch \
  --output out/apex-35b-v1.0-q4tp-v2.cmf
echo GPTQ_EXIT $?
