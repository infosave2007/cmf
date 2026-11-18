#!/bin/bash
# gate.sh TAG: the baseline binary and the current one, 6 steps each, same seed: checkpoint sha256 must match
cd /root/hc
[ -f run/gate-old.out ] || SAVE=1 E=/root/hc/emb-base ./bench.sh gate-old 8 6 > run/gate-old.out 2>&1
SAVE=1 ./bench.sh gate-$1 8 6 > run/gate-$1.out 2>&1
CMF_VULKAN_PROFILE_OPS=1 ./bench.sh enc-$1 8 16 > run/enc-$1.out 2>&1
echo done > run/gate-$1.done
