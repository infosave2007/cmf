#!/bin/bash
# Tower timing on the big image (dev build): Mage and SenseNova, GPU.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
bash /root/four/devbuild.sh
B=/root/cmf/target-dev/release/cortiq
I=/root/four/img/assets; cd /root/four/out
for c in 1 0; do
echo "== mage gpu CMF_COOP=$c"; CMF_COOP=$c CMF_OV2_PROF=1 timeout 1800 $B run mage-q4tp.cmf --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 4 2>&1 | grep -E "onevision2:|tok/s" | tail -2
done
echo "== sn gpu"; CMF_SN_PROF=1 timeout 1800 $B run sn-q4tp.cmf --no-think --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 4 2>&1 | grep -E "qwen3vis:|sensenova:|tok/s" | tail -3
echo "== towers done"
