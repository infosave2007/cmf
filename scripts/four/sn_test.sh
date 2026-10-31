#!/bin/bash
# Build; SenseNova reference (Transformers float32) then cortiq parity.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A7 | head -30
# wait for the Mage test to release the GPU
while pgrep -f test_mage.sh >/dev/null; do sleep 5; done
cd /root/four
echo "== sn ref $(date)"
SN_GEN=48 python3 /root/cmf/scripts/four/sensenova_ref.py /root/four/dl/SenseNova-Loop-9B-Preview /root/four/ref/sn img/assets/img01.png 2>&1 | grep -v "Warning\|warn" | tail -12
echo "== sn ref done $(date)"
