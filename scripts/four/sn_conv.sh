#!/bin/bash
# Build, then convert SenseNova-Loop-9B to f16 / q8_2f / q4tp.
. ~/.cargo/env
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A7 | head -30
B=/root/cmf/target/release/cortiq
cd /root/four/out
echo "== convert sensenova $(date)"
$B convert --model /root/four/dl/SenseNova-Loop-9B-Preview --quant q8_2f --output sn-q8_2f.cmf --also-output q4tp=sn-q4tp.cmf --also-output f16=sn-f16.cmf 2>&1 | tail -4
ls -la /root/four/out/sn-*
echo "== done $(date)"
