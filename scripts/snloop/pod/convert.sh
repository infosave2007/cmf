#!/bin/bash
# convert.sh BIN: SenseNova-Loop-9B -> f16 / q8_2f / q4tp (round-to-nearest) in /root/sn/out.
set -u
B=${1:-/root/sn/bin/cortiq-fast}
cd /root/sn/out || exit 1
echo "== convert $(date)"
$B convert --model /root/sn/dl/SenseNova-Loop-9B --quant q8_2f --output sn9-q8_2f.cmf \
  --also-output f16=sn9-f16.cmf --also-output q4tp=sn9-q4tp-rtn.cmf 2>&1 | grep -v "^\s*$" | tail -25
ls -la /root/sn/out/*.cmf
$B info sn9-q8_2f.cmf 2>&1 | grep -i -E "arch|yarn|rope|mtp|loop|mhc|layers|tensors" | head -20
echo "CONVERT_DONE $(date)"
