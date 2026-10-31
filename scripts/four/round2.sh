#!/bin/bash
# Rebuild, re-check GPU parity (SenseNova image/text, Mage image), then the bench set.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A7 | head -30
B=/root/cmf/target/release/cortiq
O=/root/four/out; R=/root/four/ref; I=/root/four/img/assets
cd $O
echo "== sn q8_2f gpu image (batched MRoPE prefill)"
d=/tmp/sn_gpu2; rm -rf $d; mkdir -p $d
CMF_SN_DUMP=$d CMF_SN_PROF=1 timeout 1800 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -v "^\s*$" | tail -4
python3 -c "import json;print('REF:', json.load(open('$R/sn/img01.json'))['gen_text'][:300])"
echo "== sn q8_2f gpu image, host tower GEMMs (CMF_COOP=0)"
CMF_COOP=0 CMF_SN_PROF=1 timeout 1800 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -v "^\s*$" | tail -3
echo "== sn q8_2f gpu text (batched mHC prefill)"
timeout 900 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 40 --prompt "Write a haiku about autumn leaves." 2>&1 | grep -v "^\s*$" | tail -4
echo "== mage q8_2f gpu image"
CMF_OV2_PROF=1 timeout 1800 $B run mage-q8_2f.cmf --image $I/img01.png --prompt "Describe this image in two sentences." --greedy --max-tokens 48 2>&1 | grep -v "^\s*$" | tail -4
python3 -c "import json;print('REF:', json.load(open('$R/mage/img01_describe.json'))['gen_text'][:300])"
echo "== round2 done"
bash /root/cmf/scripts/four/bench_pod.sh
