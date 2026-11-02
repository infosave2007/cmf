#!/bin/bash
# SenseNova image prompt through the token-axis MRoPE path: parity and TTFT.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
bash /root/four/devbuild.sh | tail -1
B=/root/cmf/target-dev/release/cortiq
I=/root/four/img/assets; cd /root/four/out
for c in 0 1; do
  echo "-- sn q8_2f image CMF_COOP=$c"
  CMF_COOP=$c timeout 900 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -E "^A |tok/s"
done
python3 -c "import json;print('REF:', json.load(open('/root/four/ref/sn/img01.json'))['gen_text'][:300])"
for m in sn-q4tp sn-q8_2f; do
  echo -n "-- $m screenshot: "; timeout 900 $B run $m.cmf --no-think --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 32 2>&1 | grep -E "tok/s"
done
echo "== sn mrope done"
