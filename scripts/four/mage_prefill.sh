#!/bin/bash
# Mage-VL batched-graph prefill: speed (2k prompt) and image/text parity.
export XDG_RUNTIME_DIR=/tmp
B=${B:-/root/cmf/target-dev/release/cortiq}
I=/root/four/img/assets; R=/root/four/ref; cd /root/four/out
for q in q8_2f q4tp; do
  echo "-- mage $q 2k"; timeout 900 $B bench mage-$q.cmf --ctx 2048 --tokens 32 --ignore-eos 2>&1 | grep -E "Prompt|Decode"
done
for q in q8_2f q4tp; do
  echo "-- mage $q image"; timeout 900 $B run mage-$q.cmf --image $I/img01.png --prompt "Describe this image in two sentences." --greedy --max-tokens 48 2>&1 | grep -E "This|tok/s" | tail -2
done
python3 -c "import json;print('REF:', json.load(open('$R/mage/img01_describe.json'))['gen_text'][:300])"
echo "-- mage q8_2f big screenshot"; CMF_OV2_PROF=1 timeout 900 $B run mage-q8_2f.cmf --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 64 2>&1 | grep -E "mage-vl:|tok/s|chart|This" | tail -4
echo "-- mage q8_2f text"; timeout 600 $B run mage-q8_2f.cmf --prompt "What is the capital of France? Answer in one word." --greedy --max-tokens 48 2>&1 | grep -v "^\s*$" | tail -2
echo "== mage prefill done"
