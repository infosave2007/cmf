#!/bin/bash
# Release-build check of the last change (MRoPE token-axis path) + 2k prompts.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A7 | head -20
B=/root/cmf/target/release/cortiq
I=/root/four/img/assets; cd /root/four/out
echo "-- sn q8_2f image"; timeout 900 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -E "^A |tok/s"
echo "-- mage q8_2f image"; timeout 900 $B run mage-q8_2f.cmf --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -E "^This |tok/s"
for m in mage-q8_2f mage-q4tp sn-q4tp sn-q8_2f; do
  echo -n "-- $m screenshot: "; timeout 900 $B run $m.cmf --no-think --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 32 2>&1 | grep -E "tok/s"
done
for m in mage-q4tp mage-q8_2f sn-q4tp sn-q8_2f; do echo "-- $m 2k"; timeout 900 $B bench $m.cmf --ctx 2048 --tokens 32 --ignore-eos 2>&1 | grep -E "Prompt|Decode"; done
echo "== final3 done"
