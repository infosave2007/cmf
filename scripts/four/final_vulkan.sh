#!/bin/bash
# Final Vulkan numbers (release build): decode, 2k prefill, towers, server warm image.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A7 | head -20
B=/root/cmf/target/release/cortiq
I=/root/four/img/assets; cd /root/four/out
echo "== decode (bench --tokens 128 --ignore-eos)"
for m in mage-q4tp mage-q8_2f sn-q4tp sn-q8_2f; do echo -n "$m: "; timeout 900 $B bench $m.cmf --tokens 128 --ignore-eos 2>&1 | grep -E "Decode" | tail -1; done
echo "== 2k prompt"
for m in mage-q4tp mage-q8_2f sn-q4tp sn-q8_2f; do echo "$m: "; timeout 900 $B bench $m.cmf --ctx 2048 --tokens 32 --ignore-eos 2>&1 | grep -E "Prompt|Decode" | tail -2; done
echo "== towers"
for img in img01.png mage-vl-cowork-teaser.png; do
  echo "-- mage $img"; CMF_OV2_PROF=1 timeout 900 $B run mage-q4tp.cmf --image $I/$img --prompt "Describe the image." --greedy --max-tokens 8 2>&1 | grep -E "vit|onevision2:|mage-vl:|tok/s" | tail -4
  echo "-- sn $img"; CMF_SN_PROF=1 timeout 900 $B run sn-q4tp.cmf --no-think --image $I/$img --prompt "Describe the image." --greedy --max-tokens 8 2>&1 | grep -E "vit|qwen3vis:|sensenova:|tok/s" | tail -4
done
echo "== screenshot TTFT (2206x1048, 2277 tokens)"
for m in mage-q8_2f mage-q4tp sn-q4tp sn-q8_2f; do echo -n "$m: "; CMF_OV2_PROF=1 CMF_SN_PROF=1 timeout 900 $B run $m.cmf --no-think --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 32 2>&1 | grep -E "tok/s" | tail -1; done
echo "== primal_vae"
python3 -c "
from PIL import Image
Image.open('$I/img01.png').convert('RGB').resize((1024,1024), Image.BICUBIC).save('/root/four/img1024.png')"
for v in 2x1 2x2; do for img in $I/img01.png /root/four/img1024.png; do echo -n "$v $(basename $img): "; $B vae --model primal_vae_$v.cmf --mode roundtrip --input $img --output /tmp/pv.png 2>&1 | tail -1; done; done
echo "== server warm image (img01)"
B=$B bash /root/cmf/scripts/four/serve_test.sh 2>&1 | tail -10
echo "== final done"
