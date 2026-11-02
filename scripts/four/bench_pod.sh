#!/bin/bash
# Speed and quality numbers on the pod (A40 Vulkan + 96-core CPU).
export XDG_RUNTIME_DIR=/tmp
B=/root/cmf/target/release/cortiq
O=/root/four/out; I=/root/four/img/assets; W=/root/four/wiki_test.txt
cd $O
python3 -c "
from PIL import Image
im=Image.open('$I/img01.png').convert('RGB').resize((1024,1024), Image.BICUBIC); im.save('/root/four/img1024.png')
"
echo "== ppl (cortiq, 2048 tokens of wikitext-2 test)"
for m in mage-q8_2f mage-q4tp sn-q8_2f sn-q4tp; do
  echo -n "$m: "; timeout 1800 $B ppl $m.cmf -f $W --tokens 2048 2>&1 | grep -i "ppl\|perplex" | tail -1
done
echo "== bench gpu"
for m in mage-q4tp mage-q8_2f sn-q4tp sn-q8_2f; do
  echo "-- $m"; timeout 900 $B bench $m.cmf --tokens 128 2>&1 | grep -iE "tok/s|prefill|decode" | tail -3
done
echo "== bench cpu"
for m in mage-q4tp sn-q4tp; do
  echo "-- $m"; CMF_GPU=0 timeout 900 $B bench $m.cmf --tokens 32 2>&1 | grep -iE "tok/s|prefill|decode" | tail -3
done
echo "== towers (2206x1048 screenshot-like image)"
for mode in gpu cpu; do
  env=""; [ $mode = cpu ] && env="CMF_GPU=0"
  echo "-- mage $mode"; env $env CMF_OV2_PROF=1 timeout 1800 $B run mage-q4tp.cmf --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 24 2>&1 | grep -E "onevision2:|mage-vl:|tok/s" | tail -3
  echo "-- sn $mode"; env $env CMF_SN_PROF=1 timeout 2400 $B run sn-q4tp.cmf --no-think --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 24 2>&1 | grep -E "sensenova:|tok/s" | tail -3
done
echo "== primal_vae"
for v in 2x1 2x2; do for img in $I/img01.png /root/four/img1024.png; do
  for mode in gpu cpu; do
    env=""; [ $mode = cpu ] && env="CMF_GPU=0"
    echo -n "$v $(basename $img) $mode: "; env $env $B vae --model primal_vae_$v.cmf --mode roundtrip --input $img --output /tmp/pv.png 2>&1 | tail -1
  done
done; done
echo "== done"
