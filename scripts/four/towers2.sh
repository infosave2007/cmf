#!/bin/bash
# Resident ViT check: rows vs reference (img01) and timing (big image).
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
bash /root/four/devbuild.sh
B=/root/cmf/target-dev/release/cortiq
I=/root/four/img/assets; R=/root/four/ref; cd /root/four/out
d=/tmp/ov2_res; rm -rf $d; mkdir -p $d
echo "== mage img01"; CMF_OV2_DUMP=$d CMF_OV2_PROF=1 timeout 900 $B run mage-q8_2f.cmf --image $I/img01.png --prompt "Describe this image in two sentences." --greedy --max-tokens 48 2>&1 | grep -E "vit|onevision2:|tok/s|This" | tail -4
python3 /root/cmf/scripts/four/cmp_rows.py $d/rows_0.f32 $R/mage/img01_describe_rows.npy 2560
d=/tmp/sn_res; rm -rf $d; mkdir -p $d
echo "== sn img01"; CMF_SN_DUMP=$d CMF_SN_PROF=1 timeout 900 $B run sn-q8_2f.cmf --no-think --image $I/img01.png --prompt "Describe this image in two sentences." --greedy --max-tokens 48 2>&1 | grep -E "vit|qwen3vis:|tok/s|A diverse" | tail -4
python3 /root/cmf/scripts/four/cmp_rows.py $d/rows_0.f32 $R/sn/img01_rows.npy 4096
echo "== mage big"; CMF_OV2_PROF=1 timeout 900 $B run mage-q4tp.cmf --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 4 2>&1 | grep -E "vit|onevision2:|tok/s" | tail -3
echo "== sn big"; CMF_SN_PROF=1 timeout 900 $B run sn-q4tp.cmf --no-think --image $I/mage-vl-cowork-teaser.png --prompt "What does this chart show?" --greedy --max-tokens 4 2>&1 | grep -E "vit|qwen3vis:|sensenova:|tok/s" | tail -3
echo "== towers2 done"
