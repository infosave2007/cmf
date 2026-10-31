#!/bin/bash
# Mage-VL tower parity (CPU/GPU) and greedy text against Transformers.
export XDG_RUNTIME_DIR=/tmp
B=${B:-/root/cmf/target/release/cortiq}
O=/root/four/out; R=/root/four/ref; I=/root/four/img/assets
cd $O
P="Describe this image in two sentences."
for mode in cpu gpu; do
  d=/tmp/ov2_$mode; rm -rf $d; mkdir -p $d
  env=""; [ $mode = cpu ] && env="CMF_GPU=0"
  echo "== mage f16 $mode"
  env $env CMF_OV2_DUMP=$d CMF_OV2_PROF=1 timeout 1200 $B run mage-f16.cmf --image $I/img01.png --prompt "$P" --greedy --max-tokens 48 2>&1 | grep -v "^\s*$" | tail -7
  python3 /root/cmf/scripts/four/cmp_rows.py $d/pixels_0.f32 $R/mage/img01_describe_pixels.npy 768
  python3 /root/cmf/scripts/four/cmp_rows.py $d/rows_0.f32 $R/mage/img01_describe_rows.npy 2560
done
python3 -c "import json;print('REF:', json.load(open('$R/mage/img01_describe.json'))['gen_text'][:400])"
echo "== mage text f16 gpu"
timeout 600 $B run mage-f16.cmf --prompt "What is the capital of France? Answer in one word." --greedy --max-tokens 48 2>&1 | grep -v "^\s*$" | tail -4
python3 -c "import json;print('REF:', json.load(open('$R/mage/text.json'))['gen_text'][:400])"
echo "== done"
