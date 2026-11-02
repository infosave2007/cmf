#!/bin/bash
# SenseNova-Loop image parity (CPU, wgpu) after a rebuild.
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A7 | head -30
B=/root/cmf/target/release/cortiq
O=/root/four/out; R=/root/four/ref/sn; I=/root/four/img/assets
cd $O
for mode in gpu cpu; do
  env=""; [ $mode = cpu ] && env="CMF_GPU=0"
  d=/tmp/sn_$mode; rm -rf $d; mkdir -p $d
  echo "== sn q8_2f $mode image"
  env $env CMF_SN_DUMP=$d timeout 1800 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -v "^\s*$" | tail -4
  python3 /root/cmf/scripts/four/cmp_rows.py $d/pixels_0.f32 $R/img01_pixels.npy 1536
  python3 /root/cmf/scripts/four/cmp_rows.py $d/rows_0.f32 $R/img01_rows.npy 4096
done
python3 -c "import json;print('REF:', json.load(open('$R/img01.json'))['gen_text'][:300])"
echo "== done"
