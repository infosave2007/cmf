#!/bin/bash
# SenseNova-Loop parity: CPU and wgpu, text and image, against Transformers.
export XDG_RUNTIME_DIR=/tmp
B=${B:-/root/cmf/target/release/cortiq}
O=/root/four/out; R=/root/four/ref/sn; I=/root/four/img/assets
cd $O
ref() { python3 -c "import json;print('REF:', json.load(open('$R/$1.json'))['gen_text'][:300].replace(chr(10),' | '))"; }
for mode in cpu gpu; do
  env=""; [ $mode = cpu ] && env="CMF_GPU=0"
  for q in q8_2f q4tp; do
    echo "== sn $q $mode text"
    env $env timeout 900 $B run sn-$q.cmf --no-think --greedy --max-tokens 40 --prompt "What is the capital of France? Answer in one sentence." 2>&1 | grep -v "^\s*$" | tail -2
  done
  ref text
  echo "== sn q8_2f $mode haiku"
  env $env timeout 900 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 40 --prompt "Write a haiku about autumn leaves." 2>&1 | grep -v "^\s*$" | tail -2
  ref text2
  d=/tmp/sn_$mode; rm -rf $d; mkdir -p $d
  echo "== sn q8_2f $mode image"
  env $env CMF_SN_DUMP=$d timeout 1500 $B run sn-q8_2f.cmf --no-think --greedy --max-tokens 48 --image $I/img01.png --prompt "Describe this image in two sentences." 2>&1 | grep -v "^\s*$" | tail -3
  python3 /root/cmf/scripts/four/cmp_rows.py $d/pixels_0.f32 $R/img01_pixels.npy 1536
  python3 /root/cmf/scripts/four/cmp_rows.py $d/rows_0.f32 $R/img01_rows.npy 4096
  ref img01
done
echo "== done"
