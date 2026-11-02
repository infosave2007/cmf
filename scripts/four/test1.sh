#!/bin/bash
# primal_vae convert + parity, Mage-VL tower/greedy parity (CPU and GPU).
. ~/.cargo/env
export XDG_RUNTIME_DIR=/tmp
cd /root/cmf && cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A5 | head -20
B=/root/cmf/target/release/cortiq
O=/root/four/out; R=/root/four/ref; I=/root/four/img/assets
cd $O
for v in 2x1 2x2; do
  $B convert --model /root/four/dl/primal_vae/primal_vae_$v.safetensors --output primal_vae_$v.cmf 2>&1 | tail -1
done
for v in 2x1 2x2; do for img in img01 img02 img03; do
  echo "== pvae $v $img cpu"; CMF_GPU=0 $B vae --model primal_vae_$v.cmf --mode roundtrip --input $I/$img.png --output /tmp/pv_${v}_$img.png 2>&1 | tail -1
  echo "== pvae $v $img gpu"; $B vae --model primal_vae_$v.cmf --mode roundtrip --input $I/$img.png --output /tmp/pv_${v}_${img}_g.png 2>&1 | tail -1
done; done
$B vae --model primal_vae_2x2.cmf --mode encode --input $I/img01.png --output /tmp/z.npy 2>&1 | tail -1
python3 - <<'PY'
import numpy as np
z=np.load('/tmp/z.npy'); r=np.load('/root/four/ref/pvae/2x2_img01_z.npy')
print('2x2 img01 latent vs ref: max|d|', np.abs(z-r).max(), 'rel', np.abs(z-r).max()/np.abs(r).max())
PY
echo "== mage tower cpu f16"
mkdir -p /tmp/ov2c /tmp/ov2g
CMF_GPU=0 CMF_OV2_DUMP=/tmp/ov2c timeout 900 $B run --model mage-f16.cmf --image $I/img01.png --prompt "Describe this image in two sentences." --greedy --max-tokens 40 2>&1 | grep -v "^\s*$" | tail -6
python3 /root/cmf/scripts/four/cmp_rows.py /tmp/ov2c/rows_0.f32 $R/mage/img01_describe_rows.npy 2560
python3 /root/cmf/scripts/four/cmp_rows.py /tmp/ov2c/pixels_0.f32 $R/mage/img01_describe_pixels.npy 768
echo "== mage gpu f16"
CMF_OV2_DUMP=/tmp/ov2g CMF_OV2_PROF=1 timeout 900 $B run --model mage-f16.cmf --image $I/img01.png --prompt "Describe this image in two sentences." --greedy --max-tokens 40 2>&1 | grep -v "^\s*$" | tail -8
python3 /root/cmf/scripts/four/cmp_rows.py /tmp/ov2g/rows_0.f32 $R/mage/img01_describe_rows.npy 2560
python3 -c "import json;print('REF:', json.load(open('$R/mage/img01_describe.json'))['gen_text'][:300])"
echo "== done"
