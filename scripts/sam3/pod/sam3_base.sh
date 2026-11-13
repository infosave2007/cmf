#!/bin/bash
# SAM3 baseline on the pod: one 1008 x 1008 image, the text prompt, per stage
# (SAM3_STAGES=1) on the host, Vulkan and CUDA. Downloads the file once.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
py=$root/venv/bin/python
f=cmf/sam3-q8_2f.cmf
if [ ! -s $f ]; then
  $py - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("infosave/SAM3-cmf", "sam3-q8_2f.cmf", local_dir="/root/h3i/cmf")
PY
fi
ls -la $f
[ -s ref/sam3_img.png ] || $py - <<'PY'
from PIL import Image, ImageDraw
im = Image.new("RGB", (1008, 1008), (40, 90, 160))
d = ImageDraw.Draw(im)
d.rectangle([300, 360, 620, 640], fill=(220, 30, 30))
d.ellipse([680, 120, 900, 340], fill=(30, 200, 60))
im.save("/root/h3i/ref/sam3_img.png")
PY
bin=${BIN:-cortiq}
exec 9>"$root/gpu.lock"; flock 9
for be in ${BACKENDS:-cuda wgpu 0}; do
  tag=sam3_$be
  t0=$(date +%s.%N)
  CMF_GPU=$be SAM3_STAGES=1 bin/$bin segment $f ref/sam3_img.png --prompt "red square" --out out/$tag.png --json out/$tag.json > out/$tag.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  echo "== $be rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")"
  grep -E "sam3-prof|box|score|error|panick" out/$tag.log | head -24 | cut -c1-160
done
echo SAM3_BASE_DONE
