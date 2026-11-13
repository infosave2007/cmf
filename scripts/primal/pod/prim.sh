#!/bin/bash
# Primal VAE (2x2) round trips at 512 and 1024: CUDA, Vulkan and the host,
# on the given build (BIN), under gpu.lock. Prints the CLI's timings and
# PSNR lines.
#   prim.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-pv}
cd "$root" || exit 1
bin=${BIN:-$root/bin/cortiq-fast}
f=cmf/primal_vae_2x2.cmf
[ -f "$f" ] || venv/bin/hf download infosave/primal_vae-cmf primal_vae_2x2.cmf --local-dir cmf > logs/pv_dl.log 2>&1
venv/bin/python - <<'PY'
from PIL import Image
im = Image.open("/root/h3i/ref/edit_in.png").convert("RGB")
im.resize((1024, 1024)).save("/root/h3i/ref/pv_1024.png")
im.resize((512, 512)).save("/root/h3i/ref/pv_512.png")
PY
export XDG_RUNTIME_DIR=/tmp
exec 9>"$root/gpu.lock"; flock 9
for s in 512 1024; do
  for be in cuda 1 0; do
    t0=$(date +%s.%N)
    CMF_GPU=$be "$bin" vae --model $f --mode roundtrip --input ref/pv_$s.png --output out/${p}_${be}_$s.png > out/${p}_${be}_$s.log 2>&1
    t1=$(date +%s.%N)
    echo "size=$s backend=$be rc=$? wall_s=$(awk "BEGIN{printf \"%.2f\", $t1-$t0}") :: $(grep -iE "encode|decode|psnr|declin" out/${p}_${be}_$s.log | grep -v INFO | tr '\n' ' ' | cut -c1-300)"
  done
done
echo PV_DONE
