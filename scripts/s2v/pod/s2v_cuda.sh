#!/bin/bash
# Kandinsky-S2V on its CUDA arm (fast build): one 5 s chunk at 512 x 512, 16
# steps; frames 10, 60 and 110 against the Vulkan run's.
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-s2vc}
steps=${STEPS:-16}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
py=$root/venv/bin/python
bash scripts/build.sh ${BUILD:-fast} > logs/${p}_build.log 2>&1 || { echo BUILD_FAILED; tail -40 logs/${p}_build.log; exit 1; }
f=cmf/kandinsky-s2v-pro-q4tp.cmf
tag=${p}_512
exec 9>"$root/gpu.lock"; flock 9
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > out/$tag.vram_trace 2>/dev/null &
tr=$!; t0=$(date +%s.%N)
CMF_GPU=cuda CMF_CUDA_DEBUG=1 CMF_S2V_PROF=1 CMF_S2V_CUDA_PROF=${PROF:-0} RUST_LOG=info bin/${BIN:-cortiq-fast} s2v --model $f --image ref/s2v_ref.png --audio ref/s2v_track.wav \
  --height 512 --width 512 --steps $steps --chunks 1 --out out/$tag.avi > out/$tag.log 2>&1
rc=$?; t1=$(date +%s.%N); kill $tr 2>/dev/null; exec 9>&-
echo "tag=$tag rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$(sort -n out/$tag.vram_trace | tail -1)"
tr '\r' '\n' < out/$tag.log | grep -iE "step|vae|declin|error|panick|device|s2v cuda" | grep -v "INFO.*Opened" | tail -${TAIL:-8} | cut -c1-220
[ "$steps" = 16 ] || { echo S2V_CUDA_DONE; exit 0; }
for n in 10 60 110; do
  for t in $tag s2vb_vk_512; do ffmpeg -loglevel error -y -i out/$t.avi -vf "select=eq(n\,$n)" -vframes 1 out/${t}_f$n.png; done
  echo "frame $n cuda vs vulkan: $($py - out/${tag}_f$n.png out/s2vb_vk_512_f$n.png <<'PY'
import sys
import numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=float)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=float)
mse = ((a - b) ** 2).mean()
print("identical" if mse == 0 else "psnr %.2f dB" % (10 * np.log10(255 ** 2 / mse)))
PY
)"
done
echo S2V_CUDA_DONE
