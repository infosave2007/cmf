#!/bin/bash
# After the Kimi round: the Vulkan VSA gate in f32 (as CUDA keeps it). Sparsity
# 0 against the host and CUDA, then the default sparsity at both sizes with
# the peak VRAM, the frames beside CUDA's.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 14400 ))
until grep -q -E 'KIMI_DONE|_FAILED' logs/kimi_prompt.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo KIMI_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
bash scripts/build.sh release > logs/fhvk_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/fhvk_build.log; exit 1; }
echo "build ok"
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
run() { # tag width height frames extra-env...
  local tag=$1 w=$2 h=$3 n=$4; shift 4
  mkdir -p out/fhvk_$tag
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > logs/fhvk_vram_$tag.txt 2>/dev/null &
  local smi=$! t0
  t0=$(date +%s)
  flock gpu.lock env CMF_GPU=1 "$@" CMF_ANIM_DUMP=out/fhvk_$tag/d timeout 3600 bin/cortiq animate $m --prompt "$prompt" \
    --width $w --height $h --frames $n --steps 4 --seed 7 --out out/fhvk_$tag/clip.avi --frames-dir out/fhvk_$tag/frames > logs/fhvk_$tag.log 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  echo "$tag: exit $rc in $(( $(date +%s) - t0 )) s; peak $(sort -n logs/fhvk_vram_$tag.txt | tail -1) MiB; $(grep -oE 'stages: .*' logs/fhvk_$tag.log | tail -1)"
  grep -iE "declin" logs/fhvk_$tag.log | sort -u | head -2
}
run s0 512 288 39 CMF_VSA_SPARSITY=0
run s 512 288 39 CMF_VSA_SPARSITY=0.9
run s2 512 288 39 CMF_VSA_SPARSITY=0.9
run b 1344 768 124 CMF_VSA_SPARSITY=0.9
venv/bin/python - <<'PY'
import array, math
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
for s in ('v', 'a'):
    print(f'sparsity 0, {s} step 0: vulkan vs host {rel(load(f"out/fhvk_s0/d.{s}0.f32"), load(f"out/fh_s0host/d.{s}0.f32")):.4f}'
          f'   vulkan vs cuda {rel(load(f"out/fhvk_s0/d.{s}0.f32"), load(f"out/fh_s0cuda/d.{s}0.f32")):.4f}'
          f'   (was 0.0217 / 0.0222 video, 0.0066 / 0.0067 audio)')
    print(f'sparsity 0.9, {s} step 0: vulkan vs cuda {rel(load(f"out/fhvk_s/d.{s}0.f32"), load(f"out/fh_cuda/d.{s}0.f32")):.4f}')
from PIL import Image, ImageDraw
def row(d, label, idx, w=None):
    ims = [Image.open(f'{d}/frames/frame_{i:04d}.ppm') for i in idx]
    if w:
        ims = [im.resize((w, im.size[1] * w // im.size[0])) for im in ims]
    W, H = ims[0].size
    r = Image.new('RGB', (W * len(ims), H + 20), 'white')
    for k, im in enumerate(ims):
        r.paste(im, (k * W, 20))
    ImageDraw.Draw(r).text((6, 4), label, fill='black')
    return r
rows = [row('out/fh5_full_b', 'CUDA 1344x768', (0, 60, 123), 512), row('out/fhvk_b', 'Vulkan 1344x768', (0, 60, 123), 512)]
s = Image.new('RGB', (rows[0].size[0], sum(r.size[1] for r in rows)), 'white')
y = 0
for r in rows:
    s.paste(r, (0, y)); y += r.size[1]
s.save('out/fhvk_big_sheet.png')
PY
echo FHVK_DONE
