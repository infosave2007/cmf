#!/bin/bash
# Is Vulkan's 2.2 % at sparsity 0 the VSA kernels or the chain's own class?
# Vulkan dense against CUDA dense (same seed) beside Vulkan s0 against CUDA s0.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
mkdir -p out/fh_vkdense
t0=$(date +%s)
flock gpu.lock env CMF_GPU=1 CMF_VSA=0 CMF_ANIM_DUMP=out/fh_vkdense/d timeout 3600 bin/cortiq animate $m --prompt "$prompt" \
  --width 512 --height 288 --frames 39 --steps 4 --seed 7 --out out/fh_vkdense/clip.avi > logs/fh_vkdense.log 2>&1
echo "vulkan dense: exit $? in $(( $(date +%s) - t0 )) s; $(grep -oE 'stages: .*' logs/fh_vkdense.log | tail -1)"
venv/bin/python - <<'PY'
import array, math
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
for s in ('v', 'a'):
    print(f'{s} step 0: dense vulkan vs cuda {rel(load(f"out/fh_vkdense/d.{s}0.f32"), load(f"out/fh_dense/d.{s}0.f32")):.4f}'
          f'   s0 vulkan vs cuda {rel(load(f"out/fh_s0vk/d.{s}0.f32"), load(f"out/fh_s0cuda/d.{s}0.f32")):.4f}')
PY
echo VK_DENSE_DONE
