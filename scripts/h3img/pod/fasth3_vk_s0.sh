#!/bin/bash
# Vulkan VSA kernels against the host reference at sparsity 0 (every tile kept:
# the selection drops out, the pool / flash / compress / gate kernels remain).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
mkdir -p out/fh_s0vk
t0=$(date +%s)
flock gpu.lock env CMF_GPU=1 CMF_GPU_DEBUG=1 CMF_VSA_SPARSITY=0 CMF_ANIM_DUMP=out/fh_s0vk/d timeout 3600 bin/cortiq animate $m --prompt "$prompt" \
  --width 512 --height 288 --frames 39 --steps 4 --seed 7 --out out/fh_s0vk/clip.avi > logs/fh_s0vk.log 2>&1
echo "s0 vulkan: exit $? in $(( $(date +%s) - t0 )) s; $(grep -oE 'stages: .*' logs/fh_s0vk.log | tail -1)"
grep -iE "declin" logs/fh_s0vk.log | sort -u | head -3
venv/bin/python - <<'PY'
import array, math
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
for s in ('v', 'a'):
    print(f'sparsity 0, {s} step 0: vulkan vs host {rel(load(f"out/fh_s0vk/d.{s}0.f32"), load(f"out/fh_s0host/d.{s}0.f32")):.4f}'
          f'   cuda vs host {rel(load(f"out/fh_s0cuda/d.{s}0.f32"), load(f"out/fh_s0host/d.{s}0.f32")):.4f}')
PY
echo VK_S0_DONE
