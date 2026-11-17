#!/bin/bash
# Where Vulkan's extra error at sparsity 0 sits: without the compression branch
# (CMF_VSA_GATE=0) Vulkan against CUDA. ~0.8 % (the dense pair) = the branch,
# ~2.2 % = the sparse flash.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/fh_gate_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/fh_gate_build.log; exit 1; }
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
for be in cuda 1; do
  mkdir -p out/fh_ng_$be
  flock gpu.lock env CMF_GPU=$be CMF_VSA_SPARSITY=0 CMF_VSA_GATE=0 CMF_ANIM_DUMP=out/fh_ng_$be/d timeout 3600 bin/cortiq animate $m \
    --prompt "$prompt" --width 512 --height 288 --frames 39 --steps 4 --seed 7 --out out/fh_ng_$be/clip.avi > logs/fh_ng_$be.log 2>&1
  echo "no gate $be: exit $?; $(grep -oE 'stages: .*' logs/fh_ng_$be.log | tail -1)"
  grep -iE "declin" logs/fh_ng_$be.log | sort -u | head -2
done
venv/bin/python - <<'PY'
import array, math
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
for s in ('v', 'a'):
    ng = rel(load(f'out/fh_ng_1/d.{s}0.f32'), load(f'out/fh_ng_cuda/d.{s}0.f32'))
    br = rel(load(f'out/fh_ng_cuda/d.{s}0.f32'), load(f'out/fh_s0cuda/d.{s}0.f32'))
    print(f'{s} step 0: no-gate vulkan vs cuda {ng:.4f}   (the branch moves cuda by {br:.4f})')
PY
echo VK_GATE_DONE
