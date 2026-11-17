#!/bin/bash
# FastH3 parity at a small size: the CUDA chain's VSA against the host
# reference (per-op path) and against dense attention, one seed. Dumps every
# step's velocities (CMF_ANIM_DUMP) and the frames; prints the relative
# differences of step 0. fasth3_test.sh [W H FRAMES]
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
W=${1:-512}; H=${2:-288}; F=${3:-39}
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
run() { # tag env...
  local tag=$1; shift
  mkdir -p out/fh_$tag
  local t0=$(date +%s)
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 CMF_ANIM_DUMP=out/fh_$tag/d "$@" timeout 5400 bin/cortiq animate $m --prompt "$prompt" \
    --width $W --height $H --frames $F --steps 4 --seed 7 --out out/fh_$tag/clip.avi --frames-dir out/fh_$tag/frames > logs/fh_$tag.log 2>&1
  echo "$tag: exit $? in $(( $(date +%s) - t0 )) s; $(grep -oE 'cuda h3 chain: [^,]*, [^,]*, [0-9]+ of [0-9]+ blocks resident|vsa: [^\n]*' logs/fh_$tag.log | sort -u | head -2 | tr '\n' ' ')"
  grep -E "denoise|vae|total|stage" logs/fh_$tag.log | tail -4
}
run cuda
run dense CMF_VSA=0
run host CMF_MMH3_CHAIN=0
venv/bin/python - <<'EOF'
import array, math, os
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    num = sum((x - y) ** 2 for x, y in zip(a, b)); den = sum(y * y for y in b)
    return math.sqrt(num / max(den, 1e-30))
for s in ('v', 'a'):
    for i in range(4):
        p = {t: f'out/fh_{t}/d.{s}{i}.f32' for t in ('cuda', 'host', 'dense')}
        if not all(os.path.exists(x) for x in p.values()):
            continue
        c, h, d = (load(p[t]) for t in ('cuda', 'host', 'dense'))
        print(f'{s} step {i}: cuda vs host {rel(c, h):.4f}   dense vs host {rel(d, h):.4f}   cuda vs dense {rel(c, d):.4f}')
EOF
echo FH_TEST_DONE
