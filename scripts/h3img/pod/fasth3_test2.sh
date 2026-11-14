#!/bin/bash
# FastH3 round 2: the release build with CMF_VSA_SPARSITY, the kernels' parity
# with every tile kept (CUDA chain vs host reference, step 0), the default size
# (768x1344, 124 frames) with VSA and dense, then LFM3's long prompt on the
# CUDA prompt path (greedy text against the CPU, the 4096-token bench).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/fh2_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/fh2_build.log; exit 1; }
echo "build ok"
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
run() { # tag W H F env...
  local tag=$1 W=$2 H=$3 F=$4; shift 4
  mkdir -p out/fh_$tag
  local t0=$(date +%s)
  flock gpu.lock env CMF_GPU=cuda CMF_ANIM_DUMP=out/fh_$tag/d "$@" timeout 7200 bin/cortiq animate $m --prompt "$prompt" \
    --width $W --height $H --frames $F --steps 4 --seed 7 --out out/fh_$tag/clip.avi --frames-dir out/fh_$tag/frames > logs/fh_$tag.log 2>&1
  echo "$tag: exit $? in $(( $(date +%s) - t0 )) s; $(grep -oE 'stages: .*' logs/fh_$tag.log | tail -1)"
}
run s0cuda 512 288 39 CMF_VSA_SPARSITY=0
run s0host 512 288 39 CMF_VSA_SPARSITY=0 CMF_MMH3_CHAIN=0
venv/bin/python - <<'EOF'
import array, math
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
for s in ('v', 'a'):
    print(f'sparsity 0, {s} step 0: cuda vs host {rel(load(f"out/fh_s0cuda/d.{s}0.f32"), load(f"out/fh_s0host/d.{s}0.f32")):.4f}')
EOF
run big 1344 768 124
run bigdense 1344 768 124 CMF_VSA=0
# LFM3-1.7B: the prompt path now takes the short convolution and the head gate
f=cmf/dl/lfm3-1.7b-q4tp.cmf
mkdir -p cmf/dl
[ -s $f ] || timeout 1800 venv/bin/hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q4tp.cmf --local-dir cmf/dl > logs/fh2_dl_lfm3.log 2>&1
long="$(head -c 2400 srcrs/docs/ppl_nat.txt) Summarize the text above in three sentences."
for be in cuda 0; do
  flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq run $f --prompt "$long" -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/lfm3_long_$be.txt
done
cmp -s out/lfm3_long_cuda.txt out/lfm3_long_0.txt && echo "lfm3 long prompt: CUDA text == CPU" || { echo "lfm3 long prompt: CUDA text DIFFERS"; tail -3 out/lfm3_long_cuda.txt; tail -3 out/lfm3_long_0.txt; }
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
for be in cuda 1; do
  flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq bench $f --tokens 32 --ctx 4096 --json > logs/fh2_lfm3_$be.log 2>&1
  echo "lfm3 4k $be: $(grep -oE "$keys" logs/fh2_lfm3_$be.log | tr '\n' ' ')"
done
echo FH_TEST2_DONE
