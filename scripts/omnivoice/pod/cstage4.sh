#!/bin/bash
# CUDA's staged prompt, third form: each staged layer keeps its most picked
# experts' share of the slots, the rest give way first (cstage3 within 24 GB:
# decode 20.1 tok/s over 256 after the staged prompt, 34.6 after the small
# path). Ornith q4tp within 24 GB (64 and 256 tokens), 12 / 16 GB and the whole
# card (64), TTFT and decode after 512.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cstage4
bash scripts/build.sh fast > logs/cstage4/build.log 2>&1
grep -q BUILD_FAST_DONE logs/cstage4/build.log || { echo CSTAGE4_BUILD_FAILED; grep -E '^error|error:' -A8 logs/cstage4/build.log | head -60; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB(0 = the card) tokens
  local tag=$1 cap=$2 n=$3
  local l=logs/cstage4/${tag}_${cap}_$n.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B bench $M --tokens $n --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB 512+$n: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(grep -E 'expert stream:' $l | tail -1 | cut -c1-160)"
}
run stage 24576 256
run stage 24576 64
run stage 12288 64
run stage 16384 64
run stage 0 64
echo CSTAGE4_DONE
