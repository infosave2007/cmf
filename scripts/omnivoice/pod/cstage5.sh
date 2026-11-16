#!/bin/bash
# CUDA's staged prompt, fourth form: each staged layer keeps the slots of its
# last rows' experts (cstage4 kept the most picked: within 24 GB 26.8 / 19.0
# tok/s over 256 / 64, within 12 / 16 GB 3.8 / 4.6 against the small path's
# 5.6 / 7.5). Same runs, after mimo2.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cstage5
end=$((SECONDS + 21600))
until grep -qE "MIMO2_DONE|MIMO2_DOWNLOAD_FAILED" logs/mimo2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/cstage5/build.log 2>&1
grep -q BUILD_FAST_DONE logs/cstage5/build.log || { echo CSTAGE5_BUILD_FAILED; grep -E '^error|error:' -A8 logs/cstage5/build.log | head -60; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB(0 = the card) tokens
  local tag=$1 cap=$2 n=$3
  local l=logs/cstage5/${tag}_${cap}_$n.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 3600 $B bench $M --tokens $n --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB 512+$n: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(grep -E 'expert stream:' $l | tail -1 | cut -c1-160)"
}
run stage 24576 256
run stage 24576 64
run stage 12288 64
run stage 16384 64
run stage 0 64
echo CSTAGE5_DONE
