#!/bin/bash
# CUDA: a call's missing experts copied into the RAM tier on several host
# threads before the builds (a build from the mapping took 3.8 ms, 5096 of
# them most of the 19-20 s first token of a 512-token prompt). Ornith q4tp
# within 24 / 32 GB and the whole card, the threads on (default) and one
# (CMF_HYB_RAM_FILL_THREADS=1), TTFT and decode after 512. After vq2.sh
# (whose build carries the change).
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/cfill
end=$((SECONDS + 10800))
until grep -qE "VQ2_DONE|VQ2_BUILD_FAILED" logs/vq2.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
grep -q VQ2_BUILD_FAILED logs/vq2.out && { echo CFILL_SKIPPED; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
run() { # tag capMB(0 = the card) env...
  local tag=$1 cap=$2; shift 2
  local l=logs/cfill/${tag}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x CMF_CUDA_DEBUG=1 "$@" RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'expert stream: [0-9]+ slots.*' | head -1 | cut -c1-200)"
}
for cap in 32768 24576 0; do
  run par $cap
  run one $cap CMF_HYB_RAM_FILL_THREADS=1
done
echo CFILL_DONE
