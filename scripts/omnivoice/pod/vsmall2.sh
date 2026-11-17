#!/bin/bash
# Vulkan within 12 / 16 / 24 GB on a host short of RAM (Ornith q4tp 68 GB on
# the 46 GB container): the plan keeps 15 / 8 / 5 layers whole on the card
# and reads 33–43 banks over the bus with no card slots (7.9 / 8.3 / 9.6
# tok/s; CUDA within 24 GB 26.8 with 3246 slots). Against it: every bank in
# system memory and the card's expert bytes as LRU slots
# (CMF_WGPU_EXPERT_OUT=all), and within 24 GB the stream with the RAM tier
# now filled on several threads (CMF_WGPU_EXPERT_STREAM=1; was 9.6, TTFT
# 84 s). Decode after 512 over 256. Builds first. After cstage5.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vsmall2
end=$((SECONDS + 21600))
until grep -qE "CSTAGE5_DONE|CSTAGE5_BUILD_FAILED" logs/cstage5.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
# rebuilt: the RAM tier keeps each staged layer's share (CUDA and Vulkan)
bash scripts/build.sh fast > logs/vsmall2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vsmall2/build.log || { echo VSMALL2_STALE_BINARY; grep -E '^error|error:' -A8 logs/vsmall2/build.log | head -40; exit 1; }
B=bin/cortiq-fast; f=cmf/big/ornith-1.5-122b-q4tp.cmf
strings $B | grep -q CMF_WGPU_EXPERT_OUT || { echo VSMALL2_STALE_BINARY; exit 1; }
run() { # tag capMB env
  local tag=$1 cap=$2 x=$3
  local l=logs/vsmall2/${tag}_$cap.log
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=$cap $x CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 256 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB 512+256: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -hoE 'routed experts: .*|wgpu expert stream: .*' | tail -1 | cut -c1-170)"
}
for cap in 24576 16384 12288; do
  run allout $cap CMF_WGPU_EXPERT_OUT=all
  run inplace $cap CMF_X=1
done
run stream 24576 CMF_WGPU_EXPERT_STREAM=1
# the streamed text with seeded pools against the in-place one (same tokens)
P="Explain in two sentences why the sky is blue."
for x in CMF_WGPU_EXPERT_STREAM=1 CMF_X=1; do
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=24576 $x timeout 3600 $B run $f --prompt "$P" -n 24 --greedy 2>/dev/null | grep -v '^\[' > logs/vsmall2/text_${x%%=*}.txt
done
echo "text: $(tail -n +5 logs/vsmall2/text_CMF_X.txt | tr '\n' ' ' | cut -c1-160)"
if cmp -s logs/vsmall2/text_CMF_X.txt logs/vsmall2/text_CMF_WGPU_EXPERT_STREAM.txt; then echo "   stream == in place"; else echo "   stream DIFFERS from in place"; fi
run stream_cold 24576 "CMF_WGPU_EXPERT_STREAM=1 CMF_WGPU_EXPERT_WARM=0"
echo VSMALL2_DONE
