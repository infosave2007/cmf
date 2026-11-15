#!/bin/bash
# 256-row prompt chunks for a MoE file past the card on wgpu whatever its
# expert codec (q2tp-v2 took 32: 43 staged banks copied every 32 tokens,
# 512 tokens in 30 s within 32 GB). Ornith q2tp-v2 within 32 GB and on the
# whole card, TTFT and decode after 512, then its perplexity within 32 GB
# against the CPU. After vfinal.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vq2
end=$((SECONDS + 7200))
until grep -q VFINAL_DONE logs/vfinal.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
bash scripts/build.sh fast > logs/vq2/build.log 2>&1
grep -q BUILD_FAST_DONE logs/vq2/build.log || { echo VQ2_BUILD_FAILED; grep -E '^error' -A8 logs/vq2/build.log | head -40; exit 1; }
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
for cap in 32768 0; do
  l=logs/vq2/q2_$cap.log x="CMF_GPU_VRAM_MB=$cap"; [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $x RUST_LOG=info timeout 2400 $B bench $M --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "q2 ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
done
for v in cpu vk; do
  case $v in cpu) e="CMF_GPU=0";; vk) e="CMF_GPU=1 CMF_GPU_VRAM_MB=32768";; esac
  flock gpu.lock env $e timeout 3600 $B ppl $M --file prompts/rail.txt --tokens 256 > logs/vq2/ppl_$v.log 2>&1
  echo "ppl $v: exit $? | $(grep -hiE 'ppl|perplexity' logs/vq2/ppl_$v.log | tail -1 | cut -c1-200)"
done
echo VQ2_DONE
