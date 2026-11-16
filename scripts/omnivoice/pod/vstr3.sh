#!/bin/bash
# The Vulkan expert stream past its warm-up: decode 256 tokens after 512
# (the 64-token bench started from cold slots: the prompt's staging bank
# fills none), with the stream's counters (CMF_STREAM_STATS=1), against the
# banks in place (CMF_WGPU_EXPERT_STREAM=0), Ornith q4tp within 24 GB. After
# csmall1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vstr3
end=$((SECONDS + 14400))
until grep -qE "CSMALL1_DONE" logs/csmall1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast
M=cmf/big/ornith-1.5-122b-q4tp.cmf
for v in str inplace; do
  case $v in str) e="CMF_STREAM_STATS=1";; inplace) e="CMF_WGPU_EXPERT_STREAM=0";; esac
  l=logs/vstr3/${v}.log
  flock gpu.lock env CMF_GPU=1 CMF_GPU_VRAM_MB=24576 $e RUST_LOG=info timeout 3600 $B bench $M --tokens 256 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$v 24576MB 512+256: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(grep -E 'wgpu expert stream:' $l | tail -2 | tr '\n' ';' | cut -c1-300)"
done
echo VSTR3_DONE
