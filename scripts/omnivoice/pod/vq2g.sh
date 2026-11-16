#!/bin/bash
# Vulkan prefill of the mixed q2tp profile (gate/up q2tp, down q4tp) on the
# expert-grouped arm (`moeg_gu_q2`: each expert read once for its tokens; the
# per-row loop issued ~7 commands a row and layer: whole card TTFT 7–8.4 s,
# 32 GB 9.5–9.8 s). Ornith q2tp-v2 again (MiniMax goes, measured in mm4.sh):
# perplexity (3.180), TTFT and decode after 512 on the whole card and within
# 32 GB, grouped against CMF_MOEG_Q2=0. Uses mm5.sh's build. After mm5.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/vq2g
end=$((SECONDS + 36000))
until grep -qE "MM5_DONE|MM5_BUILD_FAILED" logs/mm5.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
rm -f cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
f=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
[ -s $f ] || timeout 7200 venv/bin/hf download infosave/Ornith-1.5-122B-cmf ornith-1.5-122b-q2tp-v2.cmf --local-dir cmf/big > logs/vq2g/dl.log 2>&1
ls -la $f || { echo VQ2G_DOWNLOAD_FAILED; tail -5 logs/vq2g/dl.log; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q CMF_MOEG_Q2 || { echo VQ2G_STALE_BINARY; exit 1; }
flock gpu.lock env CMF_GPU=1 RUST_LOG=info timeout 3600 $B ppl $f --file prompts/rail.txt --tokens 512 > logs/vq2g/ppl.log 2>&1
echo "ppl grouped (3.180): exit $? | $(grep -hiE 'ppl|perplexity' logs/vq2g/ppl.log | tail -1 | cut -c1-200)"
run() { # tag capMB(0 = the card) env
  local tag=$1 cap=$2 x=$3
  local l=logs/vq2g/${tag}_$cap.log c="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && c="CMF_X=1"
  flock gpu.lock env CMF_GPU=1 $c $x RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag ${cap}MB 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
for cap in 0 32768; do
  run grouped $cap CMF_X=1
  run rows $cap CMF_MOEG_Q2=0
  run grouped2 $cap CMF_X=1
done
echo VQ2G_DONE
