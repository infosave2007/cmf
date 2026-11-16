#!/bin/bash
# MiMo-V2.6-Flash-RL q4tp (164 GB) on Vulkan, never measured (its file was gone
# from the pod): the experts stream through card slots over the RAM tier
# (past the card and the 46 GB container). Room first: Ornith q2tp-v2 and
# Flash-Next go (their matrices are measured). Texts on Vulkan and CUDA, decode
# after 512 on the whole card and within 24 GB (CUDA 13.6 tok/s on the whole
# card in 0.8.99's campaign). After cstage4.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mimo2 out/mimo2
end=$((SECONDS + 14400))
until grep -qE "CSTAGE4_DONE|CSTAGE4_BUILD_FAILED" logs/cstage4.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
rm -f cmf/big/ornith-1.5-122b-q2tp-v2.cmf cmf/big/qwen38-flash-next-fast-code.cmf
df -h /root/h3i | tail -1
f=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
[ -s $f ] || timeout 10800 venv/bin/hf download infosave/MiMo-V2.6-Flash-RL-cmf MiMo-V2.6-Flash-RL-q4tp.cmf --local-dir cmf/big > logs/mimo2/dl.log 2>&1
ls -la $f || { echo MIMO2_DOWNLOAD_FAILED; tail -5 logs/mimo2/dl.log; exit 1; }
B=bin/cortiq-fast
for be in 1 cuda; do
  flock gpu.lock env CMF_GPU=$be RUST_LOG=info timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." -n 24 --greedy > out/mimo2/two_$be.txt 2> logs/mimo2/two_$be.err
  echo "$be: 2+2 $(grep -v '^\[' out/mimo2/two_$be.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
  echo "   $(sed -E 's/\x1b\[[0-9;]*m//g' logs/mimo2/two_$be.err | grep -iE 'routed experts|expert stream|declin|refus|WARN' | cut -c30- | sort | uniq -c | head -6 | tr '\n' ';' | cut -c1-600)"
done
run() { # tag backend capMB(0 = the card)
  local tag=$1 be=$2 cap=$3
  local l=logs/mimo2/${tag}_${be}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=$be $x RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$tag $be ${cap}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
}
run mimo 1 0
run mimo 1 24576
run mimo cuda 0
echo MIMO2_DONE
