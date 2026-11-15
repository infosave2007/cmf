#!/bin/bash
# The CUDA spill rule: stream past an eighth of the experts. Ornith-122B q4tp
# within 32 GB and on the whole card (was pinned and read over the bus:
# 11.4 / 15.4 tok/s), Ornith q2tp-v2 (48 GB: a small spill stays pinned, 38.7
# in 0.8.99) on the whole card and within 32 GB, decode after 512. After sweep3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/sweep4
end=$((SECONDS + 14400))
until grep -qE "SWEEP3_DONE" logs/sweep3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
q2=cmf/big/ornith-1.5-122b-q2tp-v2.cmf
( [ -s $q2 ] || timeout 3600 venv/bin/hf download infosave/Ornith-1.5-122B-cmf ornith-1.5-122b-q2tp-v2.cmf --local-dir cmf/big > logs/sweep4/dl.log 2>&1 ) &
dl=$!
bash scripts/build.sh fast > logs/sweep4/build.log 2>&1
grep -q BUILD_FAST_DONE logs/sweep4/build.log || { echo SWEEP4_BUILD_FAILED; tail -30 logs/sweep4/build.log; exit 1; }
wait $dl
B=bin/cortiq-fast
run() { # tag file capMB(0 = the card)
  local l=logs/sweep4/$1_$3.log x="CMF_GPU_VRAM_MB=$3"
  [ $3 = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=cuda $x CMF_CUDA_DEBUG=1 RUST_LOG=info timeout 2400 $B bench $2 --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$1 ${3}MB: exit $? | $(grep -hoE '"(decode_tok_s_steady|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')| $(sed -E 's/\x1b\[[0-9;]*m//g' $l | grep -oE 'cuda hybrid: [0-9.]+ GB of a plan.s[^|;]*|expert stream: [0-9]+ slots.*' | sort -u | tail -2 | tr '\n' ';' | cut -c1-300)"
}
run orn4 cmf/big/ornith-1.5-122b-q4tp.cmf 32768
run orn4 cmf/big/ornith-1.5-122b-q4tp.cmf 0
[ -s $q2 ] && { run orn2 $q2 0; run orn2 $q2 32768; }
for cap in 24576 32768 0; do run fnx cmf/big/qwen38-flash-next-fast-code.cmf $cap; done
echo SWEEP4_DONE
