#!/bin/bash
# MiniMax-M3.1 (426B, 157 GB q2tp-mix) again after the expert work: CUDA's
# RAM tier with parallel fills and staged prompt picks, Vulkan's stream
# (pools + RAM tier). MiMo goes for room (measured in mimo2.sh). Greedy texts
# against the CPU's (out/mm3/cpu.txt), decode after 512 on the whole card and
# within 24 GB on both backends. After glmv3.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm4 out/mm4
end=$((SECONDS + 28800))
until grep -qE "GLMV3_DONE|GLMV3_STALE_BINARY" logs/glmv3.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
# first, rebuilt (CUDA staging marks the file pages it no longer needs cold):
# Ornith q4tp's CUDA perplexity within 12 GB with the packed RAM tier (CPU
# 3.049 over 256), then MiMo on Vulkan through the generic graph with banks
# in system memory (CMF_MIMO_MOE=prefix; its own dynamic bank: 1.9 tok/s, TTFT
# 189 s) and MiMo's CUDA staged prompt again (6.7 tok/s, TTFT 20 s; the small
# path 10.8 and 54 s) — then MiMo goes
bash scripts/build.sh fast > logs/mm4/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm4/build.log || { echo MM4_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm4/build.log | head -40; exit 1; }
B=bin/cortiq-fast
o=cmf/big/ornith-1.5-122b-q4tp.cmf
flock gpu.lock env CMF_GPU=cuda CMF_GPU_VRAM_MB=12288 RUST_LOG=info timeout 3600 $B ppl $o --file prompts/rail.txt --tokens 256 > logs/mm4/orn_ppl12.log 2>&1
echo "ornith cuda 12288MB packed ppl (CPU 3.049): exit $? | $(grep -hiE 'ppl|perplexity' logs/mm4/orn_ppl12.log | tail -1 | cut -c1-120)"
mi=cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
flock gpu.lock env CMF_GPU=1 CMF_MIMO_MOE=prefix RUST_LOG=info timeout 3600 $B run $mi --prompt "What is 2+2? Answer briefly." -n 24 --greedy > out/mm4/mimo_vk.txt 2> logs/mm4/mimo_vk.err
echo "mimo vulkan prefix 2+2: $(grep -v '^\[' out/mm4/mimo_vk.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
sed -E 's/\x1b\[[0-9;]*m//g' logs/mm4/mimo_vk.err | grep -hoE 'routed experts: .*|MiMo MoE placement: [a-z]+|system memory (falls|counts)[^(]*|declin.*' | sort -u | cut -c1-200 | sed 's/^/   /'
for x in "CMF_GPU=1 CMF_MIMO_MOE=prefix" "CMF_GPU=cuda CMF_HYB_STREAM_STAGE=1"; do
  l=logs/mm4/mimo_$(echo $x | tr ' =' '__').log
  flock gpu.lock env $x RUST_LOG=info timeout 3600 $B bench $mi --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "mimo $x 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ')"
done
rm -f cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
m=cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
[ -s $m ] || timeout 14400 venv/bin/hf download infosave/MiniMax-M3.1-Flash-cmf minimax-m3.1-flash-q2tp-mix.cmf --local-dir cmf/big > logs/mm4/dl.log 2>&1
ls -la $m || { echo MM4_DOWNLOAD_FAILED; tail -5 logs/mm4/dl.log; exit 1; }
P="Explain in two sentences why the sky is blue."
echo "cpu text: $(tail -n +6 out/mm3/cpu.txt | tr '\n' ' ' | cut -c1-200)"
for be in 1 cuda; do
  flock gpu.lock env CMF_GPU=$be RUST_LOG=info timeout 3600 $B run $m --prompt "$P" -n 24 --greedy > out/mm4/t_$be.txt 2> logs/mm4/t_$be.err
  echo "$be text exit $?: $(grep -v '^\[' out/mm4/t_$be.txt | tail -n +6 | tr '\n' ' ' | cut -c1-200)"
  if cmp -s <(grep -v '^\[' out/mm4/t_$be.txt) out/mm3/cpu.txt; then echo "   == CPU"; else echo "   DIFFERS from CPU"; fi
  sed -E 's/\x1b\[[0-9;]*m//g' logs/mm4/t_$be.err | grep -iE 'routed experts|expert stream|declin|refus|WARN|panick' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | sort -rn | head -6 | cut -c1-230 | sed 's/^/      /'
done
run() { # backend capMB(0 = the card)
  local be=$1 cap=$2
  local l=logs/mm4/${be}_$cap.log x="CMF_GPU_VRAM_MB=$cap"
  [ $cap = 0 ] && x="CMF_X=1"
  flock gpu.lock env CMF_GPU=$be $x CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $m --tokens 64 --ctx 512 --ignore-eos --json > $l 2>&1
  echo "$be ${cap}MB 512+64: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(grep -hoE 'wgpu expert stream: .*' $l | tail -1 | cut -c1-120)"
}
run cuda 0
run 1 0
run cuda 24576
run 1 24576
echo MM4_DONE
