#!/bin/bash
# Short runs after the long probe: decode only (32 tokens) for the models past
# the pod's card + RAM (GLM-5.3 Vulkan; MiniMax-M3.1, MiMo-V2.6 both), then
# Ripple's 4k prompt on CUDA under CMF_CUDA_PROF=1, then Edge0-120B on Vulkan
# with a backtrace of its out-of-memory panic. One file on disk at a time.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
B=bin/cortiq-fast
mkdir -p cmf/big logs/q4 out/q4
dec() { # file backend tag
  flock gpu.lock env CMF_GPU=$2 CMF_CUDA_DEBUG=1 timeout 1800 $B bench cmf/big/$1 --tokens 32 --ignore-eos --json > logs/q4/$3.log 2>&1
  echo "$3: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/q4/$3.log | tr '\n' ' ')"
  sed -E 's/\x1b\[[0-9;]*m//g' logs/q4/$3.log | grep -iE 'declin|refus|mapped host|panick|wgpu error|Out of Memory' | sed -E 's/^.*(INFO|WARN|ERROR) //' | sort | uniq -c | head -4
}
get() { [ -s cmf/big/$2 ] || timeout 7200 venv/bin/hf download $1 $2 --local-dir cmf/big > logs/q4/${2%.cmf}_dl.log 2>&1; }
dec glm-5.3-flash-q2tp.cmf 1 glm_vk
rm -f cmf/big/glm-5.3-flash-q2tp.cmf
dec minimax-m3.1-flash-q2tp-mix.cmf cuda mm_cu
dec minimax-m3.1-flash-q2tp-mix.cmf 1 mm_vk
rm -f cmf/big/minimax-m3.1-flash-q2tp-mix.cmf
get infosave/MiMo-V2.6-Flash-RL-cmf MiMo-V2.6-Flash-RL-q4tp.cmf
dec MiMo-V2.6-Flash-RL-q4tp.cmf cuda mimo_cu
dec MiMo-V2.6-Flash-RL-q4tp.cmf 1 mimo_vk
rm -f cmf/big/MiMo-V2.6-Flash-RL-q4tp.cmf
get infosave/Ripple-65B-Preview-cmf ripple-65b-q4tp.cmf
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_PROF=1 timeout 1800 $B bench cmf/big/ripple-65b-q4tp.cmf --tokens 32 --ctx 4096 --json > logs/q4/rip_4k_prof.log 2>&1
echo "ripple 4k prof: exit $? | $(grep -hoE '"(prefill_tok_s|ttft_s)": [0-9.e+-]+' logs/q4/rip_4k_prof.log | tr '\n' ' ')"
get infosave/Edge0-120B-cmf edge0-120b.cmf
flock gpu.lock env CMF_GPU=1 RUST_BACKTRACE=1 CMF_GPU_DEBUG=1 timeout 1800 $B bench cmf/big/edge0-120b.cmf --tokens 32 --ignore-eos --json > logs/q4/edge0_vk_bt.log 2>&1
echo "edge0 vk: exit $?"
grep -nE 'panicked|wgpu error|Out of Memory|^ +[0-9]+: ' logs/q4/edge0_vk_bt.log | head -40
echo Q4_DONE
