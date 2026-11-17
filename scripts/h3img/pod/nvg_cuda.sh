#!/bin/bash
# NVG (blr) on the CUDA hybrid arm: after the fast build, Qwen3.8-27B-NVG on
# CUDA (bin/cortiq-fast) — decode, the 4k prompt, the decline reasons, the
# greedy answer against Vulkan's (probe4) — then the kernels' card test.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 3600 ))
until grep -q -E 'BUILD_FAST_DONE|BUILD_FAST_FAILED' logs/${NVG_BUILD_LOG:-build_blr.log} 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo BUILD_WAIT_TIMEOUT; exit 1; }
  sleep 20
done
grep -q BUILD_FAST_DONE logs/${NVG_BUILD_LOG:-build_blr.log} || { echo BUILD_FAILED; tail -40 logs/${NVG_BUILD_LOG:-build_blr.log}; exit 1; }
echo "build ok"
mkdir -p cmf/dl logs/nvg out/nvg
f=cmf/dl/qwen38-27b-nvg055-10m.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf qwen38-27b-nvg055-10m.cmf --local-dir cmf/dl > logs/nvg/dl.log 2>&1
[ -s $f ] || { echo DOWNLOAD_FAILED; exit 1; }
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
b=bin/cortiq-fast
for i in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 1200 $b bench $f --tokens 64 --ignore-eos --json > logs/nvg/d_$i.log 2>&1
  echo "cuda decode run$i: exit $?; $(grep -oE "$keys" logs/nvg/d_$i.log | tr '\n' ' ')"
done
grep -h -E "cuda decode:|declin|refused" logs/nvg/d_2.log | sed -E 's/\x1b\[[0-9;]*m//g' | sort | uniq -c | head -8
flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 1800 $b bench $f --tokens 32 --ctx 4096 --json > logs/nvg/4k.log 2>&1
echo "cuda 4k: exit $?; $(grep -oE "$keys" logs/nvg/4k.log | tr '\n' ' ')"
grep -h -E "cuda prompt:|declin" logs/nvg/4k.log | sed -E 's/\x1b\[[0-9;]*m//g' | sort | uniq -c | head -6
flock gpu.lock env CMF_GPU=cuda timeout 1200 $b run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/nvg/cuda.txt
if cmp -s out/nvg/cuda.txt out/p4/qwen38-27b-nvg055-10m_1.txt; then echo "CUDA text == Vulkan"; else
  echo "CUDA text differs from Vulkan"; tail -3 out/nvg/cuda.txt; echo ---; tail -3 out/p4/qwen38-27b-nvg055-10m_1.txt; fi
rm -f $f
# the stage kernels against the host (same flags as the fast build: its dependencies are reused)
. "$HOME/.cargo/env"
cd srcrs || exit 1
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true
timeout 2400 cargo test --release -p cortiq-engine --features cuda --target-dir target-fast --lib --no-run 2>&1 | grep -E "^error|Finished" -A8 | tail -12
flock ../gpu.lock timeout 900 cargo test --release -p cortiq-engine --features cuda --target-dir target-fast --lib -- card_blr card_q1_tiles 2>&1 | grep -E "^test |test result|panicked|error" | head -20
echo NVG_DONE
