#!/bin/bash
# chain4.sh: rebuild cortiq, then the 0.8.43 release tree, and the 2k/16k prompt A/B of both (Vulkan),
# with CMF_PREFILL_PROF splits; llama.cpp CUDA bench once its build is done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
bash build.sh > logs/build4.out 2>&1
[ -d old0843 ] || git clone -q --depth 1 --branch v0.8.43 https://github.com/infosave2007/cmf old0843
( cd old0843 && . "$HOME/.cargo/env" && CARGO_TARGET_DIR=/root/spark26/target0843 CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli --features gpu > /root/spark26/logs/build0843.log 2>&1 )
for b in /root/spark26/target0843/release/cortiq ./cortiq; do for n in 2000 16000; do echo "== $b $n"; CMF_PREFILL_PROF=1 $b bench hf/cmf/spark-x2.6-4b-q4tp.cmf --core --ignore-eos --tokens 32 --ctx $n 2>&1 | grep -E "Prompt|Decode|prefill-split" | tail -3; done; done > logs/ab0843.txt 2>&1
until grep -q CUDA_DONE logs/llama_cuda.log; do sleep 20; done
for q in Q4_K_M Q8_0; do llama.cpp/build-cuda/bin/llama-bench -m hf/gguf/Spark-X2.6-4B-$q.gguf -p 2000,16000 -n 0 -r 1 -o md 2>/dev/null | grep -E "pp|model"; llama.cpp/build-cuda/bin/llama-bench -m hf/gguf/Spark-X2.6-4B-$q.gguf -p 0 -n 128 -d 0,16000 -r 1 -o md 2>/dev/null | grep tg; done > logs/llama_cuda_bench.txt 2>&1
echo CHAIN4_DONE > logs/chain4.done
