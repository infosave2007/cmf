#!/bin/bash
# setup.sh: toolchains and builds for the CPU comparison on the 8-core test server (Ubuntu 24.04):
# build tools, rustup, cortiq (CPU build), llama.cpp b12202 (native CPU build), a venv with
# huggingface_hub. Logs in /root/spark26cpu/logs, marker logs/setup.done.
cd /root/spark26cpu; mkdir -p logs
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > logs/apt.log 2>&1; apt-get install -y -qq build-essential cmake pkg-config libssl-dev python3-venv python3-pip curl git >> logs/apt.log 2>&1; echo "apt $?" >> logs/setup.log
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > logs/rustup.log 2>&1; echo "rustup $?" >> logs/setup.log
. "$HOME/.cargo/env"
cd /root/spark26cpu/cmf && cargo build --release -p cortiq-cli > /root/spark26cpu/logs/build_cortiq.log 2>&1; st=$?; echo "cortiq build $st" >> /root/spark26cpu/logs/setup.log
[ $st -eq 0 ] && cp target/release/cortiq /root/spark26cpu/cortiq
cd /root/spark26cpu && git clone --depth 1 --branch b12202 https://github.com/ggml-org/llama.cpp.git > logs/llama_clone.log 2>&1 && cd llama.cpp && cmake -B build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF > ../logs/llama_cmake.log 2>&1 && cmake --build build --config Release -j 8 --target llama-bench llama-cli llama-perplexity > ../logs/llama_build.log 2>&1; echo "llama build $?" >> /root/spark26cpu/logs/setup.log
python3 -m venv /root/spark26cpu/venv && /root/spark26cpu/venv/bin/pip install -q huggingface_hub > /root/spark26cpu/logs/pip.log 2>&1; echo "venv $?" >> /root/spark26cpu/logs/setup.log
touch /root/spark26cpu/logs/setup.done
