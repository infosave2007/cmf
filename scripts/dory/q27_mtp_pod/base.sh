#!/bin/bash
# pod base: Vulkan userspace, Rust, hf; the model and the sources meanwhile; a fast build (no LTO)
set -uo pipefail
cd /root/q27
apt-get update -qq
apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 git curl build-essential pkg-config zstd python3-pip > /dev/null
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|apiVersion" | head -4
pip install -q -U --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | grep -v WARNING | tail -1
(setsid nohup hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir /root/q27/m > hf-model.log 2>&1 < /dev/null &)
[ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
. "$HOME/.cargo/env"; cargo --version
[ -d cmf ] || git clone -q --depth 1 --branch v0.8.58 https://github.com/infosave2007/cmf.git cmf 2>/dev/null
tar xzmf overlay.tgz -C cmf 2>/dev/null
echo BASE_DONE
cd cmf
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A6 | tail -30
cp target/release/cortiq /root/q27/cortiq && /root/q27/cortiq --version && echo BUILD_DONE
