#!/bin/bash
# setup.sh: tools, cortiq build, source models for the Fast Code campaign -> /root/fc
set -uo pipefail
mkdir -p /root/fc/{fn,logs,data,out,hot} && cd /root/fc
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 build-essential pkg-config libssl-dev python3-pip python3-venv git curl ca-certificates zstd jq >/dev/null 2>&1
echo "APT_DONE rc=$?"
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary 2>/dev/null | grep -i "deviceName" | head -2
[ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
. "$HOME/.cargo/env"; cargo --version
python3 -m venv /root/fc/venv && /root/fc/venv/bin/pip install -q -U "huggingface_hub[cli]" hf_xet >/dev/null 2>&1
/root/fc/venv/bin/hf version 2>/dev/null || /root/fc/venv/bin/huggingface-cli version
echo TOOLS_DONE
# source models: q4tp Flash-Next (the kept experts come from here) + profile + MTP head
( cd /root/fc/fn && /root/fc/venv/bin/hf download infosave/Qwen3.8-Flash-Next-cmf qwen38-flash-next-q4tp.cmf qwen38-flash-next-q4tp.mtp.cmf flashnext-q4tp.profile flashnext.profile --local-dir . > /root/fc/logs/dl_q4tp.log 2>&1; echo "DL_Q4TP_DONE rc=$?" >> /root/fc/logs/dl_q4tp.log ) &
git clone -q https://github.com/infosave2007/cmf.git /root/fc/cmf && cd /root/fc/cmf && git log --oneline -1
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 10 cargo build --release -p cortiq-cli --features gpu 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -20
cp target/release/cortiq /root/fc/cortiq.m && /root/fc/cortiq.m --version
echo BUILD_DONE
wait
echo SETUP_DONE
