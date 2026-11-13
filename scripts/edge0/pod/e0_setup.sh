#!/bin/bash
# pod setup for the Edge0 -> CMF campaign; log to /root/e0/setup.log
set -x
mkdir -p /root/e0/src /root/e0/out /root/e0/logs
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq --no-install-recommends build-essential pkg-config git curl ca-certificates python3-pip python3-venv \
  vulkan-tools libglvnd0 libegl1 libgl1 libglx0 libvulkan1 zstd htop >/dev/null
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName" | head -4
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable >/dev/null 2>&1
. /root/.cargo/env; rustc --version
python3 -m venv /root/venv && /root/venv/bin/pip install -q -U "huggingface_hub[hf_xet]" && /root/venv/bin/hf version
echo SETUP-DONE
