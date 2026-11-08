#!/bin/bash
# OrcaCyber-Nano pod setup: Vulkan (GLVND), rustup, cmf master, release build. setup.sh LOG
L=$1; : > $L
export DEBIAN_FRONTEND=noninteractive XDG_RUNTIME_DIR=/tmp
mkdir -p /root/o && cd /root/o
(apt-get update -qq && apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 libvulkan1 build-essential pkg-config git curl ca-certificates python3-pip libssl-dev) >> $L 2>&1
echo "vulkan: $(vulkaninfo --summary 2>/dev/null | grep -m1 deviceName)" >> $L
[ -x ~/.cargo/bin/cargo ] || (curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal) >> $L 2>&1
. ~/.cargo/env
[ -d src ] || git clone -q https://github.com/infosave2007/cmf.git src >> $L 2>&1
cd src && git fetch -q origin && git checkout -q origin/master && echo "src: $(git log --oneline -1)" >> $L
t0=$(date +%s); cargo build --release -p cortiq-cli >> /root/o/build.log 2>&1; echo "build rc=$? $(( $(date +%s)-t0 )) s" >> $L
./target/release/cortiq --version >> $L 2>&1
pip3 install -q --break-system-packages huggingface_hub hf_transfer >> $L 2>&1; echo "hf: $(python3 -c 'import huggingface_hub;print(huggingface_hub.__version__)')" >> $L
echo SETUP_DONE >> $L
