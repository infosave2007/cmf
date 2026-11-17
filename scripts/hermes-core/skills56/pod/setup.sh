#!/bin/bash
# Toolchain for the Embryo trainer on this pod: build tools, the Vulkan userspace, Rust.
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential pkg-config clang cmake curl ca-certificates libssl-dev vulkan-tools libvulkan1 libglvnd0 libegl1 libgl1 libglx0 > /root/sk/setup.apt.log 2>&1
echo "APT_EXIT $?"
[ -x /root/.cargo/bin/cargo ] || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /root/sk/setup.rust.log 2>&1
echo "RUST_EXIT $? $(/root/.cargo/bin/rustc --version 2>&1)"
XDG_RUNTIME_DIR=/tmp vulkaninfo --summary 2>&1 | grep -E "deviceName|apiVersion|ERROR" | head -4
echo SETUP_DONE
