#!/bin/bash
# Toolchain of the B200 pod: build tools, Rust, a venv with huggingface_hub (hf_get.py).
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential pkg-config clang cmake curl ca-certificates libssl-dev rsync python3-pip python3-venv > /root/sk/setup.apt.log 2>&1
echo "APT_EXIT $?"
[ -x /root/.cargo/bin/cargo ] || curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /root/sk/setup.rust.log 2>&1
echo "RUST_EXIT $? $(/root/.cargo/bin/rustc --version 2>&1)"
python3 -m venv /root/venv > /root/sk/venv.log 2>&1 && /root/venv/bin/pip install -q huggingface_hub hf_xet >> /root/sk/venv.log 2>&1
echo "VENV_EXIT $?"
echo SETUP_DONE
