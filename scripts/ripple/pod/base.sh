#!/bin/bash
# base.sh [BRANCH]: a card for Ripple-65B. Vulkan userspace, Rust, the BF16 source from its public page
# (130 GB, in the background, log hf-src.log), and BRANCH (default: master) built as ./cortiq with no LTO.
# Run it again with a branch once that branch is pushed; what is in place is kept.
set -uo pipefail
mkdir -p /root/rip && cd /root/rip
br=${1:-master}
export XDG_RUNTIME_DIR=/tmp
if [ ! -f base.ok ]; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 git curl build-essential pkg-config zstd python3-pip > /dev/null
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|apiVersion" | head -4
  pip install -q -U --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | grep -v WARNING | tail -1
  (setsid nohup hf download ricursiveAI/Ripple-65B-Preview --local-dir /root/rip/src > hf-src.log 2>&1 < /dev/null &)
  [ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
  touch base.ok
fi
. "$HOME/.cargo/env"; cargo --version
[ -d cmf ] || git clone -q --depth 1 --branch "$br" https://github.com/infosave2007/cmf.git cmf || exit 1
(cd cmf && git fetch -q --depth 1 origin "$br" && git checkout -q FETCH_HEAD && git log --oneline -1 | cut -c1-100 &&
  CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A6 | tail -20) &&
  rm -f cortiq && cp cmf/target/release/cortiq cortiq && ./cortiq --version
echo BASE_DONE
