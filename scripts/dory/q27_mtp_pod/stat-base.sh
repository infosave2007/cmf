#!/bin/bash
# stat-base.sh [BRANCH]: a card for statistics. Vulkan userspace, Rust, the Turbo file from its public page,
# the release built as cortiq-old and BRANCH (default: ane-ffn) as cortiq, both with no LTO.
# Run it again with a branch once that branch is pushed; what is in place is kept.
set -uo pipefail
mkdir -p /root/q27 && cd /root/q27
br=${1:-ane-ffn}
export XDG_RUNTIME_DIR=/tmp
if [ ! -f base.ok ]; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 git curl build-essential pkg-config zstd python3-pip > /dev/null
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|apiVersion" | head -4
  pip install -q -U --break-system-packages "huggingface_hub[hf_xet]" 2>&1 | grep -v WARNING | tail -1
  (setsid nohup hf download infosave/Qwen3.8-27B-Turbo-cmf qwen38-27b-turbo.cmf --local-dir /root/q27/m > hf-model.log 2>&1 < /dev/null &)
  [ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null 2>&1
  touch base.ok
fi
. "$HOME/.cargo/env"; cargo --version
build() { # DIR REF BINARY
  [ -d "$1" ] || git clone -q --depth 1 --branch "$2" https://github.com/infosave2007/cmf.git "$1" || return 1
  (cd "$1" && git fetch -q --depth 1 origin "$2" && git checkout -q FETCH_HEAD && git log --oneline -1 | cut -c1-100 &&
    CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A6 | tail -20) &&
    rm -f "$3" && cp "$1/target/release/cortiq" "$3" && "./$3" --version
}
[ -x cortiq-old ] || build rel v0.8.58 cortiq-old
build new "$br" cortiq
echo STAT_BASE_DONE
