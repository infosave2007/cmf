#!/bin/bash
# build.sh: the synced tree (rsync of crates/ and Cargo.* into /root/rip/cmf) as ./cortiq, no LTO.
set -uo pipefail
cd /root/rip/cmf && . "$HOME/.cargo/env"
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -30
cp target/release/cortiq /root/rip/cortiq.new && mv /root/rip/cortiq.new /root/rip/cortiq && /root/rip/cortiq --version
echo BUILD_DONE
