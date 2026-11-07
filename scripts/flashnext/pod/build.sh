#!/bin/bash
cd /root/rip/cmf && . "$HOME/.cargo/env"
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 10 cargo build --release -p cortiq-cli --features gpu 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -30
cp target/release/cortiq /root/rip/cortiq.ft4 && /root/rip/cortiq.ft4 --version
echo BUILD_DONE
