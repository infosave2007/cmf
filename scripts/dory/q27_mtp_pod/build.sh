#!/bin/bash
# fast build (no LTO, 16 codegen units) into cortiq.new, swapped in when it is whole
. "$HOME/.cargo/env"
cd /root/q27/cmf
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -30
cp target/release/cortiq /root/q27/cortiq.new && mv /root/q27/cortiq.new /root/q27/cortiq3 && /root/q27/cortiq3 --version && echo BUILD_DONE
