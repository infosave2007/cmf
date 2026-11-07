#!/bin/bash
# Build the rsynced cherrypit tree (no LTO), reusing the deps already compiled in /root/rip/cmf/target.
cd /root/cherrypit/cmf && . "$HOME/.cargo/env"
export CARGO_TARGET_DIR=/root/rip/cmf/target
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 10 cargo build --release -p cortiq-cli --features gpu 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -30
cp $CARGO_TARGET_DIR/release/cortiq /root/cherrypit/cortiq.new && mv /root/cherrypit/cortiq.new /root/cherrypit/cortiq && /root/cherrypit/cortiq --version
echo BUILD_DONE
