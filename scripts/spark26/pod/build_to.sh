#!/bin/bash
# build_to.sh NAME: build the rsynced tree on the pod (no LTO) into /root/spark26/NAME, leaving
# the binaries other chains run untouched. Log logs/build_NAME.log, marker logs/build_NAME.done.
N=${1:?name}
cd /root/spark26/cmf && . "$HOME/.cargo/env"
export CARGO_TARGET_DIR=/root/spark26/target
rm -f /root/spark26/logs/build_$N.done
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli --features gpu > /root/spark26/logs/build_$N.log 2>&1
st=$?
grep -E "^error|could not" -A8 /root/spark26/logs/build_$N.log | head -30
[ $st -eq 0 ] && cp $CARGO_TARGET_DIR/release/cortiq /root/spark26/$N.new && mv /root/spark26/$N.new /root/spark26/$N && /root/spark26/$N --version
echo BUILD_DONE status=$st
touch /root/spark26/logs/build_$N.done
