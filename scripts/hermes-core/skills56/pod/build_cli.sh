#!/bin/bash
# The engine (probe-utility, ppl) for measurements on this pod: CPU build. Log: /root/sk/build_cli.log
cd /root/sk/cmf && source /root/.cargo/env
cargo build --release -p cortiq-cli > /root/sk/build_cli.log 2>&1
echo "BUILD_EXIT $?" >> /root/sk/build_cli.log
