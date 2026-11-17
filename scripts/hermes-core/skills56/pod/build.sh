#!/bin/bash
# Build the Embryo trainer for CUDA (which carries the Vulkan backend too). Log: /root/sk/build.log
cd /root/sk/cmf && source /root/.cargo/env
cargo build --release -p cortiq-embryo --features cuda > /root/sk/build.log 2>&1
echo "BUILD_EXIT $?" >> /root/sk/build.log
