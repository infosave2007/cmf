#!/bin/bash
# The trainer with the parquet reader (`shard` over parquet needs --features data), CPU only. Log /root/sk/build_data.log
cd /root/sk/cmf && source /root/.cargo/env
cargo build --release -p cortiq-embryo --features data --target-dir target-data > /root/sk/build_data.log 2>&1
echo "BUILD_EXIT $?" >> /root/sk/build_data.log
