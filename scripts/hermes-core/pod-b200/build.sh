#!/bin/bash
# The CUDA trainer, built within the pod's CPU quota (20 jobs).
source /root/.cargo/env
cd /root/hc/cmf
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
CARGO_BUILD_JOBS=20 cargo build --release -p cortiq-embryo --features cuda > /root/sk/build.log 2>&1
echo "BUILD_EXIT $?" >> /root/sk/build.log
