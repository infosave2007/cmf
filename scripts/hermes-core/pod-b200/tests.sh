#!/bin/bash
source /root/.cargo/env
cd /root/hc/cmf
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
CARGO_BUILD_JOBS=8 cargo test --release -p cortiq-embryo --features cuda --no-fail-fast > /root/hc/run/tests.log 2>&1
echo "TESTS_EXIT $?" >> /root/hc/run/tests.log
