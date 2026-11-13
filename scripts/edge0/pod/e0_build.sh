#!/bin/bash
# e0_build.sh TAG: build /root/e0/cmf into /root/e0/cortiq.TAG
. /root/.cargo/env
cd /root/e0/cmf && cargo build --release -p cortiq-cli --features cuda > /root/e0/logs/build_$1.log 2>&1; echo "build rc=$?"
grep -E "^error" -A6 /root/e0/logs/build_$1.log | head -20
cp /root/e0/cmf/target/release/cortiq /root/e0/cortiq.$1
echo BUILD-DONE
