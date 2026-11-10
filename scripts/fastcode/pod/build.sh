#!/bin/bash
# build.sh NAME: master + hybrid.patch (hybrid-fixes c8c59011) + fc.patch, build, copy to /root/fc/cortiq.NAME
set -uo pipefail
cd /root/fc/cmf && . "$HOME/.cargo/env"
git checkout -q -- . && git clean -qfd crates scripts docs 2>/dev/null
git apply --whitespace=nowarn /root/fc/hybrid.patch || { echo PATCH_FAILED hybrid; exit 1; }
git apply --whitespace=nowarn /root/fc/fc.patch || { echo PATCH_FAILED fc; exit 1; }
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 10 cargo build --release -p cortiq-cli --features gpu 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -20
cp target/release/cortiq /root/fc/cortiq.$1 && /root/fc/cortiq.$1 --version
echo BUILD_DONE
