#!/bin/bash
# build.sh NAME PATCH: branch fast-code-v2 + PATCH (git diff of the worktree), build, /root/fc/cortiq.NAME
set -uo pipefail
cd /root/fc/cmf && . "$HOME/.cargo/env"
git checkout -q -- . && git clean -qfd crates 2>/dev/null
git apply --whitespace=nowarn "$2" || { echo PATCH_FAILED; exit 1; }
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 19 cargo build -j 12 --release -p cortiq-cli --features gpu 2>&1 | grep -E "^error|Finished|could not" -A8 | tail -20
cp target/release/cortiq /root/fc/cortiq.$1.new && mv /root/fc/cortiq.$1.new /root/fc/cortiq.$1 && /root/fc/cortiq.$1 --version
echo BUILD_DONE
