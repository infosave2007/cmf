#!/bin/bash
# dflash-check.sh: build, convert DFlash 2 into a sidecar beside the turbo file, and run
# the engine's drafter against the reference kit, on the host and on the device
. "$HOME/.cargo/env"
cd /root/q27/cmf
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 timeout 900 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A12 | tail -40
cp target/release/cortiq /root/q27/cortiq4 || exit 1
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
[ -f kit/kit.json ] || python3 q27_dflash_kit.py ref kit
Q=${1:-q4tp}
rm -f m/qwen38-27b-turbo.dflash.cmf
timeout 600 ./cortiq4 convert --model ext/dflash2 --output m/qwen38-27b-turbo.cmf --dflash-sidecar --quant $Q
ls -la m/qwen38-27b-turbo.dflash.cmf
echo "== host"; timeout 600 ./cortiq4 dflash-check m/qwen38-27b-turbo.cmf kit --host 2>&1 | grep -v "^\s*$" | tail -40
echo "== device"; timeout 600 ./cortiq4 dflash-check m/qwen38-27b-turbo.cmf kit 2>&1 | grep -v "^\s*$" | tail -40
echo DFLASH_CHECK_DONE
