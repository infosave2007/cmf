#!/bin/bash
# dflash-acc.sh: the engine's drafter on every anchor of the kit, its layers' matrices as
# q4tp, q8_2f and f16: the drafts a round keeps against the published drafter's
. "$HOME/.cargo/env"
cd /root/q27/cmf
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 timeout 900 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished|could not" -A12 | tail -40
cp target/release/cortiq /root/q27/cortiq4 || exit 1
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
rm -rf kit; python3 q27_dflash_kit.py ref kit
for Q in f16 q8_2f q4tp; do
  rm -f m/qwen38-27b-turbo.dflash.cmf
  timeout 600 ./cortiq4 convert --model ext/dflash2 --output m/qwen38-27b-turbo.cmf --dflash-sidecar --quant $Q 2>&1 | tail -1
  ls -la m/qwen38-27b-turbo.dflash.cmf
  echo "== $Q device"; timeout 600 ./cortiq4 dflash-check m/qwen38-27b-turbo.cmf kit 2>&1 | grep -E "^window|^lowest|^drafts|^a round"
done
echo "== q4tp host"; timeout 600 ./cortiq4 dflash-check m/qwen38-27b-turbo.cmf kit --host 2>&1 | grep -E "^lowest|^drafts|^a round"
echo DFLASH_ACC_DONE
