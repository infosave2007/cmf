#!/bin/bash
# The 0.8.84 closing runs: release build, the Primal VAE round trips on it
# (twice), clippy; the file deleted.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
bash scripts/build.sh release > logs/r84_build_release.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/r84_build_release.log; exit 1; }
BIN=$root/bin/cortiq bash scripts/prim.sh r84a
BIN=$root/bin/cortiq bash scripts/prim.sh r84b
rm -f cmf/primal_vae_2x2.cmf
. "$HOME/.cargo/env"
( cd srcrs && cargo clippy --locked --workspace --all-targets --features cuda 2>&1 | grep -cE "^error"; echo "clippy cuda rc=${PIPESTATUS[0]}"; cargo clippy --locked --workspace --all-targets 2>&1 | grep -cE "^error"; echo "clippy rc=${PIPESTATUS[0]}" )
echo REL84_DONE
