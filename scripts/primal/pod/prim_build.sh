#!/bin/bash
# Fast build, then prim.sh.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
bash scripts/build.sh fast > logs/pv_build.log 2>&1 || { echo BUILD_FAILED; tail -40 logs/pv_build.log; exit 1; }
bash scripts/prim.sh pvf
