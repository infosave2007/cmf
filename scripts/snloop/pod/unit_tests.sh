#!/bin/bash
# The campaign's unit tests on the pod (fast profile, the iteration build's target dir).
set -u
. "$HOME/.cargo/env"
cd /root/sn/src
export CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true
cargo test --release -p cortiq-engine --features cuda --target-dir target-fast --lib yarn_scale_fold 2>&1 | grep -E "^test |test result|error|panicked" | head -20
cargo test --release -p cortiq-cli --features cuda --target-dir target-fast mtp_sidecar_outside_the_index 2>&1 | grep -E "^test |test result|error\[|panicked" | head -20
cargo test --release -p cortiq-cli --features cuda --target-dir target-fast calibration_forward_reaches 2>&1 | grep -E "^test |test result|error\[|panicked" | head -20
echo UNIT_TESTS_DONE
