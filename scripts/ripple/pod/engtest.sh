#!/bin/bash
# engtest.sh: the wgpu kernel parity tests of the engine on this card (no LTO).
cd /root/rip/cmf && . "$HOME/.cargo/env" && export XDG_RUNTIME_DIR=/tmp
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CMF_GPU=wgpu cargo test --release -p cortiq-engine --features gpu --lib -- --test-threads=1 wgpu_q82_mm_coop_matches_host wgpu_q8_2f_matvec4_matches_one_row 2>&1 | grep -E "^test |test result|relative rms|panicked|error\[" 
echo ENGTEST_DONE
