#!/bin/bash
# The Mac's sequence for MiniMax-H3-Turbo, one run at a time: the clip on
# the per-operation path with the binary already built, the build of the
# current sources, the clip with the resident chain, and the two compared.
#   mac_queue.sh [TAGPREFIX]
set -u
here=$(cd "$(dirname "$0")" && pwd)
runs=${RUNS:-$HOME/dev/h3img-runs}
p=${1:-mt}
CMF_H3_METAL=0 bash "$here/mac_anim.sh" ${p}_perop
( cd "$here/../.." && CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_INCREMENTAL=true \
    cargo build --release -p cortiq-cli --target-dir target-fast 2>&1 | grep -E "^error|Finished|could not compile" -A8 | tail -30
  [ "${PIPESTATUS[0]}" = 0 ] ) || { echo MAC_BUILD_FAILED; exit 1; }
bash "$here/mac_anim.sh" ${p}_chain
python3 "$here/pod/cmp_frames.py" "$runs/turbo/${p}_chain.frames" "$runs/turbo/${p}_perop.frames" "$runs/turbo/${p}_chain.wav" "$runs/turbo/${p}_perop.wav"
echo MAC_QUEUE_DONE
