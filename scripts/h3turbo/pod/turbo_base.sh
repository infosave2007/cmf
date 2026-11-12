#!/bin/bash
# Where MiniMax-H3-Turbo stands before the chain: the container's hash, then
# the reference clip (512 x 288, 39 frames, 4 steps) on Vulkan and on CUDA
# with the binary given in BIN, then the fast build of the current sources.
#   turbo_base.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-tb}
cd "$root" || exit 1
want=$(curl -s "https://huggingface.co/api/models/infosave/MiniMax-H3-Turbo-cmf/tree/main?expand=1" | python3 -c "
import sys, json
for e in json.load(sys.stdin):
    if e['path'] == 'mmh3-turbo-q4tp.cmf': print(e['lfs']['oid'])")
got=$(sha256sum cmf/mmh3-turbo-q4tp.cmf | cut -d' ' -f1)
echo "sha256 want=$want got=$got $([ "$want" = "$got" ] && echo OK || echo MISMATCH)"
export BIN=${BIN:-$root/bin/cortiq}
bash scripts/anim.sh ${p}_vk vulkan
bash scripts/anim.sh ${p}_cuda cuda
"$root/venv/bin/python" scripts/cmp_frames.py out/${p}_vk.frames out/${p}_cuda.frames out/${p}_vk.wav out/${p}_cuda.wav
unset BIN
bash scripts/build.sh fast
echo TURBO_BASE_DONE
