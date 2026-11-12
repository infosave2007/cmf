#!/bin/bash
# Where LTX-2.5 stands: the container from Hugging Face (checked by its
# sha256), then the reference clip (384 x 256, 49 frames) over Vulkan and on
# CUDA with the binary given in BIN.
#   ltx_base.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-lb}
cd "$root" || exit 1
if [ ! -f cmf/ltx25-q4tp.cmf ]; then
  venv/bin/hf download infosave/LTX-2.5-cmf ltx25-q4tp.cmf --local-dir cmf > logs/dl_ltx.log 2>&1
fi
want=$(curl -s "https://huggingface.co/api/models/infosave/LTX-2.5-cmf/tree/main?expand=1" | python3 -c "
import sys, json
for e in json.load(sys.stdin):
    if e['path'] == 'ltx25-q4tp.cmf': print(e['lfs']['oid'])")
got=$(sha256sum cmf/ltx25-q4tp.cmf | cut -d' ' -f1)
echo "sha256 want=$want got=$got $([ "$want" = "$got" ] && echo OK || echo MISMATCH)"
export BIN=${BIN:-$root/bin/cortiq}
bash scripts/ltx.sh ${p}_vk vulkan
bash scripts/ltx.sh ${p}_cuda cuda
venv/bin/python scripts/cmp_y4m.py out/${p}_vk.y4m out/${p}_cuda.y4m
echo LTX_BASE_DONE
