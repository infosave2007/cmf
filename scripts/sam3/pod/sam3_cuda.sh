#!/bin/bash
# SAM3 on CUDA (fast build unless BIN=cortiq): one 1008 x 1008 image, stages,
# the boxes and scores against the host run's (out/sam3_0.json) and the
# overlay's difference.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
py=$root/venv/bin/python
tag=${1:-sam3c}
bin=${BIN:-cortiq-fast}
if [ "$bin" = cortiq-fast ]; then
  bash scripts/build.sh fast > logs/${tag}_build.log 2>&1 || { echo BUILD_FAILED; tail -40 logs/${tag}_build.log; exit 1; }
fi
f=cmf/sam3-q8_2f.cmf
exec 9>"$root/gpu.lock"; flock 9
for be in ${BACKENDS:-cuda}; do
  t=${tag}_$be
  t0=$(date +%s.%N)
  CMF_GPU=$be CMF_CUDA_DEBUG=1 SAM3_STAGES=1 bin/$bin segment $f ref/sam3_img.png --prompt "red square" --out out/$t.png --json out/$t.json > out/$t.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  echo "== $be rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")"
  grep -E "sam3-prof\] +(text|trunk|fpn|enc|dec|score|masks|TOTAL)|cuda trunk|declin|error|panick" out/$t.log | tail -12 | cut -c1-160
  $py - out/$t.json out/sam3_0.json out/$t.png out/sam3_0.png <<'PY'
import json, sys
import numpy as np
from PIL import Image
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def objs(d):
    o = d.get("objects") or d.get("detections") or d
    return o if isinstance(o, list) else []
oa, ob = objs(a), objs(b)
print("objects", len(oa), "vs host", len(ob))
for x, y in zip(oa, ob):
    print(" score", round(x.get("score", 0), 4), "host", round(y.get("score", 0), 4), "box", [round(v, 4) for v in x.get("box", [])], "host", [round(v, 4) for v in y.get("box", [])])
ia = np.asarray(Image.open(sys.argv[3]).convert("RGB"), dtype=float)
ib = np.asarray(Image.open(sys.argv[4]).convert("RGB"), dtype=float)
diff = (np.abs(ia - ib).sum(-1) > 0).mean()
print("overlay pixels that differ: %.4f%%" % (100 * diff))
PY
done
echo SAM3_CUDA_DONE
