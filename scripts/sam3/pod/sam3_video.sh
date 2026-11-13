#!/bin/bash
# SAM3's tracker over 8 synthetic 1008 x 1008 frames (a red square moving
# right) on CUDA and on the host: wall time and the tracks against each other.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
py=$root/venv/bin/python
bin=${BIN:-cortiq-fast}
mkdir -p ref/sam3_frames
[ -s ref/sam3_frames/f007.png ] || $py - <<'PY'
from PIL import Image, ImageDraw
for i in range(8):
    im = Image.new("RGB", (1008, 1008), (40, 90, 160))
    d = ImageDraw.Draw(im)
    x = 200 + 40 * i
    d.rectangle([x, 360, x + 300, 640], fill=(220, 30, 30))
    d.ellipse([680, 120, 900, 340], fill=(30, 200, 60))
    im.save("/root/h3i/ref/sam3_frames/f%03d.png" % i)
PY
exec 9>"$root/gpu.lock"; flock 9
for be in ${BACKENDS:-cuda 0}; do
  t=sam3v_$be
  rm -rf out/$t; mkdir -p out/$t
  t0=$(date +%s.%N)
  CMF_GPU=$be bin/$bin segment cmf/sam3-q8_2f.cmf ref/sam3_frames --prompt "red square" --out out/$t --json out/$t.json > out/$t.log 2>&1
  rc=$?; t1=$(date +%s.%N)
  echo "== $be rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")"
  grep -iE "frame|track|segment:|declin|error|panick" out/$t.log | tail -4 | cut -c1-200
done
for be in ${BACKENDS:-cuda 0}; do [ "$be" = 0 ] && continue; echo "-- $be vs host"; $py - out/sam3v_$be.json out/sam3v_0.json <<'PY'
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def boxes(d):
    t = d.get("tracks", [])
    return {f["frame"]: f["box_xyxy"] for f in (t[0]["frames"] if t else [])}, (t[0]["score"] if t else None)
(ba, sa), (bb, sb) = boxes(a), boxes(b)
print("tracks", len(a.get("tracks", [])), len(b.get("tracks", [])), "score", sa, sb)
for i in sorted(set(ba) | set(bb)):
    print(i, ba.get(i), bb.get(i))
PY
done
echo SAM3_VIDEO_DONE
