#!/bin/bash
# What makes an edit come out over-processed: the reference picture or the
# prompt? The same 2048 x 2048 edit over crossed factors, one line each.
#   cafe1024       our own generated picture, reduced to 1024
#   cafe1024_soft  the same, slightly blurred (its micro-texture removed)
#   reference1     the model authors' example picture
#   flamingo1024   a crop of the authors' showcase, enlarged to 1024
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
python3 - <<"PY"
from PIL import Image, ImageFilter
im = Image.open("out/te4_cafe.png").convert("RGB").resize((1024, 1024), Image.LANCZOS)
im.save("refs/cafe1024.png")
im.filter(ImageFilter.GaussianBlur(1.0)).save("refs/cafe1024_soft.png")
f = Image.open("refs/showcase_photo.jpg").convert("RGB")
f.crop((2200, 0, 2757, 557)).resize((1024, 1024), Image.LANCZOS).save("refs/flamingo1024.png")
PY
one() { # tag image prompt [seed]
  IMAGE=refs/$2 PROMPT=prompts/$3.txt bash scripts/gen.sh "$1" cuda 2048 2048 "${4:-42}" > /dev/null
  echo "$1 [$2, $3, seed ${4:-42}] $(grep -E "image\(s\)" out/$1.log | cut -c1-120)"
}
one em_cafe_min cafe1024.png edit_min_cafe
one em_cafe_soft cafe1024_soft.png edit_cafe_local
one em_their_ours reference1.png edit_their_ours
one em_flamingo flamingo1024.png edit_flamingo
one em_cafe_seed7 cafe1024.png edit_cafe_local 7
python3 - <<"PY"
from PIL import Image
for t in ("em_cafe_min", "em_cafe_soft", "em_their_ours", "em_flamingo", "em_cafe_seed7"):
    try:
        Image.open(f"out/{t}.png").convert("RGB").resize((1024, 1024), Image.LANCZOS).save(f"out/{t}_s.jpg", quality=90)
    except Exception as e:
        print(t, e)
PY
echo EDIT_MATRIX_DONE
