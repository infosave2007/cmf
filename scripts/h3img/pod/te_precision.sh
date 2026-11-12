#!/bin/bash
# Is the 4-bit prompt encoder what shapes the pictures' fine texture? The
# conditioning of an 8-bit (q8_2f) encoder, packed from the bf16 source, under
# the same q4tp DiT, seed and prompts: the states' relative difference, the
# pictures' PSNR and 100 % crops of both.
#
# The pod's disk holds one large source at a time, so the published container
# is removed while the encoder packs and fetched back afterwards.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
B=${BIN:-$root/bin/cortiq-fast}
M=cmf/minimax-h3-image-q4tp.cmf
SHA=f4af395ada66213cd401a455fcc7fd7252d2d4c4813f8d1437ff58a89740f309
( ./dl.sh Comfy-Org/MiniMax-H3 text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors > logs/dl_tebf16.log 2>&1 ) &
for p in lighthouse cafe portrait; do
  CMF_GPU=cuda CMF_TE_DUMP=out/te4_$p.bin "$B" imagine $M --prompt @prompts/$p.txt --width 2048 --height 2048 --seed 42 \
    --out out/te4_$p.png > out/te4_$p.log 2>&1
  echo "te4_$p $(grep -E "image\(s\)" out/te4_$p.log | cut -c1-110)"
done
wait
tail -1 logs/dl_tebf16.log
[ -s refs/tokenizer.json ] || curl -sSL -m 120 -o refs/tokenizer.json \
  https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/resolve/main/tokenizer/tokenizer.json
rm -f $M
"$B" animate-pack --out cmf/te-q8.cmf --te src/qwen3vl_32b_minimax_h3_bf16.safetensors --te-layers 50 --quant q8_2f \
  --tokenizer refs/tokenizer.json > logs/pack_teq8.log 2>&1
tr "\r" "\n" < logs/pack_teq8.log | tail -1 | cut -c1-160
rm -f src/qwen3vl_32b_minimax_h3_bf16.safetensors
for p in lighthouse cafe portrait; do
  CMF_GPU=cuda CMF_TE_DUMP=out/te8_$p.bin "$B" imagine cmf/te-q8.cmf --prompt @prompts/$p.txt --width 2048 --height 2048 \
    --seed 42 --out out/x.png > out/te8_$p.log 2>&1
  python3 scripts/cmp_f32.py out/te4_$p.bin out/te8_$p.bin --skip 16
done
./dl.sh infosave/MiniMax-H3-Image-cmf minimax-h3-image-q4tp.cmf > logs/dl_q4tp.log 2>&1
mv src/minimax-h3-image-q4tp.cmf cmf/ && echo "$SHA  $M" | sha256sum -c -
for p in lighthouse cafe portrait; do
  CMF_GPU=cuda CMF_TE_LOAD=out/te8_$p.bin "$B" imagine $M --prompt @prompts/$p.txt --width 2048 --height 2048 --seed 42 \
    --out out/te8_$p.png > out/te8run_$p.log 2>&1
  echo "te8_$p $(grep -E "image\(s\)" out/te8run_$p.log | cut -c1-110)"
  python3 scripts/cmp_png.py out/te4_$p.png out/te8_$p.png
done
python3 - <<"PY"
from PIL import Image
for p, box in (("lighthouse", (1100, 500, 1612, 1012)), ("cafe", (900, 900, 1412, 1412)), ("portrait", (640, 600, 1152, 1112))):
    for q in ("te4", "te8"):
        try:
            im = Image.open(f"out/{q}_{p}.png").convert("RGB")
            im.crop(box).save(f"out/crop_{q}_{p}.png")
            im.resize((1024, 1024), Image.LANCZOS).save(f"out/{q}_{p}_s.jpg", quality=90)
        except Exception as e:
            print(q, p, e)
PY
echo TE_PRECISION_DONE
