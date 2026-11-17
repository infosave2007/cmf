#!/bin/bash
# The 0.8.96 sweep's leftovers (release build): MiniCPM5 q8_2f decode with
# --ignore-eos (it ends its answer at once otherwise), and Unlimited-OCR-2 on
# one generated document page: `run --image`, wall time and the decode line.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bin=bin/cortiq
mkdir -p cmf/dl ref
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
f=cmf/dl/MiniCPM5-2B-q8_2f.cmf
[ -s "$f" ] || timeout 1800 venv/bin/hf download infosave/MiniCPM5-2B-cmf MiniCPM5-2B-q8_2f.cmf --local-dir cmf/dl > logs/r96b_dl1.log 2>&1
for be in cuda 1; do
  for i in 1 2; do
    flock gpu.lock env CMF_GPU=$be timeout 600 $bin bench "$f" --tokens 64 --ignore-eos --json > logs/r96b_mc5_${be}_$i.log 2>&1
    echo "minicpm5-q8_2f $be d$i: $(grep -oE "$keys" logs/r96b_mc5_${be}_$i.log | tr '\n' ' ')"
  done
done
rm -f "$f"

# a document page: a title and paragraphs, 1024 x 1320
[ -s ref/ocr_page.png ] || venv/bin/python - <<'EOF'
from PIL import Image, ImageDraw, ImageFont
import textwrap
W, H = 1024, 1320
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 34)
b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
d.text((70, 60), "Quarterly Operations Report", font=t, fill="black")
paras = [
    "The third quarter closed with revenue of 4.82 million dollars, up 12 percent on the same period last year. "
    "Operating costs rose by 3 percent, mostly from the new warehouse in Rotterdam that opened on 14 August.",
    "Customer support handled 18,430 tickets, a fifth of them through the new chat assistant. The median time "
    "to a first answer fell from 41 minutes to 9, and satisfaction scores held at 4.6 out of 5.",
    "Two risks remain open: the supplier contract for packaging expires in January, and the payment provider "
    "will raise its fees by 0.4 points from March. Both are on the board's agenda for the November meeting.",
    "Next quarter the team will finish the inventory system migration, hire four engineers for the platform "
    "group and start the pilot in Lisbon with three retail partners.",
]
y = 140
for p in paras:
    for line in textwrap.wrap(p, 78):
        d.text((70, y), line, font=b, fill="black")
        y += 30
    y += 26
im.save("ref/ocr_page.png")
EOF
for file in unlimited-ocr-2-q8_2f.cmf unlimited-ocr-2-q4tp.cmf; do
  f=cmf/dl/$file
  [ -s "$f" ] || timeout 1800 venv/bin/hf download infosave/Unlimited-OCR-2-cmf "$file" --local-dir cmf/dl > logs/r96b_dl_ocr.log 2>&1
  for be in cuda 1; do
    for i in 1 2; do
      log=logs/r96b_ocr_$(basename "$file" .cmf)_${be}_$i.log
      t0=$(date +%s.%N)
      flock gpu.lock env CMF_GPU=$be timeout 900 $bin run "$f" --image ref/ocr_page.png --prompt "Multi page parsing." --max-tokens 1024 --greedy > "$log" 2>&1
      t1=$(date +%s.%N)
      echo "$(basename "$file" .cmf) $be run$i: wall $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s; $(grep -oE '[0-9.]+ tok/s( steady)?|first token[^,;]*' "$log" | tr '\n' ' ')"
    done
  done
  rm -f "$f"
done
# Embryo-56M: the resident graph, the lookup table off (it would answer the
# prompt from the table and generate nothing)
for be in cuda 1; do
  for i in 1 2; do
    flock gpu.lock env CMF_GPU=$be CMF_EMBRYO_RESIDENT=1 CMF_LOOKUP_MODE=off timeout 300 $bin bench cmf/embryo-56m.cmf --tokens 64 --ignore-eos --json > logs/r96b_emb_${be}_$i.log 2>&1
    echo "embryo-56m $be d$i: $(grep -oE "$keys" logs/r96b_emb_${be}_$i.log | tr '\n' ' ')"
  done
done
rm -rf cmf/dl/.cache
echo REL96B_DONE
