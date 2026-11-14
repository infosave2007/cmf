#!/bin/bash
# Round 3: the release build with VSA in the Vulkan chain and q8_2f experts on
# the CUDA hybrid arm. FastH3 on Vulkan at the small size (frames beside the
# CUDA run's), then Unlimited-OCR-2 q8_2f on CUDA against Vulkan (one page).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/fh3_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/fh3_build.log; exit 1; }
echo "build ok"
m=cmf/fh/fasth3-q4tp.cmf
prompt="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
mkdir -p out/fh_vk
t0=$(date +%s)
flock gpu.lock env CMF_GPU=1 CMF_GPU_DEBUG=1 CMF_ANIM_DUMP=out/fh_vk/d timeout 7200 bin/cortiq animate $m --prompt "$prompt" \
  --width 512 --height 288 --frames 39 --steps 4 --seed 7 --out out/fh_vk/clip.avi --frames-dir out/fh_vk/frames > logs/fh_vk.log 2>&1
echo "vulkan: exit $? in $(( $(date +%s) - t0 )) s; $(grep -oE 'stages: .*' logs/fh_vk.log | tail -1)"
grep -iE "declin|vsa|h3 wgpu chain" logs/fh_vk.log | sort | uniq -c | head -8
venv/bin/python - <<'EOF'
import array, math, os
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
def rel(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / max(sum(y * y for y in b), 1e-30))
for s in ('v', 'a'):
    a, b = f'out/fh_vk/d.{s}0.f32', f'out/fh_host/d.{s}0.f32'
    if os.path.exists(a) and os.path.exists(b):
        print(f'{s} step 0: vulkan vs host {rel(load(a), load(b)):.4f}   cuda vs host {rel(load(f"out/fh_cuda/d.{s}0.f32"), load(b)):.4f}')
from PIL import Image, ImageDraw
rows = []
for tag in ('cuda', 'vk', 'host'):
    if not os.path.exists(f'out/fh_{tag}/frames/frame_0019.ppm'):
        continue
    ims = [Image.open(f'out/fh_{tag}/frames/frame_{i:04d}.ppm') for i in (0, 19, 38)]
    w, h = ims[0].size
    row = Image.new('RGB', (w * 3, h + 22), 'white')
    for k, im in enumerate(ims):
        row.paste(im, (k * w, 22))
    ImageDraw.Draw(row).text((6, 4), {'cuda': 'CUDA', 'vk': 'Vulkan', 'host': 'host reference'}[tag] + ' (frames 0, 19, 38)', fill='black')
    rows.append(row)
sheet = Image.new('RGB', (rows[0].size[0], sum(r.size[1] for r in rows)), 'white')
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.size[1]
sheet.save('out/fh_sheet_vk.png')
EOF
# Unlimited-OCR-2 q8_2f: the routed experts now on the CUDA hybrid arm
f=cmf/dl/unlimited-ocr-2-q8_2f.cmf
mkdir -p cmf/dl
[ -s $f ] || timeout 1800 venv/bin/hf download infosave/Unlimited-OCR-2-cmf unlimited-ocr-2-q8_2f.cmf --local-dir cmf/dl > logs/fh3_dl_ocr.log 2>&1
for be in cuda 1; do
  for i in 1 2; do
    log=logs/fh3_ocr_${be}_$i.log
    t0=$(date +%s.%N)
    flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq run $f --image ref/ocr_page.png --prompt "Multi page parsing." --max-tokens 1024 --greedy > $log 2>&1
    t1=$(date +%s.%N)
    echo "ocr q8_2f $be run$i: wall $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s; $(grep -oE '[0-9.]+ tok/s( steady)?|first token[^,;]*' $log | tr '\n' ' ')"
    grep -iE "declined|refused" $log | sort -u | head -3
  done
done
for be in cuda 1; do grep -A200 '^Prompt:' logs/fh3_ocr_${be}_1.log | grep -v -E '^\[|tok/s|first token' > out/ocr_q8_$be.txt; done
cmp -s out/ocr_q8_cuda.txt out/ocr_q8_1.txt && echo "ocr q8_2f: CUDA text == Vulkan" || { echo "ocr q8_2f: CUDA text differs from Vulkan:"; diff out/ocr_q8_cuda.txt out/ocr_q8_1.txt | head -8; }
rm -f $f
# LFM3-1.7B: the batched prompt now reaches the CUDA hybrid prompt path
f=cmf/dl/lfm3-1.7b-q4tp.cmf
[ -s $f ] || timeout 1800 venv/bin/hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q4tp.cmf --local-dir cmf/dl > logs/fh3_dl_lfm3.log 2>&1
long="$(head -c 2400 srcrs/docs/ppl_nat.txt) Summarize the text above in three sentences."
for be in cuda 0; do
  flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq run $f --prompt "$long" -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/lfm3_long3_$be.txt
done
cmp -s out/lfm3_long3_cuda.txt out/lfm3_long3_0.txt && echo "lfm3 long prompt: CUDA text == CPU" || { echo "lfm3 long prompt: CUDA text DIFFERS"; tail -3 out/lfm3_long3_cuda.txt; tail -3 out/lfm3_long3_0.txt; }
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
for i in 1 2; do
  flock gpu.lock env CMF_GPU=cuda CMF_CUDA_DEBUG=1 timeout 900 bin/cortiq bench $f --tokens 32 --ctx 4096 --json > logs/fh3_lfm3_$i.log 2>&1
  echo "lfm3 4k cuda run$i: $(grep -oE "$keys" logs/fh3_lfm3_$i.log | tr '\n' ' ')"
done
grep -iE "declin|prompt path" logs/fh3_lfm3_1.log | sort -u | head -4
for i in 1 2; do
  flock gpu.lock env CMF_GPU=cuda timeout 900 bin/cortiq bench $f --tokens 64 --json > logs/fh3_lfm3_d$i.log 2>&1
  echo "lfm3 decode cuda run$i: $(grep -oE "$keys" logs/fh3_lfm3_d$i.log | tr '\n' ' ')"
done
rm -f $f
echo FH_TEST3_DONE
