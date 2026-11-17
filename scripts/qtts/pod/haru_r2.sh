#!/bin/bash
# Haru round 2: the hidden state from the plain step (the head separately),
# the CUDA product weights cached; a clip with 15 s of silence after the
# question; CUDA, Vulkan, CPU and the PyTorch reference, every reply through
# whisper, the text streams side by side.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
H=/root/haru
bash scripts/build.sh release > logs/haru2_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/haru2_build.log; exit 1; }
echo "build ok"
$H/ref/bin/python - <<'PY'
import numpy as np, soundfile as sf
a, r = sf.read('/root/haru/out/user_en.wav', dtype='float32')
sf.write('/root/haru/out/user_call2.wav', np.concatenate([a, np.zeros(15 * r, dtype=np.float32)]), r)
print(f'user clip: {len(a)/r:.1f} s of question, 15 s of silence')
PY
w=/root/h3i/cmf/dl/whisper-large-v3-turbo-q8.cmf
for v in "q4tp cuda" "q4tp 1" "q8_2f cuda" "q4tp 0"; do
  set -- $v; f=$1; be=$2
  t0=$(date +%s)
  flock gpu.lock env CMF_GPU=$be timeout 3600 bin/cortiq duplex $H/haru-$f.cmf $H/out/user_call2.wav $H/out/r2_${f}_$be.wav --reference $H/out/voice.wav --seed 1 --transcript $H/out/r2_${f}_$be.jsonl > logs/haru2_${f}_$be.log 2>&1
  echo "$f $be: exit $? in $(( $(date +%s) - t0 )) s | $(grep -E 'frames .*per frame' logs/haru2_${f}_$be.log)"
  grep -vE 'INFO|WARN|frames .*per frame|^\[' logs/haru2_${f}_$be.log | head -2 | cut -c1-300 | sed 's/^/    text: /'
  grep -oE '\[load.*' logs/haru2_${f}_$be.log | sed 's/^/    /'
  grep -iE "rror|panick" logs/haru2_${f}_$be.log | head -2
  [ -s $H/out/r2_${f}_$be.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $H/out/r2_${f}_$be.wav --language en 2>/dev/null | tail -1 | sed 's/^/    whisper: /'
done
timeout 7200 $H/ref/bin/python scripts/haru_ref.py talk $H/out/user_call2.wav $H/out/ref_talk2.wav $H/out/voice.wav 1 2>&1 | tail -1
[ -s $H/out/ref_talk2.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $H/out/ref_talk2.wav --language en 2>/dev/null | tail -1 | sed 's/^/    whisper (reference): /'
echo HARU2_DONE
