#!/bin/bash
# Haru round 1: build, convert (q4tp, q8_2f), inputs from our Qwen3-TTS (an
# English question, a voice reference), the codec round trip against the
# PyTorch reference, then conversations (ours on CUDA / Vulkan, the reference
# on the CPU), every reply through whisper.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
H=/root/haru
mkdir -p $H/out
# the reference venv (torch CPU, the haru package)
if [ ! -x $H/ref/bin/python ]; then
  python3 -m venv $H/ref && $H/ref/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu > $H/ref_install.log 2>&1 \
    && $H/ref/bin/pip install -q -e $H/src/haru >> $H/ref_install.log 2>&1 || { echo REF_INSTALL_FAILED; tail -5 $H/ref_install.log; }
fi
[ "${SKIP_BUILD:-0}" = 1 ] || { bash scripts/build.sh release > logs/haru_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/haru_build.log; exit 1; }; }
echo "build ok"
[ -s $H/haru-q4tp.cmf ] || timeout 3600 bin/cortiq convert --model $H/hf --quant q4tp --output $H/haru-q4tp.cmf --also-output q8_2f=$H/haru-q8_2f.cmf > logs/haru_convert.log 2>&1 || { echo CONVERT_FAILED; tail -30 logs/haru_convert.log; exit 1; }
ls -l $H/*.cmf | awk '{print $5, $9}'
bin/cortiq info $H/haru-q4tp.cmf 2>&1 | grep -E "Arch|Layers|Hidden|Vocab|Tensors"
# inputs from our Qwen3-TTS
tts=/root/qt/qtts-q4tp.cmf
flock gpu.lock env CMF_GPU=cuda bin/cortiq tts $tts "Hi! How are you doing today? Could you tell me something interesting about the ocean?" --lang en --voice ryan --seed 3 --out $H/out/user_en.wav > /dev/null 2>&1
flock gpu.lock env CMF_GPU=cuda bin/cortiq tts $tts "Hello there. I love talking about science, books and travel. Today is a beautiful day, and I am happy to chat with you for a while." --lang en --voice serena --seed 5 --out $H/out/voice.wav > /dev/null 2>&1
# a 12 s user clip: the question, then silence for the answer
$H/ref/bin/python - <<'PY'
import numpy as np, soundfile as sf
a, r = sf.read('/root/haru/out/user_en.wav', dtype='float32')
pad = np.zeros(int(12 * r) - len(a) if len(a) < 12 * r else 0, dtype=np.float32)
sf.write('/root/haru/out/user_call.wav', np.concatenate([a, pad]), r)
print(f'user clip: {len(a)/r:.1f} s of question, {len(pad)/r:.1f} s of silence')
PY
# codec round trip
flock gpu.lock env CMF_GPU=cuda CMF_HARU_CODEC_RT=$H/out/ours_means.f32 timeout 1200 bin/cortiq duplex $H/haru-q4tp.cmf $H/out/user_en.wav $H/out/ours_rt.wav > logs/haru_rt.log 2>&1
echo "ours codec: exit $? | $(grep -oE 'codec round trip.*' logs/haru_rt.log)"
grep -iE "rror|panick" logs/haru_rt.log | head -3
timeout 1800 $H/ref/bin/python scripts/haru_ref.py codec $H/out/user_en.wav $H/out/ref_rt.wav $H/out/ref_means.f32 2>&1 | tail -1
$H/ref/bin/python - <<'PY'
import numpy as np, soundfile as sf
a = np.fromfile('/root/haru/out/ours_means.f32', np.float32); b = np.fromfile('/root/haru/out/ref_means.f32', np.float32)
n = min(len(a), len(b))
print(f'codec means: {len(a)} vs {len(b)} values, rel {np.sqrt(((a[:n]-b[:n])**2).sum()/(b[:n]**2).sum()):.3e}')
x, _ = sf.read('/root/haru/out/ours_rt.wav', dtype='float32'); y, _ = sf.read('/root/haru/out/ref_rt.wav', dtype='float32')
n = min(len(x), len(y))
print(f'codec round trip audio: {len(x)} vs {len(y)} samples, rel {np.sqrt(((x[:n]-y[:n])**2).sum()/(y[:n]**2).sum()):.3e}')
PY
w=/root/h3i/cmf/dl/whisper-large-v3-turbo-q8.cmf
for f in q4tp q8_2f; do
  for be in cuda 1; do
    t0=$(date +%s)
    flock gpu.lock env CMF_GPU=$be timeout 3600 bin/cortiq duplex $H/haru-$f.cmf $H/out/user_call.wav $H/out/r1_${f}_$be.wav --reference $H/out/voice.wav --seed 1 --stereo --transcript $H/out/r1_${f}_$be.jsonl > logs/haru_${f}_$be.log 2>&1
    echo "$f $be: exit $? in $(( $(date +%s) - t0 )) s | $(grep -E 'frames .*per frame' logs/haru_${f}_$be.log)"
    sed -n '2p' logs/haru_${f}_$be.log | cut -c1-300 | sed 's/^/    text: /'
    grep -oE '\[load.*' logs/haru_${f}_$be.log | sed 's/^/    /'
    grep -iE "rror|panick|declin" logs/haru_${f}_$be.log | sed -E 's/^.*Z[^ ]* +//' | sort -u | head -3
    [ -s $H/out/r1_${f}_$be.wav ] && $H/ref/bin/python -c "
import soundfile as sf; a, r = sf.read('$H/out/r1_${f}_$be.wav', dtype='float32'); sf.write('$H/out/r1_${f}_${be}_model.wav', a[:, 1], r)" \
      && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $H/out/r1_${f}_${be}_model.wav --language en 2>/dev/null | tail -1 | sed 's/^/    whisper (model channel): /'
  done
done
timeout 7200 $H/ref/bin/python scripts/haru_ref.py talk $H/out/user_call.wav $H/out/ref_talk.wav $H/out/voice.wav 1 2>&1 | tail -1
[ -s $H/out/ref_talk.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $H/out/ref_talk.wav --language en 2>/dev/null | tail -1 | sed 's/^/    whisper (reference reply): /'
echo HARU1_DONE
