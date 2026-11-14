#!/bin/bash
# Qwen3-TTS round 1: convert (f16), greedy codes against qwen_tts, the decoder
# on the reference codes against its waveform, then CUDA.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/qt_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/qt_build.log; exit 1; }
echo "build ok"
q=/root/qt
mkdir -p $q/out
text="Привет! Это пример синтеза речи на русском языке."
[ -s $q/out/ref.json ] || (cd $q && timeout 3600 ref/bin/python /root/h3i/scripts/ref.py hf out "$text" russian serena) 2>$q/out/ref.err | tail -2
[ -s $q/out/ref.json ] || { echo REF_FAILED; tail -20 $q/out/ref.err; }
t0=$(date +%s)
timeout 3600 bin/cortiq convert --model $q/hf --quant f16 --output $q/qtts-f16.cmf > logs/qt_convert.log 2>&1 || { echo CONVERT_FAILED; tail -30 logs/qt_convert.log; exit 1; }
echo "converted in $(( $(date +%s) - t0 )) s: $(ls -l $q/qtts-f16.cmf | awk '{print $5}') bytes"
bin/cortiq info $q/qtts-f16.cmf 2>&1 | head -12
for be in 0 cuda; do
  t0=$(date +%s.%N)
  flock gpu.lock env CMF_GPU=$be CMF_QTTS_PROF=1 CMF_QTTS_CODES_DUMP=$q/out/ours_$be.json timeout 3600 bin/cortiq tts $q/qtts-f16.cmf "$text" --lang ru --voice serena --temperature 0 --out $q/out/ours_$be.wav > logs/qt_tts_$be.log 2>&1
  echo "ours $be: exit $?, $(awk "BEGIN{printf \"%.1f\", $(date +%s.%N) - $t0}") s"
  grep -E "wrote|generated|qwen3-tts:|rror|panick" logs/qt_tts_$be.log | head -5
done
$q/ref/bin/python - <<'PY'
import json
r = json.load(open('/root/qt/out/ref.json'))
json.dump(r['codes'], open('/root/qt/out/ref_codes.json', 'w'))
print('ref ids', r['ids'][:4], '...', r['ids'][-6:], 'frames', len(r['codes']))
for be in ('0', 'cuda'):
    try:
        o = json.load(open(f'/root/qt/out/ours_{be}.json'))
    except Exception as e:
        print(be, 'no codes', e); continue
    first = next((i for i, (a, b) in enumerate(zip(o, r['codes'])) if a != b), None)
    print(f'{be}: {len(o)} frames vs ref {len(r["codes"])}; first differing frame {first}' + (f' (ours {o[first][:4]} ref {r["codes"][first][:4]})' if first is not None else ''))
PY
CMF_GPU=0 CMF_QTTS_CODES_IN=$q/out/ref_codes.json timeout 1800 bin/cortiq tts $q/qtts-f16.cmf "x" --out $q/out/dec_ref.wav > logs/qt_dec.log 2>&1
grep -E "wrote|generated|rror|panick" logs/qt_dec.log | head -3
$q/ref/bin/python - <<'PY'
import numpy as np, soundfile as sf
ref = np.fromfile('/root/qt/out/ref_wav.f32', np.float32)
ours, fs = sf.read('/root/qt/out/dec_ref.wav', dtype='float32')
n = min(len(ref), len(ours))
d = ours[:n] - ref[:n]
print(f'decoder on the reference codes: {len(ours)} vs {len(ref)} samples, rel {np.sqrt((d**2).sum()/(ref[:n]**2).sum()):.3e}, max abs {np.abs(d).max():.3e} (16-bit wav quantum 3e-5)')
PY
echo QT1_DONE
