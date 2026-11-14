#!/bin/bash
# Qwen3-TTS round 2: the talker quantized (q8_2f, q4tp) so the device token
# graphs take it; greedy codes against the reference, time per backend.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
q=/root/qt
text="Привет! Это пример синтеза речи на русском языке."
[ -s $q/qtts-q8_2f.cmf ] || timeout 3600 bin/cortiq convert --model $q/hf --quant q8_2f --output $q/qtts-q8_2f.cmf --also-output q4tp=$q/qtts-q4tp.cmf > logs/qt_convert2.log 2>&1 || { echo CONVERT_FAILED; tail -20 logs/qt_convert2.log; exit 1; }
ls -l $q/*.cmf | awk '{print $5, $9}'
for f in q8_2f q4tp; do
  for be in cuda 1 0; do
    flock gpu.lock env CMF_GPU=$be CMF_QTTS_PROF=1 CMF_QTTS_CODES_DUMP=$q/out/r2_${f}_$be.json timeout 3600 bin/cortiq tts $q/qtts-$f.cmf "$text" --lang ru --voice serena --temperature 0 --out $q/out/r2_${f}_$be.wav > logs/qt2_${f}_$be.log 2>&1
    echo "$f $be: exit $? | $(grep -oE 'qwen3-tts:.*' logs/qt2_${f}_$be.log) | $(grep -oE '\[generated.*' logs/qt2_${f}_$be.log)"
    grep -iE "declin|refused|unsupported" logs/qt2_${f}_$be.log | sed -E 's/^.*Z[^ ]* +//' | sort -u | head -3
  done
done
$q/ref/bin/python - <<'PY'
import json
r = json.load(open('/root/qt/out/ref.json'))['codes']
for f in ('q8_2f', 'q4tp'):
    for be in ('cuda', '1', '0'):
        try:
            o = json.load(open(f'/root/qt/out/r2_{f}_{be}.json'))
        except Exception as e:
            print(f, be, 'no codes'); continue
        first = next((i for i, (a, b) in enumerate(zip(o, r)) if a != b), None)
        print(f'{f} {be}: {len(o)} frames vs ref {len(r)}; first differing frame {first}')
PY
echo QT2_DONE
