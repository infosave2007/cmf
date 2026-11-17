#!/bin/bash
# Qwen3-TTS on CUDA after the code predictor's rewind: the card's numbers
# (two runs a file and language, the second counted), whisper on each.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
q=/root/qt
w=cmf/dl/whisper-large-v3-turbo-q8.cmf
ru="Привет! Это пример синтеза речи на русском языке."
en="The quick brown fox jumps over the lazy dog. Speech synthesis now runs natively in cortiq, without Python."
for f in q4tp q8_2f; do
  for l in ru en; do
    t=$ru; [ $l = en ] && t=$en
    for i in 1 2; do
      flock gpu.lock env CMF_GPU=cuda CMF_QTTS_PROF=1 timeout 1800 bin/cortiq tts $q/qtts-$f.cmf "$t" --lang $l --voice serena --seed 7 --out $q/out/r5_${f}_$l.wav > logs/qt5_${f}_$l.log 2>&1
    done
    echo "$f $l: $(grep -oE '\([0-9.]+s, 24000 Hz, [0-9]+ frames' logs/qt5_${f}_$l.log) | $(grep -oE '\[generated.*' logs/qt5_${f}_$l.log)"
    grep -oE 'qwen3-tts: .*' logs/qt5_${f}_$l.log | sed 's/^/    /'
    flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $q/out/r5_${f}_$l.wav --language $l 2>/dev/null | tail -1 | sed 's/^/    whisper: /'
  done
done
echo QT5_DONE
