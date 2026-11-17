#!/bin/bash
# Qwen3-TTS round 4: stage clocks (code predictor prefill / steps / heads,
# decoder stages) after the host decoder went parallel; a longer English
# text too, every wav through whisper.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/qt4_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/qt4_build.log; exit 1; }
echo "build ok"
q=/root/qt
w=cmf/dl/whisper-large-v3-turbo-q8.cmf
ru="Привет! Это пример синтеза речи на русском языке."
en="The quick brown fox jumps over the lazy dog. Speech synthesis now runs natively in cortiq, without Python."
run() { # tag file backend lang text
  local tag=$1 f=$2 be=$3 lang=$4 text=$5
  flock gpu.lock env CMF_GPU=$be CMF_QTTS_PROF=1 timeout 3600 bin/cortiq tts $q/qtts-$f.cmf "$text" --lang $lang --voice serena --seed 7 --out $q/out/r4_$tag.wav > logs/qt4_$tag.log 2>&1
  echo "$tag: exit $? | $(grep -oE '✓ wrote.*' logs/qt4_$tag.log | grep -oE '\(.*\)') | $(grep -oE '\[generated.*' logs/qt4_$tag.log)"
  grep -oE 'qwen3-tts: .*|qwen3-tts stages: .*' logs/qt4_$tag.log | sed 's/^/    /'
  [ -s $q/out/r4_$tag.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $q/out/r4_$tag.wav --language $lang 2>/dev/null | tail -1 | sed 's/^/    whisper: /'
}
for f in q8_2f q4tp; do
  run ${f}_cuda_ru $f cuda ru "$ru"
  run ${f}_cuda_en $f cuda en "$en"
done
run q4tp_vk_en q4tp 1 en "$en"
run q4tp_cpu_en q4tp 0 en "$en"
echo QT4_DONE
