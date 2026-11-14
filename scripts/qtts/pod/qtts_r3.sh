#!/bin/bash
# Qwen3-TTS round 3: the code predictor as a pipeline of its own (its layers
# quantized with the profile), the decoder's GEMMs on the device. Greedy codes
# against the reference (f16), sampled speech per file and backend, every wav
# transcribed back by whisper.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh release > logs/qt3_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -40 logs/qt3_build.log; exit 1; }
echo "build ok"
q=/root/qt
rm -f $q/qtts-*.cmf
timeout 3600 bin/cortiq convert --model $q/hf --quant f16 --output $q/qtts-f16.cmf --also-output q8_2f=$q/qtts-q8_2f.cmf --also-output q4tp=$q/qtts-q4tp.cmf > logs/qt3_convert.log 2>&1 || { echo CONVERT_FAILED; tail -20 logs/qt3_convert.log; exit 1; }
ls -l $q/*.cmf | awk '{print $5, $9}'
w=cmf/dl/whisper-large-v3-turbo-q8.cmf
[ -s $w ] || timeout 1800 venv/bin/hf download infosave/whisper-large-v3-turbo-cmf whisper-large-v3-turbo-q8.cmf --local-dir cmf/dl > /dev/null 2>&1
text="Привет! Это пример синтеза речи на русском языке."
run() { # tag file backend temperature
  local tag=$1 f=$2 be=$3 t=$4
  flock gpu.lock env CMF_GPU=$be CMF_QTTS_PROF=1 CMF_QTTS_CODES_DUMP=$q/out/r3_$tag.json timeout 3600 bin/cortiq tts $q/qtts-$f.cmf "$text" --lang ru --voice serena --temperature $t --seed 7 --out $q/out/r3_$tag.wav > logs/qt3_$tag.log 2>&1
  echo "$tag: exit $? | $(grep -oE 'qwen3-tts:.*' logs/qt3_$tag.log) | $(grep -oE '\[generated.*' logs/qt3_$tag.log)"
  grep -iE "declin|refused|unsupported|rror" logs/qt3_$tag.log | sed -E 's/^.*Z[^ ]* +//' | sort -u | head -3
  if [ -s $q/out/r3_$tag.wav ]; then
    flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $q/out/r3_$tag.wav --language ru 2>/dev/null | tail -1 | sed 's/^/    whisper: /'
  fi
}
run f16_cpu_g f16 0 0
run f16_cuda_g f16 cuda 0
for f in q8_2f q4tp; do
  for be in cuda 1 0; do run ${f}_$be $f $be 0.9; done
done
$q/ref/bin/python - <<'PY'
import json
r = json.load(open('/root/qt/out/ref.json'))['codes']
for t in ('f16_cpu_g', 'f16_cuda_g'):
    try:
        o = json.load(open(f'/root/qt/out/r3_{t}.json'))
    except Exception:
        print(t, 'no codes'); continue
    first = next((i for i, (a, b) in enumerate(zip(o, r)) if a != b), None)
    print(f'{t}: {len(o)} frames vs ref {len(r)}; first differing frame {first}')
PY
flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $q/out/ref.wav --language ru 2>/dev/null | tail -1 | sed 's/^/reference wav, whisper: /'
echo QT3_DONE
