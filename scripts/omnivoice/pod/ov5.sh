#!/bin/bash
# OmniVoice round 5: the card's command lines (CUDA and Vulkan) with the
# faster codec decoder (transposed convolutions as one GEMM), their speech
# length and times, a long text, a clone, Whisper on all of them.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
B=bin/cortiq-fast; F=cmf/omnivoice-q8_2f.cmf
mkdir -p logs/ov5 out/ov5
LONG="The history of the printing press is a story of patience and invention. Long before movable type, scribes copied every book by hand, and a single volume could take months. In the fifteenth century, Johannes Gutenberg combined a screw press with metal letters that could be arranged and reused. Suddenly a page could be printed hundreds of times in a day. Books became cheaper, ideas travelled faster, and literacy slowly spread across Europe. Within fifty years, millions of copies had been printed, and the world of learning was never the same again."
run() { # tag backend text args...
  local tag=$1 be=$2 t=$3; shift 3
  flock gpu.lock env CMF_GPU=$be timeout 1200 $B tts $F "$t" "$@" --out out/ov5/$tag.wav > logs/ov5/$tag.log 2>&1
  echo "$tag exit $? | $(grep -hE 'wrote|generated in' logs/ov5/$tag.log | tr '\n' ' ')"
}
for be in cuda 1; do
  run en_$be $be "Hello! This voice comes from a single file." --lang en
  run en13_$be $be "Hello, this is a test of the OmniVoice model running natively in cortiq." --lang en
  run ru_$be $be "Привет! Это проверка синтеза речи на русском языке." --lang ru --instruct "female, low pitch"
  run zh_$be $be "你好，这是一个语音合成测试。" --lang zh
  run tag_$be $be "I can't believe it [laughter] that is so funny!" --lang en --instruct "male, british accent"
  run sich_$be $be "今天天气真好。" --lang zh --instruct "男，四川话"
done
run long_cuda cuda "$LONG" --lang en
run clone_cuda cuda "This sentence is spoken in the cloned voice of the first sample." --lang en --ref-audio ov/ref/en_auto.wav --ref-text "Hello, this is a test of the OmniVoice model running natively in cortiq."
flock gpu.lock venv/bin/python - <<'PY' 2>logs/ov5/whisper.err
import glob, torch
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda:0")
for f in sorted(glob.glob("out/ov5/*.wav")):
    print(f"{f}: {asr(f, return_timestamps=True)['text'].strip()}", flush=True)
PY
echo OV5_DONE
