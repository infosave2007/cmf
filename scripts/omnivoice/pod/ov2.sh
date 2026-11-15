#!/bin/bash
# OmniVoice round 2: q8_2f with f16 audio embeddings and heads (the default
# file), the head as a device GEMM. Step-0 logits against the reference and
# the step time on CUDA / Vulkan / CPU, deterministic tokens of every case,
# the voice-clone codes, the release's sampling (seed 1) for listening, and
# Whisper large-v3-turbo on our wavs and the reference's.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/ov_build.log 2>&1
grep -q BUILD_FAST_DONE logs/ov_build.log || { echo OV2_BUILD_FAILED; tail -40 logs/ov_build.log; exit 1; }
B=bin/cortiq-fast
mkdir -p logs/ov2 out/ov2
rm -f cmf/omnivoice-q8_2f.cmf cmf/omnivoice-q8a16.cmf cmf/omnivoice-q8.cmf cmf/omnivoice-q4tp.cmf
timeout 1800 $B convert --model ov/src --quant q8_2f --output cmf/omnivoice-q8_2f.cmf > logs/ov2/convert.log 2>&1 || { echo OV2_CONVERT_FAILED; tail -20 logs/ov2/convert.log; exit 1; }
ls -la cmf/omnivoice-*.cmf
F=cmf/omnivoice-q8_2f.cmf
txt() { venv/bin/python -c "import json;print(json.load(open('ov/ref/$1.json'))['text'])"; }
ins() { venv/bin/python -c "import json;v=json.load(open('ov/ref/$1.json'))['instruct'];print(v or '')"; }
lng() { venv/bin/python -c "import json;print(json.load(open('ov/ref/$1.json'))['lang'] or '')"; }
run() { # case backend tag [extra env...]
  local c=$1 be=$2 tag=$3; shift 3
  T="$(txt $c)"; I="$(ins $c)"; L="$(lng $c)"
  args=(--lang "$L"); [ -n "$I" ] && args+=(--instruct "$I")
  flock gpu.lock env CMF_GPU=$be CMF_OV_STEP0_DUMP=out/ov2/${c}_${tag}_step0.bin CMF_OV_TOKENS_DUMP=out/ov2/${c}_${tag}_tok.json CMF_OV_PROF=1 "$@" \
    timeout 1200 $B tts $F "$T" "${args[@]}" --out out/ov2/${c}_${tag}.wav > logs/ov2/${c}_${tag}.log 2>&1
  echo "$c $tag exit $? | $(grep -hE 'step (1|31):' logs/ov2/${c}_${tag}.log | sed 's/omnivoice: //' | tr '\n' ' ') | $(grep -hE 'generated in' logs/ov2/${c}_${tag}.log)"
}
run en_auto cuda cu CMF_OV_POS_TEMP=0
run en_auto 1 vk CMF_OV_POS_TEMP=0
run en_auto 0 cpu CMF_OV_POS_TEMP=0
for c in ru_design zh_auto en_tag; do run $c cuda cu CMF_OV_POS_TEMP=0; done
for c in en_auto ru_design zh_auto en_tag; do run $c cuda s1; done
RT="$(venv/bin/python -c "import json;print(json.load(open('ov/ref/clone_ref.json'))['ref_text'])")"
CT="$(txt en_clone)"
flock gpu.lock env CMF_GPU=cuda CMF_OV_POS_TEMP=0 CMF_OV_REF_DUMP=out/ov2/clone_ref_cu.json CMF_OV_TOKENS_DUMP=out/ov2/en_clone_cu_tok.json CMF_OV_STEP0_DUMP=out/ov2/en_clone_cu_step0.bin \
  timeout 1200 $B tts $F "$CT" --lang en --ref-audio ov/ref/en_auto.wav --ref-text "$RT" --out out/ov2/en_clone_cu.wav > logs/ov2/en_clone_cu.log 2>&1
echo "en_clone cu exit $? | $(grep -hE 'wrote|generated in' logs/ov2/en_clone_cu.log | tr '\n' ' ')"
flock gpu.lock env CMF_GPU=cuda CMF_OV_REF_DUMP=out/ov2/clone_ref_s1.json \
  timeout 1200 $B tts $F "$CT" --lang en --ref-audio ov/ref/en_auto.wav --ref-text "$RT" --out out/ov2/en_clone_s1.wav > logs/ov2/en_clone_s1.log 2>&1
echo "en_clone s1 exit $?"
venv/bin/python - <<'PY'
import numpy as np, json, os
def ls(x):
    m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
for c in ["en_auto", "ru_design", "zh_auto", "en_tag", "en_clone"]:
    rc = np.load(f"ov/ref/{c}_step0_cond.npy"); T = rc.shape[1]; lc = ls(rc)
    rt = np.load(f"ov/ref/{c}_tokens.npy")
    for tag in ["cu", "vk", "cpu"]:
        f = f"out/ov2/{c}_{tag}_step0.bin"
        if os.path.exists(f):
            x = np.fromfile(f, dtype=np.float32).reshape(2, T, 8, 1025).transpose(0, 2, 1, 3)
            oc = ls(x[0]); d = np.abs(oc - lc)
            print(f"{c} {tag}: step0 mean|dlogp| {d.mean():.4f} max {d.max():.3f} argmax agree {(oc.argmax(-1) == lc.argmax(-1)).mean() * 100:.1f}%")
        f = f"out/ov2/{c}_{tag}_tok.json"
        if os.path.exists(f):
            t = np.array(json.load(open(f)))
            if t.shape == rt.shape:
                print(f"{c} {tag}: deterministic tokens equal {float((t == rt).mean()) * 100:.1f}%")
for tag in ["cu", "s1"]:
    f = f"out/ov2/clone_ref_{tag}.json"
    if os.path.exists(f):
        a = np.array(json.load(open(f))["codes"]); b = np.load("ov/ref/clone_ref_codes.npy")
        print(f"clone ref codes {tag}: {a.shape} vs {b.shape}", (f"equal {float((a == b).mean()) * 100:.1f}% per book {[round(float((a[i] == b[i]).mean()) * 100) for i in range(len(a))]}" if a.shape == b.shape else ""))
PY
flock gpu.lock venv/bin/python - <<'PY' 2>logs/ov2/whisper.err
import glob, os, torch
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device="cuda:0")
files = sorted(glob.glob("out/ov2/*_cu.wav") + glob.glob("out/ov2/*_s1.wav") + glob.glob("out/ov2/*_vk.wav") + glob.glob("out/ov2/*_cpu.wav")) + sorted(glob.glob("ov/ref/*.wav"))
for f in files:
    if f.endswith("_raw.wav"): continue
    print(f"{f}: {asr(f)['text'].strip()}", flush=True)
PY
echo OV2_DONE
