#!/bin/bash
# OmniVoice round 1c: where q8_2f loses accuracy — layers q8_2f with the audio
# embeddings and heads in f16 (CMF_OV_F16_AUDIO=1), and q4tp — step-0 logits
# against the reference on CUDA.
set -u
cd /root/h3i || exit 1
bash scripts/build.sh fast > logs/ov_build.log 2>&1
grep -q BUILD_FAST_DONE logs/ov_build.log || { echo OV1C_BUILD_FAILED; tail -40 logs/ov_build.log; exit 1; }
B=bin/cortiq-fast
mkdir -p logs/ov1c out/ov1c
CMF_OV_F16_AUDIO=1 timeout 1800 $B convert --model ov/src --quant q8_2f --output cmf/omnivoice-q8a16.cmf > logs/ov1c/convert_q8a16.log 2>&1 || { echo CONVERT_FAILED; tail -20 logs/ov1c/convert_q8a16.log; }
timeout 1800 $B convert --model ov/src --quant q4tp --output cmf/omnivoice-q4tp.cmf > logs/ov1c/convert_q4tp.log 2>&1 || { echo CONVERT_FAILED q4tp; tail -20 logs/ov1c/convert_q4tp.log; }
ls -la cmf/omnivoice-*.cmf
T="$(venv/bin/python -c "import json;print(json.load(open('ov/ref/en_auto.json'))['text'])")"
for v in q8a16 q4tp q8_2f; do
  flock gpu.lock env CMF_GPU=cuda CMF_OV_POS_TEMP=0 CMF_OV_STEP0_DUMP=out/ov1c/en_auto_$v.bin CMF_OV_PROF=1 timeout 900 $B tts cmf/omnivoice-$v.cmf "$T" --lang en --out out/ov1c/en_auto_$v.wav > logs/ov1c/en_auto_$v.log 2>&1
  echo "$v exit $? | $(grep -hE 'step (1|31):' logs/ov1c/en_auto_$v.log | sed 's/omnivoice: //' | tr '\n' ' ') $(grep -h 'generated in' logs/ov1c/en_auto_$v.log)"
done
venv/bin/python - <<'PY'
import numpy as np
def ls(x):
    m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
rc = ls(np.load("ov/ref/en_auto_step0_cond.npy")); T = rc.shape[1]
for v in ["q8a16", "q4tp", "q8_2f"]:
    try:
        x = np.fromfile(f"out/ov1c/en_auto_{v}.bin", dtype=np.float32).reshape(2, T, 8, 1025).transpose(0, 2, 1, 3)
    except Exception as e:
        print(v, e); continue
    oc = ls(x[0]); d = np.abs(oc - rc)
    print(f"{v}: step0 cond mean|dlogp| {d.mean():.4f} max {d.max():.3f} argmax agree {(oc.argmax(-1) == rc.argmax(-1)).mean() * 100:.1f}%")
PY
echo OV1C_DONE
