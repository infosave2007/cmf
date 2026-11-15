#!/bin/bash
# OmniVoice round 1d: row-wise int8 (`--quant q8`, Q8Row) for the whole file —
# step-0 logits against the reference and the step time on CUDA and Vulkan.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
B=bin/cortiq-fast
mkdir -p logs/ov1d out/ov1d
[ -s cmf/omnivoice-q8.cmf ] || timeout 1800 $B convert --model ov/src --quant q8 --output cmf/omnivoice-q8.cmf > logs/ov1d/convert_q8.log 2>&1 || { echo CONVERT_FAILED; tail -20 logs/ov1d/convert_q8.log; }
ls -la cmf/omnivoice-q8.cmf
T="$(venv/bin/python -c "import json;print(json.load(open('ov/ref/en_auto.json'))['text'])")"
for be in cuda 1; do
  flock gpu.lock env CMF_GPU=$be CMF_OV_POS_TEMP=0 CMF_OV_STEP0_DUMP=out/ov1d/en_auto_q8_$be.bin CMF_OV_PROF=1 timeout 900 $B tts cmf/omnivoice-q8.cmf "$T" --lang en --out out/ov1d/en_auto_q8_$be.wav > logs/ov1d/en_auto_q8_$be.log 2>&1
  echo "q8 $be exit $? | $(grep -hE 'step (1|31):' logs/ov1d/en_auto_q8_$be.log | sed 's/omnivoice: //' | tr '\n' ' ') $(grep -h 'generated in' logs/ov1d/en_auto_q8_$be.log)"
done
venv/bin/python - <<'PY'
import numpy as np
def ls(x):
    m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
rc = ls(np.load("ov/ref/en_auto_step0_cond.npy")); T = rc.shape[1]
for be in ["cuda", "1"]:
    x = np.fromfile(f"out/ov1d/en_auto_q8_{be}.bin", dtype=np.float32).reshape(2, T, 8, 1025).transpose(0, 2, 1, 3)
    oc = ls(x[0]); d = np.abs(oc - rc)
    print(f"q8 {be}: step0 cond mean|dlogp| {d.mean():.4f} max {d.max():.3f} argmax agree {(oc.argmax(-1) == rc.argmax(-1)).mean() * 100:.1f}%")
PY
echo OV1D_DONE
