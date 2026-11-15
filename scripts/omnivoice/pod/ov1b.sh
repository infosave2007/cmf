#!/bin/bash
# OmniVoice round 1b: the step-0 logits against the reference (f16 on the
# CPU, q8_2f on Vulkan and CUDA), the device path's step time, then the
# deterministic tokens of every case on CUDA (q8_2f) and the clone codes.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/ov_build.log 2>&1
grep -q BUILD_FAST_DONE logs/ov_build.log || { echo OV1B_BUILD_FAILED; tail -40 logs/ov_build.log; exit 1; }
B=bin/cortiq-fast
mkdir -p logs/ov1b out/ov1b
txt() { venv/bin/python -c "import json;print(json.load(open('ov/ref/$1.json'))['text'])"; }
ins() { venv/bin/python -c "import json;v=json.load(open('ov/ref/$1.json'))['instruct'];print(v or '')"; }
lng() { venv/bin/python -c "import json;print(json.load(open('ov/ref/$1.json'))['lang'] or '')"; }
run() { # case backend file tag
  local c=$1 be=$2 f=$3 tag=$4
  T="$(txt $c)"; I="$(ins $c)"; L="$(lng $c)"
  args=(--lang "$L"); [ -n "$I" ] && args+=(--instruct "$I")
  CMF_GPU=$be CMF_OV_POS_TEMP=0 CMF_OV_STEP0_DUMP=out/ov1b/${c}_${tag}_step0.bin CMF_OV_TOKENS_DUMP=out/ov1b/${c}_${tag}_tok.json CMF_OV_PROF=1 \
    timeout 900 $B tts $f "$T" "${args[@]}" --out out/ov1b/${c}_${tag}.wav > logs/ov1b/${c}_${tag}.log 2>&1
  echo "$c $tag exit $? | $(grep -hE 'step (0|1|31):' logs/ov1b/${c}_${tag}.log | sed 's/omnivoice: //' | tr '\n' ' ') | $(grep -hE 'generated in' logs/ov1b/${c}_${tag}.log)"
}
run en_auto 1 cmf/omnivoice-q8_2f.cmf vk8
run en_auto cuda cmf/omnivoice-q8_2f.cmf cu8
run en_auto cuda cmf/omnivoice-f16.cmf cu16
run en_auto 0 cmf/omnivoice-q8_2f.cmf cpu8
for c in ru_design zh_auto en_tag; do run $c cuda cmf/omnivoice-q8_2f.cmf cu8; done
venv/bin/python - <<'PY'
import numpy as np, json, os
def ls(x):
    m = x.max(-1, keepdims=True); return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))
for c in ["en_auto", "ru_design", "zh_auto", "en_tag"]:
    rc = np.load(f"ov/ref/{c}_step0_cond.npy"); ru = np.load(f"ov/ref/{c}_step0_uncond.npy")  # [8,T,1025]
    T = rc.shape[1]
    for tag in ["cpu8", "vk8", "cu8", "cu16", "f16cpu"]:
        f = f"out/ov1b/{c}_{tag}_step0.bin"
        if not os.path.exists(f): continue
        x = np.fromfile(f, dtype=np.float32).reshape(2, T, 8, 1025).transpose(0, 2, 1, 3)
        oc, ou = ls(x[0]), ls(x[1])
        lc, lu = ls(rc), ls(ru)
        d = np.abs(oc - lc)
        agree = (oc.argmax(-1) == lc.argmax(-1)).mean() * 100
        agree_u = (ou.argmax(-1) == lu.argmax(-1)).mean() * 100
        print(f"{c} {tag}: step0 cond max|dlogp| {d.max():.3f} mean {d.mean():.4f} argmax agree {agree:.1f}% uncond {agree_u:.1f}%")
    rt = np.load(f"ov/ref/{c}_tokens.npy")
    for tag in ["cpu8", "vk8", "cu8", "cu16"]:
        f = f"out/ov1b/{c}_{tag}_tok.json"
        if os.path.exists(f):
            t = np.array(json.load(open(f)))
            print(f"{c} {tag}: tokens equal {float((t == rt).mean()) * 100:.1f}%")
PY
echo OV1B_DONE
