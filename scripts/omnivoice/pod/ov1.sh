#!/bin/bash
# OmniVoice round 1: build (fast, CUDA), convert f16 + q8_2f, then against the
# reference dumps (ov/ref): the prompt, the codec on the reference tokens,
# deterministic generation (CMF_OV_POS_TEMP=0) on CPU / Vulkan / CUDA, the
# post-processing length, and the voice-clone codes.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/ov_build.log 2>&1
grep -q BUILD_FAST_DONE logs/ov_build.log || { echo OV1_BUILD_FAILED; tail -40 logs/ov_build.log; exit 1; }
mkdir -p cmf logs/ov1 out/ov1
B=bin/cortiq-fast
for q in f16 q8_2f; do
  [ -s cmf/omnivoice-$q.cmf ] || timeout 1800 $B convert --model ov/src --quant $q --output cmf/omnivoice-$q.cmf > logs/ov1/convert_$q.log 2>&1 || { echo "OV1_CONVERT_FAILED $q"; tail -20 logs/ov1/convert_$q.log; exit 1; }
  ls -la cmf/omnivoice-$q.cmf
done
venv/bin/python - <<'PY'
import numpy as np, json, glob, os
for f in glob.glob("ov/ref/*_tokens.npy"):
    json.dump(np.load(f).tolist(), open(f.replace(".npy", ".json"), "w"))
json.dump(np.load("ov/ref/clone_ref_codes.npy").tolist(), open("ov/ref/clone_ref_codes.json", "w"))
PY
F=cmf/omnivoice-f16.cmf
txt() { venv/bin/python -c "import json;print(json.load(open('ov/ref/$1.json'))['text'])"; }
ins() { venv/bin/python -c "import json;v=json.load(open('ov/ref/$1.json'))['instruct'];print(v or '')"; }
lng() { venv/bin/python -c "import json;print(json.load(open('ov/ref/$1.json'))['lang'] or '')"; }
for c in en_auto ru_design zh_auto en_tag; do
  T="$(txt $c)"; I="$(ins $c)"; L="$(lng $c)"
  args=(--lang "$L"); [ -n "$I" ] && args+=(--instruct "$I")
  # the codec alone on the reference tokens
  CMF_GPU=0 CMF_OV_TOKENS_IN=ov/ref/${c}_tokens.json CMF_OV_RAW_DUMP=out/ov1/${c}_dec_raw.wav timeout 600 $B tts $F "$T" "${args[@]}" --out out/ov1/${c}_dec.wav > logs/ov1/${c}_dec.log 2>&1
  for be in 0 1 cuda; do
    CMF_GPU=$be CMF_OV_POS_TEMP=0 CMF_OV_PROMPT_DUMP=out/ov1/${c}_prompt.json CMF_OV_TOKENS_DUMP=out/ov1/${c}_tok_$be.json CMF_OV_RAW_DUMP=out/ov1/${c}_raw_$be.wav CMF_OV_PROF=1 \
      timeout 900 $B tts $F "$T" "${args[@]}" --out out/ov1/${c}_$be.wav > logs/ov1/${c}_$be.log 2>&1
    echo "$c $be exit $? | $(grep -hE 'wrote|generated in' logs/ov1/${c}_$be.log | tr '\n' ' ')"
  done
done
# voice clone: the reference's codes of en_auto.wav, then a clone on them
RT="$(venv/bin/python -c "import json;print(json.load(open('ov/ref/clone_ref.json'))['ref_text'])")"
CT="$(txt en_clone)"
for be in 0 cuda; do
  CMF_GPU=$be CMF_OV_POS_TEMP=0 CMF_OV_REF_DUMP=out/ov1/clone_ref_$be.json CMF_OV_TOKENS_DUMP=out/ov1/en_clone_tok_$be.json \
    timeout 900 $B tts $F "$CT" --lang en --ref-audio ov/ref/en_auto.wav --ref-text "$RT" --out out/ov1/en_clone_$be.wav > logs/ov1/en_clone_$be.log 2>&1
  echo "en_clone $be exit $? | $(grep -hE 'wrote|generated in' logs/ov1/en_clone_$be.log | tr '\n' ' ')"
done
# the same generation from the reference's own clone codes
CMF_GPU=cuda CMF_OV_POS_TEMP=0 CMF_OV_REF_CODES=ov/ref/clone_ref_codes.json CMF_OV_REF_RMS=$(venv/bin/python -c "import json;print(json.load(open('ov/ref/clone_ref.json'))['ref_rms'])") CMF_OV_PROMPT_DUMP=out/ov1/en_clone_prompt.json CMF_OV_TOKENS_DUMP=out/ov1/en_clone_tok_refcodes.json \
  timeout 900 $B tts $F "$CT" --lang en --ref-text "$RT" --out out/ov1/en_clone_refcodes.wav > logs/ov1/en_clone_refcodes.log 2>&1
echo "en_clone refcodes exit $?"
venv/bin/python scripts/ov_cmp.py > logs/ov1/cmp.txt 2>&1; cat logs/ov1/cmp.txt
echo OV1_DONE
