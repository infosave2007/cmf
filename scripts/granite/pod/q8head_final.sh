#!/bin/bash
# Release measurement of the hybrid arm's q8_2f head: MiniCPM-V-4.7, MiniCPM-OCR,
# Apex-35B (downloaded alongside the build), CUDA against Vulkan, two more
# bench runs each.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
[ -s cmf/apex-35b-v1.0-q4tp-v2.cmf ] || venv/bin/hf download infosave/Apex-35B-V1.0-cmf apex-35b-v1.0-q4tp-v2.cmf --local-dir cmf > logs/apex_dl.log 2>&1 &
dl=$!
bash scripts/build.sh release > logs/q8h_build.log 2>&1 || { echo BUILD_FAILED; exit 1; }
wait $dl
for f in MiniCPM-V-4.7-1B-q4tp minicpm-ocr-1.0-q4tp apex-35b-v1.0-q4tp-v2; do
  echo "#### $f"
  BIN=cortiq bash scripts/llm_check.sh cmf/$f.cmf ${f}_r "Write a Python function that checks whether a number is prime."
  for i in 1 2; do
    for be in cuda 1; do
      echo "$be bench $i: $(flock gpu.lock env CMF_GPU=$be timeout 900 bin/cortiq bench cmf/$f.cmf --tokens 64 --json 2>/dev/null | grep -oE '(decode_tok_s_steady|prefill_tok_s).: *[0-9.]*' | tr '\n' ' ')"
    done
  done
done
echo Q8H_DONE
