#!/bin/bash
# OmniVoice decoder speed: the transposed convolutions as one GEMM and an
# overlap-add, Snake over the pool. Codes from one CUDA generation each
# (4.5 s and ~17 s of speech), then decoded from those codes on CUDA, Vulkan
# and the CPU, the new arm against CMF_OV_CONVT_REF=1, stage timings
# (CMF_OV_PROF) and the waveform difference.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
bash scripts/build.sh fast > logs/ov4_build.log 2>&1
grep -q BUILD_FAST_DONE logs/ov4_build.log || { echo OV4_BUILD_FAILED; tail -40 logs/ov4_build.log; exit 1; }
B=bin/cortiq-fast; F=cmf/omnivoice-q8_2f.cmf
mkdir -p logs/ov4 out/ov4
MID="The history of the printing press is a story of patience and invention. Long before movable type, scribes copied every book by hand, and a single volume could take months. Johannes Gutenberg combined a screw press with metal letters that could be reused."
gen() { # tag text
  flock gpu.lock env CMF_GPU=cuda CMF_OV_TOKENS_DUMP=out/ov4/$1.json timeout 1200 $B tts $F "$2" --lang en --out out/ov4/$1_gen.wav > logs/ov4/$1_gen.log 2>&1
  echo "$1 gen exit $? | $(grep -hE 'wrote|generated in' logs/ov4/$1_gen.log | tr '\n' ' ')"
}
dec() { # tag backend arm
  local x=""; [ "$3" = ref ] && x="CMF_OV_CONVT_REF=1"
  flock gpu.lock env CMF_GPU=$2 CMF_OV_PROF=1 CMF_OV_TOKENS_IN=out/ov4/$1.json CMF_OV_RAW_DUMP=out/ov4/$1_$2_$3_raw.wav $x timeout 1800 $B tts $F "x" --lang en --out out/ov4/$1_$2_$3.wav > logs/ov4/$1_$2_$3.log 2>&1
  echo "$1 $2 $3 exit $? | $(grep -hE 'decoded in|ov codec\] (block|conv2)' logs/ov4/$1_$2_$3.log | sed -E 's/\[reference.*decoded in/decoded in/' | tr '\n' ' ')"
}
gen en13 "Hello, this is a test of the OmniVoice model running natively in cortiq."
gen mid "$MID"
for t in en13 mid; do
  for be in cuda 1 0; do
    for arm in ref gemm; do dec $t $be $arm; done
  done
done
venv/bin/python - <<'PY'
import glob, numpy as np, soundfile as sf
for t in ("en13", "mid"):
    base, _ = sf.read(f"out/ov4/{t}_0_ref_raw.wav", dtype="float32")
    for f in sorted(glob.glob(f"out/ov4/{t}_*_raw.wav")):
        y, _ = sf.read(f, dtype="float32")
        n = min(len(y), len(base))
        d = y[:n] - base[:n]
        snr = 10 * np.log10((base[:n] ** 2).sum() / max((d ** 2).sum(), 1e-30))
        print(f"{f}: len {len(y)} vs {len(base)}, max|d| {np.abs(d).max():.2e}, SNR vs CPU ref {snr:.1f} dB")
PY
echo OV4_DONE
