#!/bin/bash
# The audio decoder on CUDA with its convolutions gathered on the card,
# against the clip of the build before (sound and frames compared).
#   turbo_audio.sh [TAGPREFIX] [REFTAG]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-ta}; ref=${2:-tg_cuda}
cd "$root" || exit 1
py=$root/venv/bin/python
bash scripts/build.sh fast || exit 1
bash scripts/anim.sh ${p}_cuda cuda
[ -f out/$ref.wav ] && $py - "$root/out/${p}_cuda.wav" "$root/out/$ref.wav" <<'PY'
import sys, wave
import numpy as np
def pcm(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
a, b = pcm(sys.argv[1]), pcm(sys.argv[2])
n = min(len(a), len(b)); d = a[:n] - b[:n]
snr = float("inf") if (d ** 2).mean() == 0 else 10 * np.log10((a[:n] ** 2).mean() / (d ** 2).mean())
print(f"audio vs the build before: {n} samples  snr {snr:.2f} dB  max abs diff {np.abs(d).max():.0f}")
PY
CMF_AVAE_CONV_GPU=0 bash scripts/anim.sh ${p}_cuda_hostconv cuda
$py scripts/cmp_frames.py out/${p}_cuda.frames out/${p}_cuda_hostconv.frames out/${p}_cuda.wav out/${p}_cuda_hostconv.wav
bash scripts/anim.sh ${p}_vk vulkan
rm -rf out/${p}_*.frames
echo TURBO_AUDIO_DONE
