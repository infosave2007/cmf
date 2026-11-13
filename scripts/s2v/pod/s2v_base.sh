#!/bin/bash
# Kandinsky-S2V before its CUDA arm (release build of 0.8.84): the q4tp file,
# a 5 s test track and the reference frame; one 5 s chunk at 512 x 512 over
# Vulkan (16 steps); CUDA at 256 x 256 for 2 steps (the DiT on the host).
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
bin=$root/bin/cortiq
f=cmf/kandinsky-s2v-pro-q4tp.cmf
[ -f "$f" ] || venv/bin/hf download infosave/Kandinsky-S2V-1.0-Pro-distill-cmf kandinsky-s2v-pro-q4tp.cmf --local-dir cmf > logs/s2v_dl.log 2>&1
ls -la "$f" || { echo DOWNLOAD_FAILED; exit 1; }
venv/bin/python - <<'PY'
import numpy as np, wave
from PIL import Image
sr = 16000
t = np.arange(int(sr * 5.0)) / sr
# a voiced-like test track: a 140 Hz carrier with formants, syllable envelope
x = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate([140, 280, 700, 1200, 2400], 1))
x *= 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t)) * 0.3
with wave.open("/root/h3i/ref/s2v_track.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
Image.open("/root/h3i/ref/edit_in.png").convert("RGB").resize((512, 512)).save("/root/h3i/ref/s2v_ref.png")
PY
run() {  # tag backend size steps
  local tag=$1 be=$2 size=$3 steps=$4
  exec 9>"$root/gpu.lock"; flock 9
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > out/$tag.vram_trace 2>/dev/null &
  local tr=$! t0=$(date +%s.%N)
  CMF_GPU=$be RUST_LOG=info "$bin" s2v --model $f --image ref/s2v_ref.png --audio ref/s2v_track.wav --height $size --width $size \
    --steps $steps --chunks 1 --out out/$tag.avi > out/$tag.log 2>&1
  local rc=$? t1=$(date +%s.%N)
  kill $tr 2>/dev/null; exec 9>&-
  echo "tag=$tag backend=$be size=$size steps=$steps rc=$rc wall_s=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}") peak_vram_mib=$(sort -n out/$tag.vram_trace | tail -1)"
  tr '\r' '\n' < out/$tag.log | grep -iE "step|vae|chunk|declin|error|text|thinker" | grep -v "INFO.*Opened" | tail -6 | cut -c1-220
}
run s2vb_vk_512 1 512 16
run s2vb_cuda_256 cuda 256 2
echo S2V_BASE_DONE
