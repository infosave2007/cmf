#!/bin/bash
# Add fl2va to an existing MiniMax-H3 CMF.
#
# The published file is text-to-video: it carries the DiT, the prompt
# encoder and both VAE DECODERS. Conditioning on a keyframe needs two
# more stacks — Qwen3-VL's vision tower and the video VAE's encoder —
# and this adds them without repacking what is already there.
#
#   tools/mmh3_fl2va_pack.sh mmh3-turbo-q4tp.cmf out.cmf [video_vae.safetensors]
#
# The tower is 1.2 GB inside a 51.5 GB file and comes down by ranged
# read. The encoder needs the video VAE (5.2 GB), which is downloaded
# if not given. Budget ~7 GB of transfer and ~0.5 GB of growth.
set -euo pipefail

IN="${1:?usage: mmh3_fl2va_pack.sh <in.cmf> <out.cmf> [video_vae.safetensors]}"
OUT="${2:?}"
VAE="${3:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "── vision tower (ranged read, ~1.2 GB of a 51.5 GB file) ──"
python3 "$ROOT/tools/mmh3_fetch.py" visual --out "$WORK/visual.safetensors"

if [ -z "$VAE" ]; then
  echo "── video VAE (5.2 GB; its encoder half is what we need) ──"
  hf download Comfy-Org/MiniMax-H3 vae/minimax_h3_video_vae_fp16.safetensors \
      --local-dir "$WORK"
  VAE="$WORK/vae/minimax_h3_video_vae_fp16.safetensors"
fi

echo "── pack ──"
cortiq animate-pack --in "$IN" --out "$OUT" \
    --vision "$WORK/visual.safetensors" \
    --video-vae "$VAE" \
    --quant q4tp

cortiq verify "$OUT"
echo
echo "$OUT is fl2va-capable:"
echo "  cortiq animate $OUT --prompt '…' --first-frame frame.ppm --out clip.avi"
