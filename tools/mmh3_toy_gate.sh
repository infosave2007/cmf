#!/bin/bash
# The MiniMax-H3 port against the reference forward, all four stacks.
#
# Needs torch and a ComfyUI checkout for the fixtures; give it the path.
# Everything after that is this repository:
#
#   tools/mmh3_toy_gate.sh /path/to/ComfyUI [workdir]
#
# The packs are EXACT (--quant f32) on purpose: q4tp's noise floor sits
# an order of magnitude above the arithmetic difference this is looking
# for, so quantizing here would pass a broken port.
set -euo pipefail

COMFY="${1:?usage: mmh3_toy_gate.sh /path/to/ComfyUI [workdir]}"
WORK="${2:-${TMPDIR:-/tmp}/mmh3toy}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORTIQ=(cargo run -q -p cortiq-cli --bin cortiq --)

# ── the DiT ────────────────────────────────────────────────────────
python3 "$ROOT/tools/mk_mmh3_toy.py" --comfyui "$COMFY" --out "$WORK/dit"
"${CORTIQ[@]}" animate-pack --dit "$WORK/dit/dit.safetensors" \
    --quant f32 --out "$WORK/dit/toy.cmf"

# ── the prompt encoder ─────────────────────────────────────────────
python3 "$ROOT/tools/mk_qwen3te_toy.py" --comfyui "$COMFY" --out "$WORK/te"
"${CORTIQ[@]}" animate-pack --te "$WORK/te/te.safetensors" \
    --quant f32 --out "$WORK/te/toy.cmf"

# ── both VAE decoders ──────────────────────────────────────────────
python3 "$ROOT/tools/mk_vae_toy.py" --comfyui "$COMFY" --out "$WORK/vae"
# The ViT3D's head count is an architecture constant of the release,
# not something the checkpoint records; the toy is narrower.
"${CORTIQ[@]}" animate-pack --video-vae "$WORK/vae/video_vae.safetensors" \
    --vvae-heads 2 --quant f32 --out "$WORK/vae/vvae.cmf"
"${CORTIQ[@]}" animate-pack --audio-vae "$WORK/vae/audio_vae.safetensors" \
    --quant f32 --out "$WORK/vae/avae.cmf"

CMF_MMH3_TOY="$WORK/dit" \
CMF_QWEN3TE_TOY="$WORK/te" \
CMF_MMH3_VAE_TOY="$WORK/vae" \
    cargo test -q -p cortiq-engine --release \
        --test mmh3_parity --test qwen3te_parity --test mmh3_vae_parity \
        -- --nocapture
