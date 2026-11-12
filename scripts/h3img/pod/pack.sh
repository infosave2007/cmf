#!/bin/bash
# Pack MiniMax-H3-Image into one CMF: the DiT (ComfyUI's pruned bf16 file,
# q4tp) and the image VAE (q8_2f) over the prompt encoder and vision tower of
# our published MiniMax-H3 container, which the image model shares unchanged.
#   pack.sh [OUT] [QUANT]
set -u
root=${H3I_ROOT:-/root/h3i}
out=${1:-$root/cmf/minimax-h3-image-q4tp.cmf}
quant=${2:-q4tp}
bin=${H3I_BIN:-$root/bin/cortiq-fast}
mkdir -p "$(dirname "$out")"
"$bin" animate-pack --out "$out" \
  --in "$root/src/mmh3-turbo-fl2va-q4tp.cmf" \
  --dit "$root/src/minimax_h3_image_pruned_bf16.safetensors" \
  --image-vae "$root/src/minimax_h3_image_vae_fp16.safetensors" --image-vae-quant q8_2f \
  --drop avae --quant "$quant" && sha256sum "$out" | tee "$out.sha256" && echo PACK_DONE
