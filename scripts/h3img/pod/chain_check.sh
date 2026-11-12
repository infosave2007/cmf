#!/bin/bash
# The resident chains against the per-op path on the real weights: the same
# 512 x 512 run with the DiT, the prompt encoder and attention on the card as
# chains, then with every product a separate TF32 GEMM and norms, rotation,
# modulation, residuals and attention on the host. Prints the latents'
# relative difference and the pictures' PSNR.
#   chain_check.sh [BACKEND]
set -u
root=${H3I_ROOT:-/root/h3i}
backend=${1:-cuda}
cd "$root" || exit 1
export MIN_AREA=1024
CMF_LATENT_OUT=$root/out/chk_chain.f32 bash scripts/gen.sh chk_chain "$backend" 512 512 | tail -1
CMF_MMH3_CHAIN=0 CMF_TE_CHAIN=0 CMF_VAE3D_CHAIN=0 CMF_MMH3_ATTN=cpu CMF_LATENT_OUT=$root/out/chk_perop.f32 \
  bash scripts/gen.sh chk_perop "$backend" 512 512 | tail -1
python3 scripts/cmp_f32.py out/chk_chain.f32 out/chk_perop.f32
python3 scripts/cmp_png.py out/chk_chain.png out/chk_perop.png
