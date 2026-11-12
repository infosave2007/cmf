#!/bin/bash
# The second Vulkan diagnosis on a 16 GB card: fc2 of the fused FFN on the
# matrix units (it is on the scalar tile kernel by default), alone and with
# the other switches, the same 1024 x 1024 run, one line each plus the phases.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
one() { # tag env...
  local tag=$1; shift
  env "$@" CMF_MMH3_PROF=1 CMF_FFN_COUNT=1 bash scripts/gen.sh "$tag" vulkan 1024 1024 > /dev/null
  echo "$tag [$*] $(grep -E "image\(s\)" out/$tag.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$tag.vram_trace | tail -1) MiB"
  grep -E "q4tp_ffn split|gemm|attention  " out/$tag.log | tail -5 | cut -c1-160
}
one vk2_fc2coop CMF_FFN_FC2_COOP=1
one vk2_fc2coop_v2 CMF_FFN_FC2_COOP=1 CMF_BATCH_MM=v2
python3 scripts/cmp_png.py out/vkd_default.png out/vk2_fc2coop.png
echo VK_DIAG2_DONE
