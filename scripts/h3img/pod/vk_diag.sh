#!/bin/bash
# Where a Vulkan step goes on a 16 GB card: the same 1024 x 1024 run under
# different plane-cache and weight-budget settings, one line each.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
one() { # tag env...
  local tag=$1; shift
  env "$@" CMF_MMH3_PROF=1 CMF_FFN_COUNT=1 bash scripts/gen.sh "$tag" vulkan 1024 1024 > /dev/null
  echo "$tag [$*] $(grep -E "image\(s\)" out/$tag.log | sed -E "s/.*in ([0-9.]+)s \(text ([0-9.]+)s.*denoise ([0-9.]+)s, median step ([0-9.]+)s.*vae ([0-9.]+)s.*/total \1 text \2 denoise \3 step \4 vae \5/") peak $(sort -n out/$tag.vram_trace | tail -1) MiB"
  grep -E "ffn_packed:|resident profile|declined|refus" out/$tag.log | tail -3 | cut -c1-220
}
one vkd_default X=1
one vkd_coop0 CMF_COOP=0
one vkd_plane0 CMF_PLANE_CACHE_MB=0
one vkd_coop0_plane0 CMF_COOP=0 CMF_PLANE_CACHE_MB=0
echo VK_DIAG_DONE
