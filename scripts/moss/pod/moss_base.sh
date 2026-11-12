#!/bin/bash
# MOSS-Image-Base q4tp on the release build: 512 x 512 and 1024 x 1024 at
# 8 steps on CUDA and over Vulkan (the Base card's figures), then every MOSS
# file is deleted.
#   moss_base.sh [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
p=${1:-mb}
cd "$root" || exit 1
export BIN=$root/bin/cortiq
f=cmf/moss-image-base-q4tp.cmf
[ -f "$f" ] || venv/bin/hf download infosave/MOSS-Image-Base-cmf moss-image-base-q4tp.cmf --local-dir cmf > logs/${p}_dl.log 2>&1
sumline() { tr '\r' '\n' < out/$1.log | grep -E "MOSS image|declined|stood down" | sort -u | tr '\n' ' ' | cut -c1-500; echo; }
for s in 512 1024; do
  for b in cuda vulkan; do
    MODEL=$f bash scripts/moss.sh ${p}_${b}_$s $b $s 8 | head -1; sumline ${p}_${b}_$s
  done
done
rm -f cmf/moss-image-base-q4tp.cmf cmf/moss-image-sft-q4tp.cmf cmf/moss-image-sft-q8_2f.cmf
echo MOSS_BASE_DONE
