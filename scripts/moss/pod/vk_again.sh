#!/bin/bash
# The Vulkan MOSS figures again on the 0.8.81 release build (VAE chain fixed):
# SFT q8_2f at 512 x 512, Base q4tp at 512 and 1024; deletes the files.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export BIN=$root/bin/cortiq
sumline() { tr '\r' '\n' < out/$1.log | grep -E "MOSS image|declined|stood down" | sort -u | tr '\n' ' ' | cut -c1-500; echo; }
venv/bin/hf download infosave/MOSS-Image-SFT-cmf moss-image-sft-q8_2f.cmf --local-dir cmf > logs/va_dl1.log 2>&1
venv/bin/hf download infosave/MOSS-Image-Base-cmf moss-image-base-q4tp.cmf --local-dir cmf > logs/va_dl2.log 2>&1
MODEL=cmf/moss-image-sft-q8_2f.cmf bash scripts/moss.sh va_vk_512_q8 vulkan 512 8 | head -1; sumline va_vk_512_q8
for s in 512 1024; do MODEL=cmf/moss-image-base-q4tp.cmf bash scripts/moss.sh va_base_vk_$s vulkan $s 8 | head -1; sumline va_base_vk_$s; done
rm -f cmf/moss-image-sft-q8_2f.cmf cmf/moss-image-base-q4tp.cmf
echo VK_AGAIN_DONE
