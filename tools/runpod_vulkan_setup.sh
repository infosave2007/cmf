#!/usr/bin/env bash
set -euo pipefail

# Known-good RunPod setup. Install the complete GLVND/Vulkan set unconditionally:
# probing first can incorrectly report that the NVIDIA Vulkan driver is absent.
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y libglvnd0 libgl1 libegl1 libvulkan1 vulkan-tools

export XDG_RUNTIME_DIR=/tmp
XDG_RUNTIME_DIR=/tmp vulkaninfo --summary | grep deviceName
