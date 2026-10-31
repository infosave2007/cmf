#!/bin/bash
# Pod setup for the four-model campaign (SenseNova-Loop, Mage-VL, primal_vae, idol).
# Runs once on a fresh RunPod box; logs to /root/four/logs/setup.log.
set -u
W=/root/four
mkdir -p $W/dl $W/logs $W/out $W/ref
exec >>$W/logs/setup.log 2>&1
echo "== start $(date)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -y -q --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 \
  build-essential pkg-config git curl ca-certificates libssl-dev python3-pip python3-venv rsync time
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverName" | head -4
if [ ! -x ~/.cargo/bin/cargo ]; then curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; fi
. ~/.cargo/env; cargo --version
python3 -m pip install -q --break-system-packages -U pip 2>/dev/null || python3 -m pip install -q -U pip
python3 -m pip install -q --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
  || python3 -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3 -m pip install -q --break-system-packages -U "transformers>=5" accelerate safetensors diffusers pillow \
  "huggingface_hub[hf_xet]" einops sentencepiece av qwen-vl-utils 2>&1 | tail -3
python3 -c "import torch,transformers,diffusers;print('torch',torch.__version__,torch.cuda.is_available(),'tf',transformers.__version__,'diffusers',diffusers.__version__)"
cd $W/dl
for m in ostris/primal_vae microsoft/Mage-VL-Cowork sensenova/SenseNova-Loop-9B-Preview; do
  echo "== dl $m $(date)"
  hf download $m --local-dir $W/dl/${m##*/} --exclude "examples/*" "assets/*" || echo "DL FAIL $m"
done
du -sh $W/dl/*
echo "== done $(date)"
