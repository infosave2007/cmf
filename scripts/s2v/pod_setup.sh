#!/bin/bash
# Pod bootstrap for the Kandinsky S2V run: Vulkan userspace, Rust, the reference
# environment, the weights. Runs from /workspace/s2v; each part logs on its own.
cd /workspace/s2v
mkdir -p out cmf assets
(
  set -x
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 libvulkan1 build-essential pkg-config git cmake ffmpeg libsndfile1 >/dev/null
  export XDG_RUNTIME_DIR=/tmp
  vulkaninfo --summary | grep -E "deviceName|driverVersion|apiVersion" | head
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
  python3 -m venv venv
  . venv/bin/activate
  pip -q install -U pip "huggingface_hub[hf_xet]" safetensors numpy einops librosa soundfile accelerate sentencepiece protobuf pillow
  pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || pip -q install torch torchvision
  pip -q install "git+https://github.com/huggingface/transformers" "git+https://github.com/huggingface/diffusers"
  python -c "import torch,diffusers,transformers;print(torch.__version__,torch.cuda.is_available(),diffusers.__version__,transformers.__version__)"
  echo SETUP_DONE
) > setup.log 2>&1 &
(
  while ! [ -x venv/bin/hf ]; do sleep 3; done
  . venv/bin/activate
  hf download kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers --local-dir hf
  echo FETCH_DONE
) > fetch.log 2>&1 &
(
  while ! grep -q SETUP_DONE setup.log 2>/dev/null; do sleep 3; done
  . ~/.cargo/env
  cd src
  export CARGO_TARGET_DIR=/workspace/s2v/target
  timeout 1500 cargo build --release -p cortiq-cli --config profile.release.lto=false --config profile.release.codegen-units=16
  echo BUILD_EXIT $?
) > build.log 2>&1 &
wait
