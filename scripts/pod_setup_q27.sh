#!/bin/bash
# Bring up a fresh RunPod RTX PRO 6000 for scripts/compress_q27_14b_pod.sh.
# Expects in $ROOT: q27-code-t27.tdata.tgz (teacher data, from pack_q27_pod.sh). cortiq is the release
# binary of $TAG (no build on the pod); with cmf-changes.tgz present (code newer than the tag) it is
# built from the tag plus those files instead. The checkpoint download starts first and runs meanwhile;
# never kill `hf download` (xet restarts from zero).
set -euo pipefail
ROOT=${ROOT:-/root/q27}
TAG=${TAG:-v0.8.42}
mkdir -p "$ROOT/hf"
cd "$ROOT"

# Vulkan on RunPod: the GLVND userspace (the render-node permissions are a red herring)
apt-get update -qq
apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 git curl build-essential pkg-config > /dev/null
export XDG_RUNTIME_DIR=/tmp
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|apiVersion" | head -4

# checkpoint (text model; the vision tower is dropped by convert anyway)
pip install -q -U "huggingface_hub[hf_xet]"
nohup hf download Qwen/Qwen3.8-27B --local-dir "$ROOT/hf/Qwen3.8-27B" > "$ROOT/hf-download.log" 2>&1 &
HFPID=$!

if [ -f cmf-changes.tgz ]; then
  [ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal > /dev/null
  . "$HOME/.cargo/env"
  [ -d cmf ] || git clone -q --depth 1 --branch "$TAG" https://github.com/infosave2007/cmf.git cmf
  tar xzmf cmf-changes.tgz -C cmf
  find cmf -name "._*" -delete
  (cd cmf && CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo build --release -p cortiq-cli 2>&1 | grep -E "^error|Finished" -A5 | tail -8)
  cp cmf/target/release/cortiq "$ROOT/cortiq"
  cp cmf/scripts/compress_q27_14b_pod.sh "$ROOT/"
else
  base=https://github.com/infosave2007/cmf/releases/download/$TAG
  curl -sSfL -o cortiq.tgz "$base/cortiq-x86_64-unknown-linux-gnu.tar.gz"
  curl -sSfL -o cortiq.tgz.sha256 "$base/cortiq-x86_64-unknown-linux-gnu.tar.gz.sha256"
  [ "$(sha256sum cortiq.tgz | cut -d' ' -f1)" = "$(cut -d' ' -f1 cortiq.tgz.sha256)" ] || { echo "cortiq.tgz: sha256 mismatch"; exit 3; }
  tar xzf cortiq.tgz && chmod +x cortiq
  curl -sSfL -o compress_q27_14b_pod.sh "https://raw.githubusercontent.com/infosave2007/cmf/$TAG/scripts/compress_q27_14b_pod.sh"
  chmod +x compress_q27_14b_pod.sh
fi
./cortiq --version
[ -d q27-code.tdata ] || { tar xzf q27-code-t27.tdata.tgz && mv q27-code-t27.tdata q27-code.tdata; }

# wait for the checkpoint (progress by the .incomplete files, not du)
while kill -0 "$HFPID" 2> /dev/null; do sleep 30; done
tail -3 "$ROOT/hf-download.log"
ls "$ROOT/hf/Qwen3.8-27B"/*.safetensors | wc -l # 18 shards
ROOT=$ROOT "$ROOT/compress_q27_14b_pod.sh" check
