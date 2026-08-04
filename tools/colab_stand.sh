#!/bin/bash
# One-shot recovery of the DSV4 measurement stand on a fresh Colab box.
# Run over the bore.pub tunnel: ssh -p <PORT> root@bore.pub 'bash -s' < tools/colab_stand.sh
# Then: scp the source tarball to /root/cmf-src.tgz (or let the git clone below serve),
# and watch /root/setup_build.log + /root/setup_model.log for the two .done flags.
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -q libvulkan1 aria2 >/dev/null 2>&1 || {
  apt-get update -q >/dev/null 2>&1
  apt-get install -y -q libvulkan1 aria2 >/dev/null
}
# The NVIDIA driver is present but its Vulkan/EGL manifests are not (HANDOFF §2).
echo /usr/lib64-nvidia > /etc/ld.so.conf.d/nvidia-colab.conf && ldconfig 2>/dev/null || true
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

cat > /root/setup_build.sh <<'SB'
#!/bin/bash
set -e
curl --proto '=https' -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.85.0 --profile minimal
source $HOME/.cargo/env
mkdir -p /root/cmf
if [ -f /root/cmf-src.tgz ]; then
  tar xzf /root/cmf-src.tgz -C /root/cmf
else
  git clone https://github.com/infosave2007/cmf.git /root/cmf
fi
cd /root/cmf && cargo build --release --features gpu
touch /root/build.done
SB

cat > /root/setup_model.sh <<'SM'
#!/bin/bash
set -e
# parts-q2tp-v3: the draft stack's expert inputs are q2tp IN THE FILE —
# no runtime requant, no surgery, any VRAM size.
BASE=https://huggingface.co/infosave/DeepSeek-V4-Flash-0731-cmf/resolve/main/parts-q2tp-v3
: > /content/dsv4-q2tp.cmf
for i in $(seq 0 14); do
  for try in 1 2 3; do
    aria2c -x8 -s8 -q --allow-overwrite=true -d /content -o tmp "$BASE/$(printf part_%03d $i)" && break
    echo "retry part $i (try $try)"; sleep 5
  done
  cat /content/tmp >> /content/dsv4-q2tp.cmf && rm /content/tmp
  echo "part $i done $(stat -c%s /content/dsv4-q2tp.cmf)"
done
dd if=/content/dsv4-q2tp.cmf of=/dev/null bs=64M
touch /root/model.done
SM
chmod +x /root/setup_build.sh /root/setup_model.sh
nohup /root/setup_build.sh > /root/setup_build.log 2>&1 &
nohup /root/setup_model.sh > /root/setup_model.log 2>&1 &
echo "LAUNCHED: build + model (watch /root/*.log, flags /root/build.done /root/model.done)"
nvidia-smi -L 2>&1 | head -1
df -h / | tail -1
