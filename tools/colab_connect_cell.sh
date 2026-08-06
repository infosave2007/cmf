#!/bin/bash
# The Colab cell that opens the stand: sshd + bore tunnel + a usable Vulkan.
# Paste it into a notebook cell under `%%bash`; it prints the bore.pub port.
#
# The Vulkan block belongs HERE rather than in colab_stand.sh, which runs
# over the tunnel this cell creates — by the time that script lands, a
# `cortiq gpu` typed by hand has already reported llvmpipe and sent
# somebody looking for a driver problem that is really a missing manifest.

apt-get install -y -qq --no-install-recommends openssh-server >/dev/null 2>&1
ssh-keygen -A
mkdir -p /run/sshd /root/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDxBBWi/aiu0hIeffkfY6VpCI28rSPgjKvCqEtJUEw24 serverpsy" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
sed -i 's/#*PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
pkill sshd 2>/dev/null; /usr/sbin/sshd -e

# ── Vulkan ───────────────────────────────────────────────────────────
# Colab ships the NVIDIA driver in /usr/lib64-nvidia but neither the
# loader, nor the ICD manifest, nor that directory on the library path.
# All three are needed, and the api_version must read 1.3.0: a higher
# one (1.3.289, say) leaves the loader unable to pull `vkCreateInstance`
# out of the ICD, and it falls back to llvmpipe without saying why.
apt-get install -y -qq libvulkan1 >/dev/null 2>&1 || {
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq libvulkan1 >/dev/null 2>&1
}
echo /usr/lib64-nvidia > /etc/ld.so.conf.d/nvidia-colab.conf && ldconfig 2>/dev/null
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"/usr/lib64-nvidia/libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
  > /usr/share/vulkan/icd.d/nvidia_icd.json
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# ── tunnel ───────────────────────────────────────────────────────────
wget -qO /tmp/bore.tar.gz https://github.com/ekzhang/bore/releases/download/v0.5.3/bore-v0.5.3-x86_64-unknown-linux-musl.tar.gz
tar xzf /tmp/bore.tar.gz -C /usr/local/bin && chmod +x /usr/local/bin/bore
cat > /root/tunnel.sh <<'EOF'
#!/bin/bash
while true; do
  bore local 2222 --to bore.pub >> /tmp/bore.log 2>&1
  sleep 5
done
EOF
chmod +x /root/tunnel.sh
setsid nohup /root/tunnel.sh > /dev/null 2>&1 < /dev/null &
sleep 8
grep -o 'bore.pub:[0-9]*' /tmp/bore.log | tail -1

# Verify before trusting a GPU measurement from this box:
#   cortiq gpu   # must name the card, not "llvmpipe"
