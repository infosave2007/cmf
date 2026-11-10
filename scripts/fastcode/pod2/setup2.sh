#!/bin/bash
# setup2.sh: the 5090 pod for the Fast Code VRAM-tier campaign -> /root/fc.
# Tools, cortiq (branch fast-code-v2), the published fcaq256 (private repo), the q2tp-v2 source for a 2-bit cut,
# Strata engine 0.1.58 + Strata Coder IQ1_M. HF_TOKEN arrives on stdin, never on disk.
set -uo pipefail
mkdir -p /root/fc/{fn,logs,data,out,hot,res,strata/engine,strata/control} && cd /root/fc
read -r HF_TOKEN; export HF_TOKEN
export DEBIAN_FRONTEND=noninteractive XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a /root/fc/JOURNAL.md; }
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq --no-install-recommends vulkan-tools libglvnd0 libegl1 libgl1 libglx0 libvulkan1 build-essential pkg-config libssl-dev python3-pip python3-venv git curl ca-certificates zstd jq >/dev/null 2>&1
log "APT_DONE rc=$? $(vulkaninfo --summary 2>/dev/null | grep -m1 deviceName | tr -s ' ')"
python3 -m venv venv && venv/bin/pip install -q -U "huggingface_hub[cli]" hf_xet >/dev/null 2>&1
log "VENV_DONE $(venv/bin/hf version 2>&1 | tail -1)"
HF=/root/fc/venv/bin/hf
( cd out && $HF download infosave/Qwen3.8-Flash-Next-Fast-Code-cmf qwen38-flash-next-fast-code.cmf fast-code.profile qwen38-flash-next-fast-code.cmf.sha256 --local-dir . ; echo "DL_FC_DONE rc=$?" ) > logs/dl_fc.log 2>&1 &
( cd fn && $HF download infosave/Qwen3.8-Flash-Next-cmf qwen38-flash-next-q2tp-v2.cmf qwen38-flash-next-q2tp-v2.mtp.cmf --local-dir . ; echo "DL_Q2_DONE rc=$?" ) > logs/dl_q2.log 2>&1 &
( cd strata && $HF download ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF --include "IQ1_M/*" "tensor-allocation/*" --local-dir . ; echo "DL_ST_DONE rc=$?" ) > logs/dl_strata.log 2>&1 &
# Strata engine + control (release v0.2.12 = engine 0.1.58, CUDA 13 build), checksums from the release
( cd strata/engine && curl -sSL -o eng.tgz https://github.com/Niko1221/Strata/releases/download/v0.2.12/strata-engine-0.1.58-linux-x86_64-cuda-cuda13.tar.gz \
  && curl -sSL -o SHA256SUMS.txt https://github.com/Niko1221/Strata/releases/download/v0.2.12/SHA256SUMS.txt \
  && tar xzf eng.tgz && cd ../control && curl -sSL -o install.sh https://github.com/Niko1221/Strata/releases/download/v0.2.12/install.sh \
  && curl -sSL -o ctl.tgz https://github.com/Niko1221/Strata/releases/download/v0.2.12/strata-control-0.1.58-linux-x86_64.tar.gz \
  && tar xzf ctl.tgz; cd /root/fc/strata/engine; for f in eng.tgz ../control/ctl.tgz; do s=$(sha256sum $f | cut -d' ' -f1); grep -q "$s" SHA256SUMS.txt && echo "SHA_OK $f" || echo "SHA_BAD $f"; done; echo STRATA_BIN_DONE ) > logs/strata_bin.log 2>&1 &
[ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
. "$HOME/.cargo/env"
git clone -q -b fast-code-v2 https://github.com/infosave2007/cmf.git /root/fc/cmf && cd /root/fc/cmf && log "clone $(git log --oneline -1)"
CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 nice -n 10 cargo build --release -p cortiq-cli --features gpu > /root/fc/logs/build_v1.log 2>&1
cp target/release/cortiq /root/fc/cortiq.v1 && log "BUILD_DONE $(/root/fc/cortiq.v1 --version)" || log "BUILD_FAILED $(grep -m3 -E '^error' /root/fc/logs/build_v1.log)"
cd /root/fc
# Strata install needs the GGUF; it fetches the bf16 MTP layer itself
for i in $(seq 1 720); do grep -q DL_ST_DONE logs/dl_strata.log 2>/dev/null && grep -q STRATA_BIN_DONE logs/strata_bin.log && break; sleep 10; done
log "strata files: $(grep -E 'SHA_|DONE' logs/strata_bin.log | tr '\n' ' ') $(tail -1 logs/dl_strata.log)"
( cd strata/control && STRATA_DATA_DIR=/root/fc/strata/data ./bin/strata install --family coder --model IQ1_M --gguf-dir /root/fc/strata/IQ1_M --data-dir /root/fc/strata/data --yes --no-start --vision no --context 65536 --prebuilt /root/fc/strata/engine/eng.tgz --port 8090; echo "PREP_RC=$?" ) > logs/strata_prep.log 2>&1
log "strata install: $(tail -1 logs/strata_prep.log)"
wait
log "downloads: $(tail -1 logs/dl_fc.log) $(tail -1 logs/dl_q2.log) $(df -h /root | tail -1 | awk '{print $4}') free"
log SETUP2_DONE
