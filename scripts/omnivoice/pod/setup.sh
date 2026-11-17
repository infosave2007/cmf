#!/bin/bash
# OmniVoice pod: tools (bootstrap.sh nomodel), the reference stack (torch cu128 +
# omnivoice) in the venv, the k2-fsa/OmniVoice snapshot, then a release build.
set -u
cd /root/h3i || exit 1
bash scripts/bootstrap.sh nomodel || { echo SETUP_FAILED_BOOTSTRAP; exit 1; }
venv/bin/pip -q install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 > logs/pip_torch.log 2>&1 || { echo SETUP_FAILED_TORCH; tail -5 logs/pip_torch.log; }
venv/bin/pip -q install omnivoice soundfile > logs/pip_ov.log 2>&1 || { echo SETUP_FAILED_OMNIVOICE; tail -5 logs/pip_ov.log; }
mkdir -p ov/src
timeout 3600 venv/bin/hf download k2-fsa/OmniVoice --local-dir ov/src > logs/dl_ov.log 2>&1 && echo OV_DOWNLOAD_DONE || echo OV_DOWNLOAD_FAILED
du -sh ov/src
bash scripts/build.sh release
echo SETUP_DONE
