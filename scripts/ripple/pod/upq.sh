#!/bin/bash
# upq.sh: after the measurements and the eight-bit upload — the two low-bit files and their drafter
# sidecars into the private Hub repo, verified by size and sha256. HF_TOKEN comes in the environment.
cd /root/rip
timeout 14400 bash -c "until grep -q CARDQ5_DONE logs/cardq5.out && grep -q UPLOAD_ /root/rip/logs/up_q8.out; do sleep 30; done"
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 scripts/hf_upload.py infosave/Ripple-65B-Preview-cmf \
  out/ripple-65b-q4kv8.cmf ripple-65b-q4tp.cmf \
  out/ripple-65b-q4kv8.dflash.cmf ripple-65b-q4tp.dflash.cmf \
  out/ripple-65b-q2hot5kv.cmf ripple-65b-q2tp.cmf \
  out/ripple-65b-q2hot5kv.dflash.cmf ripple-65b-q2tp.dflash.cmf
echo UPQ_DONE
