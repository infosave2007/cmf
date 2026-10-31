#!/bin/bash
# Upload the CMF files to private infosave/*-cmf repos once the bench ends.
# HF_TOKEN comes from the environment (piped in over stdin by the caller).
set -e
while pgrep -f "four/final_vulkan" >/dev/null; do sleep 10; done
O=/root/four/out
cd $O
ln -f mage-q8_2f.cmf mage-vl-cowork-q8_2f.cmf
ln -f mage-q4tp.cmf mage-vl-cowork-q4tp.cmf
ln -f sn-q4tp.cmf sensenova-loop-9b-q4tp.cmf
ln -f sn-q8_2f.cmf sensenova-loop-9b-q8_2f.cmf
up() {  # repo file...
  local repo=$1; shift
  # private from the first byte (an `hf upload` into a missing repo creates it public)
  python3 -c "from huggingface_hub import HfApi; HfApi().create_repo('$repo', private=True, exist_ok=True)"
  for f in "$@"; do
    [ -f "$f.sha256" ] || sha256sum "$f" > "$f.sha256"
    echo "== upload $repo $f $(date)"
    hf upload "$repo" "$f" "$f" >/dev/null
    hf upload "$repo" "$f.sha256" "$f.sha256" >/dev/null
  done
}
up infosave/primal_vae-cmf primal_vae_2x2.cmf primal_vae_2x1.cmf
up infosave/Mage-VL-Cowork-cmf mage-vl-cowork-q8_2f.cmf mage-vl-cowork-q4tp.cmf
up infosave/SenseNova-Loop-9B-Preview-cmf sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-q8_2f.cmf
echo "== uploads done $(date)"
