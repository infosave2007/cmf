#!/bin/bash
# ref.sh: the references on the card. A small random model with every feature of the architecture
# (tiny/, ref/tiny.npz over 1400 tokens: past its window of 320 and its ring of 1024), then the published BF16 weights over the first 2048 tokens
# of wikitext-2 test (ref/wiki2048.npz), layers streamed from the safetensors.
set -uo pipefail
cd /root/rip
pip install -q --break-system-packages accelerate 2>&1 | grep -v WARNING | tail -1
[ -f tiny/model.safetensors ] || python3 scripts/ripple_ref.py tiny src tiny
[ -f ref/tiny.npz ] || python3 scripts/ripple_ref.py ppl tiny data/wiki.test.raw 1400 ref/tiny.npz
[ -f ref/wiki2048.npz ] || python3 scripts/ripple_ref.py ppl src data/wiki.test.raw 2048 ref/wiki2048.npz
echo REF_DONE
