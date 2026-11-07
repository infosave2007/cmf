#!/bin/bash
# fit.sh NAME [N] [CHARS]: the exact-row weights of Ripple's far field (16 full-attention layers,
# 8 KV heads, 256 rank features), fitted over N prompts with a planted fact; out/NAME.cmf is read
# with a seal after 4096 rows, on the batch graph. Writes keep_ripple.bin.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp CMF_O1_PREFILL=4096 CMF_O1_M=32
python3 scripts/q27_feat.py ./cortiq out/$1.cmf src/tokenizer.json data/wiki.test.raw keep_ripple.bin ${2:-15} ${3:-21000} 256
echo FIT_DONE
