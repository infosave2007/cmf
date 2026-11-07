#!/bin/bash
# fit.sh FILE N CHARS: exact-row weights for CherryPit (42 full layers x 2 KV heads x 2 = 168 rank features)
# over N prompts with a planted fact, sealed after 4096 rows; writes keep_cherrypit.bin.
cd /root/cherrypit
export XDG_RUNTIME_DIR=/tmp CMF_O1_PREFILL=4096 CMF_O1_M=32
venv/bin/python scripts/q27_feat.py ./cortiq $1 hf/CherryPit-1.2-2B/tokenizer.json /root/rip/data/wiki.test.raw keep_cherrypit.bin ${2:-20} ${3:-21000} 168
echo FIT_DONE
