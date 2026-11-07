#!/bin/bash
# tiny.sh [ENV=VALUE ...]: the small random model as an f16 CMF against its PyTorch numbers,
# position by position (ref/tiny.npz from ref.sh). Extra arguments are the run's environment,
# e.g. `tiny.sh CMF_GPU=0` for the host walk.
set -uo pipefail
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
[ -f tiny.cmf ] && [ tiny.cmf -nt cortiq ] || ./cortiq convert --model tiny --output tiny.cmf --quant f16 2>&1 | tail -3
env "$@" CMF_PPL_TRACE=1 ./cortiq ppl tiny.cmf -f data/wiki.test.raw --tokens 1400 > logs/tiny_ppl.log 2>&1
grep -a -E "PPL|rror|panic|declin" logs/tiny_ppl.log | head -5
python3 -I scripts/ppl_cmp.py ref/tiny.npz logs/tiny_ppl.log
