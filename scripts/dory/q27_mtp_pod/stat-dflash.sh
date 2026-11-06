#!/bin/bash
# stat-dflash.sh: the DFlash block on a card whose memory holds it beside the Turbo file. The sidecar from the
# published drafter (ext/dflash2), then three prompts with every round taken (CMF_GRAPH_SPEC_PIN=1): the chain
# pinned 3, 5 and 7 deep, and the block pinned. A round's draft, verify and commit times are in st-dflash.log.
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
M=m/qwen38-27b-turbo.cmf
[ -s m/qwen38-27b-turbo.dflash.cmf ] ||
  timeout 900 ./cortiq convert --model ext/dflash2 --output $M --dflash-sidecar --quant q4tp 2>&1 | tail -1 | cut -c1-200
ls -la m/qwen38-27b-turbo.dflash.cmf
rm -f st-dflash.log st-dflash.done
(while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done > st-dflash.smi) &
smi=$!
CMF_GRAPH_SPEC_PIN=1 ./q27_dflash_ab.sh ./cortiq $M p st-dflash.log 0 220 code edit prose -- k3 k5 k7 dflash
kill $smi
echo "vram peak $(sort -n st-dflash.smi | tail -1) MiB" >> st-dflash.log
echo DONE > st-dflash.done
