#!/bin/bash
# chain5.sh (f6): parity + speed of fc256, server tests (Strata engine, fc256, imported Strata file),
# ppl on code written after Strata's release, the RCO expert set on our weights.
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
export BIN=/root/fc/cortiq.f9
SRC=fn/qwen38-flash-next-q4tp.cmf
EX=fastcode-2026-11-09
O1="CMF_QWEN_O1=32,1800,4,12288"
log "$(SRV_ENV=$O1 ./serve_tests.sh fc256 cortiq out/fc256.cmf out/fc256.profile 2>&1 | tr '\n' ' ' | cut -c1-1500)"
log "$(SRV_ENV=$O1 ./serve_tests.sh sc cortiq out/strata_coder_q4tp.cmf none 2>&1 | tr '\n' ' ' | cut -c1-1500)"
for M in fc256:out/fc256.cmf sc:out/strata_coder_q4tp.cmf q4tp:$SRC; do
  log "$(./ppl.sh ${M%%:*}_fresh ${M#*:} data/fresh_code.txt 30000 --windows 10 --window-len 2048 --window-stride 2048 | head -1)"
done
grep -q "UPLOAD_OK $EX/strata_coder_q4tp.cmf " $J && rm -f out/strata_coder_q4tp.cmf && log "strata_coder_q4tp.cmf removed locally (on HF)"
log "defrag RCO set: $($BIN moe-defrag $SRC --stats hot/rco_kept.profile --keep 256 --output out/fc256rco.cmf 2>&1 | tail -3 | tr '\n' ' ')"
log "$(./ppl.sh rco_es out/fc256rco.cmf data/expert_store_v089.rs 1536 | head -1)"
log "$(./ppl.sh rco_fresh out/fc256rco.cmf data/fresh_code.txt 30000 --windows 10 --window-len 2048 --window-stride 2048 | head -1)"
rm -f out/fc256rco.cmf
log CHAIN5_DONE
