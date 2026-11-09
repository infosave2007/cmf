#!/bin/bash
# chain3.sh (after chain2): Strata Coder -> CMF import + its ppl/speed; server tests of Strata's engine, fc256 and the
# imported file (speed by context, HumanEval, aquarium); the RCO expert set on our q4tp weights (ppl only).
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 720); do grep -q CHAIN2_DONE $J && break; sleep 20; done
export BIN=/root/fc/cortiq.f4
SRC=fn/qwen38-flash-next-q4tp.cmf
EX=fastcode-2026-11-09
uploaded() { grep -q "UPLOAD_OK $EX/$1 " $J; }
uploaded fc128.cmf && rm -f out/fc128.cmf && log "fc128.cmf removed locally (on HF)"
log "import: $($BIN import-gguf strata/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf --output out/strata_coder_q4tp.cmf --template $SRC --alloc strata/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.rco-allocation.txt 2>&1 | tee logs/import_strata.log | grep -a -E "^  [a-z_]+ +n=|import:|rror|wrote" | tr '\n' ' ' | cut -c1-1500)"
if [ -f out/strata_coder_q4tp.cmf ]; then
  log "$(./upload.sh out/strata_coder_q4tp.cmf $EX/strata_coder_q4tp.cmf 2>&1 | tail -1)"
  log "$(./ppl.sh sc_es out/strata_coder_q4tp.cmf data/expert_store_v089.rs 1536 | head -1)"
  log "$(./ppl.sh sc_ev out/strata_coder_q4tp.cmf data/eval_code.txt 20000 --windows 8 --window-len 2048 --window-stride 2048 | head -1)"
  for C in 1000 4000; do log "$(./bench.sh sc_c$C out/strata_coder_q4tp.cmf $C)"; done
fi
log "$(./serve_tests.sh strata strata 2>&1 | tr '\n' ' ' | cut -c1-900)"
log "$(./serve_tests.sh fc256 cortiq out/fc256.cmf out/fc256.profile 2>&1 | tr '\n' ' ' | cut -c1-900)"
[ -f out/strata_coder_q4tp.cmf ] && log "$(./serve_tests.sh sc cortiq out/strata_coder_q4tp.cmf none 2>&1 | tr '\n' ' ' | cut -c1-900)"
uploaded strata_coder_q4tp.cmf && rm -f out/strata_coder_q4tp.cmf && log "strata_coder_q4tp.cmf removed locally (on HF)"
log "defrag RCO set: $($BIN moe-defrag $SRC --stats hot/rco_kept.profile --keep 256 --output out/fc256rco.cmf 2>&1 | tail -3 | tr '\n' ' ')"
log "$(./ppl.sh rco_es out/fc256rco.cmf data/expert_store_v089.rs 1536 | head -1)"
log "$(./ppl.sh rco_ev out/fc256rco.cmf data/eval_code.txt 20000 --windows 8 --window-len 2048 --window-stride 2048 | head -1)"
rm -f out/fc256rco.cmf
log CHAIN3_DONE
