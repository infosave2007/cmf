#!/bin/bash
# chain3c.sh: takes over from chain3b after its sc_ev ppl: a stage-cost sweep of fc256 (CMF_Q4_SKIP, timings only),
# the imported Strata file's speed, server tests (Strata engine, fc256, imported file), the RCO set on our weights.
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
for p in $(pgrep -f "chain3b[.]sh"); do kill $p; done
for i in $(seq 1 120); do pgrep -f "ppl out/strata_coder_q4tp[.]cmf" > /dev/null || break; sleep 5; done
grep -q "sc_ev:" $J || log "sc_ev: $(grep -a "PPL" logs/ppl_sc_ev.log | tail -1 | cut -c1-120)"
export BIN=/root/fc/cortiq.f5
SRC=fn/qwen38-flash-next-q4tp.cmf
EX=fastcode-2026-11-09
# stage costs: decode (no drafts) and 8-token prompt frames, one stage left out at a time
for S in none experts hc gdn qsa route ple; do
  log "skip=$S $(TOK=64 ./bench.sh skip_$S out/fc256.cmf 1000 CMF_QWEN_PROFILE=out/fc256.profile CMF_QWEN_MTP=0 CMF_Q4_SKIP=$S | sed 's/.*ctx=1000://' | cut -c1-200)"
done
for C in 1000 4000; do log "$(./bench.sh sc_c$C out/strata_coder_q4tp.cmf $C)"; done
log "$(./serve_tests.sh strata strata 2>&1 | tr '\n' ' ' | cut -c1-1200)"
log "$(./serve_tests.sh fc256 cortiq out/fc256.cmf out/fc256.profile 2>&1 | tr '\n' ' ' | cut -c1-1200)"
log "$(./serve_tests.sh sc cortiq out/strata_coder_q4tp.cmf none 2>&1 | tr '\n' ' ' | cut -c1-1200)"
grep -q "UPLOAD_OK $EX/strata_coder_q4tp.cmf " $J && rm -f out/strata_coder_q4tp.cmf && log "strata_coder_q4tp.cmf removed locally (on HF)"
log "defrag RCO set: $($BIN moe-defrag $SRC --stats hot/rco_kept.profile --keep 256 --output out/fc256rco.cmf 2>&1 | tail -3 | tr '\n' ' ')"
log "$(./ppl.sh rco_es out/fc256rco.cmf data/expert_store_v089.rs 1536 | head -1)"
log "$(./ppl.sh rco_ev out/fc256rco.cmf data/eval_code.txt 20000 --windows 8 --window-len 2048 --window-stride 2048 | head -1)"
rm -f out/fc256rco.cmf
log CHAIN3_DONE
