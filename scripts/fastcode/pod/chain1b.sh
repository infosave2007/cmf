#!/bin/bash
# chain1.sh: code profile -> Fast Code cuts K=256/128 -> code ppl + speed vs the q4tp source
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 240); do grep -q PPL_DONE logs/run_prof_code.txt 2>/dev/null && grep -q BUILD_DONE logs/build_f3.log 2>/dev/null && break; sleep 15; done
export BIN=/root/fc/cortiq.f3
log "profile: $(cat logs/run_prof_code.txt | head -1)"
cp hot/code_q4tp.profile hot/code_q4tp.final.profile
SRC=fn/qwen38-flash-next-q4tp.cmf
for K in 256 128; do
  log "defrag K=$K: $($BIN moe-defrag $SRC --stats hot/code_q4tp.final.profile --keep $K --merge fn/qwen38-flash-next-q4tp.mtp.cmf --output out/fc$K.cmf 2>&1 | tail -4 | tr '\n' ' ')"
  log "$(./ppl.sh fc${K}_es out/fc$K.cmf data/expert_store_v089.rs 1536 CMF_QWEN_PROFILE=out/fc$K.profile | head -1)"
  log "$(./ppl.sh fc${K}_ev out/fc$K.cmf data/eval_code.txt 20000 CMF_QWEN_PROFILE=out/fc$K.profile --windows 8 --window-len 2048 --window-stride 2048 | head -1)"
  for C in 1000 4000; do log "$(./bench.sh fc${K}_c$C out/fc$K.cmf $C CMF_QWEN_PROFILE=out/fc$K.profile)"; log "$(./bench.sh fc${K}_c${C}_old out/fc$K.cmf $C CMF_QWEN_PROFILE=out/fc$K.profile CMF_QWEN_RESIDENT=0)"; done
  [ $K = 256 ] && { log "$(./ppl.sh q4tp_ev $SRC data/eval_code.txt 20000 CMF_QWEN_PROFILE=fn/flashnext-q4tp.profile --windows 8 --window-len 2048 --window-stride 2048 | head -1)"; for C in 1000 4000; do log "$(./bench.sh q4tp_c$C $SRC $C CMF_QWEN_PROFILE=fn/flashnext-q4tp.profile)"; done; }
done
log CHAIN1_DONE
