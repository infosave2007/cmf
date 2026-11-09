#!/bin/bash
# chain2.sh: code profile -> fc256 (+backup) -> ppl/speed vs q4tp -> fc128 (+backup) -> Strata Coder import -> its ppl/speed
set -uo pipefail
cd /root/fc; export XDG_RUNTIME_DIR=/tmp
J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 240); do grep -q BUILD_DONE logs/build_f4.log 2>/dev/null && break; sleep 15; done
export BIN=/root/fc/cortiq.f4


SRC=fn/qwen38-flash-next-q4tp.cmf
EX=fastcode-2026-11-09
cut() { log "defrag K=$1: $($BIN moe-defrag $SRC --stats hot/code_q4tp.final.profile --keep $1 --merge fn/qwen38-flash-next-q4tp.mtp.cmf --output out/fc$1.cmf 2>&1 | tail -4 | tr '\n' ' ')"; }
measure() { # NAME FILE PROFILE
  log "$(./ppl.sh $1_es $2 data/expert_store_v089.rs 1536 CMF_QWEN_PROFILE=$3 | head -1)"
  log "$(./ppl.sh $1_ev $2 data/eval_code.txt 20000 CMF_QWEN_PROFILE=$3 --windows 8 --window-len 2048 --window-stride 2048 | head -1)"
  for C in 1000 4000; do log "$(./bench.sh $1_c$C $2 $C CMF_QWEN_PROFILE=$3)"; done
}
[ -f out/fc256.cmf ] || cut 256
grep -q "UPLOAD_OK $EX/fc256.cmf " $J || log "$(./upload.sh out/fc256.cmf $EX/fc256.cmf 2>&1 | tail -1)"
measure fc256 out/fc256.cmf out/fc256.profile
log "$(./bench.sh fc256_c1000_old out/fc256.cmf 1000 CMF_QWEN_PROFILE=out/fc256.profile CMF_QWEN_RESIDENT=0)"
log "$(./bench.sh fc256_c4000_nopipe out/fc256.cmf 4000 CMF_QWEN_PROFILE=out/fc256.profile CMF_QWEN_PIPELINE=0)"
measure q4tp $SRC fn/flashnext-q4tp.profile
cut 128
log "$(./upload.sh out/fc128.cmf $EX/fc128.cmf 2>&1 | tail -1)"
measure fc128 out/fc128.cmf out/fc128.profile
log CHAIN2_DONE
