#!/bin/bash
# fix_f5.sh: hold chain3b, build f5 (rollback guard, far-field chaining) once the q4tp benches are done, swap it in
# as cortiq.f4 for chain2b's remaining fc128 runs, then redo the fc256 speed rows and start chain3b.
cd /root/fc; J=/root/fc/JOURNAL.md
log() { echo "$(date +%T) $*" | tee -a $J; }
for p in $(pgrep -f "chain3b[.]sh"); do kill $p; done
for i in $(seq 1 240); do grep -q "q4tp_c4000" $J && break; sleep 10; done
./build.sh f5 > logs/build_f5.log 2>&1
grep -q BUILD_DONE logs/build_f5.log || { log "BUILD f5 FAILED"; exit 1; }
cp cortiq.f5 cortiq.f4.tmp && mv -f cortiq.f4.tmp cortiq.f4
log "f5 built (bench rollback guard; far field chains in resident mode) and swapped in as cortiq.f4"
for i in $(seq 1 720); do grep -q CHAIN2_DONE $J && break; sleep 20; done
export BIN=/root/fc/cortiq.f5
for C in 1000 4000; do log "$(./bench.sh fc256_c$C out/fc256.cmf $C CMF_QWEN_PROFILE=out/fc256.profile)"; done
log "$(./bench.sh fc256_c1000_old out/fc256.cmf 1000 CMF_QWEN_PROFILE=out/fc256.profile CMF_QWEN_RESIDENT=0)"
log "$(./bench.sh fc256_c4000_nopipe out/fc256.cmf 4000 CMF_QWEN_PROFILE=out/fc256.profile CMF_QWEN_PIPELINE=0)"
sed -i 's#export BIN=/root/fc/cortiq.f4#export BIN=/root/fc/cortiq.f5#' chain3b.sh
exec ./chain3b.sh
