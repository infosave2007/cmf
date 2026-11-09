#!/bin/bash
# final_bench.sh (after aq_multi): clean speed of Fast Code on f11 — bench by context (exact to 64k), the MTP
# draft depth, the stage costs of a prompt frame, and the server speed test through the same client as Strata
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 1080); do grep -q AQ_MULTI_DONE $J && break; sleep 20; done
export BIN=/root/fc/cortiq.f11
P="CMF_QWEN_PROFILE=out/fc256.profile"
for C in 1000 4000 16000 32000 64000; do log "$(./bench.sh fin_c$C out/fc256.cmf $C $P)"; done
for K in 2 4 5; do log "$(./bench.sh fin_k${K}_c4000 out/fc256.cmf 4000 $P CMF_QWEN_MTP_K=$K)"; done
for S in none experts hc gdn qsa ple; do
  log "fin prefill skip=$S $(TOK=8 ./bench.sh finskip_$S out/fc256.cmf 4000 $P CMF_QWEN_MTP=0 CMF_Q4_SKIP=$S | grep -o "prefill [0-9.]* tok/s") decode $(grep -a -o "Decode: .*tok/s steady" logs/bench_finskip_$S.log | head -1)"
done
log "$(./serve_tests.sh fin_fc256 cortiq out/fc256.cmf out/fc256.profile 2>&1 | tr '\n' ' ' | cut -c1-1500)"
log FINAL_BENCH_DONE
