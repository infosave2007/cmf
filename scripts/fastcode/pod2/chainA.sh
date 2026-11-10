#!/bin/bash
# chainA.sh: baselines on the 5090. fcq2 = the published code+task expert set cut from q2tp-v2 (fits a 32 GB card);
# its ppl, quality and speed by card size; Strata at the same card sizes; fcaq256 (q4tp) by card size.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
B=/root/fc/cortiq.v1
for i in $(seq 1 360); do grep -q DL_FC_DONE logs/dl_fc.log && grep -q PREP_RC logs/strata_prep.log && break; sleep 10; done
log "chainA start: $(tail -1 logs/dl_fc.log) $(tail -1 logs/strata_prep.log) fcaq256 sha $(cd out && sha256sum -c qwen38-flash-next-fast-code.cmf.sha256 2>&1 | tail -1)"
ln -sf qwen38-flash-next-fast-code.cmf out/fcaq256.cmf; ln -sf fast-code.profile out/fcaq256.profile
log "defrag fcq2 K=256: $(timeout 1200 $B moe-defrag fn/qwen38-flash-next-q2tp-v2.cmf --stats hot/code_aq.profile --keep 256 --merge fn/qwen38-flash-next-q2tp-v2.mtp.cmf --output out/fcq2.cmf 2>&1 | tail -3 | tr '\n' ' ')"
[ -f out/fcq2.cmf ] && log "fcq2 size $(stat -c %s out/fcq2.cmf)"
log "$(BIN=$B timeout 1200 ./ppl.sh fcq2_rust out/fcq2.cmf data/expert_store_v089.rs 1536 CMF_QWEN_PROFILE=out/fcq2.profile | head -1)"
log "$(BIN=$B timeout 1800 ./ppl.sh fcq2_fresh out/fcq2.cmf data/fresh_code.txt 8192 CMF_QWEN_PROFILE=out/fcq2.profile | head -1)"
run() { while read -r l; do log "$l"; done < <(timeout 3600 ./tier.sh "$@" 2>&1); }
run fcq2_c32 cortiq 32 1000,4000,16000,32000 out/fcq2.cmf out/fcq2.profile
run st_c32 strata 32 1000,4000,16000,32000
for c in 24 16 12; do
  run st_c$c strata $c 1000,4000,16000
  run fcq2_c$c cortiq $c 1000,4000 out/fcq2.cmf out/fcq2.profile
done
for c in 32 24 16 12; do run fcaq_c$c cortiq $c 1000,4000 out/fcaq256.cmf out/fcaq256.profile; done
log CHAINA_DONE
