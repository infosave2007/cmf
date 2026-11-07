#!/bin/bash
# ppl.sh NAME [TOKENS] [ENV=VALUE ...]: code perplexity of $FILE on data/expert_store_v089.rs (default 1536 tokens).
set -uo pipefail
cd /root/rip && . fnx/env.sh
n=$1; t=${2:-1536}; shift; shift 2>/dev/null || true
t0=$(date +%s)
env "$@" $BIN ppl $FILE -f data/expert_store_v089.rs --tokens $t > logs/ppl_$n.log 2>&1
echo "$n: $(grep -a "PPL" logs/ppl_$n.log | tail -1 | cut -c1-120) | $(( $(date +%s) - t0 )) s | $(grep -a -c -i "declin\|refus\|panic\|error" logs/ppl_$n.log) decline/error lines"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo PPL_DONE
