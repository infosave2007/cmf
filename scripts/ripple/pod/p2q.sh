#!/bin/bash
# p2q.sh (the second pod, an RTX PRO 6000): once the q4tp file has arrived and the eight-bit
# perplexity pass is over — the drafter sidecar, a perplexity check of the copied file, the drafter
# decode runs, device memory with and without the drafter, the fact check.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
S=34605573081
timeout 7200 bash -c "until [ \"\$(stat -c %s out/ripple-65b-q4kv8.cmf 2>/dev/null)\" = $S ]; do sleep 15; done"; sleep 30
CMF_GPU=0 ./cortiq convert --model hfdl/dflash2 --output out/ripple-65b-q4kv8.cmf --dflash-sidecar --quant q4tp 2>&1 | grep -a "wrote\|rror" | cut -c1-200
ls -la out/
for p in code prose; do
  FILE=out/ripple-65b-q4kv8.cmf bash scripts/dflash.sh plain $p 160
  FILE=out/ripple-65b-q4kv8.cmf bash scripts/dflash.sh k7 $p 160 CMF_GRAPH_SPEC_K=7 CMF_DFLASH_TIME=1
  FILE=out/ripple-65b-q4kv8.cmf bash scripts/dflash.sh auto $p 160 CMF_DFLASH_TIME=1
done
bash scripts/vram_peak.sh "dflash q4tp code" logs/final_dflash.log /root/rip/sp -- env CMF_DFLASH=1 RUST_LOG=info CMF_GRAPH_SPEC_TIME=1 ./cortiq run out/ripple-65b-q4kv8.cmf --prompt "$(cat data/code.txt)" --greedy --max-tokens 160
bash scripts/vram_peak.sh "plain q4tp code" logs/final_dflash.log /root/rip/sp -- env CMF_GRAPH_SPEC=0 ./cortiq run out/ripple-65b-q4kv8.cmf --prompt "$(cat data/code.txt)" --greedy --max-tokens 160
bash scripts/quality.sh out/ripple-65b-q4kv8.cmf logs/quality_q4tp.log
echo P2Q_DONE
