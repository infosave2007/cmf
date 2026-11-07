#!/bin/bash
# cardq5.sh: after the q2tp drafter runs — peak device memory of the q4tp file with the drafter's
# rounds on (the code prompt, 160 tokens), for the card's "fits" column.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
timeout 10800 bash -c "until grep -q CARDQ4_DONE logs/cardq4.out; do sleep 15; done"
bash scripts/vram_peak.sh "dflash q4tp code" logs/final_dflash.log /root/rip/sp -- env CMF_DFLASH=1 RUST_LOG=info CMF_GRAPH_SPEC_TIME=1 ./cortiq run out/ripple-65b-q4kv8.cmf --prompt "$(cat data/code.txt)" --greedy --max-tokens 160
grep -a -c "solo-round:" /tmp/vp.out
bash scripts/vram_peak.sh "plain q4tp code" logs/final_dflash.log /root/rip/sp -- env CMF_GRAPH_SPEC=0 ./cortiq run out/ripple-65b-q4kv8.cmf --prompt "$(cat data/code.txt)" --greedy --max-tokens 160
echo CARDQ5_DONE
