#!/bin/bash
# cardq.sh: the q4tp file as it goes out (K and V projections from the 8-bit file, no neuron cut:
# the hot neurons gave it nothing), then the card tables of the two low-bit files after the drafter
# reference run.
cd /root/rip
export XDG_RUNTIME_DIR=/tmp
./cortiq hot-bake hf/ripple-65b-q4tp.cmf hf/ripple-65b-q8_2f.cmf hot/energy.mass out/ripple-65b-q4kv8.cmf --frac 0 --take self_attn.k_proj --take self_attn.v_proj 2>&1 | grep -a "hot-bake\|GB\|rror" | cut -c1-220
timeout 3000 bash -c "until grep -q DFREF_DONE logs/dfref.out; do sleep 10; done"
bash scripts/ppl.sh ripple-65b-q4kv8 2048
bash scripts/final.sh logs/final_q4tp.log out/ripple-65b-q4kv8.cmf 4000 16000 32000 64000
MODES="turbo" bash scripts/final.sh logs/final_q2tp.log out/ripple-65b-q2hot5kv.cmf 4000 64000
MODES="standard" bash scripts/final.sh logs/final_q2tp.log out/ripple-65b-q2hot5kv.cmf 4000
echo CARDQ_DONE
