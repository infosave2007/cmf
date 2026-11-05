#!/bin/bash
# Targeted q4tp mixes: head and embedding at 8 bits, then the attention too.
cd /root/dory
export XDG_RUNTIME_DIR=/tmp
B=./src/target/release/cortiq
L=logs/variants.log
: > $L
conv() { # name, then --tensor-quant values
  n=$1; shift
  a=(); for t in "$@"; do a+=(--tensor-quant "$t"); done
  timeout 900 $B convert --model hf/bf16 --quant q4tp --output out/dory-$n.cmf "${a[@]}" > logs/conv_$n.log 2>&1
  echo "$n size $(stat -c %s out/dory-$n.cmf)" >> $L
  timeout 300 $B ppl out/dory-$n.cmf -f wiki_test.txt --windows 12 --window-len 512 2>&1 | grep -E "PPL" | cut -c1-120 >> $L
}
conv q4e 'lm_head.weight=q8_2f' 'model.embed_tokens.weight=q8_2f'
conv q4eh 'lm_head.weight=q8_2f'
conv q4a 'lm_head.weight=q8_2f' 'model.embed_tokens.weight=q8_2f' 'model.layers.*.mixer.q_proj.weight=q8_2f' 'model.layers.*.mixer.k_proj.weight=q8_2f' 'model.layers.*.mixer.v_proj.weight=q8_2f' 'model.layers.*.mixer.g_proj.weight=q8_2f' 'model.layers.*.mixer.o_proj.weight=q8_2f'
echo DONE >> $L
