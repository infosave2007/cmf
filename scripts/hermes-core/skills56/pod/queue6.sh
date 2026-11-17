#!/bin/bash
# After E4: E5 — growth by experts (2 new experts in every layer trained on the skill-1 corpus over the frozen trunk,
# 1500 steps of B4 x 512, as a1 on the Mac: R 51/362 from the release 56M) on the E1 base after chat SFT and, as its
# control, on the release 56M after the same chat SFT; their R answers and general-text perplexity (CPU, while
# the card has nothing else queued). Log /root/sk/run/queue6.log; marker queue6.done
cd /root/sk/run
L=/root/sk/run/queue6.log
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0
E=/root/sk/cmf/target-kd/release/cortiq-embryo; D=/root/sk/data; C=/root/sk/cmf/target/release/cortiq
until [ -f queue5.done ]; do sleep 20; done
{
  echo "== queue6 $(date -u +%T)"
  for n in e1s f0s; do
    SECONDS=0
    $E grow --ckpt $n-sft.ckpt --tokenizer $D/gen1.tokenizer.json --corpus $D/skill1.jsonl.gz --held $D/skill1.held.jsonl.gz \
      --general $D/rawval-tails-2m.u16 --experts 2 --steps 1500 --batch 4 --seq 512 \
      --trace-tokens 300000 --trace-docs 300 --out-ckpt g-$n.ckpt > g-$n.grow.log 2>&1 || { echo "GROW $n FAILED"; tail -3 g-$n.grow.log | cut -c1-300; continue; }
    echo "grow $n: $SECONDS s; $(grep -oE '"held_before":[0-9.]*|"held_after":[0-9.]*|"improvement":[0-9.]*' g-$n.grow.log | tr '\n' ' ')"
    $E export --ckpt g-$n.ckpt --tokenizer $D/gen1.tokenizer.json --out g-$n.cmf --dtype f16 --chat 2>&1 | tail -1
  done
  for n in e1s f0s; do
    [ -f g-$n.cmf ] || continue
    /root/sk/measure.sh g-$n backbone
    echo "g-$n heldout-en: $(CMF_GPU=0 CMF_GROWTH=off $C ppl --file $D/heldout-en.txt --windows 16 --window-len 512 g-$n.cmf 2>&1 | tail -1)"
  done
  echo "== queue6 done $(date -u +%T)"
  touch queue6.done
} >> $L 2>&1
