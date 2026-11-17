#!/bin/bash
# E3: chat SFT of a base by the release recipe (run-sft.sh of the 56M: chat-train.sft with a fifth of raw replay,
# B10 x 1024, lr 3e-5, stages of at most 500 steps, the first with fresh moments), 2000 steps; then the genome
# export and skill 1 by the CUDA recipe on it.
#   e3.sh NAME START_CKPT        writes run/NAME-sft.ckpt, run/fNAME.cmf, run/c1-NAME.cmf (+ logs)
cd /root/sk/run
export LD_LIBRARY_PATH=/usr/local/cuda/lib64 CMF_VULKAN_HEAD_TABLE=0
E=/root/sk/cmf/target-kd/release/cortiq-embryo; D=/root/sk/data
NAME="$1"; prev="$2"; p=0
R="--layers 1 2 3 4 5 6 7 --phi-layer 0 --mask-init 4.0 --lr-a 0.05 --l1 0 --soft-fold --lm-frac 0 --steps-a 1000 --eval-every 100 --steps-b 2 --lr-b 0.000001 --fcd-layers 2 --eval-every-b 1 --batch 16 --dev-batches 16 --final-batches 16"
echo "== $NAME: chat SFT from $prev $(date -u +%T)"
for next in 100 500 1000 1500 2000; do
  if [ $p -eq 0 ]; then extra="--fresh-moments"; else extra="--allow-extended --sampler-skip $p"; fi
  out=$NAME-sft-$next.ckpt; rm -f $out
  $E sft --ckpt $prev --shard $D/chat-train.sft --val $D/chat-dev.sft --final-val $D/chat-final.sft \
    --replay $D/replay-16m.u16 --raw-val $D/rawval-tails-2m.u16 --out $out --batch 10 --seq 1024 \
    --steps $next --val-batches 32 --raw-val-batches 16 --lr 3e-5 --dropless $extra > $NAME-sft-$next.log 2>&1 \
    || { echo "SFT stage $next FAILED"; tail -3 $NAME-sft-$next.log; exit 1; }
  echo "  stage $next: $(grep -o 'sft_gate.*' $NAME-sft-$next.log | tail -1 | cut -c1-200)"
  [ "$prev" != "$2" ] && rm -f $prev
  prev=$out; p=$next
done
mv $prev $NAME-sft.ckpt
echo "  final: $(grep -o 'sft_terminal_final_response_nll=[0-9.]*' $NAME-sft-2000.log | tail -1)"
rm -f f$NAME.cmf
$E export --ckpt $NAME-sft.ckpt --tokenizer $D/gen1.tokenizer.json --out f$NAME.cmf --genome-id embryo56-$NAME --genome-status sealed 2>&1 | tail -1
E=$E CKPT=/root/sk/run/$NAME-sft.ckpt /root/sk/bake.sh c1-$NAME f$NAME s1 cards $R
echo "== $NAME done $(date -u +%T)"
