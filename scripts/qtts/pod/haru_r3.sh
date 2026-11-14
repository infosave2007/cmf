#!/bin/bash
# Haru round 3: q4tp breaks the model (garbage text at every frame on every
# backend), q8_2f talks. Two mixed q4tp files — the tied embedding at q8_2f;
# and the attention (GDN and full) at q8_2f as well, the FFN alone at 4 bits —
# against q8_2f on CUDA; q8_2f on Vulkan and the CPU for the card.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
H=/root/haru
end=$(( $(date +%s) + 7200 ))
until grep -q HARU2_DONE logs/haru_r2.out 2>/dev/null; do [ "$(date +%s)" -gt $end ] && break; sleep 30; done
[ -s $H/haru-q4tp-e8.cmf ] || timeout 3600 bin/cortiq convert --model $H/hf --quant q4tp --output $H/haru-q4tp-e8.cmf --tensor-quant 'model.embed_tokens.weight=q8_2f' > logs/haru3_conv1.log 2>&1 || { echo CONVERT1_FAILED; tail -5 logs/haru3_conv1.log; }
[ -s $H/haru-q4tp-a8.cmf ] || timeout 3600 bin/cortiq convert --model $H/hf --quant q4tp --output $H/haru-q4tp-a8.cmf --tensor-quant 'model.embed_tokens.weight=q8_2f' --tensor-quant 'model.layers.*.linear_attn.*=q8_2f' --tensor-quant 'model.layers.*.self_attn.*=q8_2f' > logs/haru3_conv2.log 2>&1 || { echo CONVERT2_FAILED; tail -5 logs/haru3_conv2.log; }
ls -l $H/*.cmf | awk '{print $5, $9}'
w=/root/h3i/cmf/dl/whisper-large-v3-turbo-q8.cmf
for v in "q4tp-e8 cuda" "q4tp-a8 cuda" "q8_2f 1" "q8_2f 0" "q8_2f cuda"; do
  set -- $v; f=$1; be=$2
  flock gpu.lock env CMF_GPU=$be timeout 3600 bin/cortiq duplex $H/haru-$f.cmf $H/out/user_call2.wav $H/out/r3_${f}_$be.wav --reference $H/out/voice.wav --seed 2 > logs/haru3_${f}_$be.log 2>&1
  echo "$f $be: exit $? | $(grep -E 'frames .*per frame' logs/haru3_${f}_$be.log)"
  grep -vE 'INFO|WARN|frames .*per frame|^\[' logs/haru3_${f}_$be.log | head -2 | cut -c1-260 | sed 's/^/    text: /'
  [ -s $H/out/r3_${f}_$be.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w $H/out/r3_${f}_$be.wav --language en 2>/dev/null | tail -1 | cut -c1-260 | sed 's/^/    whisper: /'
done
echo HARU3_DONE
