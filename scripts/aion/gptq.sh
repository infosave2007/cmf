#!/usr/bin/env bash
# gptq.sh TAG [extra quantize-gptq args...]: q4tp GPTQ (scale + refine), embeddings q8_2f, then ppl
source /root/aion/scripts/env.sh
TAG=$1; shift
OUTF=out/aion-q4tp-$TAG.cmf
s=$(date +%s)
$B quantize-gptq out/aion-f16.cmf --codec q4tp --calib-model out/aion-q8_2f.cmf \
  ${CALIB:---calib data/wiki.train.raw=0.25 --calib data/code_calib.txt=0.25 --calib data/chat_calib.txt=0.5} \
  --tokens ${TOKENS:-4096} --window 2048 --lambda 0.01 \
  --channel-scale 0.75 --channel-scale-qkv --refine 256 --refine-ridge 0.3 --refine-q4tp \
  --tensor-quant model.embed_tokens.weight=q8_2f "$@" --output $OUTF > gptq_$TAG.log 2>&1
echo "$TAG gptq rc $? $(( $(date +%s) - s )) s size $(stat -c %s $OUTF)"
bash scripts/ppl_all.sh $OUTF q4tp-$TAG
