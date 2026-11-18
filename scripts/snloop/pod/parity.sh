#!/bin/bash
# parity.sh BIN TAG ENV FILE...: greedy answers to the reference prompts (sn_ref.py), 64 tokens, no thinking.
# ENV is "cpu" (CMF_GPU=0 CMF_SDOT=0: host float activations), "cuda" or "vulkan".
set -u
B=$1; TAG=$2; MODE=$3; shift 3
export XDG_RUNTIME_DIR=/tmp
case $MODE in
  cpu) E="CMF_GPU=0 CMF_SDOT=0" ;;
  cuda) E="CMF_GPU=cuda" ;;
  *) E="" ;;
esac
R=/root/sn/ref/f32; O=/root/sn/logs/parity_$TAG; mkdir -p $O
py() { python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$@"; }
for F in "$@"; do
  f=$(basename $F .cmf)
  for p in text text2 code image; do
    [ -f $R/$p.json ] || continue
    msg=$(python3 -I -c "import json,sys;m=json.load(open(sys.argv[1]));import re;print(re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>',m['prompt'],re.S).group(1).replace('<|vision_start|><|image_pad|><|vision_end|>',''))" $R/$p.json)
    img=""; [ $p = image ] && img="--image /root/sn/data/img/hero1024.png"
    env $E timeout 1800 $B run $F --no-think --greedy --max-tokens 64 $img --prompt "$msg" > $O/$f.$p.out 2>&1
    got=$(sed -n '/^$/,$p' $O/$f.$p.out | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-300)
    echo "[$MODE $f $p] $(grep -E 'tok/s|Decode' $O/$f.$p.out | tail -1 | tr -s ' ')"
    echo "   got: $(tail -n +1 $O/$f.$p.out | grep -v -E '^\s*$|tok/s|loaded|Loading|backend|cortiq|^\[' | tr '\n' ' ' | cut -c1-260)"
    echo "   ref: $(py $R/$p.json gen_text | tr '\n' ' ' | cut -c1-260)"
  done
done
echo "PARITY_DONE $TAG"
