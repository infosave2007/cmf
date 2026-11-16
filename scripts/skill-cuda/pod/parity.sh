#!/bin/bash
# parity.sh TAG MODEL [SKILL]: greedy 64 tokens on CUDA (ids + every step's logits), the float host
# (CMF_GPU=0 CMF_SDOT=0) teacher-forced along the CUDA ids and free; prompts from PROMPTS_DIR.
set -u
tag=$1; m=$2; sk=${3:-}
B=${B:-/root/h3i/bin/cortiq-skill}
pd=${PROMPTS_DIR:-/root/h3i/srcskill/scripts/cuda/prompts}
out=/root/h3i/skill/parity/$tag; rm -rf $out; mkdir -p $out
args=(); [ -n "$sk" ] && args=(--skill "$sk")
export XDG_RUNTIME_DIR=/tmp CMF_GPU_PROBE=0 CMF_MTP=0
for p in $pd/p1_short_en.txt $pd/p2_ru.txt $pd/p3_code.txt $pd/p5_quotes.txt; do
  n=$(basename $p .txt)
  for arm in cuda cpu_forced cpu_free; do
    mkdir -p $out/$arm/$n.logits
    case $arm in
      cuda) env CMF_GPU=cuda CMF_GEN_IDS_OUT=$out/$arm/$n.ids CMF_LOGIT_DUMP_ALL=$out/$arm/$n.logits \
              ${LOCK:-flock /root/h3i/gpu.lock} timeout 600 $B run $m --prompt-file $p --greedy --no-think -n ${N:-64} "${args[@]}" > $out/$arm/$n.txt 2>/dev/null
            steps=$(ls $out/$arm/$n.logits | wc -l)
            python3 -c "import sys; ids=open(sys.argv[1]).read().split(); open(sys.argv[2],'w').write(' '.join(ids[len(ids)-int(sys.argv[3]):]))" $out/$arm/$n.ids $out/$arm/$n.gen $steps ;;
      cpu_forced) env CMF_GPU=0 CMF_SDOT=0 CMF_FORCE_IDS=$out/cuda/$n.gen CMF_LOGIT_DUMP_ALL=$out/$arm/$n.logits \
              timeout 1200 $B run $m --prompt-file $p --greedy --no-think -n ${N:-64} "${args[@]}" > $out/$arm/$n.txt 2>/dev/null ;;
      cpu_free) env CMF_GPU=0 CMF_SDOT=0 CMF_GEN_IDS_OUT=$out/$arm/$n.ids \
              timeout 1200 $B run $m --prompt-file $p --greedy --no-think -n ${N:-64} "${args[@]}" > $out/$arm/$n.txt 2>/dev/null ;;
    esac
  done
done
python3 /root/h3i/skill/parity_cmp.py $out
