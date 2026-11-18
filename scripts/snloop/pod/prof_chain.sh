#!/bin/bash
# prof_chain.sh: decode profiles of the q4tp file on CUDA and Vulkan (plain and with drafts), then GPTQ v2
# (calibration over every visit of the loop) and its ppl.
set -u
cd /root/sn
B=bin/cortiq-fast
export XDG_RUNTIME_DIR=/tmp
until grep -q GPTQ_CHAIN_DONE logs/gptq_chain.log 2>/dev/null; do sleep 15; done
F=out/sn9-q4tp-rtn.cmf
echo "== cuda plain prof $(date)"
CMF_GPU=cuda CMF_MTP=0 CMF_CUDA_PROF=1 $B bench $F --core --ignore-eos --tokens 128 > logs/prof_cuda_plain.log 2>&1
grep -E -i "decode|prefill" logs/prof_cuda_plain.log | tail -4
echo "== vulkan plain $(date)"
CMF_MTP=0 $B bench $F --core --ignore-eos --tokens 128 > logs/prof_vk_plain.log 2>&1
grep -E -i "decode|prefill" logs/prof_vk_plain.log | tail -4
echo "== vulkan plain graph prof $(date)"
CMF_MTP=0 CMF_GRAPH_PROF=1 CMF_GRAPH_HOSTPROF=1 $B bench $F --core --ignore-eos --tokens 64 > logs/prof_vk_graph.log 2>&1
grep -E -i "decode|prefill" logs/prof_vk_graph.log | tail -3
echo "== gptq v2 $(date)"
$B quantize-gptq out/sn9-f16.cmf --codec q4tp --calib-model out/sn9-q8_2f.cmf \
  --calib data/wiki.train.raw=0.5 --calib data/chat_calib.txt=0.5 --tokens 16384 --window 2048 --lambda 0.01 \
  --calib-chunk-layers 8 --output out/sn9-q4tp-gptq2.cmf > logs/gptq2.log 2>&1
grep -E "WARNING|no Hessian|wrote" logs/gptq2.log | cut -c1-200
for w in "--tokens 2048" "--windows 8 --window-len 2048"; do
  echo "sn9-q4tp-gptq2 [$w] $(CMF_GPU=cuda $B ppl out/sn9-q4tp-gptq2.cmf --file data/wiki.test.raw $w 2>&1 | grep -i ppl | tail -1)"
done
echo "sn9-q4tp-gptq2 [chat_eval 4096] $(CMF_GPU=cuda $B ppl out/sn9-q4tp-gptq2.cmf --file data/chat_eval.txt --tokens 4096 2>&1 | grep -i ppl | tail -1)"
echo PROF_CHAIN_DONE
