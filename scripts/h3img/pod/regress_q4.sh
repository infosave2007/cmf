#!/bin/bash
# The hybrid arm after the blr chain refactor: a plain q4tp stack (LFM3-1.7B)
# on CUDA, the old release binary against the new fast build — greedy text,
# decode and the 4k prompt.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p cmf/dl logs/rq out/rq
f=cmf/dl/lfm3-1.7b-q4tp.cmf
[ -s $f ] || timeout 3600 venv/bin/hf download infosave/LFM3-1.7B-cmf lfm3-1.7b-q4tp.cmf --local-dir cmf/dl > logs/rq/dl.log 2>&1
[ -s $f ] || { ls cmf/dl; echo DOWNLOAD_FAILED; exit 1; }
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
for b in cortiq cortiq-fast; do
  flock gpu.lock env CMF_GPU=cuda timeout 600 bin/$b run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/rq/$b.txt
  flock gpu.lock env CMF_GPU=cuda timeout 600 bin/$b bench $f --tokens 64 --ignore-eos --json > logs/rq/d_$b.log 2>&1
  flock gpu.lock env CMF_GPU=cuda timeout 600 bin/$b bench $f --tokens 32 --ctx 4096 --json > logs/rq/4k_$b.log 2>&1
  echo "$b: decode $(grep -oE "$keys" logs/rq/d_$b.log | tr '\n' ' ') | 4k $(grep -oE "$keys" logs/rq/4k_$b.log | tr '\n' ' ')"
done
cmp -s out/rq/cortiq.txt out/rq/cortiq-fast.txt && echo "text: identical" || { echo "text differs"; tail -2 out/rq/cortiq.txt; tail -2 out/rq/cortiq-fast.txt; }
rm -f $f
echo RQ_DONE
