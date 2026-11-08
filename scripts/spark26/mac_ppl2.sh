#!/bin/bash
# mac_ppl2.sh DIR SCRIPTS: the comparable perplexity protocol on the Mac — 64 chunks of 512 raw
# tokens from the start of wiki.test, the second half of each chunk scored (255 predictions a
# chunk, 16 320 in all): cortiq on Metal (`--window-stride 512 --window-score-from 256`), stock
# mlx-lm (mlx_ppl.py ... 512 - 256) and llama.cpp's default walk (`-c 512 --chunks 64`), q4 and q8
# files of each. Waits for the F6 check (one GPU job at a time).
D=$1; S=$2; cd "$D"; PY=cmp/venv/bin/python; LL=cmp/llama.cpp/build/bin
until grep -q F6_CHECK_DONE f6_check.log; do sleep 10; done
for q in q4tp q8_2f; do
  echo "== cortiq ppl $q (metal, chunks of 512, second half)"
  CMF_GPU=1 ./cortiq ppl hf/spark-x2.6-4b-$q.cmf data/wiki.test.raw --windows 64 --window-len 512 --window-stride 512 --window-score-from 256 2>&1 | grep -E "windows|PPL"
done
for m in mlx4 mlx8; do
  echo "== mlx ppl $m"
  $PY "$S/mlx_ppl.py" cmp/hf/$m data/wiki.test.raw 64 512 512 - 256 2>&1 | grep -E "windows|PPL"
done
for q in Q4_K_M Q8_0; do
  echo "== llama ppl $q (metal, -c 512 --chunks 64)"
  $LL/llama-perplexity -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f data/wiki.test.raw -c 512 --chunks 64 -ngl 99 2>&1 | grep -E "estimate|\[64\]" | tail -2
done
echo PPL2_DONE
