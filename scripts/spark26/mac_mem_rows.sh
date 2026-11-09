#!/bin/bash
# mac_mem_rows.sh DIR SCRIPTS: the memory rows of the engine comparison measured one way for every
# engine — peak phys_footprint and resident size of the process (fp_run.py) over the p2k and p14k
# prompts with 32 generated tokens: cortiq (q4tp, q8_2f, turbo), mlx-lm (4/8-bit) and llama.cpp
# (Q4_K_M, Q8_0 with the context sized to the prompt).
D=$1; S=$2; cd "$D"; PY=cmp/venv/bin/python; LL=cmp/llama.cpp/build/bin
fp() { python3 "$S/fp_run.py" "$@"; }
for p in p2k p14k; do
  echo "cortiq q4tp $p | $(fp cmp/fp_q4tp_$p ./cortiq run hf/spark-x2.6-4b-q4tp.cmf --prompt-file $p.txt --no-think --greedy -n 32) | $(tail -1 cmp/fp_q4tp_$p.out)"
done
echo "cortiq q8_2f p14k | $(fp cmp/fp_q8_p14k ./cortiq run hf/spark-x2.6-4b-q8_2f.cmf --prompt-file p14k.txt --no-think --greedy -n 32) | $(tail -1 cmp/fp_q8_p14k.out)"
echo "cortiq turbo p14k | $(fp cmp/fp_turbo_p14k ./cortiq run hf/spark-x2.6-4b-q4tp-turbo.cmf --prompt-file p14k.txt --no-think --greedy -n 32) | $(tail -1 cmp/fp_turbo_p14k.out)"
for m in mlx4 mlx8; do for p in p2k p14k; do
  echo "mlx $m $p | $(fp cmp/fp_${m}_$p $PY "$S/mlx_bench.py" cmp/hf/$m $p.txt 32) | $(head -1 cmp/fp_${m}_$p.out)"
done; done
for q in Q4_K_M Q8_0; do for p in p2k p14k; do
  c=2048; [ $p = p14k ] && c=16384
  echo "llama $q $p | $(fp cmp/fp_${q}_$p $LL/llama-completion -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f $p.txt -n 32 -c $c --temp 0 -no-cnv --no-warmup) | $(grep -E 'eval time' cmp/fp_${q}_$p.err | sed 's/.*print: *//' | tr -s ' ' | tr '\n' ';')"
done; done
echo MEM_ROWS_DONE
