#!/bin/bash
# mac_suite2.sh DIR SCRIPTS: the quality and memory rows of the engine comparison on the Mac.
# Perplexity under one window protocol (64 windows x 512 tokens at stride 511 of the raw token
# stream, no BOS) for llama.cpp (Metal), mlx-lm and cortiq (Metal, with and without ANE); the token
# ids of each tokenizer for the parity check; peak RSS (/usr/bin/time -l) of llama.cpp and mlx-lm on
# the p2k/p14k prompts (cortiq's rows come from mac_suite.sh). The cortiq rows wait for build_mac.done.
D=$1; S=$2; cd "$D"; W=data/wiki.test.raw
LL=cmp/llama.cpp/build/bin; PY=cmp/venv/bin/python
rss() { grep "maximum resident set size" "$1" | awk '{printf "%.2f GB", $1/1073741824}'; }
$LL/llama-tokenize -m cmp/hf/gguf/Spark-X2.6-4B-Q4_K_M.gguf -f $W --ids --no-bos --log-disable > cmp/llama_ids.txt 2>/dev/null
echo "llama ids: $(tr -cd ',' < cmp/llama_ids.txt | wc -c | tr -d ' ')"
for q in Q4_K_M Q8_0; do
  echo "== llama ppl $q"
  $LL/llama-perplexity -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f $W -c 512 -b 512 --ppl-stride 511 --chunks 64 2>&1 | grep -E "Final|estimate|chunks|\[64\]" | tail -3
done
for m in mlx4 mlx8; do
  echo "== mlx ppl $m"
  $PY "$S/mlx_ppl.py" cmp/hf/$m $W 64 512 511 cmp/${m}_ids.txt 2>&1 | tail -2
done
for q in Q4_K_M Q8_0; do for p in p2k p14k; do
  /usr/bin/time -l $LL/llama-completion -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f $p.txt -n 32 --temp 0 -no-cnv --no-warmup < /dev/null > /dev/null 2> cmp/mem_${q}_$p.err
  echo "llama $q $p | rss $(rss cmp/mem_${q}_$p.err) | $(grep -E 'prompt eval time|^.*eval time' cmp/mem_${q}_$p.err | sed 's/.*print: *//' | tr -s ' ' | tr '\n' ';')"
done; done
for m in mlx4 mlx8; do for p in p2k p14k; do
  /usr/bin/time -l $PY "$S/mlx_bench.py" cmp/hf/$m $p.txt 32 > cmp/mem_${m}_$p.out 2> cmp/mem_${m}_$p.err
  echo "mlx $m $p | rss $(rss cmp/mem_${m}_$p.err) | $(head -1 cmp/mem_${m}_$p.out)"
done; done
until [ -f build_mac.done ]; do sleep 15; done
for v in "q4tp X=1" "q4tp CMF_ANE=1" "q8_2f X=1"; do
  set -- $v
  echo "== cortiq ppl $1 $2"
  env $2 CMF_GPU=1 CMF_PPL_IDS_OUT=cmp/cortiq_ids.txt ./cortiq ppl hf/spark-x2.6-4b-$1.cmf -f $W --windows 64 --window-len 512 --window-stride 511 2>&1 | grep -E "windows|PPL|rror"
done
$PY - <<'EOF'
import re
def ids(p, sep):
    return [int(x) for x in re.split(sep, open(p).read().strip()) if x.strip()]
n = 63 * 511 + 512
a = ids("cmp/cortiq_ids.txt", r"\s+")[:n]; b = ids("cmp/mlx4_ids.txt", r"\s+")[:n]
c = ids("cmp/llama_ids.txt", r"[,\s\[\]]+")[:n]
print("ids parity over", n, "tokens: cortiq==mlx", a == b, "cortiq==llama", a == c, "first diff", next((i for i in range(n) if i < len(c) and a[i] != c[i]), None))
EOF
echo SUITE2_DONE
