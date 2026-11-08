#!/bin/bash
# mac_llama_rows.sh DIR: the llama.cpp rows of mac_suite2.sh on their own (token ids, perplexity
# under the shared window protocol (-c 257: llama-perplexity adds ppl_stride/2 to the context when a
# stride is set, so 257 + 255 = the 512-token windows the other engines score), peak RSS on p2k/p14k with -c sized to the prompt: without it the
# context auto-fits to free memory and the RSS says nothing), run once llama-perplexity, llama-tokenize
# and llama-completion are built.
D=$1; cd "$D"; W=data/wiki.test.raw
LL=cmp/llama.cpp/build/bin
rss() { grep "maximum resident set size" "$1" | awk '{printf "%.2f GB", $1/1073741824}'; }
$LL/llama-tokenize -m cmp/hf/gguf/Spark-X2.6-4B-Q4_K_M.gguf -f $W --ids --no-bos --log-disable > cmp/llama_ids.txt 2>/dev/null
echo "llama ids: $(tr -cd ',' < cmp/llama_ids.txt | wc -c | tr -d ' ')"
for q in Q4_K_M Q8_0; do
  echo "== llama ppl $q"
  $LL/llama-perplexity -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f $W -c 257 -b 512 --ppl-stride 511 --chunks 64 2>&1 | grep -E "Final|estimate|chunks|\[64\]" | tail -3
done
for q in Q4_K_M Q8_0; do for p in p2k p14k; do
  c=2048; [ $p = p14k ] && c=16384
  /usr/bin/time -l $LL/llama-completion -m cmp/hf/gguf/Spark-X2.6-4B-$q.gguf -f $p.txt -n 32 -c $c --temp 0 -no-cnv --no-warmup < /dev/null > /dev/null 2> cmp/mem_${q}_$p.err
  echo "llama $q $p | rss $(rss cmp/mem_${q}_$p.err) | $(grep -E 'eval time' cmp/mem_${q}_$p.err | sed 's/.*print: *//' | tr -s ' ' | tr '\n' ';')"
done; done
echo LLAMA_ROWS_DONE
