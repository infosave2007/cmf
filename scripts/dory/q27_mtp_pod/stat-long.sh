#!/bin/bash
# stat-long.sh: the Turbo page's rows on this card, the release (cortiq-old) against the branch (cortiq):
# 1. a 300-token summary after 30 000 characters of Wikipedia (pf30k.txt: wiki_test.txt[5000:35000] and
#    "In one paragraph, what is the last article above about?"), the two binaries in turn, three times;
# 2. `bench --core` at 1 000, 4 000 and 16 000 tokens (the last is past the seal).
# Log st-long.log; st-long.done at the end.
cd /root/q27; export XDG_RUNTIME_DIR=/tmp
M=m/qwen38-27b-turbo.cmf; : > st-long.log
for rep in 1 2 3; do for b in cortiq-old cortiq; do
  r=$(CMF_MAX_SEQ=70000 timeout 900 ./$b run $M --prompt-file pf30k.txt --greedy --no-think --max-tokens 300 2>/dev/null | grep -E "^\[" | tr '\n' ' ' | cut -c1-230)
  echo "real 30000 chars $b: $r" >> st-long.log
done; done
for c in 1000 4000 16000; do for b in cortiq-old cortiq; do
  echo "$b ctx $c: $(timeout 900 ./$b bench $M --core --tokens 60 --ignore-eos --ctx $c 2>/dev/null | grep -E 'Decode:|MTP:' | tr '\n' ' ' | cut -c1-200)" >> st-long.log
done; done
echo DONE > st-long.done
