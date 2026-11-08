#!/bin/bash
# chain10.sh: what the far field keeps of distant text — a 100 000-character prompt of several
# Wikipedia articles (about 24 000 tokens) asked to summarise the first article (far back, in the
# far field apart from its first rows) and the last (mostly in the exact tail), on the standard and
# the turbo file, Vulkan, 160 greedy tokens. Outputs to logs/chain10.out. Waits for chain9.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
until [ -f logs/chain9.done ]; do sleep 20; done
python3 - <<'PY'
text = open("data/wiki.test.raw", encoding="utf-8").read().replace("\r", "")
body = text[:100000]
body = body[:body.rfind("\n")]
for name, q in (("first", "Summarise the first article above in five sentences."), ("last", "Summarise the last article above in five sentences.")):
    open(f"logs/sum_{name}.txt", "w", encoding="utf-8").write(body + "\n\n" + q + "\n")
PY
{
for f in std:hf/cmf/spark-x2.6-4b-q4tp.cmf turbo:out/spark-x2.6-4b-q4tp-turbo.cmf; do
  tag=${f%%:*}; m=${f#*:}
  for q in first last; do
    echo "== $tag $q"
    CMF_MAX_SEQ=131072 timeout 1200 ./cortiq run $m --prompt-file logs/sum_$q.txt --no-think --greedy -n 160 2>/dev/null
    echo
  done
done
} > logs/chain10.out 2>&1
echo CHAIN10_DONE > logs/chain10.done
