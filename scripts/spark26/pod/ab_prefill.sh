#!/bin/bash
# ab_prefill.sh FILE CHARS: greedy 64 tokens after a wikitext prompt of CHARS characters,
# chunked host walk (CMF_DENSE_GRAPH_PREFILL=0) against the default; prints timing and whether the texts match.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp
F=$1; N=${2:-9000}; B=${BIN:-./cortiq}
venv/bin/python -c "t=open('data/wiki.test.raw').read(); open('logs/p_ab.txt','w').write(t[:$N]+'\n\nSummarize the text above.')"
CMF_DENSE_GRAPH_PREFILL=0 $B run $F --prompt-file logs/p_ab.txt --no-think --greedy -n 64 > logs/ab_walk.txt 2>logs/ab_walk.err
CMF_BATCH_DEBUG=1 $B run $F --prompt-file logs/p_ab.txt --no-think --greedy -n 64 > logs/ab_graph.txt 2>logs/ab_graph.err
tail -1 logs/ab_walk.txt; tail -1 logs/ab_graph.txt
grep -i -E "batch graph|refus|declin" logs/ab_graph.err | sort | uniq -c | head
head -c 1500 logs/ab_walk.txt | md5sum; head -c 1500 logs/ab_graph.txt | md5sum
diff <(head -n -1 logs/ab_walk.txt) <(head -n -1 logs/ab_graph.txt) > /dev/null && echo SAME_TEXT || echo DIFF_TEXT
