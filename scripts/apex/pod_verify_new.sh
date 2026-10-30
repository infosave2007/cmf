#!/bin/bash
# The engine change against the release binary on one file: default bench
# (no env: speculation policy and prompt chunk as shipped), greedy output
# identity on a 3.2k-token prompt (plain decode on both sides), and the
# wikitext-2 perplexity.
#
# usage: pod_verify_new.sh MODEL.cmf OLD_CORTIQ NEW_CORTIQ
set -u
M=$1; OLD=$2; NEW=$3
export XDG_RUNTIME_DIR=/tmp
cd /root/apex
P="$(head -c 14000 data/wiki.test.raw) Summarize the text above in one sentence."
for b in "$OLD" "$NEW"; do
  echo "== $($b --version) bench, no env"
  "$b" bench "$M" --core --ignore-eos --tokens 256 2>&1 | grep -a -E "Prompt:|Decode:|MTP:"
  "$b" bench "$M" --core --ignore-eos --tokens 32 --ctx 4096 2>&1 | grep -a -E "Prompt:|Decode:|MTP:"
done
for b in "$OLD" "$NEW"; do
  CMF_MTP=0 "$b" run "$M" --prompt "$P" --no-think --greedy --max-tokens 96 2>/dev/null \
    | sed -n '/^Prompt:/,$p' | grep -v "tokens ·" > "greedy_$(basename $b).txt"
  CMF_MTP=0 "$b" run "$M" --prompt "$P" --no-think --greedy --max-tokens 96 2>&1 | grep -a "tokens ·"
done
cmp "greedy_$(basename $OLD).txt" "greedy_$(basename $NEW).txt" && echo "GREEDY IDENTICAL" || echo "GREEDY DIFFERS"
"$NEW" ppl "$M" -f data/wiki.test.raw --tokens 4096 2>&1 | grep -a "PPL ="
echo VERIFY_DONE
