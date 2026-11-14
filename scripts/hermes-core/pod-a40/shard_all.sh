#!/usr/bin/env bash
# Tokenize the prepared sources into training and held-out shards.
#
#   shard_all.sh            (reads /root/hc/prep/*.jsonl.gz, writes /root/hc/data/b_*.u16 and bv_*.u16)
#
# Chat and tool-use sources are rendered transcripts (convert.py, lookup_gen.py):
# their special tokens are encoded as ids (--specials). Code is plain text.
# A shard is skipped when it is newer than its inputs. Token counts go to
# /root/hc/data/shards.tsv.
set -u
P=/root/hc/prep; D=/root/hc/data
E=/root/hc/embryo-tools; T=$D/gen1.tokenizer.json
mk() { # out specials inputs...
  local out="$1" sp="$2"; shift 2
  local ins=() newest=0
  for f in "$@"; do [ -s "$f" ] && ins+=("$f"); done
  [ ${#ins[@]} -eq 0 ] && { echo "$out: no inputs" >&2; return; }
  if [ -s "$D/$out" ]; then
    local stale=0; for f in "${ins[@]}"; do [ "$f" -nt "$D/$out" ] && stale=1; done
    [ $stale -eq 0 ] && return
  fi
  local flag=(); [ "$sp" = 1 ] && flag=(--specials)
  nice -n 12 "$E" shard --tokenizer "$T" --inputs "${ins[@]}" --out "$D/$out.tmp" "${flag[@]}" > "$D/$out.log" 2>&1 \
    && mv "$D/$out.tmp" "$D/$out"
}
# training shards: one per source group, so that the mix is set by weights
mk b_ultrachat.u16 1 $P/ultrachat.jsonl.gz
mk b_magpie.u16 1 $P/magpie.jsonl.gz
mk b_chat_misc.u16 1 $P/systemchats.jsonl.gz $P/everyday.jsonl.gz $P/rewrite.jsonl.gz $P/summarize.jsonl.gz $P/explore.jsonl.gz $P/mot_science.jsonl.gz
mk b_ot_answers.u16 1 $P/openthoughts_answers.jsonl.gz
mk b_multilingual.u16 1 $P/multilingual.jsonl.gz
mk b_multilingual_think.u16 1 $P/multilingual_think.jsonl.gz $P/aya_think.jsonl.gz
mk b_ot_think.u16 1 $P/openthoughts_think.jsonl.gz
mk b_think_misc.u16 1 $P/multiturn_if_think.jsonl.gz $P/systemchats_think.jsonl.gz $P/tablegpt_think.jsonl.gz $P/everyday_think.jsonl.gz
mk b_toolmind.u16 1 $P/toolmind.jsonl.gz
mk b_toucan.u16 1 $P/toucan_kimi.jsonl.gz $P/toucan_sft.jsonl.gz
mk b_nemotron.u16 1 $P/nemotron_agent.jsonl.gz $P/nemotron_search.jsonl.gz
mk b_tools_misc.u16 1 $P/glaive.jsonl.gz $P/hermes_fc.jsonl.gz $P/smolagents.jsonl.gz $P/when2call.jsonl.gz $P/tau2.jsonl.gz $P/envfactory.jsonl.gz
mk b_lookup.u16 1 $P/lookup.jsonl.gz
mk b_hermes_code.u16 0 $P/hermes_code2.jsonl.gz
mk b_harness_code.u16 0 $P/harness_code.jsonl.gz
# held-out shards
mk bv_chat.u16 1 $P/ultrachat.val.jsonl.gz $P/magpie.val.jsonl.gz $P/summarize.val.jsonl.gz $P/systemchats.val.jsonl.gz
mk bv_multilingual.u16 1 $P/multilingual.val.jsonl.gz $P/multilingual_think.val.jsonl.gz
mk bv_think.u16 1 $P/openthoughts_think.val.jsonl.gz $P/multiturn_if_think.val.jsonl.gz
mk bv_tools.u16 1 $P/toolmind.val.jsonl.gz $P/toucan_kimi.val.jsonl.gz $P/glaive.val.jsonl.gz $P/hermes_fc.val.jsonl.gz $P/nemotron_agent.val.jsonl.gz
mk bv_lookup.u16 1 $P/lookup.val.jsonl.gz
mk bv_harness.u16 0 $P/harness_code.val.jsonl.gz
( cd $D && for f in b_*.u16 bv_*.u16 en.u16 ru.u16 code.u16 math.u16 se.u16 docs.u16 hermes_code.u16 val_*.u16 rawval.u16; do
    [ -s "$f" ] && printf "%s\t%d\n" "$f" $(( $(stat -c %s "$f") / 2 )); done ) > $D/shards.tsv
cat $D/shards.tsv
