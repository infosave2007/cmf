#!/bin/bash
# Gates of an assembled chat model on the CPU runtime (or EVAL_GPU=wgpu): general false answers, popular plain questions, facts, world questions, dialogs, free chat.
set -u
D=$1; cd "$D"; R=${EMBRYO_RUN:-$HOME/embryo-data/o1/chat-sft}; K=${EMBRYO_HERBS:-$HOME/embryo-data/herbs/skill}
C=${CORTIQ:-$HOME/dev/cmf-embryo-wide/target/release/cortiq}; F=${2:-model.cmf}
export CMF_GPU=${EVAL_GPU:-0} CMF_EMBRYO_RESIDENT=1
log() { echo "$(date +%T) $*" | tee -a eval.status; }
summ() { python3 -I -c "
import json,sys; d=json.load(open(sys.argv[1])); d=d[0] if isinstance(d,list) else d
print(f\"kw {d['exact_or_keyword']}/{d['prompts']} loops {d['loops']} lookup_hits {d.get('lookup_hits')} src {d.get('lookup_src_matches')}/{d.get('lookup_src_known')} targets {d.get('route_counts')} tok/s {sum(p.get('tok_s') or 0 for p in d['per_prompt'])/max(1,len(d['per_prompt'])):.0f}\")" "$1" 2>/dev/null; }
for set in $K/general-eval.jsonl:1 $R/popular/popular.jsonl:1 $R/router-key/popular2.jsonl:1 $K/herbs-recall-train.jsonl:48 $K/herbs-recall-final.jsonl:48 world/world-test-held.jsonl:1 $R/router-key/world-seen-300.jsonl:1; do
  f=${set%%:*}; mt=${set##*:}; s=$(basename $f .jsonl)
  $C probe-utility $F --prompts-jsonl $f --route auto --lookup-mode answer --max-tokens $mt --json > e-$s.json 2> e-$s.err
  log "$s: $(summ e-$s.json)"
done
$C probe-dialog $F --dialogs-jsonl $K/herbs-dialogs-dev.jsonl --route auto --lookup-mode answer --json > e-dialogs.json 2> e-dialogs.err
log "dialogs: $(python3 -I -c "
import json; d=json.load(open('e-dialogs.json')); print({k:d.get(k) for k in ('first_turn_correct_rate','followup_hit_rate','followup_correct_rate','followup_correct_of_expected')})")"
python3 ${EMBRYO_SCRIPTS:-$HOME/dev/cmf-embryo-wide/scripts/embryo}/mk_followup.py dialogs "$PWD" $R/popular/popular.jsonl
for g in pos neg; do
  $C probe-dialog $F --dialogs-jsonl fu-dialogs-$g.jsonl --route auto --lookup-mode answer --json > e-fu-$g.json 2> e-fu-$g.err
  log "plain follow-ups $g: $(python3 -I -c "
import json,sys; d=json.load(open(sys.argv[1])); print({k:d.get(k) for k in ('followup_turns','followup_hits','followup_hit_rate','followup_correct','route_counts','field_counts')})" e-fu-$g.json)"
done
for g in fact chat; do
  $C probe-utility $F --prompts-jsonl kl-test-$g.jsonl --route auto --lookup-mode answer --max-tokens 1 --json > e-kl-$g.json 2> e-kl-$g.err
  log "keyless test $g: $(summ e-kl-$g.json)"
done
$C probe-utility $F --prompts-jsonl $R/decode/sweep.jsonl --route auto --lookup-mode answer --max-tokens 200 --rep-penalty 1.1 --json > e-sweep.json 2> e-sweep.err
log "sweep 40: $(summ e-sweep.json)"
log EVAL-DONE
