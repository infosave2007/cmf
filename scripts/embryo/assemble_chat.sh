#!/bin/bash
# Assemble the Embryo chat model from a checkpoint: sealed genome (DTYPE: f16 or q8_2f) + file sampling defaults, herbs and world lookup records,
# key_span reconstruction router (φ on the GPU, fit + calibration by route-fit), gates. usage: assemble_chat.sh <ckpt> <dir> [alpha] [rank] [margin]
set -eu
CK=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); D=$2; AL=${3:-0.5}; RK=${4:-16}; MG=${5:-0.01}
R=${EMBRYO_RUN:-$HOME/embryo-data/o1/chat-sft}; K=${EMBRYO_HERBS:-$HOME/embryo-data/herbs/skill}
T=${EMBRYO_TOOLS:-$R/embryo-tools}; C=${CORTIQ:-$HOME/dev/cmf-embryo-wide/target/release/cortiq}
TOK=$R/../sft-t27/gen1.tokenizer.json
SAMPLING=${SAMPLING:-temperature=0.7,top_p=0.9,repetition_penalty=1.1,max_tokens=200,loop_stop=1,history_turns=1}
mkdir -p "$D"; cd "$D"; : > status; log() { echo "$(date +%T) $*" | tee -a status; }
python3 -I ${EMBRYO_SCRIPTS:-$HOME/dev/cmf-embryo-wide/scripts/embryo}/build_world.py world > world.log
cat ${EMBRYO_DROP_KEYS:-$HOME/dev/cmfpublic/docs/embryo-v2-2026-09-24/s16-drop-keys.txt} $R/router-key/extra-drop-keys.txt > drop-keys.txt
rm -f g0.cmf g1.cmf model.cmf
$T export --ckpt "$CK" --tokenizer "$TOK" --out g0.cmf --dtype ${DTYPE:-f16} --genome-id embryo-chat-56m --genome-status sealed --chat --sampling "$SAMPLING" > export.log 2>&1
export CMF_GPU=0
$C lookup-build g0.cmf --entries $R/router-fit/entries-names.jsonl --id herbs --out g1.cmf --on-duplicate last --drop-keys drop-keys.txt > build-herbs.log 2>&1
RECS="countries elements people books self trivia places science howto"
for g in $RECS; do
  case $g in self) X="--route-margin ${MS:-0.02} --min-key-share 0.5";; *) X="--route-margin ${MW:-0.03}";; esac
  $C lookup-build g1.cmf --entries world/entries-$g.jsonl --id $g --out g2.cmf --on-duplicate first $X > build-$g.log 2>&1
  mv g2.cmf g1.cmf
done
$C lookup-build g1.cmf --entries world/entries-dontknow.jsonl --id dontknow --out g2.cmf --on-duplicate first --keyless > build-dontknow.log 2>&1
mv g2.cmf g1.cmf
mv g1.cmf model.cmf; log "built $(ls -la model.cmf | awk '{print $5}') bytes"
for id in herbs $RECS; do
  $T phi-rows --ckpt "$CK" --cmf model.cmf --id $id --prompts $R/phi2/chat-users.jsonl --out hard-$id --only-keyed --keys-only --layers 5 > hard-$id.log 2>&1
done
python3 -I ${EMBRYO_SCRIPTS:-$HOME/dev/cmf-embryo-wide/scripts/embryo}/mk_fit.py "$PWD" "${FIT_ALL:-1}"
for s in fit-general fit-herbs $(for g in $RECS; do echo fit-$g; done); do
  $T phi-pool --ckpt "$CK" --cmf model.cmf --prompts $s.jsonl --layer 5 --alpha $AL --out $s > $s.pool.log 2>&1
  log "$(tail -1 $s.pool.log)"
done
for id in herbs $RECS; do
  $C route-fit model.cmf --id $id --skill-prompts fit-$id.prompts.jsonl --general-prompts fit-general.prompts.jsonl --skill-phi fit-$id.phi.f32 --general-phi fit-general.phi.f32 \
    --phi-layer 5 --rank $RK --max 0 --key-alpha $AL --margin $MG --target-fpr ${FPR:-0.001} > routefit-$id.log 2>&1
  log "route-fit $id: $(tail -1 routefit-$id.log | cut -c1-200)"
done
# Follow-ups have their own form of φ and their own isolated topologies
FA=${FA:-4}; S=${EMBRYO_SCRIPTS:-$HOME/dev/cmf-embryo-wide/scripts/embryo}
python3 $S/mk_followup.py prompts "$PWD"
$T phi-pool --ckpt "$CK" --cmf model.cmf --prompts fu-keys.jsonl --layer 5 --alpha 0 --out fu-keys > fu.log 2>&1
for s in fu-pos fu-neg fu-neg2; do $T phi-pool --ckpt "$CK" --cmf model.cmf --prompts $s.jsonl --layer 5 --whole-message --out $s >> fu.log 2>&1; done
python3 $S/mk_followup.py rows "$PWD" $FA
$C route-fit-followup model.cmf --skill-phi herbs=fu-herbs.f32 --general-phi fu-general.f32 --rank ${FK:-32} --alpha $FA --margin ${FM:-0.01} > routefit-followup.log 2>&1
log "route-fit-followup: $(tail -1 routefit-followup.log | cut -c1-200)"
$C route-eval model.cmf --prompts-jsonl $K/general-eval.jsonl --expect backbone --include-quarantine --json > re-general.json 2> re-general.err
for id in herbs $RECS; do $C skill-gate model.cmf --id $id --gate re-general.json --status active > gate-$id.log 2> gate-$id.err; log "gate $id exit $?"; done
# A message with no key: the two isolated topologies say whether it asks for a fact the network does not have
python3 -I $S/mk_keyless.py "$PWD"
for s in kl-fact kl-chat; do $T phi-pool --ckpt "$CK" --cmf model.cmf --prompts $s.jsonl --layer 5 --whole-message --out $s >> fu.log 2>&1; done
$C route-fit-keyless model.cmf --record dontknow --fact-phi kl-fact.phi.f32 --chat-phi kl-chat.phi.f32 --rank ${KK:-32} --margin ${KM:--0.06} > routefit-keyless.log 2>&1
log "route-fit-keyless: $(tail -1 routefit-keyless.log | cut -c1-200)"
# The header updates of the fits leave copies behind
$C tail-trim model.cmf | tee -a status
log GATED
