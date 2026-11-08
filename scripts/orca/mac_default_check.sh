#!/bin/bash
# Drafts by default on native Metal: the code prompt without the switch and with CMF_GRAPH_SPEC=0; 16 000 tokens past
# the seal (ANE prompt) with the default and with drafts off; the policy unit test. mac_default_check.sh LOG
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf
P=/Users/oleg/dev/orca-runs/scripts/p; L=$1; : > $L
until grep -q "MATCH" $R/pdl.log 2>/dev/null && [ -f $R/build.status ]; do sleep 20; done
cat $R/build.status | tee -a $L; grep sha256 $R/pdl.log | tee -a $L
one() { # NAME FILE TOKENS ENV...
  local name=$1 f=$2 n=$3; shift 3; sleep 30
  env "$@" RUST_LOG=info $B run $M --prompt-file "$f" --greedy --no-think --max-tokens $n > $R/dc.out 2> $R/dc.err
  echo "$name: $(grep -E '^\[' $R/dc.out | head -2 | tr '\n' ' ' | cut -c1-190) | text $(grep -vE '^\[' $R/dc.out | md5 -q | cut -c1-8)" | tee -a $L
  sed -e 's/\x1b\[[0-9;]*m//g' $R/dc.err | grep -oE "metal native: spec [^,]*" | head -1 >> $L
}
one "code default" $P/p_code.txt 220
one "code CMF_GRAPH_SPEC=0" $P/p_code.txt 220 CMF_GRAPH_SPEC=0
one "16000 default" $R/p16k.txt 160 CMF_ANE=1
one "16000 CMF_GRAPH_SPEC=0" $R/p16k.txt 160 CMF_ANE=1 CMF_GRAPH_SPEC=0
cd /Users/oleg/dev/cmf-orca && cargo test --release -p cortiq-engine --lib -- native_metal_batch_spec speculative_trial > $R/tdef.log 2>&1; echo "test rc=$?" >> $L; grep -E "^test |test result" $R/tdef.log >> $L
echo DEFAULT_DONE >> $L
