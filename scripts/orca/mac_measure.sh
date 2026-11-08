#!/bin/bash
# OrcaCyber-Nano-1.0 Turbo on the MacBook Air M4 24 GB (native Metal), one run at a time with pauses:
# five prompts without and with drafts, a 3 600-token prompt on the GPU and with the Neural Engine, a 16 000-token
# prompt (past the seal) with the Neural Engine. mac_measure.sh LOG
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf
P=/Users/oleg/dev/orca-runs/scripts/p; L=$1; : > $L
until grep -q "MATCH" $R/pdl.log 2>/dev/null && [ -f $R/build.status ]; do sleep 20; done
grep -E "sha256|rc=" $R/pdl.log $R/build.status | tee -a $L
echo "binary: $($B --version); $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB" | tee -a $L
one() { # NAME PROMPT_FILE TOKENS ENV...
  local name=$1 f=$2 n=$3; shift 3
  sleep 20
  env "$@" RUST_LOG=info /usr/bin/time -l $B run $M --prompt-file "$f" --greedy --no-think --max-tokens $n > $R/one.out 2> $R/one.err
  echo "$name: $(grep -E '^\[' $R/one.out | head -2 | tr '\n' ' ' | cut -c1-200) | text $(grep -vE '^\[' $R/one.out | md5 -q | cut -c1-8) | peak footprint $(grep 'peak memory footprint' $R/one.err | awk '{printf "%.2f GB", $1/1073741824}')" | tee -a $L
  sed -e 's/\x1b\[[0-9;]*m//g' $R/one.err | grep -iE "panicked|ERROR|graph|speculation trial|ane" | sed -e 's/^.*Z  INFO //' | head -4 | cut -c1-180 >> $L
}
for p in code review logs prose ru; do
  one "$p plain" $P/p_$p.txt 220 CMF_GRAPH_SPEC=0
  one "$p drafts" $P/p_$p.txt 220 CMF_GRAPH_SPEC=1
done
python3 - > $R/p3600.txt <<'PY'
t=open("/Users/oleg/dev/orca-runs/wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(t[5000:19500]+"\n\nIn one paragraph, what is the last article above about?")
PY
one "3600 GPU" $R/p3600.txt 120 CMF_GRAPH_SPEC=1
one "3600 ANE" $R/p3600.txt 120 CMF_GRAPH_SPEC=1 CMF_ANE=1
python3 - > $R/p16k.txt <<'PY'
t=open("/Users/oleg/dev/orca-runs/wiki_test.txt",encoding="utf-8").read().replace("\r","")
print(t[5000:70000]+"\n\nIn one paragraph, what is the last article above about?")
PY
one "16000 ANE" $R/p16k.txt 160 CMF_GRAPH_SPEC=1 CMF_ANE=1
echo MAC_DONE >> $L
