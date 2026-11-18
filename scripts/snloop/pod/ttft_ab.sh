#!/bin/bash
# ttft_ab.sh BIN FILE BACKEND PROMPT_FILE MAXTOK [ENV ...]: `cortiq run --greedy --no-think` once per ENV
# (each a space-free VAR=VAL or "-" for none), after one warm run; first token, decode, drafts and the answer.
set -u
B=$1; F=$2; BK=$3; P=$4; N=$5; shift 5
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=270000
[ $BK = cuda ] && export CMF_GPU=cuda
$B run $F --prompt "Hi" --greedy --no-think --max-tokens 4 > /dev/null 2>&1
for e in "$@"; do
  E=""; [ "$e" != "-" ] && E="$e"
  o=$(env $E timeout 900 $B run $F --prompt-file $P --greedy --no-think --max-tokens $N 2>/dev/null)
  ans=$(echo "$o" | sed -n '/^Prompt:/,$p' | tail -n +2 | grep -v '^\[' | tr '\n' ' ' | tail -c 120)
  echo "$BK ${e}: $(echo "$o" | grep -oE "first token [0-9.]+ s · decode [0-9.]+ tok/s") $(echo "$o" | grep -oE "MTP draft: [0-9]+ of [0-9]+ accepted \([0-9]+%\)") | md5 $(echo "$ans" | md5sum | cut -c1-8) | ...${ans: -60}"
done
