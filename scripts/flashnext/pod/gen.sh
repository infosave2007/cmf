#!/bin/bash
# gen.sh NAME PROMPTFILE TOKENS [ENV=VALUE ...]: a greedy direct answer of $FILE with the profile; logs/gen_NAME.{out,err}.
set -uo pipefail
cd /root/rip && . fnx/env.sh
n=$1; p=$2; t=$3; shift 3
env "$@" CMF_QWEN_PROFILE=$PROFILE $BIN run $FILE --prompt-file $p --greedy --no-think --max-tokens $t > logs/gen_$n.out 2> logs/gen_$n.err
echo "exit $?"
grep -a -o "\[[0-9]* tokens[^]]*\]" logs/gen_$n.out | tail -1
sed -e "s/\x1b\[[0-9;]*m//g" logs/gen_$n.err | grep -a -i -E "o1|seal|far|refus|declin|panic|error|warn" | cut -c1-200 | tail -8
echo GEN_DONE
