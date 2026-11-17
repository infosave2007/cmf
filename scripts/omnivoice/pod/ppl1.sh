#!/bin/bash
# Is Vulkan's early greedy split on Ornith q4tp precision or a bug in the
# system-memory expert banks? Teacher-forced perplexity over 256 tokens: CPU,
# Vulkan with the banks in system memory (default), Vulkan without
# (CMF_SYS_EXPERTS=0: device prefix + host tail). After sweep1.sh.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/ppl1
end=$((SECONDS + 14400))
until grep -qE "SWEEP1_DONE|SWEEP1_BUILD_FAILED" logs/sweep1.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
B=bin/cortiq-fast; f=cmf/big/ornith-1.5-122b-q4tp.cmf
python3 - <<'PY'
text = ("In the early nineteenth century, railways changed how goods and people crossed a country. Before them, "
        "a journey of a hundred miles could take days on muddy roads, and the price of coal doubled with every "
        "dozen miles it travelled. The first lines were short and built for freight, but passengers soon crowded "
        "onto them, and within a generation towns that had never seen a stranger found themselves an hour from "
        "the capital. Clocks were set to a single national time so that timetables could be printed, newspapers "
        "arrived on the morning they were written, and fresh fish reached inland markets before it spoiled. ")
open("prompts/rail.txt", "w").write(text * 3)
PY
for v in cpu sys nosys; do
  case $v in cpu) e="CMF_GPU=0";; sys) e="CMF_GPU=1";; nosys) e="CMF_GPU=1 CMF_SYS_EXPERTS=0";; esac
  flock gpu.lock env $e timeout 3600 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/ppl1/$v.log 2>&1
  echo "$v: exit $? | $(grep -hiE 'ppl|perplexity' logs/ppl1/$v.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
done
echo PPL1_DONE
