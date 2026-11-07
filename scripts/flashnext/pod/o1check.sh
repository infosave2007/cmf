#!/bin/bash
# o1check.sh: the 6045-token prompt answered exact and with the far field sealed at 4096 rows; the texts compared.
set -uo pipefail
cd /root/rip
fnx/gen.sh o1off data/aquarium.md 160 CMF_QWEN_O1=off
fnx/gen.sh o1on data/aquarium.md 160 CMF_QWEN_O1=32,1800,4,4096
a=$(awk "/^Ready:/{f=1} f" logs/gen_o1off.out | tail -c 1200); b=$(awk "/^Ready:/{f=1} f" logs/gen_o1on.out | tail -c 1200)
echo "--- exact:"; echo "$a" | tail -c 700; echo; echo "--- sealed:"; echo "$b" | tail -c 700; echo
echo O1CHECK_DONE
