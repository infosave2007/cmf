#!/bin/bash
# Three runs of the clip with the stage checksums of the first forward
# (CMF_LTX_DBG), under the environment given: do they agree?
#   ENVVAR=1 ltx_dbg3.sh BACKEND TAG
set -u
root=${H3I_ROOT:-/root/h3i}
be=$1; p=$2
cd "$root" || exit 1
for i in 1 2 3; do
  CMF_LTX_DBG=1 bash scripts/ltx.sh ${p}_$i $be > /dev/null 2>&1
  echo "$p run $i: $(grep -m1 "ltx dbg blocks" out/${p}_$i.log) | $(grep -m1 "stage 1 denoised" out/${p}_$i.log)"
  rm -f out/${p}_$i.y4m out/${p}_$i.wav
done
