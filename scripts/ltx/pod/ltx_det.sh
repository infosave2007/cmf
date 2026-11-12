#!/bin/bash
# Does a seed reproduce? The reference clip three times on one backend, the
# prompt encoded every time, every step's velocities and the frames compared.
#   ltx_det.sh BACKEND [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
be=$1; p=${2:-ld}
cd "$root" || exit 1
py=$root/venv/bin/python
rm -rf out/dumpd; mkdir -p out/dumpd
for i in 1 2 3; do
  EXTRA=--no-context-cache CMF_LTX_DUMP=$root/out/dumpd/r$i bash scripts/ltx.sh ${p}_${be}_$i $be | head -8
done
for i in 2 3; do
  worst=0
  for f in out/dumpd/r1.*.f32; do
    t=${f##*/r1.}
    d=$($py scripts/cmp_f32.py $f out/dumpd/r$i.$t | sed 's/.*rel rms diff \([^ ]*\).*/\1/')
    worst=$($py -c "print(max($worst, $d))")
  done
  echo "run 1 vs $i: worst rel rms diff of a step's velocities $worst; frames: $($py scripts/cmp_y4m.py out/${p}_${be}_1.y4m out/${p}_${be}_$i.y4m)"
done
rm -rf out/dumpd out/${p}_${be}_2.y4m out/${p}_${be}_3.y4m
echo LTX_DET_DONE
