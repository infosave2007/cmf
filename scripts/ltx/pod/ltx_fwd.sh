#!/bin/bash
# One forward of the LTX transformer, the resident chain against the
# per-operation path, on one backend (CMF_LTX_DUMP: the velocities of every
# sampling step). Step 0 differs by the transformer alone; the later steps
# show how the sampler amplifies it. The clip of the chain is kept.
#   ltx_fwd.sh BACKEND [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
be=$1; p=${2:-lf}
cd "$root" || exit 1
py=$root/venv/bin/python
mkdir -p out/dump
CMF_LTX_DUMP=$root/out/dump/${p}_chain bash scripts/ltx.sh ${p}_${be}_chain $be | head -3
CMF_LTX_CHAIN=0 CMF_LTX_DUMP=$root/out/dump/${p}_perop bash scripts/ltx.sh ${p}_${be}_perop $be | head -3
for f in out/dump/${p}_chain.v*.f32; do
  t=${f##*chain.}
  echo "$t  $($py scripts/cmp_f32.py $f out/dump/${p}_perop.$t)"
  t2=a${t#v}
  echo "$t2  $($py scripts/cmp_f32.py out/dump/${p}_chain.$t2 out/dump/${p}_perop.$t2)"
done
$py scripts/cmp_y4m.py out/${p}_${be}_chain.y4m out/${p}_${be}_perop.y4m
rm -rf out/dump out/${p}_${be}_perop.y4m
echo LTX_FWD_DONE
