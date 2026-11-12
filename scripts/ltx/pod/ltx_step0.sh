#!/bin/bash
# Is the chain's first forward right? The chain twice (every step compared:
# is it deterministic), then the per-operation path with the op arbitration
# pinned (CMF_GPU_PROBE=0) until its first forward is written, compared with
# the chain's.
#   ltx_step0.sh BACKEND [TAGPREFIX]
set -u
root=${H3I_ROOT:-/root/h3i}
be=$1; p=${2:-l0}
cd "$root" || exit 1
py=$root/venv/bin/python
rm -rf out/dump0; mkdir -p out/dump0
CMF_LTX_DUMP=$root/out/dump0/a bash scripts/ltx.sh ${p}_${be}_a $be | head -8
CMF_LTX_DUMP=$root/out/dump0/b bash scripts/ltx.sh ${p}_${be}_b $be | head -1
for f in out/dump0/a.*.f32; do
  t=${f##*/a.}
  echo "chain twice $t  $($py scripts/cmp_f32.py $f out/dump0/b.$t)"
done
$py scripts/cmp_y4m.py out/${p}_${be}_a.y4m out/${p}_${be}_b.y4m
( CMF_GPU_PROBE=0 CMF_LTX_CHAIN=0 CMF_LTX_DUMP=$root/out/dump0/p bash scripts/ltx.sh ${p}_${be}_p $be > /dev/null 2>&1 & )
deadline=$(( $(date +%s) + 400 ))
until [ -s out/dump0/p.a1.0000.f32 ] || [ $(date +%s) -ge $deadline ]; do sleep 3; done
sleep 2
pid=$(ps -eo pid,args | awk '/[c]ortiq[a-z-]* ltx-video/ {print $1}' | head -1)
[ -n "$pid" ] && kill $pid
for s in v a; do
  echo "step 0 $s  chain / per-op, arbitration pinned: $($py scripts/cmp_f32.py out/dump0/a.${s}1.0000.f32 out/dump0/p.${s}1.0000.f32)"
done
rm -rf out/dump0 out/${p}_${be}_b.y4m out/${p}_${be}_p.*
echo LTX_STEP0_DONE
