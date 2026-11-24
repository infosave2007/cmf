#!/bin/bash
# check.sh: the frozen recipe for 4 steps with CMF_BAKE_RES_CHECK=1 (resident vs layer path on every
# training step), then 20 resident steps timed; dir target
cd ~/mskill
STEPS=4 EVAL=4 BAKE_TIMEOUT=1800 CMF_BAKE_RES_CHECK=1 ./bake.sh dir chk_dir > /dev/null
grep -E "bake-res-check|step 4:|baseline \(full\)|error|panic" runs/chk_dir/bake.log | cut -c1-200
STEPS=20 EVAL=20 BAKE_TIMEOUT=1800 ./bake.sh dir spd_dir > /dev/null
grep -E "step 20:|baseline \(full\)|^\[A\] [0-9]+s|=== bake|error|panic" runs/spd_dir/bake.log | cut -c1-200
STEPS=4 EVAL=4 BAKE_TIMEOUT=1800 CMF_BAKE_RES_CHECK=1 ./bake.sh range chk_range > /dev/null
grep -E "bake-res-check|step 4:|baseline \(full\)|error|panic" runs/chk_range/bake.log | cut -c1-200
echo CHECK_DONE
