#!/bin/bash
# 16 000 tokens past the seal: drafts against none, with and without the ANE; swap and page-outs around each run.
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf; L=$1; : > $L
po() { vm_stat | awk '/Pageouts/ {gsub(/\./,"",$2); print $2}'; }
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
for arm in "plain+ANE CMF_GRAPH_SPEC=0 CMF_ANE=1" "drafts+ANE CMF_ANE=1" "drafts-GPU CMF_ANE=0"; do
  set -- $arm; name=$1; shift; sleep 90
  p0=$(po); s0=$(sw)
  env "$@" RUST_LOG=info CMF_GRAPH_SPEC_TIME=1 /usr/bin/time -l $B run $M --prompt-file $R/p16k.txt --greedy --no-think --max-tokens 160 > $R/m16_$name.out 2> $R/m16_$name.err
  p1=$(po); s1=$(sw)
  echo "$name: $(grep -E '^\[' $R/m16_$name.out | head -2 | tr '\n' ' ' | cut -c1-170) | pageouts +$((p1-p0)) swap $s0->$s1 | footprint $(grep 'peak memory footprint' $R/m16_$name.err | awk '{printf "%.2f GB", $1/1073741824}') | faults $(grep 'page faults' $R/m16_$name.err | awk '{print $1}')" | tee -a $L
  sed -e 's/\x1b\[[0-9;]*m//g' $R/m16_$name.err | grep -E "speculation" | sed -e 's/^.*pipeline: //' | cut -c1-140 >> $L
  sed -e 's/\x1b\[[0-9;]*m//g' $R/m16_$name.err | grep -E "spec-round wall" | awk '{print $3}' | head -12 | tr '\n' ' ' | sed 's/^/round walls ms: /' >> $L; echo >> $L
done
echo M16_DONE >> $L
