#!/bin/bash
# The ANE's surfaces released after the prompt: rebuild, then 3 600 and 16 000 tokens with the ANE and drafts by default.
R=/Users/oleg/dev/orca-runs/mac; B=/Users/oleg/dev/cmf-orca/target/release/cortiq; M=$R/orcacyber-nano-1.0-turbo.cmf; L=$1; : > $L
(cd /Users/oleg/dev/cmf-orca && t0=$(date +%s) && cargo build --release -p cortiq-cli > $R/build4.log 2>&1; echo "build rc=$? $(( $(date +%s)-t0 )) s" >> $L)
sw() { sysctl -n vm.swapusage | awk '{print $6}'; }
for arm in "3600 p3600.txt 120" "16000 p16k.txt 160"; do
  set -- $arm; sleep 90; s0=$(sw)
  CMF_ANE=1 RUST_LOG=info /usr/bin/time -l $B run $M --prompt-file $R/$2 --greedy --no-think --max-tokens $3 > $R/ar_$1.out 2> $R/ar_$1.err
  echo "$1 ANE+drafts: $(grep -E '^\[' $R/ar_$1.out | head -2 | tr '\n' ' ' | cut -c1-170) | text $(grep -vE '^\[' $R/ar_$1.out | md5 -q | cut -c1-8) | faults $(grep 'page faults' $R/ar_$1.err | awk '{print $1}') | swap $s0->$(sw)" | tee -a $L
  sed -e 's/\x1b\[[0-9;]*m//g' $R/ar_$1.err | grep -E "speculation trial" | sed -e 's/^.*pipeline: //' | head -2 | cut -c1-140 >> $L
done
echo AR_DONE >> $L
