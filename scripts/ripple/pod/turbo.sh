#!/bin/bash
# turbo.sh PLAIN NAME: out/PLAIN.cmf with the far field in its header as hf/NAME.cmf (the file that
# is published): 32 landmarks, a window of 1800 rows, 4 sinks, 248 exact rows by keep_ripple.bin
# (fit.sh), on for a prompt of 8192 rows or more.
set -uo pipefail
cd /root/rip && mkdir -p hf
./cortiq far-turbo out/$1.cmf hf/$2.cmf --exact-weights keep_ripple.bin 2>&1 | tail -1 | cut -c1-140
ls -l hf/$2.cmf | awk '{print $5, $9}'
echo TURBO_DONE
