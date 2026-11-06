#!/bin/bash
# Dory's turbo file over a long prompt on Vulkan: peaks of device memory, RAM and disk, rows in RAM and
# in files, then planted facts. spill.sh BIN LOG, from a directory with m/dory/dory-turbo.cmf,
# wiki_test.txt and venv/; `src` is the repository. The first argument can be an older binary.
export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=70000 RUST_LOG=cortiq_engine=info
B=$1; L=$2
: > "$L"
V=src/scripts/dory/vram_peak.sh
T=m/dory/dory-turbo.cmf
X="$B bench $T --core --tokens 40 --ignore-eos --ctx"
SP=$PWD/sp
rm -rf "$SP"; mkdir -p "$SP"
seals() { grep -E "O\(1\) sealed|rebuilt over" /tmp/vp.out | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/.*dory: /   /' | cut -c1-110 | tail -2 | tee -a "$L"; }
for c in 1000 12000 32000 64000; do
  timeout 900 $V "RAM $c" "$L" "$SP" -- $X $c; seals
done
for c in 12000 64000; do
  CMF_O1_SPILL=$SP timeout 900 $V "files $c" "$L" "$SP" -- $X $c; seals
done
echo "files left: $(ls "$SP" | wc -l)" | tee -a "$L"
timeout 1500 venv/bin/python src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 8 2>&1 | cut -c1-170 | tee -a "$L"
timeout 1500 venv/bin/python src/scripts/dory/passkey2.py $B $T wiki_test.txt 100000 8 "CMF_O1_SPILL=$SP" 2>&1 | cut -c1-170 | tee -a "$L"
echo DONE >> "$L"
