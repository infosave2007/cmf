#!/bin/bash
# Fresh general text for the 56M research, files the 56M has not seen (it was raised on file 0 of
# each): FineWeb-Edu sample/10BT 001-002, FineMath 4plus 00001, FineWeb-2 rus_Cyrl 000_00001
# (all ODC-By 1.0). Each source in its own directory; size and sha256 against the HF tree API.
# Log /root/sk/raw/dl.log, marker /root/sk/raw/dl.done (or dl.failed).
set -u
R=/root/sk/raw; mkdir -p $R
H=https://huggingface.co/datasets
get() { # dir dataset path
  local dir="$R/$1" ds="$2" p="$3"; mkdir -p "$dir"
  local f="$dir/$(basename "$p")"
  local meta; meta=$(curl -s "https://huggingface.co/api/datasets/$ds/tree/main/$(dirname "$p")" | python3 -c "
import json,sys
for x in json.load(sys.stdin):
    if x['path']=='$p': print(x['size'], x['lfs']['oid'])")
  local size=${meta% *} sha=${meta#* }
  for try in 1 2 3 4 5 6 7 8; do
    [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -ge "$size" ] && break
    curl -sL -C - --retry 10 --retry-delay 5 --speed-limit 200000 --speed-time 60 -o "$f" "$H/$ds/resolve/main/$p"
  done
  local got; got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$sha" ] && echo "OK  $f $size" || { echo "BAD $f (size $(stat -c %s "$f"), sha $got, want $sha)"; return 1; }
}
{
  ok=0
  get fineweb-edu HuggingFaceFW/fineweb-edu sample/10BT/001_00000.parquet & 
  get fineweb-edu HuggingFaceFW/fineweb-edu sample/10BT/002_00000.parquet &
  get finemath HuggingFaceTB/finemath finemath-4plus/train-00001-of-00064.parquet &
  get fineweb2-rus HuggingFaceFW/fineweb-2 data/rus_Cyrl/train/000_00001.parquet &
  for j in $(jobs -p); do wait $j || ok=1; done
  [ $ok = 0 ] && touch $R/dl.done || touch $R/dl.failed
  df -h /root | tail -1
} > $R/dl.log 2>&1
