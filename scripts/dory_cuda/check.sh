#!/bin/bash
# Dory on CUDA against the host: the prompt's last logits (CMF_DORY_DUMP) and
# a greedy answer, per file.
#   check.sh <bin> <file> <tag> [tokens]
set -u
bin=$1; f=$2; tag=$3; n=${4:-64}
cd /root/cuda
d=out/dk_$tag; rm -rf $d; mkdir -p $d/cpu $d/cuda
p="Write a Rust function that reverses a list."
for arm in cuda cpu; do
  g=cuda; [ $arm = cpu ] && g=0
  env CMF_GPU=$g CMF_GPU_PROBE=0 XDG_RUNTIME_DIR=/tmp CMF_DORY_DUMP=$d/$arm timeout 1200 $bin run $f --prompt "$p" --greedy --no-think -n $n > $d/$arm.out 2> $d/$arm.log
  echo "$arm rc=$? $(grep -h 'tokens · first' $d/$arm.out | cut -c1-100) text $(grep -v -E 'tokens · first token|^\[' $d/$arm.out | md5sum | cut -c1-8)"
done
python3 - "$d" <<'PY'
import sys, os, struct
d = sys.argv[1]
names = sorted(os.listdir(d + "/cpu"), key=lambda s: int(s.split(".")[0]))[:1]
for nm in names:
    a = open(f"{d}/cpu/{nm}", "rb").read(); b = open(f"{d}/cuda/{nm}", "rb").read()
    n = min(len(a), len(b)) // 4
    x = struct.unpack(f"{n}f", a[:n*4]); y = struct.unpack(f"{n}f", b[:n*4])
    md = max(abs(p - q) for p, q in zip(x, y)); sc = max(abs(p) for p in x)
    ax = max(range(n), key=lambda i: x[i]); ay = max(range(n), key=lambda i: y[i])
    print(f"logits at {nm}: {n} values, max |diff| {md:.4g} of max {sc:.4g}, argmax cpu {ax} cuda {ay}")
PY
