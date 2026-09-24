#!/usr/bin/env python3
"""Compare two CMF_LOGIT_DUMP_ALL directories step by step.

Each directory holds `step{n:05}.f32` — one decode step's logits as raw
little-endian f32 (the engine writes them when CMF_LOGIT_DUMP_ALL=<dir>).
Typical use: the same greedy prompt on the CPU (CMF_GPU=0) and on a GPU
path, then

    python3 tools/logit_steps_cmp.py --ref cpu_dir --test gpu_dir [--tol 1e-3]

For every step present in both: max|a-b| / max|a| (relative to the
reference's largest logit), whether the argmax agrees, and the reference's
top-1/top-2 margin (a flip inside a near-tie is reported as such). The
comparison stops at the first step whose argmax differs — past it the two
runs decode different tokens and are no longer comparable.
Exit status 0 when every compared step agrees on top-1 and stays under
--tol, 1 otherwise.
"""
import argparse
import glob
import os
import struct
import sys


def load(path):
    with open(path, "rb") as f:
        b = f.read()
    return struct.unpack("<%df" % (len(b) // 4), b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    steps = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(a.ref, "step*.f32"))
    )
    steps = [s for s in steps if os.path.exists(os.path.join(a.test, s))]
    if not steps:
        print("no common steps")
        return 1
    worst = 0.0
    ok = True
    compared = 0
    for s in steps:
        r = load(os.path.join(a.ref, s))
        t = load(os.path.join(a.test, s))
        n = min(len(r), len(t))
        r, t = r[:n], t[:n]
        mx = max(abs(v) for v in r) or 1.0
        d = max(abs(x - y) for x, y in zip(r, t))
        rel = d / mx
        ar = max(range(n), key=r.__getitem__)
        at = max(range(n), key=t.__getitem__)
        srt = sorted(r, reverse=True)
        margin = srt[0] - srt[1]
        compared += 1
        worst = max(worst, rel)
        flag = ""
        if ar != at:
            flag = " TOP1-DIFF (ref margin %.4g)" % margin
            ok = False
        if rel >= a.tol:
            flag += " OVER-TOL"
            ok = False
        if not a.quiet or flag:
            print(
                "%s max|d| %.3e rel %.3e top1 %d/%d margin %.4g%s"
                % (s, d, rel, ar, at, margin, flag)
            )
        if ar != at:
            break
    print(
        "compared %d step(s): worst rel %.3e, %s"
        % (compared, worst, "PASS" if ok else "FAIL")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
