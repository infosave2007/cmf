#!/usr/bin/env python3
"""Per-position NLL from CMF_PPL_TRACE logs, summed in position bins.

Each log may hold several passes (positions restart: an --o1 run prints
the O(1) pass, then the exact pass). Prints one column per pass, NLL per
token in each bin, over the positions every pass scored.

Usage: ppl_trace_bins.py [--bin 512] [--start 256] LOG[:pass] ...
  LOG:1 takes the second pass of LOG (0-based), default 0.
"""
import argparse
import re
import sys

TRACE = re.compile(r"^(?:BTRACE )?pos\s+(\d+)\s+(?:tgt|target)\s+(\d+)\s+nll\s+([-\d.]+)")


def passes(path):
    out, cur, last = [], {}, -1
    for line in open(path, errors="replace"):
        m = TRACE.match(line.strip())
        if not m:
            continue
        pos, nll = int(m[1]), float(m[3])
        if pos <= last:
            out.append(cur)
            cur = {}
        cur[pos] = nll
        last = pos
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=512)
    ap.add_argument("--start", type=int, default=256)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    cols = []
    for spec in a.logs:
        path, _, idx = spec.partition(":")
        ps = passes(path)
        k = int(idx or 0)
        if k >= len(ps):
            sys.exit(f"{path}: {len(ps)} pass(es), wanted #{k}")
        cols.append((spec, ps[k]))
    common = sorted(set.intersection(*(set(c) for _, c in cols)))
    common = [p for p in common if p >= a.start]
    print(f"{'bin':>11} " + " ".join(f"{s[-18:]:>18}" for s, _ in cols))
    lo = common[0] if common else 0
    while lo <= common[-1]:
        hi = lo + a.bin
        ps = [p for p in common if lo <= p < hi]
        if ps:
            row = [sum(c[p] for p in ps) / len(ps) for _, c in cols]
            print(f"{lo:>5}-{hi - 1:<5} " + " ".join(f"{v:18.4f}" for v in row))
        lo = hi
    tot = [sum(c[p] for p in common) for _, c in cols]
    print(f"{'total':>11} " + " ".join(f"{v:18.2f}" for v in tot) + f"   ({len(common)} positions)")


if __name__ == "__main__":
    main()
