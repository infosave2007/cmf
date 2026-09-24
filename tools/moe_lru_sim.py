#!/usr/bin/env python3
"""Replay a CMF_MOE_TRACE routing log through expert caches.

The trace (pipeline.rs `moe_trace_at`) has one `layer:e1,e2,...` line per
(layer, token), layers in execution order. This answers the question a
device expert bank lives on: how often is a token's pick already resident
if each layer keeps its S most recently used experts?

    python3 tools/moe_lru_sim.py trace.txt --slots 32,64,128,192 [--warm 64]

Per slot count it prints the per-layer LRU hit rate (fill every miss) over
all tokens and over tokens after the first --warm ones, the model-wide LRU
with the same total (S x layers) and the cold picks per token those rates
imply. Compulsory misses (first sight of a (layer, expert)) are counted too:
no cache of any size avoids them.
"""
import argparse
from collections import OrderedDict, defaultdict


def load(path):
    rows = []  # (layer, [experts])
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            li, ids = line.split(":", 1)
            rows.append((int(li), [int(e) for e in ids.split(",") if e != ""]))
    return rows


def tokens_of(rows):
    """Token n = the n-th line of every layer. Order-independent: a paired
    forward (two positions per layer) interleaves lines as l:x1, l:x2, and
    only the per-layer order is the token order."""
    per = defaultdict(list)
    for li, ids in rows:
        per[li].append(ids)
    n = min((len(v) for v in per.values()), default=0)
    return [[(li, per[li][t]) for li in sorted(per)] for t in range(n)]


def per_layer_lru(toks, slots, warm):
    caches = defaultdict(OrderedDict)
    h = n = hw = nw = 0
    for t, tok in enumerate(toks):
        for li, ids in tok:
            c = caches[li]
            for e in ids:
                hit = e in c
                h += hit
                n += 1
                if t >= warm:
                    hw += hit
                    nw += 1
            for e in ids:
                if e in c:
                    c.move_to_end(e)
                else:
                    c[e] = True
                    while len(c) > slots:
                        c.popitem(last=False)
    return h / max(n, 1), hw / max(nw, 1), n


def global_lru(toks, total, warm):
    c = OrderedDict()
    h = n = hw = nw = 0
    for t, tok in enumerate(toks):
        for li, ids in tok:
            for e in ids:
                hit = (li, e) in c
                h += hit
                n += 1
                if t >= warm:
                    hw += hit
                    nw += 1
            for e in ids:
                k = (li, e)
                if k in c:
                    c.move_to_end(k)
                else:
                    c[k] = True
                    while len(c) > total:
                        c.popitem(last=False)
    return h / max(n, 1), hw / max(nw, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slots", default="32,64,128,192")
    ap.add_argument("--warm", type=int, default=64)
    a = ap.parse_args()
    rows = load(a.trace)
    toks = tokens_of(rows)
    layers = sorted({li for li, _ in rows})
    picks_per_tok = sum(len(ids) for _, ids in toks[0]) if toks else 0
    first = set()
    compulsory = 0
    for tok in toks:
        for li, ids in tok:
            for e in ids:
                if (li, e) not in first:
                    first.add((li, e))
                    compulsory += 1
    total_picks = sum(len(ids) for _, ids in rows)
    print(
        f"trace: {len(toks)} tokens x {len(layers)} MoE layers, {picks_per_tok} picks/token, "
        f"{len(first)} distinct (layer, expert) = {len(first) / max(len(layers), 1):.1f}/layer, "
        f"compulsory misses {compulsory / max(total_picks, 1) * 100:.1f}% of picks"
    )
    print(f"{'slots/layer':>11} | {'per-layer LRU':>13} | {'after warm':>10} | "
          f"{'model-wide LRU':>14} | {'after warm':>10} | cold picks/token (per-layer, after warm)")
    for s in [int(x) for x in a.slots.split(",")]:
        pl, plw, _ = per_layer_lru(toks, s, a.warm)
        gl, glw = global_lru(toks, s * len(layers), a.warm)
        print(f"{s:>11} | {pl * 100:>12.1f}% | {plw * 100:>9.1f}% | {gl * 100:>13.1f}% | "
              f"{glw * 100:>9.1f}% | {picks_per_tok * (1 - plw):.1f}")


if __name__ == "__main__":
    main()
