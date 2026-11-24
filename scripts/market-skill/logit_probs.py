#!/usr/bin/env python3
"""The logistic baseline of market_skill.baselines (same features, same 6000 train rows a market, same fit), with
its probability kept, for the accuracy-against-coverage chart.

  logit_probs.py DATA_DIR TARGET SPLIT OUT.jsonl
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import market_skill as ms  # noqa: E402


def fit(train, keys, classes, iters=300, lr=0.5, l2=1e-3):
    # market_skill.logistic, returning the model instead of the labels
    X = [[r["feats"][k] for k in keys] + [math.sin(2 * math.pi * r["hour"] / 24), math.cos(2 * math.pi * r["hour"] / 24)] for r in train]
    y = [1.0 if r["label"] == classes[1] else 0.0 for r in train]
    d = len(X[0])
    mu = [ms.mean([x[j] for x in X]) for j in range(d)]
    sd = [ms.std([x[j] for x in X]) or 1.0 for j in range(d)]
    Z = [[(x[j] - mu[j]) / sd[j] for j in range(d)] for x in X]
    w, b = [0.0] * d, 0.0
    n = len(Z)
    for _ in range(iters):
        gw, gb = [0.0] * d, 0.0
        for z, t in zip(Z, y):
            s = b + sum(wj * zj for wj, zj in zip(w, z))
            p = 1 / (1 + math.exp(-max(-30, min(30, s))))
            e = p - t
            gb += e
            for j in range(d):
                gw[j] += e * z[j]
        b -= lr * gb / n
        w = [wj - lr * (gj / n + l2 * wj) for wj, gj in zip(w, gw)]
    return w, b, mu, sd


def main():
    data, target, split, out = sys.argv[1:5]
    classes = ms.CLASSES[target]
    d = os.path.join(data, target)
    train = ms.load(os.path.join(d, "train.jsonl"))
    ev = ms.load(os.path.join(d, f"eval_{split}.jsonl"))
    rows = []
    for m in sorted({r["symbol"] for r in ev}):
        tr = [r for r in train if r["symbol"] == m]
        te = [r for r in ev if r["symbol"] == m]
        keys = [k for k in te[0]["feats"]]
        w, b, mu, sd = fit(ms.evenly(tr, 6000), keys, classes)
        for r in te:
            x = [r["feats"][k] for k in keys] + [math.sin(2 * math.pi * r["hour"] / 24), math.cos(2 * math.pi * r["hour"] / 24)]
            s = b + sum(wj * (xj - m_) / sj for wj, xj, m_, sj in zip(w, x, mu, sd))
            p1 = 1 / (1 + math.exp(-max(-30, min(30, s))))
            pred = classes[1] if s > 0 else classes[0]
            rows.append({"timestamp": r["timestamp"], "symbol": m, "label": r["label"], "pred": pred,
                         "p": {classes[1]: p1, classes[0]: 1 - p1}})
        acc = sum(x["pred"] == x["label"] for x in rows if x["symbol"] == m) / len(te)
        print(f"{m}: acc {acc*100:.1f}%", flush=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
