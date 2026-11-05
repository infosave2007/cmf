#!/usr/bin/env python3
"""How well rows of a planted fact can be found. Each feature as a per-prompt rank; the best
single ones, then a linear fit over all of them, tested on prompts and on kinds of fact it did
not see. illum_fit.py FEAT.npz"""
import sys
import numpy as np

d = np.load(sys.argv[1], allow_pickle=True)
Xs, ys, kinds, full = d["X"], d["y"], d["kind"], d["full"]
n, nf = len(Xs), Xs[0].shape[1]
names = []
for li in full:
    for what in ("res", "vnorm", "knorm"):
        names += [f"{what} L{li} h{h}" for h in range(4)]
names.append("surprise")


def pct(v):
    o = np.argsort(np.argsort(v, axis=0), axis=0)
    return o / (v.shape[0] - 1)


P = [pct(x) for x in Xs]


def score(s, y):
    """% of rows above the fact's best, mean and worst row."""
    r = np.argsort(np.argsort(-s)) / len(s) * 100
    return r[y].min(), r[y].mean(), r[y].max()


single = []
for f in range(nf):
    a = np.array([score(P[i][:, f], ys[i].astype(bool)) for i in range(n)]).mean(0)
    single.append((a[1], a[0], a[2], names[f]))
single.sort()
print("best single features: mean / best / worst row, % of the prompt above")
for a, b, c, nm in single[:12]:
    print(f"  {a:6.2f} {b:6.2f} {c:6.2f}  {nm}")


def fit(train, lam=1.0):
    """A linear score from ranks: LDA with a ridge."""
    pos = np.concatenate([P[i][ys[i].astype(bool)] for i in train])
    neg = np.concatenate([P[i][~ys[i].astype(bool)][::7] for i in train])
    mu = pos.mean(0) - neg.mean(0)
    C = np.cov(neg.T) + lam * 1e-2 * np.eye(nf)
    return np.linalg.solve(C, mu)


loo = np.array([score(P[i] @ fit([j for j in range(n) if j != i]), ys[i].astype(bool)) for i in range(n)])
print(f"linear, leave one prompt out: mean {loo[:,1].mean():.2f} best {loo[:,0].mean():.2f} worst {loo[:,2].mean():.2f}")
for k in sorted(set(kinds)):
    te = [i for i in range(n) if kinds[i] == k]
    tr = [i for i in range(n) if kinds[i] != k]
    w = fit(tr)
    a = np.array([score(P[i] @ w, ys[i].astype(bool)) for i in te]).mean(0)
    print(f"  leave kind {k} out: mean {a[1]:.2f} best {a[0]:.2f} worst {a[2]:.2f}")
for what, sel in (("res only", [i for i, x in enumerate(names) if x.startswith("res")]), ("res+vnorm", [i for i, x in enumerate(names) if x.startswith(("res", "vnorm"))]), ("res+vnorm+surprise", [i for i, x in enumerate(names) if x.startswith(("res", "vnorm", "surprise"))])):
    r = []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        pos = np.concatenate([P[j][ys[j].astype(bool)][:, sel] for j in tr])
        neg = np.concatenate([P[j][~ys[j].astype(bool)][::7][:, sel] for j in tr])
        w = np.linalg.solve(np.cov(neg.T) + 1e-2 * np.eye(len(sel)), pos.mean(0) - neg.mean(0))
        r.append(score(P[i][:, sel] @ w, ys[i].astype(bool)))
    r = np.array(r)
    print(f"{what:20s}: mean {r[:,1].mean():.2f} best {r[:,0].mean():.2f} worst {r[:,2].mean():.2f}")
