#!/usr/bin/env python3
"""Golden-data generator for the CMF Nystrom attention kernel
(crates/cortiq-engine/src/nystrom.rs).

Mirrors NystromJointAttn.head_out from the validated fp64 matrix probe
(cmf/experiments/nystrom_ppl3_06b.py), transposed to STREAMING
semantics so the Rust kernel is directly comparable:

  * landmarks are contiguous segment means of the PREFILL prefix (first
    P tokens) only, frozen afterwards (the matrix probe uses the full
    sequence);
  * for every row t: exact near window j in (t-W, t] PLUS the first
    SINK permanent sink keys (spec 5b near mask: t-j < W OR j < S),
    unnormalized skeleton over far keys S <= j <= t-W with the frozen
    landmarks, and ONE joint denominator — same shifts, same clamp,
    same pinv rtol=1e-6 as the probe.  A sink=0 output set is also
    written as the regression fixture for the sink-free path.

Inputs are rounded to f32 before all math so the f64 reference and the
f32 streaming kernel see identical numbers.

Data regime: time-segmented cluster structure (keys/queries near one of
`m` centers, segment-aligned inside the prefix).  This is deliberate:
segment-mean landmarks then span the key/query manifold, which is the
regime the kernel is validated for (real attention heads).  IID
gaussian data is adversarial for any Nystrom skeleton: the per-(t,j)
clamp of negative skeleton weights fires, and no streaming
implementation can reproduce a per-(t,j) clamp from aggregated far
accumulators.  The generator asserts the clamp never fires here.

It also simulates the streaming algebra (ridge pseudo-inverse + per-
landmark flash max shifts + delayed insertion) in numpy f64 and prints
its gap vs the matrix form, so a tolerance failure in the Rust test can
be attributed to either f32 rounding or an algebra bug.

Run:
  /Users/oleg/Documents/cortiq-bot/venv_heal/bin/python \
      tools/gen_nystrom_golden.py
Writes crates/cortiq-engine/tests/data/nystrom_golden.json.
"""
import json
import os

import numpy as np

D, DV, M, W, T, P = 16, 8, 4, 16, 96, 64
SINK = 4        # permanent exact sink keys (spec 5b); 0 = legacy path
SEED = 20260712
SCALE = 1.0 / np.sqrt(D)
EPS = 1e-30


def seg_means(x, m):
    """Nystromformer landmark recipe: contiguous segment means; x [T,d].
    Integer split (i*T)//m matches the torch probe and the Rust port."""
    t = x.shape[0]
    return np.stack([x[(i * t) // m:((i + 1) * t) // m].mean(0)
                     for i in range(m)])


def make_data(rng):
    """Keys = m cluster atoms + tiny noise, segment-aligned in the
    prefix.  With (near-)atomic keys the kernel matrix has rank ~m and
    the Nystrom skeleton is (near-)exact for ANY queries — the regime
    the kernel is validated for, and the only regime where the matrix
    probe's per-(t,j) clamp provably never fires (see module doc).  The
    tiny key noise keeps every insertion score distinct so the per-
    landmark running-max rescales are exercised; query noise is free.
    Decode tokens cycle the clusters and blend neighbouring centers."""
    centers = rng.standard_normal((M, D)) * 0.7
    q = np.empty((T, D))
    k = np.empty((T, D))
    for t in range(T):
        if t < P:
            g = (t * M) // P            # segment-aligned cluster id
            qc = centers[g]
        else:
            g = t % M                   # decode: cycle + blend clusters
            qc = 0.7 * centers[g] + 0.3 * centers[(g + 1) % M]
        q[t] = qc + 0.5 * rng.standard_normal(D)
        k[t] = centers[g] + 0.02 * rng.standard_normal(D)
    v = rng.standard_normal((T, DV))
    # Round to the f32 grid FIRST: reference f64 math and the f32 kernel
    # must consume bit-identical inputs.
    f32 = lambda a: a.astype(np.float32).astype(np.float64)
    return f32(q), f32(k), f32(v)


def golden_matrix(q, k, v, sink):
    """head_out math with prefix landmarks + streaming near/far split.
    Sink semantics (spec 5b): near mask is (t-j < W) OR (j < sink); the
    far skeleton covers sink <= j <= t-W only."""
    m_eff = max(4, min(M, P // 8))
    kL = seg_means(k[:P], m_eff)
    qL = seg_means(q[:P], m_eff)
    Au = np.exp((qL @ kL.T) * SCALE)
    Mu = np.linalg.pinv(Au, rtol=1e-6)
    Fu = np.exp((q @ kL.T) * SCALE)          # [T,m]
    E = np.exp((qL @ k.T) * SCALE)           # [m,T]
    west = (Fu @ Mu) @ E                     # skeleton estimate of e(t,j)
    n_neg = 0
    neg_mass = 0.0
    lg = (q @ k.T) * SCALE
    out = np.zeros((T, DV))
    for t in range(T):
        c = lg[t, :t + 1].max()              # full causal row shift
        w = np.zeros(t + 1)
        lo = max(0, t - W + 1)               # near: t-j < W  ->  j > t-W
        w[lo:t + 1] = np.exp(lg[t, lo:t + 1] - c)
        se = min(sink, t + 1)                # permanent exact sink keys
        w[:se] = np.exp(lg[t, :se] - c)
        if t - W >= sink:                    # far: sink <= j <= t-W
            row = west[t, sink:t - W + 1]
            n_neg += int((row < 0).sum())
            neg_mass = max(neg_mass, float(-row.clip(max=0).sum())
                           * np.exp(-c) / max(w.sum(), EPS))
            w[sink:t - W + 1] = row.clip(0.0) * np.exp(-c)
        out[t] = (w @ v[:t + 1]) / max(w.sum(), EPS)
    return out, Au, n_neg, neg_mass


def ridge_pinv(a):
    """(A^T A + lam I)^-1 A^T with lam = 1e-6*mean(diag(A^T A)) — the
    regularized pseudo-inverse the Rust kernel computes via Cholesky."""
    ata = a.T @ a
    lam = 1e-6 * np.mean(np.diag(ata))
    return np.linalg.solve(ata + lam * np.eye(a.shape[0]), a.T)


def stream_sim(q, k, v, sink):
    """f64 simulation of the streaming kernel algebra: delayed insertion,
    per-landmark running-max shifts, per-token landmark row shift, joint
    denominator.  Sinks bypass the window, so eviction never sees them.
    Validates the algebra independent of f32 rounding."""
    m_eff = max(4, min(M, P // 8))
    kL = seg_means(k[:P], m_eff)
    qL = seg_means(q[:P], m_eff)
    mu = ridge_pinv(np.exp((qL @ kL.T) * SCALE))
    t_hat = np.zeros((m_eff, DV))
    z_hat = np.zeros(m_eff)
    mmax = np.full(m_eff, -np.inf)
    win = []                                  # list of key indices
    out = np.zeros((T, DV))
    for t in range(T):
        if t >= sink:                         # sinks never enter the ring
            if len(win) == W:                 # delayed insertion at t=j+W
                j = win.pop(0)
                l = (qL @ k[j]) * SCALE
                grow = l > mmax
                r = np.exp(np.where(grow, mmax - l, 0.0))
                t_hat *= r[:, None]
                z_hat *= r
                mmax = np.maximum(mmax, l)
                e = np.exp(l - mmax)
                t_hat += e[:, None] * v[j]
                z_hat += e
            win.append(t)
        near = list(range(min(sink, t + 1))) + win
        s = np.array([(q[t] @ k[j]) * SCALE for j in near])
        c = s.max()
        if z_hat.sum() > 0:
            sl = (q[t] @ kL.T) * SCALE
            f = sl.max()
            fh = np.exp(sl - f)
            u = fh @ mu
            c_all = max(c, f + mmax.max())
            g = u * np.exp(f + mmax - c_all)
            far_den = float(g @ z_hat)
            far_num = g @ t_hat
            if far_den < 0:                   # aggregate skeleton guard
                far_den, far_num = 0.0, np.zeros(DV)
        else:
            c_all, far_den, far_num = c, 0.0, np.zeros(DV)
        p = np.exp(s - c_all)
        den = far_den + p.sum()
        num = far_num + p @ v[near]
        out[t] = num / max(den, EPS)
    return out


def main():
    rng = np.random.default_rng(SEED)
    q, k, v = make_data(rng)
    outs = {}
    for sink in (SINK, 0):
        out, Au, n_neg, neg_mass = golden_matrix(q, k, v, sink)
        sim = stream_sim(q, k, v, sink)
        gap = np.abs(sim - out)
        sv = np.linalg.svd(Au, compute_uv=False)
        print(f'[sink={sink}] Au cond {sv[0] / sv[-1]:.2f} '
              f'sv {np.round(sv, 4)}')
        print(f'[sink={sink}] negative skeleton weights: count={n_neg} '
              f'worst relative mass={neg_mass:.3e}')
        print(f'[sink={sink}] stream-sim vs matrix golden: '
              f'max|d|={gap.max():.3e} mean|d|={gap.mean():.3e}')
        assert n_neg == 0, 'clamp fired: not in the skeleton-clean regime'
        assert gap.max() < 1e-4, 'streaming algebra drifts from matrix'
        outs[sink] = out

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, '..', 'crates', 'cortiq-engine', 'tests',
                        'data', 'nystrom_golden.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {
        'd': D, 'dv': DV, 'm': M, 'w': W, 't': T, 'p': P, 'sink': SINK,
        'q': [[round(float(x), 9) for x in row] for row in q],
        'k': [[round(float(x), 9) for x in row] for row in k],
        'v': [[round(float(x), 9) for x in row] for row in v],
        'out': [[float(x) for x in row] for row in outs[SINK]],
        # sink=0 outputs: the regression fixture for the sink-free path.
        'out_sink0': [[float(x) for x in row] for row in outs[0]],
    }
    with open(path, 'w') as fh:
        json.dump(blob, fh)
    print(f'wrote {os.path.normpath(path)} '
          f'({os.path.getsize(path) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
