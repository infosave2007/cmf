#!/usr/bin/env python3
"""Calibrated skeleton for the O(1) Nystrom attention (`CMF_O1_CALIB`).

Training-free: landmarks and the mixing matrix come from a calibration
corpus in closed form, no gradient step anywhere.

Input: `CMF_O1_CAPTURE` dumps, one file per (layer, group, window):
`L{layer}_g{group}_{n}.f32` = u32 t, d, heads; f32 Q[heads][t][d],
K[t][d], V[t][d] (post-RoPE, exactly what the seal sees). Capture them
with `cortiq ppl ... --o1 all --o1-prefill <window-1>` so the whole
window is sealed.

Output, per (layer, group), the `.o1c` file `nystrom::O1Calib` loads:
  OUT/kmeans/  k-means landmarks K~ (group keys) and Q~ (per-head queries);
               the runtime still builds M = pinv(exp(Q~K~^T/sqrt d)).
  OUT/lsq/     the same landmarks plus M per head fitted by ridge least
               squares to the exact far-field attention output.

The least-squares target is the first-order output error. With D the
exact softmax denominator, n = sum_far p_j v_j and d = sum_far p_j the
exact far numerator and mass, and o the exact output,
    delta_o ~= (n_hat - n) - o (d_hat - d),
and n_hat - o d_hat = phi^T M (A_v - A_1 o^T) is linear in M. A weighted
mass column (beta) keeps the denominator honest as well.

With --val the script also scores held-out captures offline: relative L2
error of the attention output against exact attention for each operator
variant, emulating the runtime (sinks + exact window + delayed far
insertion, aggregate rectifier) over positions >= --prefill.
"""
import argparse
import glob
import multiprocessing
import os
import re
import struct
import sys
from collections import defaultdict

import numpy as np

RIDGE_REL = 1e-6  # nystrom.rs RIDGE_REL


def load(path):
    with open(path, "rb") as f:
        t, d, h = struct.unpack("<3I", f.read(12))
        x = np.frombuffer(f.read(), dtype="<f4")
    q = x[: h * t * d].reshape(h, t, d)
    k = x[h * t * d : h * t * d + t * d].reshape(t, d)
    v = x[h * t * d + t * d :].reshape(t, d)
    return q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)


def captures(directory):
    groups = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(directory, "L*_g*_*.f32"))):
        m = re.match(r"L(\d+)_g(\d+)_(\d+)\.f32$", os.path.basename(p))
        if m:
            groups[(int(m.group(1)), int(m.group(2)))].append(p)
    return groups


def kmeans(x, m, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    d2 = ((x - c[0]) ** 2).sum(1)
    for _ in range(1, m):
        c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - c[-1]) ** 2).sum(1))
    c = np.array(c)
    for _ in range(iters):
        dist = (x**2).sum(1)[:, None] - 2 * x @ c.T + (c**2).sum(1)[None, :]
        a = dist.argmin(1)
        for i in range(m):
            sel = x[a == i]
            if len(sel):
                c[i] = sel.mean(0)
    return c


def seg_means(x, m):
    t = len(x)
    return np.array([x[i * t // m : (i + 1) * t // m].mean(0) for i in range(m)])


def ridge_pinv(a):
    ata = a.T @ a
    lam = RIDGE_REL * max(np.trace(ata) / len(a), np.finfo(float).tiny)
    for _ in range(12):
        try:
            np.linalg.cholesky(ata + lam * np.eye(len(a)))
            return np.linalg.solve(ata + lam * np.eye(len(a)), a.T)
        except np.linalg.LinAlgError:
            lam *= 10
    return np.eye(len(a)) / max(np.trace(ata) / len(a), np.finfo(float).tiny)


class Window:
    """Exact attention of one head over one window, plus the per-position
    near/far split the runtime uses."""

    def __init__(self, q, k, v, w, sink):
        t, d = k.shape
        self.t, self.d, self.w, self.sink = t, d, w, sink
        self.s = 1.0 / np.sqrt(d)
        self.q, self.k, self.v = q, k, v
        lg = self.s * q @ k.T
        lg[np.triu_indices(t, 1)] = -np.inf
        self.c = lg.max(1)
        e = np.exp(lg - self.c[:, None], out=lg)
        self.den = e.sum(1)  # D / e^c
        self.o = (e @ v) / self.den[:, None]
        # far(t) = [sink, t-w]; near = sinks + (t-w, t] — the far block is
        # the strict lower band below the window, so zero e outside it.
        far_e = np.tril(e, -w)
        far_e[:, :sink] = 0
        self.far_n = (far_e @ v) / self.den[:, None]  # sum_far p v
        self.far_d = far_e.sum(1) / self.den
        self.near_num = self.o * self.den[:, None] - far_e @ v
        self.near_den = self.den - far_e.sum(1)

    def psi_prefix(self, qt):
        """Prefix sums over keys of exp(s Q~ k_j - mu)[v_j, 1] restricted
        to the far range, per query position; returns (Av [t,m,dv],
        A1 [t,m], mu [m])."""
        pl = self.s * self.k @ qt.T  # [t, m]
        mu = pl.max(0)
        ps = np.exp(pl - mu)
        cv = ps[:, :, None] * self.v[:, None, :]
        np.cumsum(cv, 0, out=cv)
        c1 = np.cumsum(ps, 0)
        t, w, sink = self.t, self.w, self.sink
        av = np.zeros((t, len(qt), self.v.shape[1]))
        a1 = np.zeros((t, len(qt)))
        # query i sees keys [sink, i - w]
        lo = w + sink
        if lo < t:
            av[lo:] = cv[sink : t - w]
            a1[lo:] = c1[sink : t - w]
            if sink:
                av[lo:] -= cv[sink - 1]
                a1[lo:] -= c1[sink - 1]
        return av, a1, mu

    def phi_bar(self, kt):
        """exp(s K~ q_t) / D_t, [t, m]."""
        fl = self.s * self.q @ kt.T
        return np.exp(fl - self.c[:, None] - np.log(self.den)[:, None])

    def output(self, phib, mt, av, a1, gain=1.0):
        """Runtime operator: near exact + gain * skeleton far, aggregate
        rectifier. mt is M~ in the normalized basis (see fit)."""
        u = phib @ mt  # [t, m]
        fn = np.einsum("tb,tbk->tk", u, av) * gain
        fd = np.einsum("tb,tb->t", u, a1) * gain
        bad = fd < 0
        fn[bad] = 0
        fd[bad] = 0
        dn = self.den
        num = self.near_num / dn[:, None] + fn
        den = self.near_den / dn + fd
        return num / den[:, None], bad.mean()


def pinv_mt(qt, kt, s, mu):
    """pinv(exp(s Q~ K~^T)) expressed in the normalized basis:
    M~ = M diag(e^mu)."""
    m = ridge_pinv(np.exp(s * qt @ kt.T))
    return m * np.exp(mu)[None, :]


def rel_err(o, ref, lo):
    return np.sqrt(((o[lo:] - ref[lo:]) ** 2).sum() / (ref[lo:] ** 2).sum())


def fit_lsq(data, w, sink, kt, qt, ridge, beta):
    """data: [(q_h [t,d], k, v)] — one Window at a time, to bound memory."""
    m = len(kt)
    g = np.zeros((m * m, m * m))
    r = np.zeros((m, m))
    # One normalization of the psi basis across windows: mu = the largest
    # calibration logit per landmark.
    mu = np.max([(k @ qt.T).max(0) / np.sqrt(k.shape[1]) for _, k, _ in data], 0)
    for q, k, v in data:
        wd = Window(q, k, v, w, sink)
        av, a1, mu_w = wd.psi_prefix(qt)
        scale = np.exp(mu_w - mu)  # this window's basis -> the common one
        av = av * scale[None, :, None]
        a1 = a1 * scale[None, :]
        ok = np.arange(wd.t) >= wd.w + wd.sink
        phib = wd.phi_bar(kt)[ok]
        o = wd.o[ok]
        b = av[ok] - a1[ok][:, :, None] * o[:, None, :]
        y = wd.far_n[ok] - wd.far_d[ok][:, None] * o
        bb = beta if beta > 0 else np.sqrt(((wd.v[None] - o[:, None, :]) ** 2).sum(-1).mean())
        b = np.concatenate([b, bb * a1[ok][:, :, None]], 2)
        y = np.concatenate([y, bb * wd.far_d[ok][:, None]], 1)
        pp = np.einsum("ta,tc->tac", phib, phib).reshape(len(phib), -1)
        qq = np.einsum("tbk,tek->tbe", b, b).reshape(len(phib), -1)
        g += (pp.T @ qq).reshape(m, m, m, m).transpose(0, 2, 1, 3).reshape(m * m, m * m)
        r += np.einsum("ta,tb->ab", phib, np.einsum("tbk,tk->tb", b, y))
        del wd, av, a1, b, y, pp, qq
    lam = ridge * np.trace(g) / len(g)
    mt = np.linalg.solve(g + lam * np.eye(len(g)), r.ravel()).reshape(m, m)
    return mt, mu


def write_o1c(path, kt, qts, mus=None):
    m, d = kt.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", m, d, len(qts), 1 if mus is not None else 0))
        f.write(kt.astype("<f4").tobytes())
        for qt in qts:
            f.write(qt.astype("<f4").tobytes())
        if mus is not None:
            for mu in mus:
                f.write(mu.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True, help="capture dir to fit on")
    ap.add_argument("--val", help="capture dir to score offline")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--w", type=int, default=128)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=256, help="runtime seal point (offline score)")
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.0, help="mass-column weight (0 = rms |v-o|)")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4, help="(layer, group) fits in parallel")
    ap.add_argument("--layers", help="comma list: fit only these layers")
    a = ap.parse_args()
    tr, va = captures(a.train), captures(a.val) if a.val else {}
    for sub in ("kmeans", "lsq"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    print(
        f"{'L':>3} {'g':>2} {'h':>2} | {'runtime':>8} {'nofar':>8} {'km+pinv':>8} {'km+lsq':>8}"
        f" | {'rt-r%':>5} {'ls-r%':>5} {'farp':>6} {'log10|M|':>9}"
    )
    keep = {int(x) for x in a.layers.split(",")} if a.layers else None
    jobs = [
        (layer, grp, paths, va.get((layer, grp), []), a)
        for (layer, grp), paths in sorted(tr.items())
        if keep is None or layer in keep
    ]
    with multiprocessing.Pool(max(1, a.jobs)) as pool:
        for rows in pool.imap(fit_group, jobs):
            print("\n".join(rows), flush=True)
    print(f"wrote {a.out}/kmeans and {a.out}/lsq", file=sys.stderr)


def fit_group(job):
    """Fit one (layer, group); write both calibrations; return table rows."""
    layer, grp, paths, vpaths, a = job
    if a.max_windows:
        paths = paths[: a.max_windows]
    data = [load(p) for p in paths]
    nh = data[0][0].shape[0]
    kt = kmeans(np.concatenate([k[a.sink :] for _, k, _ in data]), a.m)
    qts, mts, rows = [], [], []
    vdata = [load(p) for p in vpaths]
    for h in range(nh):
        qt = kmeans(np.concatenate([q[h, a.sink :] for q, _, _ in data]), a.m, seed=h + 1)
        mt, mu = fit_lsq([(q[h], k, v) for q, k, v in data], a.w, a.sink, kt, qt, a.ridge, a.beta)
        m_rt = mt * np.exp(-mu)[None, :]  # runtime units: M = M~ diag(e^-mu)
        qts.append(qt)
        mts.append(m_rt)
        row = f"{layer:>3} {grp:>2} {h:>2} |"
        if vdata:
            errs = defaultdict(list)
            rects = defaultdict(list)
            lo = a.prefill
            for q, k, v in vdata:
                wd = Window(q[h], k, v, a.w, a.sink)
                # runtime: segment means of the first `prefill` tokens + pinv
                me = max(4, min(a.m, lo // 8))
                kt_rt, qt_rt = seg_means(k[:lo], me), seg_means(q[h, :lo], me)
                av, a1, mu_w = wd.psi_prefix(qt_rt)
                o, bad = wd.output(wd.phi_bar(kt_rt), pinv_mt(qt_rt, kt_rt, wd.s, mu_w), av, a1)
                errs["runtime"].append(rel_err(o, wd.o, lo))
                rects["runtime"].append(bad)
                errs["nofar"].append(rel_err(wd.near_num / wd.near_den[:, None], wd.o, lo))
                av, a1, mu_w = wd.psi_prefix(qt)
                o, _ = wd.output(wd.phi_bar(kt), pinv_mt(qt, kt, wd.s, mu_w), av, a1)
                errs["km+pinv"].append(rel_err(o, wd.o, lo))
                # fitted M~ lives at the calibration mu; rescale to this window's mu
                o, bad = wd.output(wd.phi_bar(kt), mt * np.exp(mu_w - mu)[None, :], av, a1)
                errs["km+lsq"].append(rel_err(o, wd.o, lo))
                rects["km+lsq"].append(bad)
                errs["farmass"].append(wd.far_d[lo:].mean())
            for key in ("runtime", "nofar", "km+pinv", "km+lsq"):
                row += f" {np.mean(errs[key]):8.4f}"
            row += f" | {100 * np.mean(rects['runtime']):5.1f} {100 * np.mean(rects['km+lsq']):5.1f}"
            row += f" {np.mean(errs['farmass']):6.3f}"
        lm = np.log10(np.abs(m_rt) + 1e-300)
        row += f" {lm.min():4.0f}..{lm.max():3.0f}"
        rows.append(row)
    write_o1c(os.path.join(a.out, "kmeans", f"L{layer}_g{grp}.o1c"), kt, qts)
    write_o1c(os.path.join(a.out, "lsq", f"L{layer}_g{grp}.o1c"), kt, qts, mts)
    return rows


if __name__ == "__main__":
    main()
