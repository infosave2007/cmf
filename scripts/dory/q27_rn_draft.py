#!/usr/bin/env python3
"""A draft made of resonance and NVG, measured against the MTP block on the same rows.
[CHAIN=n] [EPOCHS=n] [GPU_GB=n] [RN="regimes,rank,binary,act;…"] q27_rn_draft.py DUMP_DIR OUT_JSON [mtp] [token] [rn] [adapter] [gate]

DUMP_DIR is what `cortiq mtp-heal --dump` writes: the trunk's hidden row and choice at every position, the
MTP chain's output rows and top-4 per step before and after its training, and the tensors a draft reads.
A row counts at depth j when every token the chain was fed is the trunk's own choice and the drafts before
it held, as in a greedy round. Held-out windows are those with index % 10 == 9.

  mtp      the MTP block's hold rate per depth, from the dump
  token    resonance alone: a topology (mean and a PCA subspace) per next token, the smallest
           reconstruction error names the token
  rn       resonance picks a regime of the trunk's state, the regime's NVG expert (a binary low-rank
           map) predicts the next state, the shared head reads it; chained on its own output
  adapter  the same routed NVG experts as a correction of the MTP block's output
  gate     two topologies, held and missed, on the MTP block's output decide how deep a round verifies
"""
import json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

D, OUT = sys.argv[1], sys.argv[2]
WHAT = set(sys.argv[3:]) or {"mtp", "token", "rn", "adapter", "gate"}
meta = json.load(open(f"{D}/meta.json"))
N, T, K, KK, CH, HS, V = (meta[k] for k in ("windows", "window", "topk", "draft_topk", "chain", "hidden", "vocab"))
CH = min(CH, int(os.environ.get("CHAIN", CH)))  # CHAIN=n measures the first n steps of a deeper dump
EPOCHS = int(os.environ.get("EPOCHS", 6))
EPS, GEMMA = meta["norm_eps"], meta["gemma_norm"]
dev = "cuda"
if os.environ.get("GPU_GB"):  # a ceiling of its own, where the card is shared with a training
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_GB"]) * 2 ** 30 / torch.cuda.get_device_properties(0).total_memory)
torch.manual_seed(0)
# what a round of k drafts costs on a MacBook Air M4 in plain tokens (JOURNAL_Q27_MAC_DECODE_2026-11-06.md)
COST = [1.0, 1.10, 1.275, 1.54, 1.59, 1.65, 1.70, 1.76]
res = {"meta": {k: meta[k] for k in ("windows", "window", "chain")}}


def raw(name, dtype, shape):
    return np.memmap(f"{D}/{name}", dtype=dtype, mode="r").reshape(shape)


def gpu(a, step=16):
    return torch.cat([torch.from_numpy(np.ascontiguousarray(a[i:i + step])).to(dev) for i in range(0, a.shape[0], step)])


def rms(x, w):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * ((1 + w) if GEMMA else w)


lens = torch.from_numpy(np.fromfile(f"{D}/len.u32", dtype=np.uint32).astype(np.int64)).to(dev)
ids = gpu(raw("ids.u32", np.uint32, (N, T)).astype(np.int64))
top1 = gpu(np.ascontiguousarray(raw("ti.u32", np.uint32, (N, T, K))[:, :, 0]).astype(np.int64))
E = gpu(raw("embed.f16", np.float16, (V, HS)), 1 << 15)
HEAD = gpu(raw("head.f16", np.float16, (V, HS)), 1 << 15)
EH = gpu(raw("eh_proj.f16", np.float16, (HS, 2 * HS))).float()
enorm, hnorm, mnorm = (torch.from_numpy(np.fromfile(f"{D}/{n}.f32", dtype=np.float32)).to(dev) for n in ("enorm", "hnorm", "mtp_norm"))

# rows (w, p) with CH positions after them; on[j-1]: the chain's tokens up to step j are the trunk's own
pos = torch.arange(T, device=dev)[None, :].expand(N, T)
ok = pos + CH < lens[:, None]
W, P = ok.nonzero(as_tuple=True)
on = []
cur = torch.ones_like(W, dtype=torch.bool)
for j in range(1, CH + 1):
    cur = cur & (ids[W, P + j] == top1[W, P + j - 1])
    on.append(cur)
val = W % 10 == 9
tr = ~val
print(f"{N} windows, {len(W)} rows ({int(val.sum())} held out); on the trunk's own tokens at depth 1..{CH}: " + " / ".join(f"{float(o.float().mean()):.3f}" for o in on), flush=True)


def hold(pred, rows=None):
    """pred[j-1][rows] = the draft's token at depth j. Returns per depth (base rows, held rows)."""
    rows = val if rows is None else rows
    held, out = rows.clone(), []
    for j in range(1, CH + 1):
        base = held & on[j - 1]
        hit = base & (pred[j - 1] == top1[W, P + j])
        out.append((int(base.sum()), int(hit.sum())))
        held = hit
    return out


def rates(c):
    return [h / max(n, 1) for n, h in c]


def speed(h):
    """Tokens per plain token's time at the best fixed depth, on the Mac's round costs."""
    best, acc, tok = (0, 1.0), 1.0, 1.0
    for k, r in enumerate(h, 1):
        acc *= r
        tok += acc
        if tok / COST[k] > best[1]:
            best = (k, tok / COST[k])
    return best


def report(name, c):
    r = rates(c)
    k, s = speed(r)
    print(f"{name:44s} holds " + " / ".join(f"{x:.3f}" for x in r) + f"   x{s:.2f} at depth {k}", flush=True)
    res[name] = {"hold": r, "rows": [n for n, _ in c], "speed": s, "depth": k}


def chain_dump(tag):
    return [gpu(np.ascontiguousarray(raw(f"{tag}/d{j}.u32", np.uint32, (N, T, KK))[:, :, 0]).astype(np.int64))[W, P] for j in range(1, CH + 1)]


# the head over the tokens the training windows hold (the runtime's draft head is a shortlist too); a
# held-out row whose token is outside counts as a miss
trw = torch.arange(N, device=dev) % 10 != 9
SL = torch.unique(torch.cat([ids[trw].flatten(), top1[trw].flatten()]))
INV = torch.full((V,), -100, device=dev, dtype=torch.long)
INV[SL] = torch.arange(len(SL), device=dev)
HEADS = HEAD[SL].to(torch.bfloat16)
del HEAD
print(f"head shortlist: {len(SL)} tokens, {float((INV[top1[W[val], P[val] + 1]] >= 0).float().mean()):.4f} of held-out rows inside", flush=True)


def logits(x):
    return rms(x, mnorm).to(torch.bfloat16) @ HEADS.T


# ── the MTP block, from the dump ──
tags = [t for t in ("before", "after") if os.path.exists(f"{D}/{t}/d1.u32")]
if "mtp" in WHAT:
    for t in tags:
        report(f"MTP block, {t} training", hold(chain_dump(t)))

Hh = None
if WHAT & {"token", "rn"}:
    parts = [np.memmap(os.path.join(s["dir"], "h.f32"), dtype=np.float32, mode="r").reshape(s["windows"], T, HS) for s in meta["trunk"]]
    Hh = torch.cat([gpu(p, 8) for p in parts])  # the trunk's last hidden rows, f32


def block_in(tok, hid):
    """The MTP block's input: eh_proj · [enorm(e); hnorm(h)]."""
    return torch.cat([rms(E[tok], enorm), rms(hid, hnorm)], -1) @ EH.T


def topologies(x, assign, k, rank=16, cap=20000):
    """Per class a mean and a PCA basis of its rows: patent 1's isolated topologies."""
    mu, B = torch.zeros(k, x.shape[1], device=dev), torch.zeros(k, x.shape[1], rank, device=dev)
    for c in range(k):
        r = (assign == c).nonzero().squeeze(1)
        if len(r) < 2 * rank:
            continue
        r = r[torch.randperm(len(r), device=dev)[:cap]]
        xc = x[r].float()
        mu[c] = xc.mean(0)
        B[c] = torch.pca_lowrank(xc - mu[c], q=rank, center=False, niter=3)[2]
    return mu, B


def recon_err(x, mu, B, chunk=4096):
    """[rows][classes]: what is left of a row after each topology rebuilds it."""
    out = []
    k, h, r = B.shape
    bf = B.permute(1, 0, 2).reshape(h, k * r)
    mb = torch.einsum("kh,khr->kr", mu, B)
    m2 = mu.pow(2).sum(1)
    for i in range(0, len(x), chunk):
        xc = x[i:i + chunk].float()
        d2 = xc.pow(2).sum(1, keepdim=True) - 2 * xc @ mu.T + m2
        pr = (xc @ bf).view(len(xc), k, r) - mb
        out.append(d2 - pr.pow(2).sum(-1))
    return torch.cat(out)


# ── resonance alone: a topology per next token ──
if "token" in WHAT:
    t0 = time.time()
    tri = tr.nonzero().squeeze(1)
    tri = tri[torch.randperm(len(tri), device=dev)[:400000]]
    z = torch.cat([block_in(ids[W[i], P[i] + 1], Hh[W[i], P[i]]) for i in tri.split(8192)])
    y = top1[W[tri], P[tri] + 1]
    cnt = torch.bincount(y, minlength=V)
    cls = (cnt >= 48).nonzero().squeeze(1)
    remap = torch.full((V,), -1, device=dev, dtype=torch.long)
    remap[cls] = torch.arange(len(cls), device=dev)
    for rank in (4, 16):
        mu, B = topologies(z, remap[y], len(cls), rank=rank, cap=4000)
        vi = (val & on[0]).nonzero().squeeze(1)
        pred = torch.cat([cls[recon_err(block_in(ids[W[i], P[i] + 1], Hh[W[i], P[i]]), mu, B, 512).argmin(1)] for i in vi.split(4096)])
        yv = top1[W[vi], P[vi] + 1]
        cov = float((remap[yv] >= 0).float().mean())
        acc = float((pred == yv).float().mean())
        name = f"resonance alone, {len(cls)} token topologies of rank {rank}"
        print(f"{name:44s} holds {acc:.3f} at depth 1 (its tokens cover {cov:.3f} of rows; {time.time() - t0:.0f}s)", flush=True)
        res[name] = {"hold": [acc], "coverage": cov, "classes": len(cls)}
    del z, mu, B


class Experts(torch.nn.Module):
    """Routed experts x → x + U_k·act(V_k·x). With `binary` the two matrices are signs with a scale per
    row and per column (NVG's form), trained through the sign; without `act` their product is NVG's
    low-rank form itself."""

    def __init__(s, k, r, binary, act=True):
        super().__init__()
        s.k, s.r, s.binary, s.act = k, r, binary, act
        # a binary matrix's latent weights start small, so that a sign can turn within a run
        s.Vt = torch.nn.Parameter(torch.randn(k, r, HS, device=dev) * (0.1 if binary else 1 / HS ** 0.5))
        s.U = torch.nn.Parameter(torch.randn(k, HS, r, device=dev) * (0.1 if binary else 1e-3 / r ** 0.5))
        if binary:
            s.vr, s.vc = torch.nn.Parameter(torch.ones(k, r, 1, device=dev) / HS ** 0.5), torch.nn.Parameter(torch.ones(k, 1, HS, device=dev))
            s.ur, s.uc = torch.nn.Parameter(torch.ones(k, HS, 1, device=dev)), torch.nn.Parameter(torch.full((k, 1, r), 1e-3 / r ** 0.5, device=dev))

    def mats(s, c):
        if not s.binary:
            return s.Vt[c], s.U[c]
        sg = lambda w: w + (torch.sign(w) - w).detach()
        return sg(s.Vt[c]) * s.vr[c] * s.vc[c], sg(s.U[c]) * s.ur[c] * s.uc[c]

    def forward(s, x, route):
        out = x.clone()
        for c in range(s.k):
            r = (route == c).nonzero().squeeze(1)
            if len(r):
                vt, u = s.mats(c)
                mid = x[r] @ vt.T
                out[r] = x[r] + (F.silu(mid) if s.act else mid) @ u.T
        return out

    def bits(s):
        return s.k * 2 * s.r * HS


def kmeans(x, k, iters=25):
    c = x[torch.randperm(len(x), device=dev)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for i in range(k):
            m = a == i
            if m.any():
                c[i] = x[m].mean(0)
    return a


def regimes(x, k):
    """k regimes of the rows `x`, found without labels, each with its topology."""
    if k == 1:
        return None
    p = torch.pca_lowrank(x.float(), q=64, niter=3)[2]
    return topologies(x, kmeans(x.float() @ p, k), k)


def route_of(x, topo):
    return torch.zeros(len(x), dtype=torch.long, device=dev) if topo is None else recon_err(x, *topo).argmin(1)


def fit(name, step_fn, groups, epochs, bs=4096):
    """step_fn(rows, j, carry) → (x, carry): the draft's output rows at depth j. Trained on the trunk's token.
    groups: (parameters, learning rate) pairs."""
    groups = [(list(p), lr) for p, lr in groups if p is not None]
    params = [q for p, _ in groups for q in p]
    opt = torch.optim.AdamW([{"params": p, "lr": lr} for p, lr in groups], weight_decay=0.0)
    tri = tr.nonzero().squeeze(1)
    steps = epochs * (len(tri) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr for _, lr in groups], total_steps=steps, pct_start=0.1)
    t0, n = time.time(), 0
    for ep in range(epochs):
        for b in tri[torch.randperm(len(tri), device=dev)].split(bs)[:len(tri) // bs]:
            loss, carry = 0.0, None
            for j in range(1, CH + 1):
                x, carry = step_fn(b, j, carry)
                loss = loss + F.cross_entropy(logits(x).float(), INV[top1[W[b], P[b] + j]], ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            n += 1
            if n % 50 == 0:
                print(f"  {name}: step {n}/{steps} loss {float(loss) / CH:.3f} ({time.time() - t0:.0f}s)", flush=True)
    pred = [[] for _ in range(CH)]
    with torch.no_grad():
        for b in torch.arange(len(W), device=dev)[val].split(bs):
            carry = None
            for j in range(1, CH + 1):
                x, carry = step_fn(b, j, carry)
                pred[j - 1].append(SL[logits(x).argmax(1)])
    full = [torch.zeros(len(W), dtype=torch.long, device=dev) for _ in range(CH)]
    for j in range(CH):
        full[j][val] = torch.cat(pred[j])
    return full


# ── resonance + NVG as the draft itself ──
if "rn" in WHAT:
    sample = tr.nonzero().squeeze(1)
    sample = sample[torch.randperm(len(sample), device=dev)[:200000]]
    sig = torch.cat([rms(Hh[W[i], P[i]], hnorm) for i in sample.split(8192)])
    # regimes, rank, binary (NVG) or float, with the activation or linear; RN="k,r,b,a;…" names other ones
    configs = ((1, 0, 0, 1), (1, 2048, 0, 1), (16, 512, 0, 1), (16, 512, 1, 1), (16, 512, 1, 0), (64, 256, 1, 1))
    if os.environ.get("RN"):
        configs = [tuple(int(x) for x in c.split(",")) for c in os.environ["RN"].split(";")]
    for k, r, binary, act in configs:
        topo = regimes(sig, k)
        ex = Experts(k, r, binary, act) if r else None
        w0 = torch.nn.Parameter(EH.clone())

        def step_fn(b, j, carry):
            hid = Hh[W[b], P[b]] if j == 1 else carry
            z = torch.cat([rms(E[ids[W[b], P[b] + j]], enorm), rms(hid, hnorm)], -1) @ w0.T
            if ex:
                with torch.no_grad():
                    rt = route_of(rms(hid, hnorm), topo)
                z = ex(z, rt)
            return z, z

        name = "linear map alone (eh_proj trained)" if not r else f"{'resonance, ' + str(k) + ' regimes' if k > 1 else 'no routing'} + {'NVG' if binary else 'float'} experts r{r}{'' if act else ' linear'}"
        pred = fit(name, step_fn, [([w0], 3e-5), (ex.parameters() if ex else None, 1e-3)], epochs=EPOCHS)
        report(name, hold(pred))
        if ex:
            res[name]["expert_MB_all"] = ex.bits() / 8e6 if binary else ex.bits() * 16 / 8e6
            res[name]["expert_MB_read_per_token"] = res[name]["expert_MB_all"] / k
        del ex, w0, topo
    del sig

# ── routed NVG experts as a correction of the MTP block's output ──
if "adapter" in WHAT:
    for t in tags:
        X = [gpu(raw(f"{t}/x{j}.f16", np.float16, (N, T, HS)), 8) for j in range(1, CH + 1)]
        sample = tr.nonzero().squeeze(1)
        sample = sample[torch.randperm(len(sample), device=dev)[:200000]]
        sig = torch.cat([rms(X[0][W[i], P[i]], mnorm) for i in sample.split(8192)])
        for k, r, binary in ((1, 1024, True), (16, 512, True)):
            topo = regimes(sig, k)
            ex = Experts(k, r, binary)

            def step_fn(b, j, carry):
                x = X[j - 1][W[b], P[b]].float()
                with torch.no_grad():
                    rt = route_of(rms(x, mnorm), topo)
                return ex(x, rt), None

            name = f"MTP {t} + {'resonance, ' + str(k) + ' regimes, ' if k > 1 else ''}NVG correction r{r}"
            report(name, hold(fit(name, step_fn, [(ex.parameters(), 1e-3)], epochs=EPOCHS)))
            del ex, topo
        del X, sig

# ── resonance as a gate on the depth of a round ──
if "gate" in WHAT:
    for t in tags:
        d = chain_dump(t)
        conf = [gpu(np.ascontiguousarray(raw(f"{t}/dp{j}.f32", np.float32, (N, T, KK))[:, :, 0]))[W, P] for j in range(1, CH + 1)]
        held, flags = torch.ones_like(W, dtype=torch.bool), []
        for j in range(CH):
            held = held & on[j] & (d[j] == top1[W, P + j + 1])
            flags.append(held)
        nheld = sum(f.long() for f in flags)
        rows = on[CH - 1]
        score = []
        for j in range(1, CH + 1):
            X = gpu(raw(f"{t}/x{j}.f16", np.float16, (N, T, HS)), 8)
            base = rows & tr & (flags[j - 2] if j > 1 else torch.ones_like(rows))
            bi = base.nonzero().squeeze(1)
            bi = bi[torch.randperm(len(bi), device=dev)[:300000]]
            f = torch.cat([rms(X[W[i], P[i]], mnorm) for i in bi.split(8192)])
            mu, B = topologies(f, flags[j - 1][bi].long(), 2)
            e = torch.cat([recon_err(rms(X[W[i], P[i]], mnorm), mu, B) for i in torch.arange(len(W), device=dev).split(8192)])
            score.append(e[:, 0] - e[:, 1])  # > 0: the held topology rebuilds the row better
            del X, f
            vb = rows & val & (flags[j - 2] if j > 1 else torch.ones_like(rows))
            lab = flags[j - 1][vb].float()

            def auc(s):
                o = s[vb].argsort()
                rk = torch.empty_like(o, dtype=torch.float)
                rk[o] = torch.arange(1, len(o) + 1, device=dev, dtype=torch.float)
                pos = lab.sum()
                return float((rk[lab > 0].sum() - pos * (pos + 1) / 2) / (pos * (len(lab) - pos)).clamp(min=1))

            print(f"gate, MTP {t}, depth {j}: AUC resonance {auc(score[-1]):.3f}, the draft's own confidence {auc(conf[j - 1]):.3f}", flush=True)
            res[f"gate AUC, MTP {t}, depth {j}"] = {"resonance": auc(score[-1]), "confidence": auc(conf[j - 1])}

        def sim(depth, m):
            dd = depth[m]
            return float((1 + torch.minimum(nheld[m], dd)).sum() / torch.tensor(COST, device=dev)[dd].sum())

        def gated(s, thr):
            go, depth = torch.ones_like(rows), torch.zeros_like(W)
            for j in range(CH):
                go = go & (s[j] >= thr[j])
                depth = depth + go.long()
            return depth

        def tune(s):
            thr = [-1e30] * CH
            for j in range(CH):
                best = (-1.0, thr[j])
                for q in np.linspace(0, 0.9, 19):
                    cand = thr[:j] + [float(torch.quantile(s[j][rows & tr][:200000].float(), float(q)))] + thr[j + 1:]
                    v = sim(gated(s, cand), rows & tr)
                    if v > best[0]:
                        best = (v, cand[j])
                thr[j] = best[1]
            return thr

        m = rows & val
        out = {f"fixed depth {k}": sim(torch.full_like(W, k), m) for k in range(1, CH + 1)}
        out["gate by the draft's confidence"] = sim(gated(conf, tune(conf)), m)
        out["gate by resonance"] = sim(gated(score, tune(score)), m)
        both = [a.float() + b for a, b in zip([(c - c[rows & tr].mean()) / c[rows & tr].std() for c in conf], [(s - s[rows & tr].mean()) / s[rows & tr].std() for s in score])]
        out["gate by both"] = sim(gated(both, tune(both)), m)
        out["oracle depth"] = sim(nheld, m)
        print(f"rounds on the Mac's costs, MTP {t}: " + ", ".join(f"{k} x{v:.2f}" for k, v in out.items()), flush=True)
        res[f"rounds, MTP {t}"] = out

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
