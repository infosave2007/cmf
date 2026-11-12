#!/usr/bin/env python3
"""Price a far-field approximation of the DiT's attention on the model's own
operands, before any kernel is written for it.

The chain dumps q, k, v of chosen blocks (CMF_H3_DUMP=<dir>, CMF_H3_FLASH=0).
For every dump this computes the exact attention output of the image rows and
the output of a two-level scheme:

  * text rows (and reference rows): exact, for every query;
  * image keys in the cells within `r` cells of the query's cell: exact;
  * every other cell of c x c tokens: ONE pooled key and value, weighted by
    the cell's size (optionally with the isotropic second-order term of the
    mean of exponentials).

and reports the relative error of the output, the share of the exact
attention mass that sits in the approximated far field, and how many keys a
query visits. Needs torch with CUDA.

With --nystrom M[,M...] it also prices the landmark form of the far field
(the one the language models' turbo files use): M landmark keys and queries
(the means of a g x g grid of the image, M = g * g), the kernel between a
query and a far key as k(q, K~) W^+ k(Q~, k), the near cells and the text
exact as above. The row's `cell` is then -M.

  far_price.py DIR [--cells 2,4] [--radius 1,2,4] [--nystrom 64,256] [--json OUT]
"""
import glob
import json
import os
import sys

import numpy as np
import torch

d = sys.argv[1]
arg = lambda k, dv: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else dv
cells = [int(v) for v in arg("--cells", "2,4").split(",")]
radii = [int(v) for v in arg("--radius", "1,2,4").split(",")]
nys = [int(v) for v in arg("--nystrom", "").split(",") if v]
lay = json.load(open(os.path.join(d, "layout.json")))
T, R, hp, wp = lay["text"], lay["refs"], lay["hp"], lay["wp"]
dev = "cuda"
out = []
print(f"layout: {T} text rows, {R} reference rows, image {hp} x {wp} tokens")
print("| call | block | cell | radius | iso | far mass | rel err | worst head | keys a query | of |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for meta_path in sorted(glob.glob(os.path.join(d, "c*_b*.json")), key=lambda p: [int(x) for x in os.path.basename(p)[1:-5].split("_b")]):
    m = json.load(open(meta_path))
    n, ns, nh, hd = m["n"], m["ns"], m["nh"], m["hd"]
    base = meta_path[:-5]
    ld = lambda name, shape: torch.from_numpy(np.fromfile(f"{base}_{name}.f16", dtype=np.float16).reshape(shape)).to(dev).float()
    Q, K = ld("q", (nh, n, hd)), ld("k", (nh, n, hd))
    V = ld("v", (nh, hd, ns))[:, :, :n].transpose(1, 2).contiguous()
    scale = hd ** -0.5
    ni = hp * wp
    i0 = n - ni  # the target image is the last segment
    assert i0 == T + R, (n, ni, T, R)
    yy, xx = torch.meshgrid(torch.arange(hp, device=dev), torch.arange(wp, device=dev), indexing="ij")
    Qi = Q[:, i0:]
    # exact output of the image rows, a few heads at a time
    O = torch.empty(nh, ni, hd, device=dev)
    S_all = []
    for h in range(nh):
        S = (Qi[h] @ K[h].T) * scale
        S_all.append(None)
        O[h] = torch.softmax(S, dim=-1) @ V[h]
    den = O.pow(2).sum(dim=(1, 2)).sqrt()
    for c in cells:
        if hp % c or wp % c:
            continue
        ch, cw = hp // c, wp // c
        cell = (yy // c) * cw + (xx // c)  # [hp, wp]
        cell_flat = cell.reshape(-1)
        cy, cx = (yy // c).reshape(-1), (xx // c).reshape(-1)
        ccy, ccx = torch.arange(ch * cw, device=dev) // cw, torch.arange(ch * cw, device=dev) % cw
        # pooled keys / values of every cell, and the keys' spread inside it
        Ki = K[:, i0:].reshape(nh, ch, c, cw, c, hd).permute(0, 1, 3, 2, 4, 5).reshape(nh, ch * cw, c * c, hd)
        Vi = V[:, i0:].reshape(nh, ch, c, cw, c, hd).permute(0, 1, 3, 2, 4, 5).reshape(nh, ch * cw, c * c, hd)
        Kc, Vc = Ki.mean(2), Vi.mean(2)
        var = Ki.var(dim=2, unbiased=False).mean(-1)  # [nh, cells]: mean per-dim variance
        for r in radii:
            near_cell = ((cy[:, None] - ccy[None, :]).abs() <= r) & ((cx[:, None] - ccx[None, :]).abs() <= r)  # [ni, cells]
            near_tok = near_cell[:, cell_flat]  # [ni, ni]
            keys = T + R + int(near_tok.sum(1).float().mean()) + int((~near_cell).sum(1).float().mean())
            for iso in (0, 1):
                num = torch.zeros(nh, device=dev)
                farm = torch.zeros(nh, device=dev)
                for h in range(nh):
                    S = (Qi[h] @ K[h].T) * scale  # [ni, n]
                    P = torch.softmax(S, dim=-1)
                    farm[h] = P[:, i0:].masked_fill(near_tok, 0.0).sum(1).mean()
                    Lc = (Qi[h] @ Kc[h].T) * scale + float(np.log(c * c))
                    if iso:
                        Lc = Lc + 0.5 * scale * scale * Qi[h].pow(2).sum(-1, keepdim=True) * var[h][None, :]
                    L = torch.cat([S[:, :i0], S[:, i0:].masked_fill(~near_tok, float("-inf")), Lc.masked_fill(near_cell, float("-inf"))], dim=1)
                    Oa = torch.softmax(L, dim=-1) @ torch.cat([V[h], Vc[h]], dim=0)
                    num[h] = (Oa - O[h]).pow(2).sum().sqrt()
                rel = (num.pow(2).sum().sqrt() / den.pow(2).sum().sqrt()).item()
                worst = (num / den).max().item()
                row = dict(call=m["call"], block=m["block"], cell=c, radius=r, iso=iso, far_mass=farm.mean().item(),
                           rel_err=rel, worst_head=worst, keys=keys, of=n)
                out.append(row)
                print(f"| {row['call']} | {row['block']} | {c} | {r} | {iso} | {row['far_mass']:.3f} | {rel:.4f} | {worst:.4f} | {keys} | {n} |", flush=True)
    for mland in nys:
        gsz = int(round(mland ** 0.5))
        if gsz * gsz != mland or hp % gsz or wp % gsz or hp % 4 or wp % 4:
            continue
        # near: 4 x 4 cells within 2 cells of the query's (as the two-level rows with cell 4, radius 2)
        c, r = 4, 2
        ch, cw = hp // c, wp // c
        cell_flat = ((yy // c) * cw + (xx // c)).reshape(-1)
        cy, cx = (yy // c).reshape(-1), (xx // c).reshape(-1)
        ccy, ccx = torch.arange(ch * cw, device=dev) // cw, torch.arange(ch * cw, device=dev) % cw
        near_tok = (((cy[:, None] - ccy[None, :]).abs() <= r) & ((cx[:, None] - ccx[None, :]).abs() <= r))[:, cell_flat]
        by, bx = hp // gsz, wp // gsz
        pool = lambda X: X.reshape(nh, gsz, by, gsz, bx, hd).mean(dim=(2, 4)).reshape(nh, mland, hd)
        Kl, Ql = pool(K[:, i0:]), pool(Q[:, i0:])
        num = torch.zeros(nh, device=dev)
        for h in range(nh):
            S = ((Qi[h] @ K[h].T) * scale).double()  # [ni, n]
            shift = S.max()
            E = torch.exp(S - shift)  # the exact kernel
            Kq = torch.exp(((Qi[h] @ Kl[h].T) * scale).double() - shift)  # [ni, m]
            Wm = torch.exp(((Ql[h] @ Kl[h].T) * scale).double() - shift)  # [m, m]
            Kk = torch.exp(((Ql[h] @ K[h, i0:].T) * scale).double() - shift)  # [m, ni]
            A = (Kq @ torch.linalg.pinv(Wm, rcond=1e-6) @ Kk).clamp_min(0.0)  # the landmark kernel over image keys
            Ei = torch.where(near_tok, E[:, i0:], A)
            Ea = torch.cat([E[:, :i0], Ei], dim=1)
            Oa = ((Ea / Ea.sum(1, keepdim=True)) @ V[h].double()).float()
            num[h] = (Oa - O[h]).pow(2).sum().sqrt()
        rel = (num.pow(2).sum().sqrt() / den.pow(2).sum().sqrt()).item()
        worst = (num / den).max().item()
        keys = T + R + int(near_tok.sum(1).float().mean()) + mland
        row = dict(call=m["call"], block=m["block"], cell=-mland, radius=r, iso=0, far_mass=float("nan"), rel_err=rel,
                   worst_head=worst, keys=keys, of=n)
        out.append(row)
        print(f"| {row['call']} | {row['block']} | nys {mland} | {r} | 0 | - | {rel:.4f} | {worst:.4f} | {keys} | {n} |", flush=True)
    del Q, K, V, O
if "--json" in sys.argv:
    json.dump(out, open(arg("--json", ""), "w"), indent=1)
