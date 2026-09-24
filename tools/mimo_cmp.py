#!/usr/bin/env python3
"""Compare engine dumps with oracle dumps for MiMo-V2 (or any model that
writes the same raw layout).

    python3 tools/mimo_cmp.py --engine /dev/shm/eng_p3 --ref /root/mimo/ref_p3_raw \
        --positions 0,1,127-129,last --engine-trace m.txt --ref-trace /root/mimo/ref_p3_raw/moe_trace.txt

Inputs (raw little-endian f32, one vector per file):
  p{pos:06}_l{li:02}.f32            hidden state after layer li (CMF_LAYER_DUMP)
  p{pos:06}_l{li:02}_{label}.f32    optional sub-dumps (attn, ffn, ...), compared per label
  p{pos:06}_logits.f32              logits at pos (either side; the engine may
                                    instead give --engine-logits in CMF_LOGIT_DUMP
                                    format: final hidden then logits, one position)
  moe_trace.txt                     CMF_MOE_TRACE lines 'li:e1,...,ek', one per
                                    (token, MoE layer), position-major (CMF_PREFILL=seq)
  picks.jsonl                       oracle margins; a mismatched pick whose oracle
                                    k-th vs (k+1)-th margin is below --tie-margin is
                                    reported as a near-tie, not an error

Reports per layer (worst over the chosen positions): relative L2 ||e-r||/||r||,
cosine, max|e-r|; flags the first layer above --tol and any layer whose error
jumps by more than --jump x the previous layer's. Logits: top-1 / top-5
agreement, KL(ref || eng), max|delta| and max|delta| / max|ref|. Picks: mean
|A n B| / k overlap per layer. Exit status 1 when a gate fails (--tol on
hidden states, top-1 on logits, picks outside near-ties); 0 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np

NAME = re.compile(r"^p(\d{6})_l(\d{2,3})(?:_([A-Za-z0-9]+))?\.f32$")
LOGITS = re.compile(r"^p(\d{6})_logits\.f32$")


def scan(d):
    """{(pos, li, label): path}, {pos: logits path}"""
    hid, lg = {}, {}
    for fn in os.listdir(d):
        m = NAME.match(fn)
        if m:
            hid[(int(m.group(1)), int(m.group(2)), m.group(3) or "")] = os.path.join(d, fn)
            continue
        m = LOGITS.match(fn)
        if m:
            lg[int(m.group(1))] = os.path.join(d, fn)
    return hid, lg


def load(p):
    return np.fromfile(p, dtype="<f4").astype(np.float64)


def parse_positions(spec, avail):
    if spec in (None, "", "all"):
        return sorted(avail)
    n = max(avail) + 1 if avail else 0
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if part == "last":
            out.add(n - 1)
        elif re.fullmatch(r"-?\d+", part):
            p = int(part)
            out.add(p + n if p < 0 else p)
        elif re.fullmatch(r"\d+-\d+", part):
            a, b = map(int, part.split("-"))
            out.update(range(a, b + 1))
        else:
            raise SystemExit(f"bad position spec {part!r}")
    return sorted(p for p in out if p in avail)


def metrics(e, r):
    d = e - r
    nr = np.linalg.norm(r)
    ne = np.linalg.norm(e)
    rel = float(np.linalg.norm(d) / nr) if nr > 0 else float("inf")
    cos = float(e @ r / (ne * nr)) if ne > 0 and nr > 0 else float("nan")
    return rel, cos, float(np.abs(d).max())


def compare_hidden(eng, ref, positions, a):
    keys = sorted(k for k in ref if k in eng and k[0] in positions)
    if not keys:
        print("hidden: no common (pos, layer) files", file=sys.stderr)
        return True, None
    by = defaultdict(list)                      # (label, li) -> [(pos, rel, cos, max)]
    for k in keys:
        e, r = load(eng[k]), load(ref[k])
        if e.shape != r.shape:
            raise SystemExit(f"{eng[k]}: {e.size} values vs {r.size} in {ref[k]}")
        if not np.isfinite(e).all():
            by[(k[2], k[1])].append((k[0], float("inf"), float("nan"), float("inf")))
            continue
        by[(k[2], k[1])].append((k[0],) + metrics(e, r))
    ok, first_bad = True, None
    for label in sorted({lab for lab, _ in by}):
        name = label or "hidden"
        npos = len({k[0] for k in keys if k[2] == label})
        gate = f"tol rel-L2 {a.tol:g}" if label == "" else "informational"
        print(f"== {name}: {npos} positions, {gate}")
        print(f"{'layer':>5} {'worst rel-L2':>13} {'@pos':>6} {'min cos':>10} {'max|d|':>10} {'mean rel':>10}")
        prev = None
        for li in sorted(li for lab, li in by if lab == label):
            rows = by[(label, li)]
            worst = max(rows, key=lambda r: r[1])
            mean = sum(r[1] for r in rows) / len(rows)
            mincos = min(r[2] for r in rows)
            maxd = max(r[3] for r in rows)
            flag = ""
            if worst[1] > a.tol:
                flag = "  <-- above tol"
                if label == "" and first_bad is None:
                    first_bad = li
                if label == "":
                    ok = False
            elif prev is not None and prev > 0 and worst[1] > a.jump * prev and worst[1] > a.tol / 10:
                flag = f"  <-- jump x{worst[1] / prev:.1f}"
            print(f"{li:5d} {worst[1]:13.3e} {worst[0]:6d} {mincos:10.6f} {maxd:10.3e} {mean:10.3e}{flag}")
            prev = worst[1]
    if first_bad is not None:
        print(f"FIRST LAYER ABOVE TOL: {first_bad}")
    return ok, first_bad


def log_softmax(x):
    m = x.max()
    return x - m - math.log(np.exp(x - m).sum())


def compare_logits(eng_lg, ref_lg, positions, a, vocab=None):
    pos = [p for p in positions if p in eng_lg and p in ref_lg]
    if not pos:
        return True
    ok = True
    print(f"== logits at {len(pos)} positions")
    print(f"{'pos':>6} {'top1 e/r':>17} {'top5 overlap':>12} {'KL(r||e)':>10} {'max|d|':>10} {'max|d|/max|r|':>13} {'r margin':>9}")
    t1 = 0
    for p in pos:
        e, r = eng_lg[p], ref_lg[p]
        if vocab:
            e, r = e[:vocab], r[:vocab]
        if e.shape != r.shape:
            raise SystemExit(f"logits at {p}: {e.size} vs {r.size}")
        te, tr = np.argsort(-e)[:5], np.argsort(-r)[:5]
        rs = np.sort(r)[::-1]
        margin = float(rs[0] - rs[1])
        lr, le = log_softmax(r), log_softmax(e)
        kl = float((np.exp(lr) * (lr - le)).sum())
        d = np.abs(e - r)
        same = te[0] == tr[0]
        t1 += same
        if not same and margin >= a.logit_tie:
            ok = False
        print(f"{p:6d} {te[0]:>8d}/{tr[0]:<8d} {len(set(te) & set(tr)):>12d} {kl:10.3e} {d.max():10.3e} "
              f"{d.max() / np.abs(r).max():13.3e} {margin:9.4f}{'' if same else '  <-- top1 differs'}")
    print(f"top-1 agreement {t1}/{len(pos)}")
    return ok


def read_trace(path, pos0=0):
    """CMF_MOE_TRACE -> {(pos, li): [experts]}. Position advances whenever the
    layer index does not increase (position-major order, CMF_PREFILL=seq)."""
    out = {}
    pos, last = pos0 - 1, None
    for line in open(path):
        line = line.strip()
        if not line or ":" not in line:
            continue
        li, ex = line.split(":", 1)
        li = int(li)
        if last is None or li <= last:
            pos += 1
        last = li
        out[(pos, li)] = [int(x) for x in ex.split(",") if x != ""]
    return out


def read_margins(path):
    m = {}
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            m[(r["pos"], r["layer"])] = r["margin"]
    return m


def compare_picks(a, positions):
    eng = read_trace(a.engine_trace, a.engine_trace_pos0)
    ref = read_trace(a.ref_trace)
    margins = read_margins(a.ref_picks or os.path.join(os.path.dirname(a.ref_trace), "picks.jsonl"))
    keys = sorted(k for k in ref if k in eng and (not positions or k[0] in positions))
    if not keys:
        print("picks: no common (pos, layer) entries")
        return True
    per_layer = defaultdict(list)
    bad, ties = [], 0
    for k in keys:
        e, r = set(eng[k]), set(ref[k])
        ov = len(e & r) / max(len(r), 1)
        per_layer[k[1]].append(ov)
        if e != r:
            if margins.get(k, float("inf")) < a.tie_margin:
                ties += 1
            else:
                bad.append((k, sorted(eng[k]), sorted(ref[k]), margins.get(k)))
    print(f"== expert picks: {len(keys)} (pos, layer) pairs, {len(per_layer)} layers")
    worst = sorted(per_layer.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[:8]
    for li, ovs in worst:
        print(f"  layer {li:3d} mean overlap {sum(ovs) / len(ovs):.4f} over {len(ovs)} tokens")
    total = sum(sum(v) for v in per_layer.values()) / sum(len(v) for v in per_layer.values())
    print(f"mean overlap {total:.4f}; mismatches: {len(bad)} real, {ties} near-ties (margin < {a.tie_margin:g})")
    for k, e, r, m in bad[:10]:
        print(f"  pos {k[0]} layer {k[1]}: engine {e} oracle {r} margin {m}")
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", help="engine dump dir (CMF_LAYER_DUMP)")
    ap.add_argument("--ref", help="oracle dump dir (tools/mimo_ref.py dump)")
    ap.add_argument("--positions", default="all", help="'all' or '0,1,127-129,last'")
    ap.add_argument("--tol", type=float, default=2e-3, help="gate on per-layer relative L2")
    ap.add_argument("--jump", type=float, default=3.0)
    ap.add_argument("--engine-logits", help="CMF_LOGIT_DUMP file (hidden then logits) for --logits-pos")
    ap.add_argument("--logits-pos", type=int, help="position of --engine-logits (default: last ref pos)")
    ap.add_argument("--vocab", type=int, help="compare only the first N logits")
    ap.add_argument("--logit-tie", type=float, default=1e-3,
                    help="a top-1 disagreement is tolerated when the ref top1-top2 margin is below this")
    ap.add_argument("--engine-trace", help="CMF_MOE_TRACE file")
    ap.add_argument("--engine-trace-pos0", type=int, default=0, help="position of the trace's first token")
    ap.add_argument("--ref-trace", help="oracle moe_trace.txt")
    ap.add_argument("--ref-picks", help="oracle picks.jsonl (default: next to --ref-trace)")
    ap.add_argument("--tie-margin", type=float, default=1e-4)
    a = ap.parse_args()

    ok = True
    positions = None
    if a.engine and a.ref:
        eh, el = scan(a.engine)
        rh, rl = scan(a.ref)
        avail = {k[0] for k in rh} | set(rl)
        positions = parse_positions(a.positions, avail)
        hid_ok, _ = compare_hidden(eh, rh, set(positions), a)
        ok &= hid_ok
        eng_lg = {p: load(f) for p, f in el.items()}
        ref_lg = {p: load(f) for p, f in rl.items() if p in positions or p in el}
        if a.engine_logits:
            raw = load(a.engine_logits)
            lp = a.logits_pos if a.logits_pos is not None else (max(rl) if rl else None)
            if lp is None or lp not in rl:
                raise SystemExit("--engine-logits needs the oracle's p{pos}_logits.f32 at --logits-pos")
            ref_lg[lp] = load(rl[lp])
            eng_lg[lp] = raw[raw.size - ref_lg[lp].size:]
            positions = sorted(set(positions) | {lp})
        ok &= compare_logits(eng_lg, ref_lg, positions, a, a.vocab)
    if a.engine_trace and a.ref_trace:
        ok &= compare_picks(a, set(positions) if positions else None)
    print("CMP", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
