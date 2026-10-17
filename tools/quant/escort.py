#!/usr/bin/env python3
"""Escort: which tensors to keep at higher precision for a bits-per-weight budget.

Sensitivity: inside the quantized model (--model), restore ONE linear to the version a
higher-precision file of the same model holds (--hi, e.g. the q8_2f conversion; several
--hi files are tried in order, the first that stores the tensor in a different dtype wins)
and measure the drop of the held-out NLL (--metric nll) or of the KL to a teacher's top-K
(--metric kl, needs --teacher). Each block's input on the held-out windows is cached, so a
tensor in layer l reruns only layers l..L-1. Gain per extra byte = drop / (hi payload bytes
- current payload bytes).

Water-filling: candidates sorted by gain per byte are taken while they fit the budget
(--bpw x body parameters / 8 bytes; body = every linear of the decoder layers) and gain > 0.
--greedy R re-measures the remaining candidates inside the escorted model after each of R
equal budget slices (single-tensor gains stop being additive when one tensor dominates).

Output (OUT/): sensitivity.json (per tensor gain, bytes, gain/byte, picks), escort.txt
(names), tensor_quant.txt (`--tensor-quant NAME=DTYPE` lines for convert / quantize-gptq),
and with --write FILE.cmf the escorted model (hi payloads patched in via cortiq).

  escort.py --model C2.cmf --hi q8_2f.cmf --corpus train.ids,train=0,held=8192:8 --bpw 0.1 --out esc
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import calib  # noqa: E402
import cmfio  # noqa: E402
import qmodel  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the quantized CMF")
    ap.add_argument("--hi", action="append", required=True, help="higher-precision CMF(s) of the same model")
    ap.add_argument("--corpus", action="append", required=True,
                    help="evaluation windows = the held-out windows of PATH[,held=OFF:CNT+...] (train= ignored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", choices=("nll", "kl"), default="nll")
    ap.add_argument("--teacher", help="reference CMF for --metric kl")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--bpw", type=float, nargs="+", default=[0.1], help="extra body bpw budget(s)")
    ap.add_argument("--include", help="regex: only these tensors are candidates")
    ap.add_argument("--exclude", help="regex: never these tensors")
    ap.add_argument("--greedy", type=int, default=1, help="re-measure rounds (1 = one-shot water-filling)")
    ap.add_argument("--wl", type=int, default=512)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--residency", choices=("resident", "stream"), default="resident")
    ap.add_argument("--vchunk", type=int, default=16384)
    ap.add_argument("--tf32", action="store_true", help="TF32 matmuls (default exact f32)")
    ap.add_argument("--write", help="write the escorted model for the first budget here")
    ap.add_argument("--cortiq", default="cortiq")
    ap.add_argument("--gpu-cap-gb", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


def log(*a, t0=time.time()):
    print(f"[{time.time() - t0:7.0f}s]", *a, flush=True)


def candidates(model, his, inc, exc):
    """[(layer, role, name, hi CmfFile, extra bytes, params)]"""
    out = []
    for layer in model.layers:
        for role, lin in layer.lin.items():
            if inc and not re.search(inc, lin.name):
                continue
            if exc and re.search(exc, lin.name):
                continue
            for h in his:
                if lin.name in h and h.tensor(lin.name).dtype != lin.dtype:
                    extra = h.tensor(lin.name).nbytes - len(lin.payload)
                    if extra > 0:
                        out.append((layer.l, role, lin.name, h, extra, lin.rows * lin.cols))
                    break
    return out


class Scorer:
    """held-out loss of the model with layers [l, L) re-run from cached block inputs"""

    def __init__(self, model, windows, metric, teacher, mb):
        self.m, self.w, self.metric, self.teacher, self.mb = model, windows, metric, teacher, mb
        self.tokens = len(windows) * (windows.shape[1] - 1)
        self.rebase()

    def rebase(self):
        xs = [self.m.embed_ids(self.w[i:i + self.mb]) for i in range(0, len(self.w), self.mb)]
        self.bounds = self.m.forward_layers(xs, keep=True)
        self.base = self._loss(self.bounds[-1])

    def _loss(self, outs):
        tot = 0.0
        for j, x in enumerate(outs):
            s0 = j * self.mb
            tg = self.w[s0:s0 + x.shape[0]]
            if self.metric == "kl":
                lp = torch.from_numpy(np.asarray(self.teacher["held_lp"][s0:s0 + x.shape[0]]))
                ix = torch.from_numpy(np.asarray(self.teacher["held_ix"][s0:s0 + x.shape[0]]))
                _, kl, _ = self.m.head_eval(x, tg, lp, ix)
                tot += sum(kl)
            else:
                ce, _, _ = self.m.head_eval(x, tg)
                tot += sum(ce)
        return tot / self.tokens

    def with_layer_from(self, l):
        return self._loss(self.m.forward_layers(self.bounds[l], l0=l))


def waterfill(sens, budget):
    order = sorted(sens, key=lambda n: -sens[n]["gain_per_byte"])
    pick, used = [], 0
    for n in order:
        if sens[n]["gain"] <= 0:
            break
        if used + sens[n]["extra_bytes"] <= budget:
            pick.append(n)
            used += sens[n]["extra_bytes"]
    return pick, used


def main(argv=None):
    args = get_args(argv)
    qmodel.setup_torch(0, tf32=args.tf32, deterministic=True)
    dev = qmodel.resolve_device(args.device, args.gpu_cap_gb)
    os.makedirs(args.out, exist_ok=True)
    corpora = [calib.parse_corpus(s) for s in args.corpus]
    for c in corpora:
        c.train = 0
    calib.place(corpora, args.wl, 0)
    _, held_w, _ = calib.build_sets(corpora, args.wl)
    teacher = None
    if args.metric == "kl":
        if not args.teacher:
            raise SystemExit("--metric kl needs --teacher")
        teacher = calib.teacher_cache(os.path.join(args.out, "teacher"), args.teacher, corpora, args.wl, args.topk,
                                      lambda c: qmodel.Model(c, dev, residency="stream", vchunk=args.vchunk, pin=False),
                                      mb=args.mb, log=log)
    src = cmfio.CmfFile(args.model)
    his = [cmfio.CmfFile(p) for p in args.hi]
    model = qmodel.Model(src, dev, residency=args.residency, vchunk=args.vchunk)
    cands = candidates(model, his, args.include, args.exclude)
    body = sum(l.rows * l.cols for l in model.linears())
    log(f"{len(held_w)} held-out windows of {args.wl}; {len(cands)} candidates; body {body} parameters")
    scorer = Scorer(model, held_w, args.metric, teacher, args.mb)
    log(f"base held-out {args.metric} {scorer.base:.5f} per token")
    budgets = [b * body / 8 for b in args.bpw]
    rounds = max(1, args.greedy)
    picked, sens_all, used_total = [], {}, 0
    remaining = list(cands)
    by_name = {c[2]: c for c in cands}
    for r in range(rounds):
        sens = {}
        for i, (l, role, name, h, extra, params) in enumerate(remaining):
            hi = qmodel.QLinear(name, h.tensor(name), h.payload(name), pin=model.pin)
            old = model.swap(l, role, hi)
            loss = scorer.with_layer_from(l)
            model.swap(l, role, old)
            hi.release()
            gain = scorer.base - loss
            sens[name] = {"gain": gain, "extra_bytes": int(extra), "gain_per_byte": gain / extra,
                          "params": int(params), "dtype": model.layers[l].lin[role].dtype, "hi_dtype": h.tensor(name).dtype,
                          "round": r}
            if (i + 1) % 25 == 0 or i + 1 == len(remaining):
                log(f"round {r}: {i + 1}/{len(remaining)} measured (last {name}: {gain:+.5f})")
        sens_all.update(sens)
        if rounds == 1:
            break
        # greedy: spend the next slice of the first budget, then re-measure inside the escorted model
        slice_budget = budgets[0] * (r + 1) / rounds - used_total
        pick, used = waterfill(sens, slice_budget)
        for name in pick:
            l, role, _, h, extra, _ = by_name[name]
            model.swap(l, role, qmodel.QLinear(name, h.tensor(name), h.payload(name), pin=model.pin))
        picked += pick
        used_total += used
        remaining = [c for c in remaining if c[2] not in set(pick)]
        scorer.rebase()
        log(f"greedy round {r}: +{len(pick)} tensors ({used_total * 8 / body:.4f} bpw), base now {scorer.base:.5f}")
    results = []
    for bpw, budget in zip(args.bpw, budgets):
        if rounds > 1 and bpw == args.bpw[0]:
            pick, used = picked, used_total
        else:
            pick, used = waterfill(sens_all, budget)
        results.append({"bpw_budget": bpw, "pick": pick, "used_bytes": int(used), "extra_bpw": used * 8 / body,
                        "predicted_gain": float(sum(sens_all[n]["gain"] for n in pick))})
        kinds = {}
        for n in pick:
            k = n.split(".")[-2]
            kinds[k] = kinds.get(k, 0) + 1
        log(f"budget +{bpw} bpw: {len(pick)} tensors, +{used * 8 / body:.4f} bpw, by kind {kinds}, "
            f"predicted {args.metric} gain {results[-1]['predicted_gain']:.5f}")
    per_kind = {}
    for n, s in sens_all.items():
        per_kind.setdefault(n.split(".")[-2], []).append(s["gain"] / (s["extra_bytes"] / 2 ** 20))
    summary = {k: {"mean_gain_per_MiB": float(np.mean(v)), "n": len(v)} for k, v in per_kind.items()}
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["mean_gain_per_MiB"]):
        log(f"  {k:10s} mean gain per MiB {v['mean_gain_per_MiB']:+.6f} ({v['n']} tensors)")
    json.dump({"args": vars(args), "metric": args.metric, "base": scorer.base, "body_params": body,
               "sens": sens_all, "per_kind": summary, "budgets": results}, open(os.path.join(args.out, "sensitivity.json"), "w"),
              indent=1)
    first = results[0]["pick"]
    with open(os.path.join(args.out, "escort.txt"), "w") as f:
        f.write("".join(n + "\n" for n in first))
    with open(os.path.join(args.out, "tensor_quant.txt"), "w") as f:
        f.write("".join(f"--tensor-quant {n}={cmfio.QUANT_NAME[sens_all[n]['hi_dtype']]}\n" for n in first))
    if args.write and first:
        groups = {}
        for n in first:
            groups.setdefault(sens_all[n]["hi_dtype"], []).append(n)
        cur = args.model
        for gi, (dt, names) in enumerate(sorted(groups.items())):
            paths = {}
            for n in names:
                h = by_name[n][3]
                p = os.path.join(args.out, "hi", f"{n}.bin")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fh:
                    fh.write(h.payload(n).tobytes())
                paths[n] = p
            dst = args.write if gi == len(groups) - 1 else os.path.join(args.out, f"tmp{gi}.cmf")
            cmfio.patch_raw(args.cortiq, cur, dst, paths, dtype=cmfio.QUANT_NAME[dt])
            if cur != args.model:
                os.remove(cur)
            cur = dst
        log(f"wrote and verified {args.write}")
    return results


if __name__ == "__main__":
    main()
