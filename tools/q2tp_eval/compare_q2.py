"""Parity of cortiq's q2tp payloads against the q2tp study's torch payloads (same tensors).
usage: compare_q2.py REF_ARM_DIR CORTIQ_DUMP_DIR OUT_JSON [--hess]
  REF_ARM_DIR     /root/q2x/arms/<ARM> (<name>.q2tp, read-only)
  CORTIQ_DUMP_DIR raw dump of a cortiq file (`cortiq dequant --dtype raw --all`, <name>.bin)
Per tensor: fraction of equal row lo / step, rung codes and 2-bit codes; weight SSE of both against the
f16 source; with --hess the held-out (and calibration) output error tr(dW H dW^T) of both under the
study's FP-stream Hessians (/root/q2x/hess/L{l}.pt, read-only)."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q2x_lib import *

ref, dump, out = sys.argv[1:4]
use_h = "--hess" in sys.argv
dev = "cuda"
Wt = Weights("cpu")
rep = {}
for l in range(CFG["layers"]):
    hh = torch.load(f"/root/q2x/hess/L{l}.pt") if use_h else None
    for lin in ("mlp.gate_proj", "mlp.up_proj"):
        n = tname(l, lin)
        w = Wt.cpu[n].to(torch.float16).float()
        rows, cols = w.shape
        a = q2tp_decode(open(f"{ref}/{n}.q2tp", "rb").read(), rows, cols)
        b = q2tp_decode(open(f"{dump}/{n}.bin", "rb").read(), rows, cols)
        r = {k: float((a[k] == b[k]).mean()) for k in ("lo", "st", "code", "q")}
        wa, wb = torch.from_numpy(q2tp_dequant(a)), torch.from_numpy(q2tp_dequant(b))
        r["sse_ref"] = float(((w - wa) ** 2).sum())
        r["sse_cortiq"] = float(((w - wb) ** 2).sum())
        if use_h:
            wd = w.to(dev)
            for key in ("val", "cal"):
                H = hh[key].to(dev).float()
                r[f"{key}_ref"] = out_err(wd, wa.to(dev), H)
                r[f"{key}_cortiq"] = out_err(wd, wb.to(dev), H)
        rep[n] = r
    v = [rep[tname(l, x)] for x in ("mlp.gate_proj", "mlp.up_proj")]
    msg = " ".join(f"{k}={np.mean([x[k] for x in v]):.4f}" for k in ("lo", "st", "code", "q"))
    if use_h:
        msg += " val_ratio=%.4f" % np.mean([x["val_cortiq"] / x["val_ref"] for x in v])
    print(f"layer {l}: {msg} sse_ratio={np.mean([x['sse_cortiq'] / x['sse_ref'] for x in v]):.4f}", flush=True)
v = list(rep.values())
summ = {k: float(np.mean([x[k] for x in v])) for k in ("lo", "st", "code", "q")}
summ["sse_ratio_total"] = sum(x["sse_cortiq"] for x in v) / sum(x["sse_ref"] for x in v)
if use_h:
    for key in ("val", "cal"):
        summ[f"{key}_ratio_median"] = float(np.median([x[f"{key}_cortiq"] / x[f"{key}_ref"] for x in v]))
        summ[f"{key}_ratio_total"] = sum(x[f"{key}_cortiq"] for x in v) / sum(x[f"{key}_ref"] for x in v)
print(json.dumps(summ, indent=1))
json.dump({"summary": summ, "tensors": rep}, open(out, "w"), indent=1)
