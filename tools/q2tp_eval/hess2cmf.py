"""The q2tp study's FP-stream gate/up Hessians (/root/q2x/hess/L{l}.pt, read-only; key "cal" = 32k train
tokens) as a cortiq CMFHESS1 cache, so `cortiq quantize-gptq --codec q2tp --hessians` runs on EXACTLY the
Hessians the torch arms (N1, W0) used. One entry per layer under both gate_proj and up_proj names.
usage: hess2cmf.py OUT [KEY]"""
import struct, sys
import numpy as np, torch

out = sys.argv[1]
key = sys.argv[2] if len(sys.argv) > 2 else "cal"
L = 42
with open(out + ".tmp", "wb") as f:
    f.write(b"CMFHESS1")
    f.write(struct.pack("<Q", L))
    for l in range(L):
        hh = torch.load(f"/root/q2x/hess/L{l}.pt")
        H = hh[key].double().numpy()
        n = H.shape[0]
        names = [f"model.layers.{l}.mlp.{p}.weight" for p in ("gate_proj", "up_proj")]
        f.write(struct.pack("<I", len(names)))
        for nm in names:
            b = nm.encode()
            f.write(struct.pack("<I", len(b)))
            f.write(b)
        count = int(hh.get("ncal", 32768))
        f.write(struct.pack("<QQQ", n, count, n * n))
        f.write(np.ascontiguousarray(np.diag(H)).astype("<f8").tobytes())   # sum x^2 per channel
        iu = np.triu_indices(n)
        f.write(np.ascontiguousarray(H[iu]).astype("<f8").tobytes())        # upper triangle, by rows
        print(l, n, flush=True)
import os
os.rename(out + ".tmp", out)
print("done", out)
