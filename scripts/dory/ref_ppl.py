#!/usr/bin/env python3
"""Reference NLL of `cortiq ppl --windows N --window-len L` windows, by ref_dory.py's
forward over the same ids (`CMF_PPL_IDS_OUT`). ref_ppl.py MODEL IDS N L [per_loop|last_loop] [loops]"""
import json, math, sys
import torch
from ref_dory import Dory

model, ids_path, n, ln = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
kv = sys.argv[5] if len(sys.argv) > 5 else "per_loop"
loops = int(sys.argv[6]) if len(sys.argv) > 6 else 2
ids = [int(x) for x in open(ids_path).read().split()]
m = Dory(model, "cuda", torch.float32)
step = (len(ids) - ln - 1) // n
tot, cnt = 0.0, 0
for w in range(n):
    x = torch.tensor(ids[w * step : w * step + ln + 1], device="cuda")
    caches = {}
    for b in range(0, ln, 64):
        e = min(b + 64, ln)
        lg = m.forward(x[b:e], b, caches, loops=loops, kv=kv)
        tot += torch.nn.functional.cross_entropy(lg, x[b + 1 : e + 1], reduction="sum").item()
        cnt += e - b
    print(w, math.exp(tot / cnt), flush=True)
print(f"ref {kv} loops {loops}: {n}x{ln} tokens {cnt} nll {tot/cnt:.5f} ppl {math.exp(tot/cnt):.4f}")
