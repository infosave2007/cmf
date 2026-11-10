"""Transformers reference ppl for MNN-Pocket-0.3B: windows of LEN consecutive ids from a cortiq ids dump.
usage: python -I ref_ppl.py MODEL_DIR IDS_FILE N_WINDOWS LEN [STRIDE]"""
import math, sys, time
import torch
from transformers import AutoModelForCausalLM
model_dir, ids_file, nw, ln = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ids = [int(t) for t in open(ids_file).read().split()]
stride = int(sys.argv[5]) if len(sys.argv) > 5 else (len(ids) - ln) // max(nw - 1, 1)
m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
tot, n, t0 = 0.0, 0, time.time()
with torch.no_grad():
    for w in range(nw):
        x = torch.tensor([ids[w * stride: w * stride + ln]])
        lp = torch.log_softmax(m(x).logits[0, :-1].double(), -1)
        nll = -lp[torch.arange(ln - 1), x[0, 1:]]
        tot += nll.sum().item(); n += ln - 1
        print(f"window {w} start {w*stride} ppl {math.exp(nll.mean().item()):.3f}", flush=True)
print(f"REF PPL (transformers f32) = {math.exp(tot / n):.3f} over {n} positions, {time.time()-t0:.1f} s")
