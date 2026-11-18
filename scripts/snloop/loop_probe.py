#!/usr/bin/env python3
"""Loop probe for SenseNova-Loop-9B (teacher-forced, one window).

1. Logits with the looped block [4, 28) run once and twice; per token: top-1
   agreement, KL(2 || 1), the state after the first pass.
2. Mixed runs: a share of tokens leaves after the first pass. Such a token's
   rows in the second pass's caches are its first-pass projections (the
   attention's q/k/v, the GDN's qkv/z/b/a inputs), and its hidden after the
   block is the first pass's. Reports ppl and top-1 agreement with the full
   model for an oracle choice (tokens whose answer the second pass keeps) and
   for random choices.

usage: loop_probe.py MODEL_DIR TEXT N OUT_NPZ [--shares 0.5,0.7,0.85]
env:   SN_DEV=cuda, SN_DTYPE=bf16|f32
"""
import argparse, math, os
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("text"); ap.add_argument("n", type=int); ap.add_argument("out")
ap.add_argument("--shares", default="0.5,0.7,0.85")
ap.add_argument("--offset", type=int, default=0)
a = ap.parse_args()
dev = os.environ.get("SN_DEV", "cuda")
dt = torch.float32 if os.environ.get("SN_DTYPE") == "f32" else torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, dtype=dt).to(dev).eval()
lm = model.model.language_model
plan2 = lm._loop_plan
plan1 = tuple(range(4)) + tuple(range(4, 28)) + tuple(range(28, 32))
text = open(a.text).read()
ids = tok(text, add_special_tokens=False).input_ids[a.offset:a.offset + a.n]
x = torch.tensor([ids]).to(dev)
T = x.shape[1]

# --- hooks -----------------------------------------------------------------
state = {"visit": {}, "store": {}, "mask": None, "capture": False}
PROJ = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")

def layer_pre(li):
    def f(mod, args, kwargs):
        state["visit"][li] = state["visit"].get(li, 0) + 1
    return f

def proj_hook(li, name):
    def f(mod, inp, out):
        v = state["visit"].get(li, 0)
        key = (li, name)
        if v == 1:
            state["store"][key] = out.detach()
        elif v == 2 and state["mask"] is not None:
            m = state["mask"]
            out = out.clone(); out[:, m] = state["store"][key][:, m]
            return out
    return f

def layer_post(li):
    def f(mod, inp, out):
        if li != 27:
            return None
        h = out[0] if isinstance(out, tuple) else out
        v = state["visit"].get(li, 0)
        if v == 1:
            state["store"][("h", 27)] = h.detach()
            if state["capture"]:
                state["h1"] = h.detach().float().cpu()
        elif v == 2 and state["mask"] is not None:
            m = state["mask"]
            h = h.clone(); h[:, m] = state["store"][("h", 27)][:, m]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return f

for li in range(4, 28):
    layer = lm.layers[li]
    layer.register_forward_pre_hook(layer_pre(li), with_kwargs=True)
    layer.register_forward_hook(layer_post(li))
    for name in PROJ:
        sub = layer
        try:
            for part in name.split("."):
                sub = getattr(sub, part)
        except AttributeError:
            continue
        sub.register_forward_hook(proj_hook(li, name))

def fwd(plan, mask=None, capture=False):
    state["visit"].clear(); state["store"].clear(); state["mask"] = mask; state["capture"] = capture
    lm._loop_plan = plan
    with torch.inference_mode():
        lg = model(input_ids=x, use_cache=False).logits[0].float().cpu()
    lm._loop_plan = plan2
    torch.cuda.empty_cache()
    return lg

xc = x[0].cpu()

def stats(lg, ref):
    nll = torch.nn.functional.cross_entropy(lg[:-1], xc[1:], reduction="mean").item()
    agree = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
    return math.exp(nll), agree

l2 = fwd(plan2, capture=True)
h1 = state["h1"]
l1 = fwd(plan1)
p2, _ = stats(l2, l2); p1, ag1 = stats(l1, l2)
t2 = l2.argmax(-1); t1 = l1.argmax(-1)
same = (t1 == t2)
lp2 = torch.log_softmax(l2, -1); lp1 = torch.log_softmax(l1, -1)
kl = (lp2.exp() * (lp2 - lp1)).sum(-1)
print(f"T {T} ppl loop2 {p2:.4f} loop1 {p1:.4f} top1 agree {ag1:.4f} KL mean {kl.mean():.4f} median {kl.median():.4f}", flush=True)
np.savez_compressed(a.out, ids=np.array(ids), h1=h1[0].numpy().astype(np.float16), same=same.cpu().numpy(),
                    kl=kl.cpu().numpy(), t1=t1.cpu().numpy(), t2=t2.cpu().numpy())
# oracle: leave where the first pass already gives the second pass's answer, lowest KL first
order = torch.argsort(kl)
for share in [float(s) for s in a.shares.split(",")]:
    k = int(share * T)
    for kind in ("oracle", "random"):
        if kind == "oracle":
            idx = order[:k]
        else:
            idx = torch.randperm(T)[:k]
        m = torch.zeros(T, dtype=torch.bool); m[idx] = True; m = m.to(dev)
        lm_ = fwd(plan2, mask=m)
        pp, ag = stats(lm_, l2)
        print(f"share {share:.2f} {kind:6s} ppl {pp:.4f} top1 agree {ag:.4f}", flush=True)
print("PROBE_DONE", flush=True)
