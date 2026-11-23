#!/usr/bin/env python3
"""Fair re-check of the patent-2 ODE example (prune_diff_eq_06b.ipynb) on a Mac (MPS).

The notebook trained DTG-MA masks on Qwen3-0.6B for templated ODEs and reported
3.3 % (base) -> 100 % (mask). Its base got a raw prompt and 32 new tokens, and its
test problems came from the training generator. Here:

  * coefficients are split: the TRAIN set and a HELD-OUT set the masks never see;
  * the base is measured three ways: the notebook's (raw prompt, 32 tokens), with the
    chat template (thinking off, 300 tokens) and with a one-shot example in the prompt;
  * the answer check accepts the usual notations (e^(2x), e^{2x}, exp(2x), 2x^2, ...);
  * masks: "all" = every nn.Linear output (the notebook), "ffn" = the MLP's
    intermediate neurons only (what `cortiq skill bake` masks).

usage: ode_repro.py --variant all|ffn [--epochs 15] [--n-test 100] --out result.json
"""
import argparse, json, math, random, re, time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------------------------------------------------------- data
TRAIN = {
    "exponential": [-3, -2, -1, 1, 2, 3],
    "linear_const": [-3, -2, -1, 1, 2, 3, 4, 5],
    "exponential_const": [(1, 2), (1, 3), (1, 4), (2, 4), (2, 6), (2, 8), (3, 6), (3, 9), (3, 12)],
    "second_order": [-4, -2, 2, 4, 6],
    "harmonic": [1],
    "simple": [None],
}
HELD = {
    "exponential": [-6, -5, -4, 4, 5, 6],
    "linear_const": [-7, -6, -5, -4, 6, 7, 8, 9],
    "exponential_const": [(1, 5), (1, 7), (2, 10), (4, 8), (4, 12), (5, 10), (5, 15)],
    "second_order": [-8, -6, 8, 10, 12],
    "harmonic": [2, 3],
}


def fmt_exp(a):
    return "x" if a == 1 else "(-x)" if a == -1 else f"({a}x)"


def problem(kind, c):
    """(prompt, canonical answer, verify(text) -> bool)."""
    num = r"\s*\*?\s*"
    if kind == "exponential":
        a = c
        eq = "y' = y" if a == 1 else "y' = -y" if a == -1 else f"y' = {a}y"
        sol = f"y = Ce^{fmt_exp(a)}"
        coef = "" if a == 1 else "-" if a == -1 else str(a)
        pats = [rf"e\^\s*[\(\{{]?\s*{re.escape(coef)}{num}x\s*[\)\}}]?", rf"exp\(\s*{re.escape(coef)}{num}x\s*\)"]
        # e^( must be followed by the coefficient itself, so 2x never matches e^(12x)
        def ok(t, pats=pats):
            return any(re.search(p, t) for p in pats)
        return f"Solve the differential equation: {eq}\nGive the general solution.", sol, ok
    if kind == "linear_const":
        a = c
        eq = f"y' = {a}"
        sol = f"y = {a}x + C"
        coef = "" if a == 1 else "-" if a == -1 else str(a)
        p = rf"(?<![\d.]){re.escape(coef)}{num}x(?!\^|²)"
        return f"Solve the differential equation: {eq}\nFind y(x).", sol, lambda t, p=p: re.search(p, t) is not None
    if kind == "exponential_const":
        a, b = c
        part = b // a
        eq = f"y' + y = {b}" if a == 1 else f"y' + {a}y = {b}"
        sol = f"y = {part} + Ce^({'-x' if a == 1 else f'-{a}x'})"
        coef = "-" if a == 1 else f"-{a}"
        e = [rf"e\^\s*[\(\{{]?\s*{re.escape(coef)}{num}x\s*[\)\}}]?", rf"exp\(\s*{re.escape(coef)}{num}x\s*\)"]
        def ok(t, e=e, part=part):
            return re.search(rf"(?<![\d.]){part}(?![\d.])", t) is not None and any(re.search(p, t) for p in e)
        return f"Solve the differential equation: {eq}\nGive the general solution.", sol, ok
    if kind == "second_order":
        a = c
        k = a // 2
        eq = f"y'' = {a}"
        sol = f"y = {k}x^2 + C1x + C2"
        coef = "" if k == 1 else "-" if k == -1 else str(k)
        p = rf"(?<![\d.]){re.escape(coef)}{num}x\s*(\^\s*\{{?2\}}?|²)"
        return f"Solve the differential equation: {eq}\nGive the general solution.", sol, lambda t, p=p: re.search(p, t) is not None
    if kind == "harmonic":
        k = c
        eq = "y'' + y = 0" if k == 1 else f"y'' + {k * k}y = 0"
        arg = "x" if k == 1 else f"{k}x"
        sol = f"y = C1 sin({arg}) + C2 cos({arg})"
        def ok(t, k=k):
            a = r"x" if k == 1 else rf"{k}\s*x"
            return re.search(rf"sin\s*[\(\{{]?\s*{a}\s*[\)\}}]?", t) is not None and re.search(rf"cos\s*[\(\{{]?\s*{a}\s*[\)\}}]?", t) is not None
        return f"Solve the differential equation: {eq}\nGive the general solution.", sol, ok
    # simple: y' = 0 -> y = C
    return ("Solve the differential equation: y' = 0\nWhat is y?", "y = C",
            lambda t: re.search(r"y\s*=\s*C\b|y\s*=\s*c\b|constant", t) is not None)


def make_set(table, n, rng):
    kinds = list(table)
    out = []
    for _ in range(n):
        k = rng.choice(kinds)
        out.append((k,) + problem(k, rng.choice(table[k])))
    return out


# ----------------------------------------------------------------------------- masks
class OutMask(nn.Module):
    """Linear(x) * sigmoid(mask) over output features (the notebook's DTGMALinear)."""
    def __init__(self, lin, init):
        super().__init__()
        self.lin = lin
        self.mask = nn.Parameter(torch.full((lin.out_features,), init, device=lin.weight.device))
        self.on = True

    def forward(self, x):
        y = self.lin(x)
        return y * torch.sigmoid(self.mask).to(y.dtype) if self.on else y


class InMask(nn.Module):
    """Linear(x * sigmoid(mask)) over input features: down_proj, i.e. the FFN neurons."""
    def __init__(self, lin, init):
        super().__init__()
        self.lin = lin
        self.mask = nn.Parameter(torch.full((lin.in_features,), init, device=lin.weight.device))
        self.on = True

    def forward(self, x):
        return self.lin(x * torch.sigmoid(self.mask).to(x.dtype)) if self.on else self.lin(x)


def inject(model, variant, init):
    mods = []
    def walk(m, path=""):
        for name, ch in list(m.named_children()):
            full = f"{path}.{name}" if path else name
            if isinstance(ch, nn.Linear):
                if variant == "all":
                    w = OutMask(ch, init)
                elif variant == "ffn" and name == "down_proj":
                    w = InMask(ch, init)
                else:
                    continue
                setattr(m, name, w)
                mods.append(w)
            else:
                walk(ch, full)
    walk(model)
    return mods


def set_on(mods, on):
    for m in mods:
        m.on = on


# ----------------------------------------------------------------------------- eval
ONE_SHOT = ("Solve the differential equation: y' = 7y\nGive the general solution.\n"
            "Answer: y = Ce^(7x)\n\n")


@torch.no_grad()
def generate(model, tok, prompt, max_new, device):
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate(model, tok, items, mode, device):
    right, rows = 0, []
    for kind, prompt, sol, ok in items:
        if mode == "raw32":
            text = generate(model, tok, prompt, 32, device)
        elif mode == "chat":
            p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            text = generate(model, tok, p, 300, device)
        elif mode == "chat_1shot":
            p = tok.apply_chat_template([{"role": "user", "content": ONE_SHOT + prompt + "\nAnswer with the solution only."}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            text = generate(model, tok, p, 300, device)
        else:
            raise ValueError(mode)
        good = bool(ok(text))
        right += good
        rows.append({"kind": kind, "prompt": prompt.splitlines()[0], "expected": sol, "got": text[:200], "ok": good})
    return right / len(items), rows


def per_kind(rows):
    agg = {}
    for r in rows:
        a = agg.setdefault(r["kind"], [0, 0])
        a[0] += r["ok"]; a[1] += 1
    return {k: f"{v[0]}/{v[1]}" for k, v in sorted(agg.items())}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--variant", choices=["all", "ffn"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--init", type=float, default=2.0)
    ap.add_argument("--l1", type=float, default=0.02)
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(device)
    model.eval()

    rng = random.Random(1000 + a.seed)
    test_in = make_set(TRAIN, a.n_test, rng)          # the notebook's kind of test
    test_held = make_set(HELD, a.n_test, rng)         # coefficients never trained on
    train = make_set(TRAIN, a.train, random.Random(a.seed))
    train_prompts = {p for _, p, _, _ in train}
    overlap = sum(p in train_prompts for _, p, _, _ in test_in) / len(test_in)
    res = {"args": vars(a), "device": device, "in_dist_test_prompts_seen_in_train": overlap}
    t0 = time.time()

    if not a.skip_base:
        for mode in ["raw32", "chat", "chat_1shot"]:
            for name, items in [("in_dist", test_in), ("held_out", test_held)]:
                acc, rows = evaluate(model, tok, items, mode, device)
                res[f"base_{mode}_{name}"] = acc
                res[f"base_{mode}_{name}_kinds"] = per_kind(rows)
                res.setdefault("examples", {})[f"base_{mode}_{name}"] = rows[:4]
                print(f"base {mode:10s} {name:8s} {acc*100:5.1f}%  {per_kind(rows)}  [{time.time()-t0:.0f}s]", flush=True)

    mods = inject(model, a.variant, a.init)
    params = [m.mask for m in mods]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    n_mask = sum(p.numel() for p in params)
    print(f"variant {a.variant}: {len(mods)} masked modules, {n_mask:,} mask parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr)
    data = [(p, " " + s) for _, p, s, _ in train]
    model.train()
    for ep in range(1, a.epochs + 1):
        random.Random(ep).shuffle(data)
        tot, nb = 0.0, 0
        for i in range(0, len(data), a.batch):
            batch = [p + s for p, s in data[i:i + a.batch]]
            t = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            t["labels"] = t["input_ids"].clone()
            lm = model(**t).loss
            l1 = sum(torch.sigmoid(p).mean() for p in params)
            loss = lm + a.l1 * l1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += lm.item(); nb += 1
        with torch.no_grad():
            active = sum((torch.sigmoid(p) > 0.5).sum().item() for p in params) / n_mask
        print(f"epoch {ep:2d}/{a.epochs}  LM {tot/nb:.4f}  active {active*100:.1f}%  [{time.time()-t0:.0f}s]", flush=True)
    model.eval()
    res["active"] = active
    for name, items in [("in_dist", test_in), ("held_out", test_held)]:
        acc, rows = evaluate(model, tok, items, "raw32", device)
        res[f"skill_{name}"] = acc
        res[f"skill_{name}_kinds"] = per_kind(rows)
        res.setdefault("examples", {})[f"skill_{name}"] = rows[:6]
        res.setdefault("wrong", {})[f"skill_{name}"] = [r for r in rows if not r["ok"]][:8]
        print(f"skill raw32 {name:8s} {acc*100:5.1f}%  {per_kind(rows)}", flush=True)
    res["seconds"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
