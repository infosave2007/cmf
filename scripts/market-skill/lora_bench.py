#!/usr/bin/env python3
"""LoRA on the market-skill task, the bake's budget and data, for the article's comparison.

  lora_bench.py HF_MODEL_DIR DATA_DIR TARGET OUT_DIR [--rank 16]

The same 12 000 balanced train records in the bake's order (market_skill.balanced/interleave), 750 steps of 8,
the exact classifier's loss (cross-entropy over the two class tokens at the answer position), the checkpoint
with the best accuracy on the bake's held set (480 records, every 50 steps), then the test set (3000) once.
bf16 weights, LoRA in f32 on every linear of the language model (PEFT `all-linear`), AdamW lr 2e-4.
Reports the training wall time, the adapter's size on disk and the accuracies, as JSON.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import market_skill as ms  # noqa: E402


def render(rec):
    sp = f"{ms.DIRECTIVE}\n\n{ms.system_prompt(rec['target'], rec['symbol'])}"
    return (f"<|im_start|>system\n{sp}<|im_end|>\n<|im_start|>user\n{rec['prompt']}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("target")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    classes = ms.CLASSES[a.target]
    tok = AutoTokenizer.from_pretrained(a.model)
    cls_ids = [tok.encode(c, add_special_tokens=False) for c in classes]
    assert all(len(x) == 1 for x in cls_ids), cls_ids
    cls_ids = [x[0] for x in cls_ids]
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    load_s = time.time() - t_load
    d = os.path.join(a.data, a.target)
    train_all = ms.load(os.path.join(d, "train.jsonl"))
    markets = list(dict.fromkeys(r["symbol"] for r in train_all))
    train = ms.interleave([ms.balanced([r for r in train_all if r["symbol"] == m], ms.TRAIN_PER_MARKET, classes) for m in markets])
    held = ms.held_records(a.data, a.target)
    test = ms.load(os.path.join(d, "eval_test.jsonl"))

    def batches(recs, bs):
        for i in range(0, len(recs), bs):
            yield recs[i:i + bs]

    def encode(recs):
        texts = [render(r) for r in recs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        last = enc["attention_mask"].sum(1) - 1
        y = torch.tensor([classes.index(r["label"]) for r in recs])
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda(), last.cuda(), y.cuda()

    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def parts(m):
        # the decoder stack and the head (a PEFT wrapper, a multimodal wrapper)
        core = m.get_base_model() if hasattr(m, "get_base_model") else m
        body = core.model
        body = getattr(body, "language_model", body)
        return body, core.lm_head

    def class_logits(m, ids, mask, last):
        # the hidden state at the answer position against the class rows of
        # the head only (the exact classifier's loss; no full-vocabulary logits)
        body, head = parts(m)
        h = body(input_ids=ids, attention_mask=mask).last_hidden_state
        h = h[torch.arange(ids.shape[0], device=ids.device), last]
        return h.float() @ head.weight[cls_ids].float().T

    @torch.no_grad()
    def evaluate(m, recs):
        m.eval()
        preds, conf = [], []
        for b in batches(recs, 16):
            ids, mask, last, _ = encode(b)
            p = torch.softmax(class_logits(m, ids, mask, last), -1)
            preds += p.argmax(-1).tolist()
            conf += p.max(-1).values.tolist()
        m.train()
        gold = [r["label"] for r in recs]
        pl = [classes[i] for i in preds]
        return ms.metrics(gold, pl, classes), pl, conf

    base_test, _, _ = evaluate(model, test)
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    best = (-1.0, 0, None)
    curve = []
    train_s = 0.0
    k = 0
    for step in range(1, a.steps + 1):
        b = [train[(k + i) % len(train)] for i in range(a.batch)]
        k += a.batch
        torch.cuda.synchronize()
        t0 = time.time()
        ids, mask, last, y = encode(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = class_logits(model, ids, mask, last)
        loss = torch.nn.functional.cross_entropy(lg, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        train_s += time.time() - t0
        if step % a.eval_every == 0:
            mh, _, _ = evaluate(model, held)
            curve.append({"step": step, "held_acc": mh["acc"], "loss": float(loss)})
            print(f"step {step}: held acc {mh['acc']*100:.2f}%  loss {float(loss):.4f}  train {train_s:.0f}s", flush=True)
            if mh["acc"] > best[0]:
                best = (mh["acc"], step, {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad})
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best[2][n].to(p.device))
    test_m, test_pred, test_conf = evaluate(model, test)
    model.save_pretrained(os.path.join(a.out, "adapter"))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(a.out, "adapter")) for f in fs)
    with open(os.path.join(a.out, "test_preds.jsonl"), "w") as f:
        for r, p, c in zip(test, test_pred, test_conf):
            f.write(json.dumps({"timestamp": r["timestamp"], "symbol": r["symbol"], "label": r["label"], "pred": p, "conf": c}) + "\n")
    rep = {"target": a.target, "rank": a.rank, "steps": a.steps, "batch": a.batch, "lr": a.lr,
           "trainable_params": n_train, "adapter_bytes": size, "train_seconds": train_s, "load_seconds": load_s,
           "best_step": best[1], "best_held_acc": best[0], "base_test": base_test, "lora_test": test_m, "curve": curve,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
