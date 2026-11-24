#!/usr/bin/env python3
"""Skill vs LoRA on the market task: test accuracy (overall, per market, balanced), top-20 % confident accuracy,
file size, training time. Prints a markdown table from DATA (serve predictions, LoRA reports)."""
import json
import os
import sys

DATA = sys.argv[1]
CLASSES = {"range": ["LOW", "HIGH"], "dir": ["DOWN", "UP"]}


def rows_serve(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append({"symbol": r["symbol"], "label": r["label"], "pred": r["pred"], "conf": max(r["p"].values())})
    return out


def rows_lora(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def acc(rs):
    return 100 * sum(r["pred"] == r["label"] for r in rs) / len(rs)


def bal(rs, classes):
    rec = []
    for c in classes:
        g = [r for r in rs if r["label"] == c]
        rec.append(sum(r["pred"] == c for r in g) / len(g) if g else 0.0)
    return 100 * sum(rec) / len(rec)


def top(rs, share=0.2):
    k = round(len(rs) * share)
    return acc(sorted(rs, key=lambda r: -r["conf"])[:k])


for target in ("range", "dir"):
    cl = CLASSES[target]
    methods = [("base", rows_serve(os.path.join(DATA, f"base_{target}_test.jsonl"))),
               ("skill", rows_serve(os.path.join(DATA, f"skill_{target}_test.jsonl")))]
    lp = os.path.join(DATA, f"lora_{target}", "test_preds.jsonl")
    if os.path.exists(lp):
        methods.append(("LoRA r16", rows_lora(lp)))
    gp = os.path.join(DATA, f"logit_{target}_test.jsonl")
    if os.path.exists(gp):
        methods.append(("logistic", rows_serve(gp)))
    print(f"\n### {target}\n")
    print("| method | acc | balanced | top-20 % | BTC bal | EURUSD bal | USDJPY bal |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, rs in methods:
        per = [bal([r for r in rs if r["symbol"] == m], cl) for m in ("BTCUSDT", "EURUSD", "USDJPY")]
        print(f"| {name} | {acc(rs):.1f} | {bal(rs, cl):.1f} | {top(rs):.1f} | " + " | ".join(f"{p:.1f}" for p in per) + " |")
    rp = os.path.join(DATA, f"lora_{target}", "report.json")
    if os.path.exists(rp):
        rep = json.load(open(rp))
        print(f"\nLoRA: {rep['trainable_params']:,} trainable params, adapter {rep['adapter_bytes']/1e6:.1f} MB, "
              f"train {rep['train_seconds']:.0f} s ({rep['train_seconds']/rep['steps']:.3f} s/step), best step "
              f"{rep['best_step']} (held {100*rep['best_held_acc']:.2f} %), {rep['gpu']}, torch {rep['torch']}")
