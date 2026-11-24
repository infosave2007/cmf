#!/usr/bin/env python3
"""Article charts for the market skill: held accuracy by step (skill vs LoRA) and test accuracy against the share
of the most confident answers. Reads the bake logs, the serve predictions and the LoRA reports from DATA."""
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SKILL = "#2a78d6"
LORA = "#eb6834"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def bake_curve(path):
    steps, acc = [0], None
    for line in open(path, encoding="utf-8"):
        m = re.search(r"baseline \(full\): [\d.]+ \| acc ([\d.]+)%", line)
        if m:
            acc = [float(m.group(1))]
        m = re.search(r"\[A\] step (\d+):.*\| acc ([\d.]+)%", line)
        if m:
            steps.append(int(m.group(1)))
            acc.append(float(m.group(2)))
    return steps, acc


def lora_report(target):
    p = os.path.join(DATA, f"lora_{target}", "report.json")
    return json.load(open(p)) if os.path.exists(p) else None


def lora_curve(rep, base0):
    return [0] + [c["step"] for c in rep["curve"]], [base0] + [100 * c["held_acc"] for c in rep["curve"]]


# 1. held accuracy by step, one panel per target, same y-axis
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, target, title in zip(axes, ("range", "dir"), ("Range of the next hour (HIGH / LOW)", "Direction of the next hour (UP / DOWN)")):
    style(ax)
    s, a = bake_curve(os.path.join(DATA, f"{target}_bake.log"))
    ends = [[a[-1], a[-1], "skill", s[-1]]]
    ax.plot(s, a, color=SKILL, linewidth=2, marker="o", markersize=4, label="cortiq skill, f32 step (1.02 s/step)")
    bp = os.path.join(DATA, f"{target}_bf16_bake.log")
    if os.path.exists(bp):
        bs, ba = bake_curve(bp)
        ax.plot(bs, ba, color=SKILL, linewidth=1.5, linestyle=(0, (3, 2)), marker="o", markersize=3,
                label="cortiq skill, bf16 step (0.57 s/step)")
        kb = max(range(len(ba)), key=lambda i: (ba[i], -i))
        ax.plot([bs[kb]], [ba[kb]], marker="o", markersize=7, color=SKILL, markerfacecolor=SURFACE, markeredgewidth=2, zorder=5)
        ends.append([ba[-1], ba[-1], "skill bf16", bs[-1]])
    rep = lora_report(target)
    if rep:
        ls, la = lora_curve(rep, a[0])
        ax.plot(ls, la, color=LORA, linewidth=2, marker="o", markersize=4,
                label=f"LoRA r=16, all linear layers ({rep['train_seconds'] / rep['steps']:.2f} s/step)")
        ends.append([la[-1], la[-1], "LoRA", ls[-1]])
    ax.axhline(50, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(8, 49.6, "chance", color=MUTED, fontsize=9, ha="left", va="top")
    # the checkpoint each method keeps: the best held accuracy
    k = max(range(len(a)), key=lambda i: (a[i], -i))
    ax.plot([s[k]], [a[k]], marker="o", markersize=9, color=SKILL, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    if rep:
        k2 = max(range(len(la)), key=lambda i: (la[i], -i))
        ax.plot([ls[k2]], [la[k2]], marker="o", markersize=9, color=LORA, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.set_title(title, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("training step (batch 8)")
    ax.set_xlim(-10, 760)
    ax.set_ylim(45, 75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    # direct labels at the right end, spread to at least 1.4 points apart
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + 1.4)
    for y0, y1, text, x1 in ends:
        ax.annotate(text, (x1, y0), xytext=(x1 + 14, y1), textcoords="data", color=INK2, fontsize=9, va="center")
axes[0].set_ylabel("accuracy on the held set (480, balanced)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper left", ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.06, 1.0), labelcolor=INK2)
fig.text(0.5, 0.015, "Qwen3.5-0.8B, the same training records in the same order, 750 steps of 8. "
         "Large dot: the checkpoint kept (best held accuracy). RTX 3060.", ha="center", fontsize=9, color=MUTED)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(os.path.join(OUT, "held_accuracy_by_step.png"), dpi=200)
plt.close(fig)


# 2. test accuracy against the share of the most confident answers (range, 3000 hours)
def serve_preds(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append((max(r["p"].values()), r["pred"] == r["label"]))
    return out


def coverage(preds):
    preds = sorted(preds, key=lambda x: -x[0])
    xs, ys = [], []
    n = len(preds)
    hit = 0
    for i, (_, ok) in enumerate(preds, 1):
        hit += ok
        if i >= n * 0.02 and (i % 15 == 0 or i == n):
            xs.append(100 * i / n)
            ys.append(100 * hit / i)
    return xs, ys


fig, ax = plt.subplots(figsize=(8.5, 4.8))
style(ax)
series = [("base model", serve_preds(os.path.join(DATA, "base_range_test.jsonl")), MUTED, (0, (4, 3)))]
series.append(("cortiq skill", serve_preds(os.path.join(DATA, "skill_range_test.jsonl")), SKILL, "-"))
gp = os.path.join(DATA, "logit_range_test.jsonl")
if os.path.exists(gp):
    series.insert(1, ("logistic regression", serve_preds(gp), INK2, (0, (1, 2))))
lp = os.path.join(DATA, "lora_range", "test_preds.jsonl")
if os.path.exists(lp):
    lpreds = [(r["conf"], r["pred"] == r["label"]) for r in map(json.loads, open(lp, encoding="utf-8"))]
    series.append(("LoRA r=16", lpreds, LORA, "-"))
ends = []
for name, preds, color, ls in series:
    x, y = coverage(preds)
    ax.plot(x, y, color=color, linewidth=2, linestyle=ls, label=name)
    ends.append([y[-1], y[-1], f"{name}  {y[-1]:.1f}%", x[-1]])
    # the 20 % point, the article's number: skill's label above, LoRA's below
    k = min(range(len(x)), key=lambda i: abs(x[i] - 20))
    if name in ("cortiq skill", "LoRA r=16"):
        ax.plot([x[k]], [y[k]], marker="o", markersize=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        up = name == "cortiq skill"
        ax.annotate(f"{y[k]:.1f}%", (x[k], y[k]), xytext=(0, 10 if up else -12), textcoords="offset points", ha="center",
                    va="bottom" if up else "top", color=INK, fontsize=9)
# end labels, spread to at least 1.9 points apart
ends.sort(key=lambda e: e[0])
for i in range(1, len(ends)):
    ends[i][1] = max(ends[i][1], ends[i - 1][1] + 1.9)
for y0, y1, text, x1 in ends:
    ax.annotate(text, (x1, y0), xytext=(x1 + 2, y1), textcoords="data", color=INK2, fontsize=9, va="center")
ax.set_xlim(0, 124)
ax.set_xticks(range(0, 101, 20))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_ylim(55, 100)
ax.set_xlabel("share of answers kept, most confident first")
ax.set_ylabel("accuracy on the kept answers")
ax.set_title("Range of the next hour, test period (3000 hours, opened once)", fontsize=12, loc="left", pad=10)
ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.86, 0.0))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_vs_coverage.png"), dpi=200)
plt.close(fig)
print("ok")
