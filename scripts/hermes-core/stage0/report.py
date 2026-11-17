"""Results table of stage 0 from the graded files of a run directory.

    python -I report.py --run <run dir> --out results.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CATS = ["slot", "os", "choose", "none", "react", "fix", "two", "arith", "direct"]
WHAT = {
    "slot": "one card: find the example, fill the values",
    "os": "one card with a section per OS: take the right one",
    "choose": "three cards: take the right one",
    "none": "three cards, none fits: say NONE",
    "react": "rules + the last output: the next action",
    "fix": "a failed command and its error: the corrected one",
    "two": "a two-step command from two cards",
    "arith": "a number read or computed from an output",
    "direct": "one card, nothing to fill",
}
ROWS = [
    ("copy_first", "Copy the card's first example (floor)", ""),
    ("smollm2-135m", "SmolLM2-135M-Instruct", "135M"),
    ("smollm2-360m", "SmolLM2-360M-Instruct", "362M"),
    ("boris2-think", "Boris-2-Think", "144M"),
    ("ref", "Reference answers (grader self-test)", ""),
]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def failure(r: dict) -> str:
    a = r["answer"].strip()
    if not a:
        return "empty answer"
    if r["check"] in ("label", "value"):
        return "wrong label or number"
    if a.upper().startswith("NONE"):
        return "said NONE"
    if "{{" in a or "}}" in a or "path/to" in a:
        return "placeholder left unfilled"
    prog = lambda c: re.sub(r"^sudo\s+", "", c.strip()).split(" ")[0] if c.strip() else ""
    if prog(a) != prog(r["ref"]):
        return "another program"
    return "right program, wrong arguments"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run)
    data = {}
    for key, _, _ in ROWS:
        f = run / f"{key}.graded.jsonl"
        if f.exists():
            data[key] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    n_by = {c: sum(r["cat"] == c for r in data["ref"]) for c in CATS}
    lines = ["# Stage 0: open models on the R set\n",
             "R: tasks of \"reasoning over a card\", the card in the prompt, two worked examples, greedy decoding. "
             "Commands are checked by running them in a sandbox (Linux) or by normalised match (macOS, Windows, "
             "system-state commands). Share of correct answers, percent.\n",
             "| Model | Params | " + " | ".join(f"{c} ({n_by[c]})" for c in CATS) + " | All | 95 % |",
             "|---|---|" + "---|" * (len(CATS) + 2)]
    for key, name, params in ROWS:
        if key not in data:
            continue
        rs = data[key]
        cells = []
        for c in CATS:
            sub = [r for r in rs if r["cat"] == c]
            cells.append(f"{100 * sum(r['ok'] for r in sub) / len(sub):.0f}")
        k, n = sum(r["ok"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        lines.append(f"| {name} | {params} | " + " | ".join(cells) + f" | **{100 * k / n:.1f}** ({k}/{n}) | {lo:.0f}–{hi:.0f} |")
    lines.append("\nCategories: " + "; ".join(f"**{c}** — {WHAT[c]}" for c in CATS) + ".\n")

    lines.append("## Where the answers go wrong\n")
    models = [(k, nme) for k, nme, _ in ROWS if k in data and k not in ("ref", "copy_first")]
    kinds = ["placeholder left unfilled", "another program", "right program, wrong arguments", "said NONE",
             "wrong label or number", "empty answer"]
    lines.append("| Failure | " + " | ".join(n for _, n in models) + " |")
    lines.append("|---|" + "---|" * len(models))
    for kd in kinds:
        row = []
        for key, _ in models:
            bad = [r for r in data[key] if not r["ok"]]
            row.append(str(sum(failure(r) == kd for r in bad)))
        lines.append(f"| {kd} | " + " | ".join(row) + " |")
    lines.append("| *wrong in all* | " + " | ".join(str(sum(not r["ok"] for r in data[k])) for k, _ in models) + " |")

    if "boris2-think" in data:
        b = data["boris2-think"]
        lines.append(f"\nBoris-2-Think wrote a `<think>` block in {sum(bool(r.get('think')) for r in b)} of {len(b)} answers "
                     f"and a `Final answer:` line in {sum(bool(r.get('final')) for r in b)}.")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
