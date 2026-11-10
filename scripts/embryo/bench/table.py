"""Embryo-56M against KLM-1.6 and the Open SLM leaderboard.

    python -I table.py --runs <dir> --space <dir>/index.html --out results.md \
        [--json out.json] [--chart-scores scores.json]

Reads the lm-eval and ArithMark-3 results of a run directory, the model
rows of the leaderboard Space's `index.html` (parsed at run time; the
Space's license allows small excerpts with attribution, so only a few
reference rows go into the output, never the table), and computes the
Intelligence Index the way the Space's methodology states it:

    N(score, chance) = 100 * (score - chance) / (100 - chance)
    ARC = mean(ARC-Easy, ARC-Challenge)  (before normalization)
    Index = (N(HellaSwag, 25) + N(ARC, 25) + N(PIQA, 50) + 0.65 * N(ArithMark-3, 25)) / 3.65

(an independent implementation; a missing component drops its weight
from numerator and denominator, at least two are needed). Checked
against KLM-1.6's published 33.95.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

CHANCE = {"hellaswag": 25.0, "arc": 25.0, "piqa": 50.0, "arithmark3": 25.0}
FIELDS = ("arc", "arcChall", "hellaswag", "piqa", "arithmark3")

EMBRYO_NAME = "Embryo-56M"
EMBRYO_PARAMS = 55_900_000  # total; 34.7M active per token
KLM_PEERS = ["KLM-1.6", "KLM-1.5", "GPT-X3-150M", "SmolLM2-135M", "LFM-2.5-230M-Base", "void.1"]
SIMILAR_SIZE = ["Supra-50M-Base", "Nexus-Erebus-50M", "Pythia-70M", "Pythia-31M", "michel-tiny"]


def n(score: float, chance: float) -> float:
    return 100.0 * (score - chance) / (100.0 - chance)


def index(m: dict) -> float | None:
    comps = []
    if m.get("hellaswag") is not None:
        comps.append(n(m["hellaswag"], CHANCE["hellaswag"]))
    a, c = m.get("arc"), m.get("arcChall")
    if a is not None or c is not None:
        arc = (a + c) / 2 if a is not None and c is not None else (a if a is not None else c)
        comps.append(n(arc, CHANCE["arc"]))
    if m.get("piqa") is not None:
        comps.append(n(m["piqa"], CHANCE["piqa"]))
    has_am = m.get("arithmark3") is not None
    if len(comps) + has_am < 2:
        return None
    w = 0.65 if has_am else 0.0
    s = sum(comps) + (n(m["arithmark3"], CHANCE["arithmark3"]) * w if has_am else 0.0)
    return s / (len(comps) + w)


def parse_space(html: str) -> list[dict]:
    start = html.index("const MODELS = [")
    end = html.index("];", start)
    rows = []
    for line in html[start:end].splitlines():
        s = line.strip()
        if not s.startswith("{ name:"):
            continue  # commented-out rows start with //
        name = re.search(r"name:\s*'((?:[^'\\]|\\.)*)'", s).group(1)
        params = int(re.search(r"params:\s*([0-9]+)", s).group(1))
        m = {"name": name, "params": params}
        for f in FIELDS:
            g = re.search(rf"\b{f}:\s*(-?[0-9.]+)", s)
            m[f] = float(g.group(1)) if g else None
        rows.append(m)
    return rows


def load_embryo(runs: Path) -> dict:
    def lm(task: str) -> dict:
        js = json.loads((runs / f"{task}.raw.results.json").read_text())
        r = dict(js["results"][task])
        r["n"] = js["n-samples"][task]["effective"]
        return r

    tasks = {"arc": "arc_easy", "arcChall": "arc_challenge", "hellaswag": "hellaswag", "piqa": "piqa"}
    m = {"name": EMBRYO_NAME, "params": EMBRYO_PARAMS, "detail": {}}
    for f, t in tasks.items():
        r = lm(t)
        m[f] = 100.0 * r["acc_norm,none"]
        m["detail"][t] = {
            "acc": 100.0 * r["acc,none"],
            "acc_stderr": 100.0 * r["acc_stderr,none"],
            "acc_norm": 100.0 * r["acc_norm,none"],
            "acc_norm_stderr": 100.0 * r["acc_norm_stderr,none"],
            "n": r["n"],
        }
    am = json.loads((runs / "arithmark-3.results.json").read_text())["results"]["arithmark-3"]
    m["arithmark3"] = am["acc_norm"]
    p = am["acc_norm"] / 100.0
    m["detail"]["arithmark-3"] = {
        "acc": am["acc"],
        "acc_norm": am["acc_norm"],
        "acc_norm_stderr": 100.0 * math.sqrt(p * (1 - p) / (am["total"] - 1)),
        "n": am["total"],
    }
    return m


def fmt(v: float | None, d: int = 2) -> str:
    return "–" if v is None else f"{v:.{d}f}"


def row(m: dict, rank: dict | None = None) -> str:
    p = m["params"] / 1e6
    return (
        f"| {m['name']} | {p:.1f}M | {fmt(m['arc'])} | {fmt(m['arcChall'])} | {fmt(m['hellaswag'])} | "
        f"{fmt(m['piqa'])} | {fmt(m['arithmark3'])} | {fmt(m['index'])} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--space", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--chart-scores", default=None, help="write the Embryo-56M scores for chart.py")
    args = ap.parse_args()

    board = parse_space(Path(args.space).read_text())
    for m in board:
        m["index"] = index(m)
    klm = next(m for m in board if m["name"] == "KLM-1.6")
    assert abs(klm["index"] - 33.95) < 0.005, f"KLM-1.6 index {klm['index']:.4f} != 33.95: formula drift"
    emb = load_embryo(Path(args.runs))
    emb["index"] = index(emb)

    def ranks(limit: float) -> tuple[int, int, list[dict]]:
        pool = [m for m in board if m["params"] < limit and m["index"] is not None] + [emb]
        pool.sort(key=lambda m: -m["index"])
        r = pool.index(emb) + 1
        return r, len(pool), pool

    lines = []
    lines.append("# Embryo-56M on the KLM-1.6 benchmarks\n")
    lines.append(
        "Zero-shot, length-normalized accuracy (acc_norm), percent. ARC / HellaSwag / PIQA: lm-eval 0.4.12; "
        "ArithMark-3: the official script's data and scoring rule. Embryo-56M: backbone of "
        "`final5-release/model.cmf` (q8_2f), raw text, through `cortiq loglik`. Other rows: Open SLM "
        "leaderboard (AxiomicLabs/Open_SLM_Leaderboard, read 2026-11-10), reference excerpt with attribution; "
        "Index recomputed from those rows by the Space's stated formula.\n"
    )
    lines.append("| Model | Params | ARC-Easy | ARC-Challenge | HellaSwag | PIQA | ArithMark-3 | Int Index |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append("| *chance* | | 25.00 | 25.00 | 25.00 | 50.00 | 25.00 | 0.00 |")
    lines.append(row(emb).replace(f"| {EMBRYO_NAME} |", f"| **{EMBRYO_NAME}** |"))
    by = {m["name"]: m for m in board}
    shown = [EMBRYO_NAME]
    for name in KLM_PEERS:
        if name in by:
            lines.append(row(by[name]))
            shown.append(name)
    lines.append("| *similar size* | | | | | | | |")
    for name in SIMILAR_SIZE:
        if name in by:
            lines.append(row(by[name]))
            shown.append(name)

    lines.append("\n## Rank by Intelligence Index\n")
    lines.append("| Size class | Embryo-56M rank | Models in class | Leader | Leader index | Neighbours (index) |")
    lines.append("|---|---|---|---|---|---|")
    ranks_out = {}
    for label, lim in (("< 100M", 100e6), ("< 150M", 150e6), ("< 250M", 250e6)):
        r, tot, pool = ranks(lim)
        nb = pool[max(0, r - 3): r - 1] + pool[r: r + 2]
        nbs = ", ".join(f"{m['name']} ({m['index']:.2f})" for m in nb)
        lines.append(f"| {label} | {r} | {tot} | {pool[0]['name']} | {pool[0]['index']:.2f} | {nbs} |")
        ranks_out[label] = {"rank": r, "of": tot, "leader": pool[0]["name"], "neighbours": [m["name"] for m in nb]}

    lines.append("\n## Embryo-56M detail\n")
    lines.append("| Task | n | acc | acc_norm | ± stderr (acc_norm) | chance | (acc_norm − chance) / stderr |")
    lines.append("|---|---|---|---|---|---|---|")
    chance = {"arc_easy": 25, "arc_challenge": 25, "hellaswag": 25, "piqa": 50, "arithmark-3": 25}
    for t, d in emb["detail"].items():
        z = (d["acc_norm"] - chance[t]) / d["acc_norm_stderr"]
        lines.append(
            f"| {t} | {d['n']} | {d['acc']:.2f} | {d['acc_norm']:.2f} | {d['acc_norm_stderr']:.2f} | {chance[t]} | {z:+.1f} |"
        )
    chat = []
    for t in ("arc_easy", "arc_challenge", "hellaswag", "piqa"):
        f = Path(args.runs) / f"{t}.chat.results.json"
        if f.exists():
            r = json.loads(f.read_text())["results"][t]
            d = emb["detail"][t]
            chat.append(
                f"| {t} | {d['acc']:.2f} / {d['acc_norm']:.2f} | "
                f"{100 * r['acc,none']:.2f} / {100 * r['acc_norm,none']:.2f} | ±{100 * r['acc_norm_stderr,none']:.2f} |"
            )
    if chat:
        lines.append("\n## Side check, not in the index: raw text vs the cmf-im-v1 chat frame\n")
        lines.append("Chat frame: the task prompt as the user turn, each choice as the assistant answer.\n")
        lines.append("| Task | raw acc / acc_norm | chat acc / acc_norm | ± stderr (chat acc_norm) |")
        lines.append("|---|---|---|---|")
        lines.extend(chat)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.chart_scores:
        Path(args.chart_scores).write_text(json.dumps({
            "ARC-Easy": emb["arc"], "ARC-Challenge": emb["arcChall"], "HellaSwag": emb["hellaswag"],
            "PIQA": emb["piqa"], "ArithMark-3": emb["arithmark3"],
        }, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps({"embryo": emb, "ranks": ranks_out, "klm16_index": klm["index"]}, indent=2))


if __name__ == "__main__":
    main()
