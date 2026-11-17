"""Add Embryo-56M as a seventh bar to the KLM-1.6 benchmark chart.

    python -I chart.py --svg benchmarks.svg --scores scores.json --out benchmarks-embryo.svg

`--scores`: {"ARC-Easy": 31.2, "ARC-Challenge": ..., "HellaSwag": ..., "PIQA": ..., "ArithMark-3": ...}
(percent, acc_norm). The input is the line-oriented SVG of the KLM-1.6
card: five panels, each with gridlines, six rounded bars (`<path>` with a
`<title>Model, Bench: v%</title>`) and their value labels. Every panel is
re-laid for seven bars at the same scale (read off the existing bars), the
legend gets a seventh entry and the pill is re-centred, a footnote states
how Embryo-56M was measured. The axis of every panel starts at chance, as
in the original; where Embryo-56M falls below the start of an axis, that
axis is extended downwards in its own tick step (plot height unchanged, all
bars of the panel rescaled) and the chance level is drawn dashed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

NAME = "Embryo-56M"
COLOR = "#e0703a"
BAR_W, PITCH, R = 26.0, 34.0, 3.0
LEGEND_TEXT_W = 80.0  # "Embryo-56M" at 13 px Inter 600, measured in the browser

BAR_RE = re.compile(
    r'<path d="M([0-9.]+),([0-9.]+) V([0-9.]+) Q[^"]*" fill="([^"]+)"><title>([^<]+), ([^<:]+): ([0-9.]+)%</title></path>'
)
LABEL_RE = re.compile(r'<text x="([0-9.]+)" y="([0-9.]+)" font-size="11.5"([^>]*)>([0-9.]+)</text>')
GRID_RE = re.compile(r'<line x1="([0-9.]+)" x2="([0-9.]+)" y1="([0-9.]+)" y2="([0-9.]+)" stroke="(#[0-9a-f]+)"')
AXIS_RE = re.compile(r'<text x="([0-9.]+)" y="([0-9.]+)" font-size="12"[^>]*>([0-9.]+)</text>')


def bar_path(x0: float, base: float, top: float) -> str:
    x1 = x0 + BAR_W
    return (
        f"M{x0:.1f},{base:.1f} V{top + R:.1f} Q{x0:.1f},{top:.1f} {x0 + R:.1f},{top:.1f} "
        f"H{x1 - R:.1f} Q{x1:.1f},{top:.1f} {x1:.1f},{top + R:.1f} V{base:.1f} Z"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svg", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scores = json.loads(Path(args.scores).read_text())
    lines = Path(args.svg).read_text().split("\n")

    # Panels in document order: a baseline gridline (#cfcec7) opens one.
    panels = []
    for i, l in enumerate(lines):
        g = GRID_RE.search(l)
        if g and g.group(5) == "#cfcec7":
            panels.append({"x1": float(g.group(1)), "x2": float(g.group(2)), "base": float(g.group(3)), "line": i,
                           "bars": [], "labels": []})
    pi = -1
    for i, l in enumerate(lines):
        while pi + 1 < len(panels) and i >= panels[pi + 1]["line"]:
            pi += 1
        if pi < 0:
            continue
        b = BAR_RE.search(l)
        if b:
            panels[pi]["bars"].append({"line": i, "top": float(b.group(3)) - R, "fill": b.group(4),
                                       "model": b.group(5), "bench": b.group(6), "v": float(b.group(7))})
    # Value labels: matched to bars by value text and x order; floors from the axis labels.
    for p in panels:
        p["bench"] = p["bars"][0]["bench"]
    used = set()
    for p in panels:
        for bar in p["bars"]:
            for i, l in enumerate(lines):
                if i in used:
                    continue
                m = LABEL_RE.search(l)
                if m and m.group(4) == f"{bar['v']:.2f}" and p["x1"] <= float(m.group(1)) <= p["x2"]:
                    bar["label_line"], bar["label_attr"] = i, m.group(3)
                    used.add(i)
                    break
            else:
                raise SystemExit(f"no value label for {bar['model']} / {p['bench']}")
        p["ticks"] = sorted(
            (float(m.group(3)), float(m.group(2)) - 4, i)
            for i, m in ((i, AXIS_RE.search(l)) for i, l in enumerate(lines))
            if m and abs(float(m.group(1)) - (p["x1"] - 8)) < 0.5 and p["base"] - 200 < float(m.group(2)) < p["base"] + 10
        )
        p["grid"] = [i for i, l in enumerate(lines)
                     for g in [GRID_RE.search(l)] if g and float(g.group(1)) == p["x1"] and float(g.group(2)) == p["x2"]]
        p["floor"] = p["ticks"][0][0]
        p["top_val"], p["top_y"] = p["ticks"][-1][0], p["ticks"][-1][1]
        bar = p["bars"][0]
        p["k"] = (p["base"] - bar["top"]) / (bar["v"] - p["floor"])

    out: list[str | None] = list(lines)
    extra_after: dict[int, list[str]] = {}
    extended = []
    for p in panels:
        v = float(scores[p["bench"]])
        step = p["ticks"][1][0] - p["ticks"][0][0]
        chance = p["floor"]
        if (v - p["floor"]) * p["k"] < 6.0:
            # Extend the axis down in its own step until the bar is visible.
            m = math.ceil((p["floor"] - v) / step + 0.2)
            p["floor"] = p["floor"] - m * step
            p["k"] = (p["base"] - p["top_y"]) / (p["top_val"] - p["floor"])
            for b in p["bars"]:
                b["top"] = p["base"] - (b["v"] - p["floor"]) * p["k"]
            # New gridlines and tick labels replace the old ones in place.
            first_grid, first_tick = p["grid"][0], p["ticks"][0][2]
            tick_attr = re.search(r'font-size="12"([^>]*)>', lines[first_tick]).group(1)
            for i in p["grid"] + [t[2] for t in p["ticks"]]:
                out[i] = None
            new_grid, new_ticks = [], []
            t = p["floor"]
            while t <= p["top_val"] + 1e-9:
                y = p["base"] - (t - p["floor"]) * p["k"]
                if abs(t - p["floor"]) < 1e-9:
                    new_grid.append(f'<line x1="{p["x1"]:.1f}" x2="{p["x2"]:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="#cfcec7" stroke-width="1"/>')
                elif abs(t - chance) < 1e-9:
                    new_grid.append(f'<line x1="{p["x1"]:.1f}" x2="{p["x2"]:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="#a9a8a0" stroke-width="1" stroke-dasharray="4 3"/>')
                else:
                    new_grid.append(f'<line x1="{p["x1"]:.1f}" x2="{p["x2"]:.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="#efeee9" stroke-width="1"/>')
                new_ticks.append(f'<text x="{p["x1"] - 8:.1f}" y="{y + 4:.1f}" font-size="12"{tick_attr}>{t:g}</text>')
                t += step
            out[first_grid] = "\n".join(new_grid)
            out[first_tick] = "\n".join(new_ticks)
            extended.append(f"{p['bench']} from {p['floor']:g} (chance {chance:g} dashed)")
        n = len(p["bars"]) + 1
        span = (n - 1) * PITCH + BAR_W
        left = p["x1"] + (p["x2"] - p["x1"] - span) / 2
        for j, bar in enumerate(p["bars"]):
            x0 = left + j * PITCH
            out[bar["line"]] = (
                f'<path d="{bar_path(x0, p["base"], bar["top"])}" fill="{bar["fill"]}">'
                f'<title>{bar["model"]}, {p["bench"]}: {bar["v"]:.2f}%</title></path>'
            )
            ly = bar["top"] - 7.0
            out[bar["label_line"]] = (
                f'<text x="{x0 + BAR_W / 2:.1f}" y="{ly:.1f}" font-size="10.5"{bar["label_attr"]}>{bar["v"]:.2f}</text>'
            )
        x0 = left + (n - 1) * PITCH
        top = p["base"] - (v - p["floor"]) * p["k"]
        extra_after.setdefault(p["bars"][-1]["line"], []).append(
            f'<path d="{bar_path(x0, p["base"], top)}" fill="{COLOR}"><title>{NAME}, {p["bench"]}: {v:.2f}%</title></path>'
        )
        extra_after.setdefault(p["bars"][-1]["label_line"], []).append(
            f'<text x="{x0 + BAR_W / 2:.1f}" y="{top - 7:.1f}" font-size="10.5" fill="{COLOR}" '
            f'text-anchor="middle" font-weight="600">{v:.2f}</text>'
        )

    # Legend: swatches (12x12 at y=43) and names (y=53.5), inside the pill.
    sw = [(i, float(m.group(1))) for i, l in enumerate(lines)
          for m in [re.search(r'<rect x="([0-9.]+)" y="43.0" width="12" height="12" rx="3"', l)] if m]
    names = [(i, float(m.group(1))) for i, l in enumerate(lines)
             for m in [re.search(r'<text x="([0-9.]+)" y="53.5" font-size="13"', l)] if m]
    pill_i, pill = next((i, m) for i, l in enumerate(lines)
                        for m in [re.search(r'<rect x="([0-9.]+)" y="32" width="([0-9.]+)" height="34"', l)] if m)
    px, pw = float(pill.group(1)), float(pill.group(2))
    pad = sw[0][1] - px
    last_text_end = px + pw - pad
    new_sw = last_text_end + 20.0
    new_pw = (new_sw + 20.0 + LEGEND_TEXT_W + pad) - px
    shift = (1000.0 - new_pw) / 2 - px
    for i, x in sw + names:
        out[i] = re.sub(r'x="[0-9.]+"', f'x="{x + shift:.1f}"', out[i], count=1)
    out[pill_i] = re.sub(r'x="[0-9.]+" y="32" width="[0-9.]+"',
                         f'x="{px + shift:.1f}" y="32" width="{new_pw:.1f}"', out[pill_i], count=1)
    extra_after.setdefault(names[-1][0], []).extend([
        f'<rect x="{new_sw + shift:.1f}" y="43.0" width="12" height="12" rx="3" fill="{COLOR}"/>',
        f'<text x="{new_sw + 20 + shift:.1f}" y="53.5" font-size="13" fill="{COLOR}" text-anchor="start" '
        f'font-weight="600">{NAME}</text>',
    ])

    axes = (
        "Axes start at chance; an axis extended below chance for Embryo-56M shows chance dashed."
        if extended else "Axes start at chance."
    )
    note = (
        f"{NAME}: backbone of the q8_2f release file, zero-shot acc_norm (lm-eval 0.4.12; "
        "ArithMark-3 by the official script's rule), measured with cortiq loglik."
    )
    result = []
    for i, l in enumerate(out):
        if l is not None and l.strip() == "</svg>":
            for y, text in ((590.0, note), (605.0, axes)):
                result.append(
                    f'<text x="500.0" y="{y:.1f}" font-size="10.5" fill="#898781" text-anchor="middle" '
                    f'font-weight="400">{text}</text>'
                )
        if l is not None:
            result.append(l)
        result.extend(extra_after.get(i, []))
    Path(args.out).write_text("\n".join(result))
    for p in panels:
        print(f"{p['bench']}: axis from {p['floor']:g}, {p['k']:.3f} px/pt, {NAME} {scores[p['bench']]:.2f}")
    for e in extended:
        print(f"extended: {e}")


if __name__ == "__main__":
    main()
