#!/usr/bin/env python3
"""Per-window o1/exact ppl ratios from `cortiq ppl ... --o1 ...` logs run with
CMF_PPL_PROGRESS=1 (cumulative `ppl-window:` lines).

Usage: ppl_windows_ratio.py LOG [LOG ...]  — one row per log: the ratio of
every window, of windows 1-4, and of all windows.
"""
import math
import re
import sys

LINE = re.compile(r"ppl-window: index=(\d+) offset=\d+ scored=(\d+) .*exact_nll=([\d.]+) o1_nll=([\d.]+)")


def main():
    for path in sys.argv[1:]:
        rows = [tuple(map(float, m.groups()[1:])) for m in map(LINE.search, open(path)) if m]
        prev = (0.0, 0.0, 0.0)
        per = []
        for n, ex, o1 in rows:
            dn, dex, do1 = n - prev[0], ex - prev[1], o1 - prev[2]
            per.append(math.exp((do1 - dex) / dn))
            prev = (n, ex, o1)
        cum = lambda k: math.exp((rows[k - 1][2] - rows[k - 1][1]) / rows[k - 1][0]) if len(rows) >= k else float("nan")
        ex_ppl = math.exp(rows[-1][1] / rows[-1][0]) if rows else float("nan")
        o1_ppl = math.exp(rows[-1][2] / rows[-1][0]) if rows else float("nan")
        print(
            f"{path}: windows {' '.join(f'x{r:.3f}' for r in per)} | 1-4 x{cum(4):.3f}"
            f" | all({len(rows)}) x{cum(len(rows)):.3f} | ppl o1 {o1_ppl:.3f} exact {ex_ppl:.3f}"
        )


if __name__ == "__main__":
    main()
