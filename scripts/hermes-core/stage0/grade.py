"""Grade model answers on the R set.

    python -I grade.py --tasks r.jsonl --outputs <name>.outputs.jsonl --out <name>.graded.jsonl
    python -I grade.py --tasks r.jsonl --baseline ref|copy_first --out <name>.graded.jsonl

Exec-checked answers run in the sandbox (sandbox.py). `--baseline ref` grades
the reference answers (must be 100 %: the grader's self-test), `copy_first`
the first example of the first card with nothing filled in (the floor).
Prints accuracy per category and overall with a 95 % Wilson interval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sandbox as S  # noqa: E402

CATS = ["slot", "os", "choose", "none", "react", "fix", "two", "arith", "direct"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--outputs")
    ap.add_argument("--baseline", choices=("ref", "copy_first"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = {t["id"]: t for t in (json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip())}
    if args.baseline:
        answers = {i: {"id": i, "answer": t["ref"] if args.baseline == "ref" else (t["copy_first"] or "")} for i, t in tasks.items()}
    else:
        answers = {o["id"]: o for o in (json.loads(l) for l in Path(args.outputs).read_text().splitlines() if l.strip())}

    try:  # a dead Docker must stop the run, not grade every answer as wrong
        probe = S.run("echo sandbox-ok")
    except RuntimeError as e:
        raise SystemExit(f"{e}\nis Docker up and the image built?")
    if probe["exit"] != 0 or "sandbox-ok" not in probe["stdout"]:
        raise SystemExit(f"the sandbox does not run commands (exit {probe['exit']}): is Docker up and the image built?")

    def one(i):
        ok, detail = S.grade(tasks[i], answers[i]["answer"])
        return {**answers[i], "cat": tasks[i]["cat"], "family": tasks[i]["family"], "ref": tasks[i]["ref"],
                "check": tasks[i]["check"]["type"], "ok": ok, **detail}

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, list(answers)))
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tot_k = tot_n = 0
    for c in CATS:
        rs = [r for r in rows if r["cat"] == c]
        if not rs:
            continue
        k, n = sum(r["ok"] for r in rs), len(rs)
        tot_k, tot_n = tot_k + k, tot_n + n
        lo, hi = wilson(k, n)
        print(f"{c:<7} {k:>3}/{n:<3} {100 * k / n:5.1f} %  [{100 * lo:.0f}–{100 * hi:.0f}]")
    lo, hi = wilson(tot_k, tot_n)
    print(f"{'all':<7} {tot_k:>3}/{tot_n:<3} {100 * tot_k / tot_n:5.1f} %  [{100 * lo:.0f}–{100 * hi:.0f}]")


if __name__ == "__main__":
    main()
