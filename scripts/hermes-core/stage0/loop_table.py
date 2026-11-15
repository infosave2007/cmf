"""Table of set R runs in the lookup loop: what passed and how the loop went.

    python -I loop_table.py GRADED.jsonl [GRADED.jsonl ...]

One column per graded file (run_lookup.py outputs after grade.py): passed tasks
in all and per category; over the tasks whose answer is a command from a card
(all but none, react, arith), how many answers start with the reference's
command; how the conversations ended; over the tasks that have cards (the same
plus none), how many called lookup and how many got the task's card back; tasks
with more than one lookup and how many of those got the card only on a later
try.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

CATS = ["slot", "os", "choose", "none", "fix", "direct", "two", "react", "arith"]


def first(s: str) -> str:
    return (s or "").split()[0] if (s or "").split() else ""


def column(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    n = collections.Counter(r["cat"] for r in rows)
    cmd = [r for r in rows if r["cat"] not in ("none", "react", "arith")]
    card = [r for r in rows if r["cat"] not in ("react", "arith")]
    again = [r for r in rows if len(r.get("lookups") or []) > 1]
    col = {"all": f"{sum(r['ok'] for r in rows)}/{len(rows)}"}
    col.update({c: f"{sum(r['ok'] for r in rows if r['cat'] == c)}/{n[c]}" for c in CATS})
    col["right command name"] = f"{sum(first(r['answer']) == first(r['ref']) for r in cmd)}/{len(cmd)}"
    for k, v in sorted(collections.Counter(r["finish"] for r in rows).items()):
        col[f"ended: {k}"] = str(v)
    col["called lookup"] = f"{sum(bool(r.get('lookups')) for r in card)}/{len(card)}"
    col["got the card back"] = f"{sum(any(r.get('hits') or []) for r in card)}/{len(card)}"
    col["more than one lookup"] = str(len(again))
    col["  the card only on a later try"] = str(sum(not r["hits"][0] and any(r["hits"][1:]) for r in again))
    return col


def main() -> None:
    paths = sys.argv[1:]
    cols = [column(p) for p in paths]
    keys = list(dict.fromkeys(k for c in cols for k in c))
    rest = [k for k in keys if k.startswith("ended: ")]
    keys = [k for k in keys if k not in rest]
    keys[keys.index("called lookup"):keys.index("called lookup")] = sorted(rest)
    names = [Path(p).name.removesuffix(".graded.jsonl") for p in paths]
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(c.get(k, "0") for c in cols) + " |")


if __name__ == "__main__":
    main()
