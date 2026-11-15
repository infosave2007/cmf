"""Which of the task's cards the answer's command comes from, over the `choose` tasks of set R.

    python -I card_choice.py --tasks r.jsonl GRADED.jsonl [GRADED.jsonl ...]

A `choose` task shows several cards and only one does the task. A model
that picks by meaning takes its command from that card; one that does not
takes it from a card by its place in the list. Per graded file (the output
of grade.py): how many answers start with the command of the right card,
and where the card they do start with sits among the task's cards (first,
middle, last, or none of them). Chance is one in the number of cards.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

HEAD = re.compile(r"^(?:\[(?:Linux|macOS|Windows)\] — )?(.+?): ")


def card_names(prompt: str) -> list[str]:
    if not prompt.startswith("Reference card") or "\n\nTask: " not in prompt:
        return []
    blocks = prompt.split("\n\nTask: ", 1)[0].split("\n", 1)[1].split("\n\n")
    return [m.group(1) for m in (HEAD.match(b.split("\n", 1)[0]) for b in blocks) if m]


def first(s: str) -> str:
    return (s or "").split()[0] if (s or "").split() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("graded", nargs="+")
    args = ap.parse_args()
    tasks = {}
    for line in Path(args.tasks).read_text().splitlines():
        t = json.loads(line)
        names = card_names(t["prompt"])
        if t["cat"] == "choose" and len(set(names)) > 1:
            tasks[t["id"]] = (names, first(t["ref"]))
    where = collections.Counter(("last" if n.index(r) == len(n) - 1 else "first" if n.index(r) == 0 else "middle")
                                for n, r in tasks.values() if r in n)
    print(f"{len(tasks)} choose tasks; the right card sits: " + ", ".join(f"{k} {v}" for k, v in sorted(where.items())))
    for path in args.graded:
        took, right = collections.Counter(), 0
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r["id"] not in tasks:
                continue
            names, ref = tasks[r["id"]]
            cmd = first(r["answer"])
            right += cmd == ref
            i = names.index(cmd) if cmd in names else -1
            took["none of them" if i < 0 else "last" if i == len(names) - 1 else "first" if i == 0 else "middle"] += 1
        print(f"{Path(path).name}: command of the right card {right}/{len(tasks)}; taken from: "
              + ", ".join(f"{k} {took[k]}" for k in ("first", "middle", "last", "none of them")))


if __name__ == "__main__":
    main()
