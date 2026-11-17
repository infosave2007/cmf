"""Inputs of `cortiq-embryo skill-bake --sft-train …` for the two skills on Embryo-56M.

    python -I build_sft.py --tasks cards.messages.jsonl --rules RULES.messages.jsonl --general GENERAL.messages.jsonl --out DIR

--tasks, --rules: as for build_corpus.py. --general: conversations whose user
turns are general questions (the backbone class of the router).

Writes to --out:
  s1.messages.jsonl   a command from a card (slot, direct, choose, os, fix, none, two), group = the tldr page
  s2.messages.jsonl   the next action by rules, the answer after the rule copied word for word;
                      group = rules.<i mod 200>: the tasks are independent draws, the groups only give
                      `sft-prepare` its train / dev / final split
  s1.prompts.jsonl, s2.prompts.jsonl, general.prompts.jsonl   {"prompt"}: the user turns the router is fitted on
                      (4000 of each skill, every general one)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

CARD = ("slot", "direct", "choose", "os", "fix", "none", "two")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--general", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3)
    n = Counter()
    p1, p2 = [], []
    with open(out / "s1.messages.jsonl", "w") as f:
        for line in open(args.tasks):
            r = json.loads(line)
            if r["cat"] in CARD:
                f.write(json.dumps({"messages": r["messages"], "group": r["group"]}, ensure_ascii=False) + "\n")
                p1.append(r["messages"][0]["content"])
                n["s1"] += 1
    k = 0
    with open(out / "s2.messages.jsonl", "w") as f:
        for line in open(args.rules):
            r = json.loads(line)
            if r["cat"] == "react":
                f.write(json.dumps({"messages": r["messages"], "group": f"rules.{k % 200}"}, ensure_ascii=False) + "\n")
                p2.append(r["messages"][0]["content"])
                k += 1
                n["s2"] += 1
    gen = []
    for line in open(args.general):
        r = json.loads(line)
        gen += [m["content"] for m in r["messages"] if m["role"] == "user"][:1]
    for name, ps in (("s1", rng.sample(p1, 4000)), ("s2", rng.sample(p2, 4000)), ("general", gen)):
        (out / f"{name}.prompts.jsonl").write_text("".join(json.dumps({"prompt": p}, ensure_ascii=False) + "\n" for p in ps))
        n[name + " prompts"] = len(ps)
    print(dict(n))


if __name__ == "__main__":
    main()
