"""Two skill corpora for the growth experiment on Embryo-56M.

    python -I build_corpus.py --tasks cards.messages.jsonl --rules RULES.messages.jsonl --out DIR

--tasks: the card tasks of `stage0/train_gen.py` (default options), the set the
trunk fine-tune of stage 0 was run on. --rules: the output of
`stage0/lookup_gen.py --reason react --labels --or-rules 0.25` over the same
tasks (only its rule conversations are taken: the answer after the rule
copied word for word).

Writes to --out, one {"text"} per line, the conversation in the cmf-im-v1
frame as the engine renders a chat:
  skill1.jsonl.gz, skill1.held.jsonl.gz   a command from a card (slot, direct, choose, os, fix, none, two);
                                          held out: one tldr page in 20
  skill2.jsonl.gz, skill2.held.jsonl.gz   the next action by rules; held out: every 20th task
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))

from convert import render  # noqa: E402

CARD = ("slot", "direct", "choose", "os", "fix", "none", "two")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = Counter()
    with gzip.open(out / "skill1.jsonl.gz", "wt", encoding="utf-8") as ft, gzip.open(out / "skill1.held.jsonl.gz", "wt", encoding="utf-8") as fh:
        for line in open(args.tasks):
            r = json.loads(line)
            if r["cat"] not in CARD:
                continue
            text = render(None, None, [("user", r["messages"][0]["content"]), ("assistant", None, r["messages"][1]["content"], [])])
            held = int(hashlib.md5(r["group"].encode()).hexdigest()[:8], 16) % 20 == 0
            (fh if held else ft).write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n["skill1 held" if held else "skill1"] += 1
            n["skill1 chars"] += len(text)
    k = 0
    with gzip.open(out / "skill2.jsonl.gz", "wt", encoding="utf-8") as ft, gzip.open(out / "skill2.held.jsonl.gz", "wt", encoding="utf-8") as fh:
        for line in open(args.rules):
            r = json.loads(line)
            if r["cat"] != "react":
                continue
            text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in r["messages"])
            held = k % 20 == 0
            k += 1
            (fh if held else ft).write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n["skill2 held" if held else "skill2"] += 1
            n["skill2 chars"] += len(text)
    print(dict(n))


if __name__ == "__main__":
    main()
