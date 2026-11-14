"""Run a CMF model on the R set through `cortiq probe-utility`.

    python -I run_cortiq.py --cortiq <bin> --model <file.cmf> --tasks r.jsonl --out <name>.outputs.jsonl

The task text is the user turn of the cmf-im-v1 chat contract, no worked
examples (zero-shot: a model fine-tuned on this format needs none). Greedy
decoding on the backbone (`--route backbone`, CMF_GROWTH=off, CMF_GPU=0):
records and their router take no part. The answer is the first line of the
reply, as for the other models; grade.py grades it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_model import extract  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[:: max(1, len(tasks) // args.limit)][: args.limit]
    with tempfile.TemporaryDirectory() as d:
        pj = Path(d) / "prompts.jsonl"
        pj.write_text("".join(json.dumps({"lang": "en", "prompt": t["prompt"], "expect": [t["ref"]], "src": t["id"]},
                                         ensure_ascii=False) + "\n" for t in tasks))
        env = dict(os.environ, CMF_GPU="0", CMF_GROWTH="off")
        p = subprocess.run([args.cortiq, "probe-utility", args.model, "--prompts-jsonl", str(pj),
                            "--max-tokens", str(args.max_tokens), "--route", "backbone", "--json"],
                           env=env, capture_output=True, text=True, check=True)
    rep = json.loads(p.stdout)
    rep = rep[0] if isinstance(rep, list) else rep
    rows = rep["per_prompt"]
    assert len(rows) == len(tasks), (len(rows), len(tasks))
    with open(args.out, "w") as f:
        for t, r in zip(tasks, rows):
            assert r["label"].endswith(t["id"]), (r["label"], t["id"])
            ans, _ = extract(r["answer"], "plain")
            f.write(json.dumps({"id": t["id"], "answer": ans, "raw": r["answer"], "prompt_tokens": r["prompt_tokens"],
                                "finish": r.get("finish_reason")}, ensure_ascii=False) + "\n")
    toks = sum(r["generated"] for r in rows)
    print(f"{len(tasks)} tasks, {toks} new tokens, mean {sum(r['tok_s'] for r in rows) / len(rows):.0f} tok/s")


if __name__ == "__main__":
    main()
