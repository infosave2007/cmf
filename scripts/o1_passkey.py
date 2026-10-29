#!/usr/bin/env python3
"""Passkey recall through the streaming O(1) kernel, next to exact attention.

One trial: a five-digit pass key stated at the start, --distance tokens of
filler, then "The pass key is" and the key again. `cortiq ppl --o1 ...
--o1-prefill P` seals after P tokens (the key sits inside them), streams the
rest through the O(1) operator, and scores the same tokens exactly as well;
CMF_PPL_TRACE gives the greedy top-1 per scored position, so the last five
positions of each pass say whether the model reproduces the key.

`cortiq probe-recall` cannot do this: it feeds the whole prompt through
`forward_ids`, which seals at the END of the prompt, so the query token is
attended exactly.

Usage:
  o1_passkey.py --cortiq BIN --model M.cmf --filler test.txt \
      --distances 512,1536 --trials 8 [--o1 all] [--prefill 256] [-- extra ppl args]
Environment (CMF_GPU=0, CMF_O1_CALIB, ...) passes through to cortiq.
"""
import argparse
import os
import random
import re
import subprocess
import sys
import tempfile

HEAD = (
    "There is an important pass key hidden inside a long text. Find it and "
    "memorize it. The pass key is {key}. Remember it. {key} is the pass key.\n\n"
)
TAIL = "\n\nWhat is the pass key? The pass key is {key}"
# Serial scorer: "pos P tgt T nll X | top1 Y"; batched exact scorer:
# "BTRACE pos P target T nll X top Y".
TRACE = re.compile(r"^(?:BTRACE )?pos\s+(\d+)\s+(?:tgt|target)\s+(\d+)\s+nll\s+([-\d.]+)\s+(?:\| )?top1?\s+(\d+)")
CHARS_PER_TOKEN = 3.6  # code-heavy filler; only sizes the slice


def passes(stderr):
    """Split CMF_PPL_TRACE lines into passes (positions restart)."""
    out, cur, last = [], [], -1
    for line in stderr.splitlines():
        m = TRACE.match(line.strip())
        if not m:
            continue
        pos, tgt, nll, top = int(m[1]), int(m[2]), float(m[3]), int(m[4])
        if pos <= last:
            out.append(cur)
            cur = []
        cur.append((pos, tgt, nll, top))
        last = pos
    if cur:
        out.append(cur)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--distances", default="512,1536")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--o1", default="all")
    ap.add_argument("--prefill", type=int, default=256)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    filler = open(a.filler).read()
    rng = random.Random(a.seed)
    env = dict(os.environ, CMF_PPL_TRACE="1")
    print(f"{'D':>6} | {'o1 key':>6} {'o1 dig':>6} {'o1 nll':>7} | {'ex key':>6} {'ex dig':>6} {'ex nll':>7}")
    for dist in [int(x) for x in a.distances.split(",")]:
        res = {"o1": [0, 0, 0.0], "ex": [0, 0, 0.0]}
        for _ in range(a.trials):
            key = str(rng.randrange(10000, 100000))
            start = rng.randrange(0, max(1, len(filler) - int(dist * CHARS_PER_TOKEN) - 1))
            body = filler[start : start + int(dist * CHARS_PER_TOKEN)]
            text = HEAD.format(key=key) + body + TAIL.format(key=key)
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
                f.write(text)
            cmd = [a.cortiq, "ppl", a.model, "-f", f.name, "--tokens", "100000",
                   "--o1", a.o1, "--o1-prefill", str(a.prefill)] + a.extra
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            os.unlink(f.name)
            ps = passes(r.stderr)
            if len(ps) != 2:
                sys.exit(f"expected an o1 and an exact pass, got {len(ps)}:\n{r.stderr[-2000:]}")
            for name, rows in zip(("o1", "ex"), ps):
                digits = rows[-5:]
                ok = [top == tgt for _, tgt, _, top in digits]
                res[name][0] += all(ok)
                res[name][1] += sum(ok)
                res[name][2] += sum(n for _, _, n, _ in digits) / 5
        t = a.trials
        print(
            f"{dist:>6} | {res['o1'][0] / t:6.2f} {res['o1'][1] / (5 * t):6.2f} {res['o1'][2] / t:7.3f}"
            f" | {res['ex'][0] / t:6.2f} {res['ex'][1] / (5 * t):6.2f} {res['ex'][2] / t:7.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
