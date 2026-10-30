#!/usr/bin/env python3
"""Rank KV groups by how much attention they put on a far pass key.

One exact pass per trial instead of one passkey run per narrowed group:
`cortiq ppl --o1 all --o1-prefill 100000` seals at the END of the prompt
(the prefill clamps to len-1), and CMF_O1_CAPTURE dumps every group's
post-RoPE Q/K/V at that seal. The queries that predict the five key digits
are inside the capture, so the softmax each Q head puts on the earlier
copies of the key, and on everything outside a narrow window, is read off
directly. A group whose heads look the key up far away is a copying group:
it loses recall the moment the key leaves its exact window, so it is the
one to widen (CMF_O1_WIDE).

Usage:
  o1_retrieval_score.py --cortiq BIN --model M.cmf --filler test.txt \
      --distances 1024,1800 --trials 4 [--window 512] [-- extra ppl args]
Prints one row per (layer, group), ranked by key mass, and writes
--out (JSON) for the wide-set builder.
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o1_passkey import CHARS_PER_TOKEN, HEAD, TAIL  # noqa: E402

NAME = re.compile(r"L(\d+)_g(\d+)_\d+\.f32$")


def read_capture(path):
    raw = np.fromfile(path, dtype=np.uint32, count=3)
    t, d, h = (int(x) for x in raw)
    a = np.fromfile(path, dtype=np.float32, offset=12)
    q = a[: h * t * d].reshape(h, t, d)
    k = a[h * t * d : (h + 1) * t * d].reshape(t, d)
    return q, k


def find_all(ids, pat, end):
    n = len(pat)
    return [i for i in range(end - n + 1) if ids[i : i + n] == pat]


def score_trial(cap_dir, ids, window, sink):
    """Per (layer, group): (key mass, far mass), max over Q heads, mean
    over the five digit queries."""
    L = len(ids)
    digits = ids[L - 5 :]
    starts = find_all(ids, digits, L - 5)
    if not starts:
        return None
    key_pos = sorted({s + j for s in starts for j in range(5)})
    queries = list(range(L - 6, L - 1))  # each predicts the next digit
    out = {}
    for path in glob.glob(os.path.join(cap_dir, "*.f32")):
        m = NAME.search(path)
        layer, group = int(m[1]), int(m[2])
        q, k = read_capture(path)
        t, d = k.shape
        key_mass, far_mass = [], []
        for qp in queries:
            if qp >= t:
                continue
            s = q[:, qp, :] @ k[: qp + 1].T / np.sqrt(d)
            s -= s.max(axis=1, keepdims=True)
            p = np.exp(s)
            p /= p.sum(axis=1, keepdims=True)
            kp = [x for x in key_pos if x <= qp]
            key_mass.append(p[:, kp].sum(axis=1))
            lo, hi = sink, max(sink, qp + 1 - window)
            far_mass.append(p[:, lo:hi].sum(axis=1))
        km = np.mean(key_mass, axis=0)
        fm = np.mean(far_mass, axis=0)
        out[(layer, group)] = (float(km.max()), float(fm.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--filler", required=True)
    ap.add_argument("--distances", default="1024,1800")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="retrieval_score.json")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    filler = open(a.filler).read()
    rng = random.Random(a.seed)
    acc = {}
    n = 0
    for dist in [int(x) for x in a.distances.split(",")]:
        for _ in range(a.trials):
            key = str(rng.randrange(10000, 100000))
            span = int(dist * CHARS_PER_TOKEN)
            start = rng.randrange(0, max(1, len(filler) - span - 1))
            text = HEAD.format(key=key) + filler[start : start + span] + TAIL.format(key=key)
            work = tempfile.mkdtemp(prefix="o1rs_")
            cap = os.path.join(work, "cap")
            os.mkdir(cap)
            txt, idsf = os.path.join(work, "p.txt"), os.path.join(work, "ids.txt")
            with open(txt, "w") as f:
                f.write(text)
            env = dict(os.environ, CMF_O1_CAPTURE=cap, CMF_PPL_IDS_OUT=idsf)
            cmd = [a.cortiq, "ppl", a.model, "-f", txt, "--tokens", "100000",
                   "--o1", "all", "--o1-prefill", "100000"] + a.extra
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"cortiq failed:\n{r.stderr[-2000:]}")
            ids = [int(x) for x in open(idsf).read().split()]
            res = score_trial(cap, ids, a.window, a.sink)
            shutil.rmtree(work)
            if res is None:
                print(f"D={dist}: key digits not found earlier in the prompt, trial skipped", flush=True)
                continue
            n += 1
            for g, (km, fm) in res.items():
                s = acc.setdefault(g, [0.0, 0.0])
                s[0] += km
                s[1] += fm
            print(f"D={dist} trial {n}: {len(ids)} tokens, {len(res)} groups", flush=True)
    rows = sorted(((g, s[0] / n, s[1] / n) for g, s in acc.items()), key=lambda r: -r[1])
    print(f"{'layer:group':>11} {'key mass':>9} {'far mass':>9}")
    for (layer, group), km, fm in rows:
        print(f"{layer:>8}:{group:<2} {km:9.4f} {fm:9.4f}")
    with open(a.out, "w") as f:
        json.dump([{"layer": l, "group": g, "key_mass": km, "far_mass": fm}
                   for (l, g), km, fm in rows], f, indent=1)


if __name__ == "__main__":
    main()
