"""Cross-check the lm-eval adapter's log-likelihoods against cortiq directly.

    python -I verify_ll.py --cortiq <bin> --model <file.cmf> \
        --requests <work>/<tag>.requests.jsonl --adapter <work>/<tag>.loglik.jsonl --out-dir <dir>

For the requests the adapter sent (an lm-eval `--limit` run), three
independent paths must agree:
  A  the adapter's numbers (`cortiq loglik --scorer shared`: context once,
     continuations walked as a token trie, one decode step per node);
  B  `cortiq loglik --scorer full`: a fresh forward of context+continuation
     per request through the engine's `nll_ids_from` (batched prefill);
  C  `cortiq probe-choice --raw --route backbone`: the existing forced-choice
     probe, one prompt per (doc, candidate) row, candidates = the doc's
     choices (it prepends the space itself).
Prints the max |A-B| and |A-C| and the per-request table.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CMF_GROWTH="off", CMF_GPU="0")

    reqs = [json.loads(l) for l in Path(args.requests).read_text().splitlines() if l.strip()]
    a = {r["k"]: r for r in (json.loads(l) for l in Path(args.adapter).read_text().splitlines() if l.strip())}

    full = out / "full.loglik.jsonl"
    with open(out / "full.loglik.log", "w") as log:
        subprocess.run([args.cortiq, "loglik", args.model, "--requests-jsonl", args.requests,
                        "--out", str(full), "--scorer", "full"], env=env, check=True, stderr=log)
    b = {r["k"]: r for r in (json.loads(l) for l in full.read_text().splitlines() if l.strip())}

    # C: probe-choice, one run per context; rows = one per candidate.
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in reqs:
        groups.setdefault(r["context"], []).append(r)
    c: dict[int, float] = {}
    for gi, (ctx, rs) in enumerate(groups.items()):
        texts = [r["continuation"] for r in rs]
        if not all(t.startswith(" ") for t in texts) or len({t.strip().lower() for t in texts}) != len(texts):
            continue  # probe-choice adds the space itself and dedupes by lower-case key
        p = out / f"pc{gi}.prompts.jsonl"
        q = out / f"pc{gi}.cands.jsonl"
        p.write_text("".join(json.dumps({"prompt": ctx, "expect": [t[1:]], "src": str(r["k"])}) + "\n" for t, r in zip(texts, rs)))
        q.write_text("".join(json.dumps({"text": t[1:]}) + "\n" for t in texts))
        res = subprocess.run([args.cortiq, "probe-choice", args.model, "--prompts-jsonl", str(p),
                              "--candidates-jsonl", str(q), "--raw", "--route", "backbone", "--json"],
                             env=env, check=True, capture_output=True, text=True)
        js = json.loads(res.stdout)
        for row in js["per_row"]:
            c[int(row["src"])] = row["lp_correct"]

    lines = ["| k | n_cont | A adapter | B full | C probe-choice | A greedy |", "|---|---|---|---|---|---|"]
    dab = dac = 0.0
    for r in reqs:
        k = r["k"]
        av, bv = a[k]["logprob"], b[k]["logprob"]
        dab = max(dab, abs(av - bv))
        cv = c.get(k)
        if cv is not None:
            dac = max(dac, abs(av - cv))
        lines.append(f"| {k} | {a[k]['n_cont']} | {av:.6f} | {bv:.6f} | {'-' if cv is None else f'{cv:.6f}'} | {a[k]['is_greedy']} |")
    summary = f"{len(reqs)} requests ({len(groups)} contexts): max |A-B| = {dab:.2e}, max |A-C| = {dac:.2e} over {len(c)} rows"
    (out / "verify.md").write_text(summary + "\n\n" + "\n".join(lines) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
