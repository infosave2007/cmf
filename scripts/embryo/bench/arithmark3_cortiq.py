"""ArithMark-3.0 on a CMF model, by the rules of the official script.

    python -I arithmark3_cortiq.py --cortiq <bin> --model <file.cmf> \
        --script <dir>/bencharithmark-3.py --data <dir>/arithmark-3.jsonl --out-dir <dir>

The official `bencharithmark-3.py` loads Hugging Face transformers models
only, so its model call cannot run on a CMF file. This driver imports the
script itself (torch / transformers / huggingface_hub / tqdm stubbed, they
are only touched by the model path) and uses its `load_arithmark_3` for
the data and validation; it then reproduces the script's scoring exactly:

* context and continuation tokenized independently, no special tokens
  (`tokenize_request`; `cortiq loglik --split separate`, no BOS);
* raw score = sum of continuation token log-likelihoods (`acc`);
  normalized = raw / number of continuation tokens (`acc_norm`, the
  script's primary metric);
* prediction = `max(range(n), key=scores.__getitem__)` (first maximum);
* per-category breakdown, dataset SHA-256, the script's result layout.

The log-likelihoods come from `cortiq loglik` on the backbone
(CMF_GROWTH=off, CMF_GPU=0).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path


def import_official(script: Path):
    for name in ("torch", "torch.nn", "torch.nn.functional", "huggingface_hub", "tqdm", "transformers"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    sys.modules["huggingface_hub"].hf_hub_download = None
    sys.modules["tqdm"].tqdm = lambda it, **_: it
    sys.modules["transformers"].AutoModelForCausalLM = None
    sys.modules["transformers"].AutoTokenizer = None
    spec = importlib.util.spec_from_file_location("bencharithmark3_official", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    official = import_official(Path(args.script))
    data_path = Path(args.data).resolve()
    examples = official.load_arithmark_3(data_path)
    if args.limit:
        examples = examples[: args.limit]
    sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    max_context = official.MAX_CONTEXT

    out = Path(args.out_dir)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    tag = "arithmark-3" + (f".limit{args.limit}" if args.limit else "")
    req = work / f"{tag}.requests.jsonl"
    with open(req, "w") as f:
        for ei, ex in enumerate(examples):
            for j, ending in enumerate(ex["endings"]):
                f.write(json.dumps({"context": ex["ctx"], "continuation": ending, "e": ei, "j": j}, ensure_ascii=False) + "\n")
    res_path = work / f"{tag}.loglik.jsonl"
    env = dict(os.environ, CMF_GROWTH="off", CMF_GPU="0")
    with open(work / f"{tag}.loglik.log", "w") as log:
        subprocess.run(
            [args.cortiq, "loglik", args.model, "--requests-jsonl", str(req), "--out", str(res_path), "--split", "separate"],
            env=env, check=True, stderr=log,
        )
    rows = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()]
    raw = defaultdict(dict)
    norm = defaultdict(dict)
    for r in rows:
        if r["n_ctx"] + r["n_cont"] > max_context:
            raise SystemExit(f"example {r['e']}: {r['n_ctx'] + r['n_cont']} tokens > MAX_CONTEXT {max_context}")
        raw[r["e"]][r["j"]] = r["logprob"]
        norm[r["e"]][r["j"]] = r["logprob"] / max(r["n_cont"], 1)

    raw_correct = norm_correct = total = 0
    grouped = defaultdict(lambda: {"raw_correct": 0, "norm_correct": 0, "total": 0})
    predictions = []
    for ei, ex in enumerate(examples):
        n = len(ex["endings"])
        rs = [raw[ei][j] for j in range(n)]
        ns = [norm[ei][j] for j in range(n)]
        rp = max(range(n), key=rs.__getitem__)
        npred = max(range(n), key=ns.__getitem__)
        rh, nh = int(rp == ex["label"]), int(npred == ex["label"])
        raw_correct += rh
        norm_correct += nh
        total += 1
        g = grouped[ex["category"]]
        g["raw_correct"] += rh
        g["norm_correct"] += nh
        g["total"] += 1
        predictions.append({
            "id": ex["id"], "category": ex["category"], "label": ex["label"],
            "raw_prediction": rp, "normalized_prediction": npred,
            "raw_scores": rs, "normalized_scores": ns,
        })
    result = {
        "acc": raw_correct / total * 100,
        "acc_norm": norm_correct / total * 100,
        "raw_correct": raw_correct,
        "norm_correct": norm_correct,
        "total": total,
        "categories": {
            c: {
                "acc": v["raw_correct"] / v["total"] * 100,
                "acc_norm": v["norm_correct"] / v["total"] * 100,
                **v,
            }
            for c, v in sorted(grouped.items())
        },
        "primary_metric": "acc_norm",
        "predictions": predictions,
    }
    result["primary_acc"] = result["acc_norm"]
    (out / f"{tag}.results.json").write_text(json.dumps({
        "model": args.model,
        "scorer": "cortiq loglik --split separate (backbone, CMF_GROWTH=off, CMF_GPU=0)",
        "official_script_sha256": hashlib.sha256(Path(args.script).read_bytes()).hexdigest(),
        "dataset": {"path": str(data_path), "sha256": sha, "items": total},
        "results": {"arithmark-3": result},
    }, indent=2))
    print(
        f"arithmark-3: raw {result['acc']:.2f}% ({raw_correct}/{total})  "
        f"normalized {result['acc_norm']:.2f}% ({norm_correct}/{total})  dataset sha256 {sha}"
    )


if __name__ == "__main__":
    main()
