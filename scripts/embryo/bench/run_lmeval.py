"""Zero-shot lm-eval 0.4.12 tasks on a CMF model through `cortiq loglik`.

    HF_HOME=<dir> python -I run_lmeval.py --cortiq <bin> --model <file.cmf> \
        --out-dir <dir> --tasks arc_easy,arc_challenge,hellaswag,piqa [--limit N]

One `simple_evaluate` per task (num_fewshot=0, no chat template), one
`cortiq loglik` call per task. Writes `<task>.results.json` (lm-eval's
results / configs / versions blocks) and `<task>.samples.jsonl` (per-doc
log-likelihoods, for analysis) into --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lm_eval  # noqa: E402
from lm_eval.utils import handle_non_serializable  # noqa: E402

from cortiq_lm import CortiqLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,piqa")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frame", default="raw", choices=("raw", "chat"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = {"CMF_GROWTH": "off", "CMF_GPU": "0"}
    for task in args.tasks.split(","):
        tag = f"{task}.{args.frame}" + (f".limit{args.limit}" if args.limit else "")
        lm = CortiqLM(
            cortiq=args.cortiq,
            model=args.model,
            workdir=str(out / "work"),
            tag=tag,
            frame=args.frame,
            env=env,
        )
        res = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task],
            num_fewshot=0,
            limit=args.limit,
            log_samples=True,
            batch_size=1,
            apply_chat_template=False,
        )
        samples = res.pop("samples", {}).get(task, [])
        keep = {k: res.get(k) for k in ("results", "configs", "versions", "n-shot", "n-samples", "lm_eval_version", "git_hash", "date")}
        (out / f"{tag}.results.json").write_text(
            json.dumps(keep, indent=2, default=handle_non_serializable, ensure_ascii=False)
        )
        with open(out / f"{tag}.samples.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps({
                    "doc_id": s.get("doc_id"),
                    "target": s.get("target"),
                    "acc": s.get("acc"),
                    "acc_norm": s.get("acc_norm"),
                    "filtered_resps": s.get("filtered_resps"),
                }, default=handle_non_serializable) + "\n")
        r = res["results"][task]
        print(
            f"{task}: acc {r.get('acc,none'):.4f} ± {r.get('acc_stderr,none'):.4f} | "
            f"acc_norm {r.get('acc_norm,none'):.4f} ± {r.get('acc_norm_stderr,none'):.4f} | "
            f"n {len(samples)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
