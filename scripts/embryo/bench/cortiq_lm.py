"""lm-evaluation-harness model backed by `cortiq loglik`.

Only the `loglikelihood` request type is served (all four tasks here are
`multiple_choice`). Every call writes the requests to a JSONL file in the
run directory, sorted so equal contexts are adjacent, runs `cortiq loglik`
once (`--scorer shared`: each context forwarded once, its continuations
walked as a token trie with KV rollback), and maps the answers back to
request order. A result
file whose request file hash matches is reused, so re-aggregation costs
nothing.

Tokenization is the engine's: `cortiq loglik --split hflm` reproduces
lm-eval's `HFLM._encode_pair` (trailing context whitespace moves into the
continuation; continuation = enc(context + continuation)[len(enc(context)):]).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from lm_eval.api.model import LM


class CortiqLM(LM):
    def __init__(
        self,
        cortiq: str,
        model: str,
        workdir: str,
        tag: str,
        split: str = "hflm",
        frame: str = "raw",
        scorer: str = "shared",
        env: dict | None = None,
    ):
        super().__init__()
        self.cortiq = cortiq
        self.model = model
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.split = split
        self.frame = frame
        self.scorer = scorer
        self.env = dict(os.environ)
        self.env.update(env or {})

    def _run(self, rows: list[dict], tag: str) -> list[dict]:
        req = self.workdir / f"{tag}.requests.jsonl"
        out = self.workdir / f"{tag}.loglik.jsonl"
        meta = self.workdir / f"{tag}.loglik.meta.json"
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        digest = hashlib.sha256(body.encode()).hexdigest()
        key = {
            "requests_sha256": digest,
            "model": self.model,
            "split": self.split,
            "frame": self.frame,
            "scorer": self.scorer,
            "CMF_GROWTH": self.env.get("CMF_GROWTH"),
            "CMF_GPU": self.env.get("CMF_GPU"),
        }
        if out.exists() and meta.exists() and json.loads(meta.read_text()) == key:
            res = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
            if len(res) == len(rows):
                return res
        req.write_text(body)
        # CORTIQ_LM_JOBS=N: N engine processes over contiguous parts cut at context boundaries
        # (a context and all its continuations stay in one part: the shared scorer forwards a
        # context once per part either way, so the numbers are those of one process)
        jobs = max(1, int(os.environ.get("CORTIQ_LM_JOBS", "1")))
        parts: list[list[dict]] = []
        if jobs > 1:
            per = -(-len(rows) // jobs)
            cur: list[dict] = []
            for r in rows:
                if len(cur) >= per and r["context"] != cur[-1]["context"]:
                    parts.append(cur)
                    cur = []
                cur.append(r)
            if cur:
                parts.append(cur)
        else:
            parts = [rows]
        procs = []
        for i, part in enumerate(parts):
            preq = req if len(parts) == 1 else self.workdir / f"{tag}.requests.part{i}.jsonl"
            pout = out if len(parts) == 1 else self.workdir / f"{tag}.loglik.part{i}.jsonl"
            if len(parts) > 1:
                preq.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
            cmd = [
                self.cortiq, "loglik", self.model,
                "--requests-jsonl", str(preq),
                "--out", str(pout),
                "--split", self.split,
                "--frame", self.frame,
                "--scorer", self.scorer,
            ]
            env = dict(self.env)
            if len(parts) > 1:
                env.setdefault("CMF_THREADS", str(max(1, (os.cpu_count() or 1) // len(parts))))
            log = open(self.workdir / f"{tag}.loglik{'' if len(parts) == 1 else f'.part{i}'}.log", "w")
            procs.append((subprocess.Popen(cmd, env=env, stderr=log), log, pout))
        for p, log, _ in procs:
            if p.wait() != 0:
                raise RuntimeError(f"cortiq loglik exited with {p.returncode}")
            log.close()
        if len(parts) > 1:
            with open(out, "w") as f:
                for _, _, pout in procs:
                    f.write(pout.read_text())
        res = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        if len(res) != len(rows):
            raise RuntimeError(f"cortiq loglik returned {len(res)} rows for {len(rows)} requests")
        meta.write_text(json.dumps(key))
        return res

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        rows = [
            {"context": r.args[0], "continuation": r.args[1], "k": k}
            for k, r in enumerate(requests)
        ]
        order = sorted(range(len(rows)), key=lambda k: (rows[k]["context"], k))
        res = self._run([rows[k] for k in order], self.tag)
        out: list[tuple[float, bool] | None] = [None] * len(rows)
        for r in res:
            out[r["k"]] = (float(r["logprob"]), bool(r["is_greedy"]))
        assert all(o is not None for o in out)
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("cortiq_lm serves loglikelihood only")

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError("cortiq_lm serves loglikelihood only")
