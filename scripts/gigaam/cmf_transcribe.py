#!/usr/bin/env python3
"""Transcribe the clips of a manifest with `cortiq transcribe` and write the
same JSON lines `ref_transcribe.py` writes (`file`, `e2e`, `raw`, `ref`).

    cmf_transcribe.py CORTIQ MODEL.cmf MANIFEST.tsv --out hyp.jsonl
        [--batch 200] [--head both] [--limit N]

Clips go to the binary in batches (one model load a batch); the binary's
`--json` line a file is parsed back. Environment (CMF_GPU, CMF_THREADS)
passes through.
"""
import argparse, json, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("cortiq")
ap.add_argument("model")
ap.add_argument("manifest")
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=200)
ap.add_argument("--head", default="both")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

base = os.path.dirname(a.manifest)
clips = []
for line in open(a.manifest, encoding="utf-8"):
    name, secs, ref = line.rstrip("\n").split("\t")
    clips.append((name, float(secs), ref))
if a.limit:
    clips = clips[: a.limit]

t0 = time.time()
audio = work = 0.0
with open(a.out, "w", encoding="utf-8") as out:
    for i in range(0, len(clips), a.batch):
        part = clips[i : i + a.batch]
        cmd = [a.cortiq, "transcribe", a.model, *[os.path.join(base, c[0]) for c in part], "--json", "--head", a.head]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"cortiq failed: {res.stderr[-2000:]}")
        lines = [l for l in res.stdout.splitlines() if l.startswith("{")]
        if len(lines) != len(part):
            sys.exit(f"{len(lines)} results for {len(part)} clips")
        for (name, secs, ref), line in zip(part, lines):
            r = json.loads(line)
            row = {"file": name, "seconds": secs, "ref": ref, "time": round(r["seconds"]["total"], 4)}
            for head in ("e2e", "raw"):
                if r.get(head):
                    row[head] = r[head]["text"]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            audio += secs
            work += r["seconds"]["total"]
        out.flush()
        print(f"{min(i + a.batch, len(clips))}/{len(clips)} clips, {audio / max(work, 1e-9):.1f}x real time in the model, wall {time.time() - t0:.0f}s", file=sys.stderr)
