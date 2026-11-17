#!/usr/bin/env python3
"""eval.py PORT TAG [N]: greedy answers of a cortiq server to test.jsonl -> accuracy.
Raw /v1/completions with the exact rendered prompt; first word of the answer vs the label."""
import json, sys, time, urllib.request, collections
port, tag = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
rows = [json.loads(l) for l in open("/root/h3i/skill/emo/test.jsonl")][:n]
out = open(f"/root/h3i/skill/emo/pred_{tag}.jsonl", "w")
hit = collections.Counter(); tot = collections.Counter(); bad = 0
t0 = time.time()
for r in rows:
    body = json.dumps({"model": "cmf", "prompt": r["prompt"], "max_tokens": 6, "temperature": 0.0,
                       "repetition_penalty": 1.0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["text"]
    w = txt.strip().split()[0].strip(".,!:;\"'").lower() if txt.strip() else ""
    if w not in LABELS:
        bad += 1
    tot[r["label"]] += 1
    hit[r["label"]] += int(w == r["label"])
    out.write(json.dumps({"label": r["label"], "pred": w, "text": txt}) + "\n")
dt = time.time() - t0
acc = sum(hit.values()) / max(1, sum(tot.values()))
bal = sum(hit[l] / tot[l] for l in tot) / max(1, len(tot))
print(json.dumps({"tag": tag, "n": sum(tot.values()), "accuracy": round(acc, 4), "balanced": round(bal, 4),
                  "invalid": bad, "per_label": {l: round(hit[l] / tot[l], 3) for l in LABELS if tot[l]},
                  "seconds": round(dt, 1)}))
