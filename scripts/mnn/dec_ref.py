"""Reference answers of MNN-Pocket-0.3B-Dec (the authors' inference.py: Transformers + PEFT, unmerged LoRA,
float32) for a JSON list of requests. usage: python -I dec_ref.py DEC_DIR PYLIB REQUESTS.json OUT.json"""
import json, sys
dec, pylib, req, out = sys.argv[1:5]
sys.path.insert(0, pylib); sys.path.insert(0, dec)
from inference import PocketDecisionModel
m = PocketDecisionModel.from_pretrained(dec, device="cpu", dtype="float32")
res = [m.predict(r["state"], r["questions"]) for r in json.load(open(req, encoding="utf-8"))]
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in res:
    print(r["input_tokens"], {k: (a.get("choice"), a.get("decision"), round(a.get("score", -1), 3), {c: round(p, 4) for c, p in a["probabilities"].items()}) for k, a in r["answers"].items()})
