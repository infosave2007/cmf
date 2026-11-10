"""Compare `cortiq decide-pointer` answers with the reference (dec_ref.py) on the same requests:
same winners, input token counts, and the largest probability difference.
usage: python3 -I dec_compare.py REF.json CMF.json"""
import json
import sys

ref, got = (json.load(open(f, encoding="utf-8")) for f in sys.argv[1:3])
assert len(ref) == len(got)
worst, same, n, tok = 0.0, 0, 0, 0
for r, g in zip(ref, got):
    tok += r["input_tokens"] == g["input_tokens"]
    assert list(r["answers"]) == list(g["answers"])
    for k, a in r["answers"].items():
        b = g["answers"][k]
        assert list(a["probabilities"]) == list(b["probabilities"]), k
        n += 1
        pa, pb = a["probabilities"], b["probabilities"]
        same += max(pa, key=pa.get) == max(pb, key=pb.get)
        worst = max(worst, max(abs(pa[c] - pb[c]) for c in pa))
        for f in ("score", "noul"):
            if f in a:
                worst = max(worst, abs(a[f] - b[f]) / (len(pa) - 1 if f == "score" else 1))
print(f"{len(ref)} requests: input tokens equal {tok}/{len(ref)}, winners equal {same}/{n}, "
      f"max |dp| {worst:.4f}, temperature {got[0]['temperature']}")
