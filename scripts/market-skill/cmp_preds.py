# cmp_preds.py A.jsonl B.jsonl: per-record class probabilities of two eval runs
import json, sys
A = [json.loads(l) for l in open(sys.argv[1])]
B = [json.loads(l) for l in open(sys.argv[2])]
n = min(len(A), len(B))
d = [max(abs(A[i]["p"][k] - B[i]["p"][k]) for k in A[i]["p"]) for i in range(n)]
same = sum(A[i]["pred"] == B[i]["pred"] for i in range(n))
print(f"{n} records: max |dp| {max(d):.2e}, mean {sum(d)/n:.2e}, same label {same}/{n}")
