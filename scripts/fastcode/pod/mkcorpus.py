# mkcorpus.py OUTDIR: code routing corpus (calib) and disjoint code eval set, ChatML-formatted
import json, random, sys, os
from huggingface_hub import hf_hub_download
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
rng = random.Random(20261109)
oss = hf_hub_download("ise-uiuc/Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl", repo_type="dataset")
evo = hf_hub_download("ise-uiuc/Magicoder-Evol-Instruct-110K", "data-evol_instruct-decontaminated.jsonl", repo_type="dataset")
def chat(q, a):
    return f"<|im_start|>user\n{q.strip()}<|im_end|>\n<|im_start|>assistant\n{a.strip()}<|im_end|>\n"
rows = [json.loads(l) for l in open(oss)]
bylang = {}
for r in rows:
    bylang.setdefault(r.get("lang", "?"), []).append(r)
print({k: len(v) for k, v in sorted(bylang.items(), key=lambda x: -len(x[1]))})
erows = [json.loads(l) for l in open(evo)]
rng.shuffle(erows)
calib, ev = [], []
cbytes = {"oss": 0, "evo": 0}
# OSS-Instruct: equal share per language (all languages present), eval rows disjoint
for lang, rs in sorted(bylang.items()):
    rng.shuffle(rs)
    k = 0; b = 0
    while k < len(rs) and b < 26000:
        t = chat(rs[k]["problem"], rs[k]["solution"]); calib.append(t); b += len(t); k += 1
    cbytes["oss"] += b
    eb = 0
    while k < len(rs) and eb < 3500:
        t = chat(rs[k]["problem"], rs[k]["solution"]); ev.append(t); eb += len(t); k += 1
# Evol-Instruct: mixed languages and longer answers
k = 0; b = 0
while b < 200000:
    t = chat(erows[k]["instruction"], erows[k]["response"]); calib.append(t); b += len(t); k += 1
cbytes["evo"] = b
eb = 0
while eb < 30000:
    t = chat(erows[k]["instruction"], erows[k]["response"]); ev.append(t); eb += len(t); k += 1
rng.shuffle(calib); rng.shuffle(ev)
open(f"{out}/calib_code.txt", "w").write("".join(calib))
open(f"{out}/eval_code.txt", "w").write("".join(ev))
print("calib bytes", sum(map(len, calib)), cbytes, "eval bytes", sum(map(len, ev)))
