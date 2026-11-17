#!/usr/bin/env python3
"""prep_code.py DATASET.json OUTDIR: the coding-skill corpora from a Cortiq-Coder dataset
([prompt, ChatML record] pairs), split 410 / 20 / 40 (seed 11) into train / validation / test:
  code_{train,val,test}.txt  plain text, ChatML markers dropped (all-token LM bake)
  chat_{train,val,test}.txt  records as cortiq renders a Qwen3.5 turn with thinking off, the final answer only
                             (answer-span bake), and chat_test_prompts.jsonl (prompt, reference answer)."""
import json, os, random, sys

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
d = json.load(open(src))
S, E = "<|im_start|>", "<|im_end|>"


def plain(text):
    for tag in ("system\n", "user\n", "assistant\n"):
        text = text.replace(S + tag, "")
    return text.replace(E, "").strip()


def parts(raw):
    sysm = raw.split(S + "system\n", 1)[1].split(E, 1)[0]
    user = raw.split(S + "user\n", 1)[1].split(E, 1)[0]
    a = raw.split(S + "assistant\n", 1)[1]
    ans = a.split("</think>", 1)[1] if "</think>" in a else a
    return sysm, user, ans.replace(E, "").strip()


def prompt(sysm, user):
    return f"{S}system\n{sysm}{E}\n{S}user\n{user}{E}\n{S}assistant\n<think>\n\n</think>\n\n"


idx = list(range(len(d)))
random.Random(11).shuffle(idx)
split = {"test": set(idx[:40]), "val": set(idx[40:60])}
which = lambda i: "test" if i in split["test"] else "val" if i in split["val"] else "train"
files = {f"{k}_{s}.txt": open(os.path.join(out, f"{k}_{s}.txt"), "w") for k in ("code", "chat") for s in ("train", "val", "test")}
tp = open(os.path.join(out, "chat_test_prompts.jsonl"), "w")
for i, x in enumerate(d):
    raw = x[1] if len(x) > 1 else x[0]
    s = which(i)
    t = plain(raw)
    if t:
        files[f"code_{s}.txt"].write(t + "\n\n")
    sysm, user, ans = parts(raw)
    files[f"chat_{s}.txt"].write(prompt(sysm, user) + ans + E + "\n")
    if s == "test":
        tp.write(json.dumps({"i": i, "prompt": prompt(sysm, user), "answer": ans}) + "\n")
for f in files.values():
    f.close()
print({k: os.path.getsize(os.path.join(out, k)) for k in files})
