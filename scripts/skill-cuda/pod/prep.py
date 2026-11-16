#!/usr/bin/env python3
"""prep.py: dair-ai/emotion -> bake corpus (train), held (validation), eval prompts (test).
Records are rendered exactly as cortiq renders a Qwen3.5 chat turn with thinking off."""
import json, sys
import pandas as pd
D = "/root/h3i/skill/data/emotion/split"
OUT = "/root/h3i/skill/emo"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SYS = "Classify the emotion of the message. Reply with one word: sadness, joy, love, anger, fear or surprise."
def prompt(text):
    return ("<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n" + text.strip()
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
def record(text, label):
    return prompt(text) + LABELS[label] + "<|im_end|>\n"
tr = pd.read_parquet(f"{D}/train-00000-of-00001.parquet").sample(frac=1.0, random_state=7)
va = pd.read_parquet(f"{D}/validation-00000-of-00001.parquet").sample(frac=1.0, random_state=7)
te = pd.read_parquet(f"{D}/test-00000-of-00001.parquet")
with open(f"{OUT}/train.txt", "w") as f:
    for t, l in zip(tr.text, tr.label):
        f.write(record(t, l))
with open(f"{OUT}/held.txt", "w") as f:
    for t, l in zip(va.text[:400], va.label[:400]):
        f.write(record(t, l))
with open(f"{OUT}/held_tags.jsonl", "w") as f:
    for l in va.label[:400]:
        f.write(json.dumps({"family": LABELS[l]}) + "\n")
with open(f"{OUT}/test.jsonl", "w") as f:
    for t, l in zip(te.text, te.label):
        f.write(json.dumps({"prompt": prompt(t), "label": LABELS[l]}) + "\n")
print(len(tr), "train,", 400, "held,", len(te), "test; label shares test:",
      {LABELS[k]: round(v, 3) for k, v in te.label.value_counts(normalize=True).sort_index().items()})
