#!/usr/bin/env python3
"""Transformers reference for sensenova/SenseNova-Loop-9B (f32).

Writes, per prompt, the last-position logits (.npy), a greedy continuation and
its ids (.json); then the perplexity of the first N tokens of a text file
(one window, no BOS) and per-position top-1 ids for that window.

usage: sn_ref.py MODEL_DIR OUT_DIR [--image PNG] [--ppl TEXT N]
env:   SN_DEV=cpu|cuda (default cpu), SN_GEN=64 tokens, SN_THREADS, SN_SKIP_TEXT=1 (image and ppl only)
"""
import argparse, json, math, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--image"); ap.add_argument("--ppl", nargs=2)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = os.environ.get("SN_DEV", "cpu")
if os.environ.get("SN_THREADS"):
    torch.set_num_threads(int(os.environ["SN_THREADS"]))
torch.backends.cuda.matmul.allow_tf32 = False
proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(a.model, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
lm = model.model.language_model
print("loaded", type(model).__name__, "plan", len(lm._loop_plan), "rope", type(lm.rotary_emb).__name__,
      "attention_scaling", getattr(lm.rotary_emb, "attention_scaling", None), flush=True)
np.save(f"{a.out}/inv_freq.npy", lm.rotary_emb.inv_freq.float().cpu().numpy())
gen_n = int(os.environ.get("SN_GEN", "64"))

def run(tag, messages, image=None, think=False):
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    kw = dict(text=[prompt], return_tensors="pt")
    if image is not None:
        kw["images"] = [image]
    inputs = {k: v.to(dev) for k, v in proc(**kw).items()}
    meta = {"prompt": prompt, "input_ids": inputs["input_ids"][0].tolist()}
    with torch.inference_mode():
        if image is not None:
            meta["grid_thw"] = inputs["image_grid_thw"].tolist()
        logits = model(**inputs).logits[0, -1].float().cpu().numpy()
        t0 = time.time()
        gen = model.generate(**inputs, max_new_tokens=gen_n, do_sample=False, repetition_penalty=1.0)
        meta["gen_s"] = time.time() - t0
    new = gen[0, inputs["input_ids"].shape[1]:].tolist()
    np.save(f"{a.out}/{tag}_logits.npy", logits)
    meta.update({"gen_ids": new, "gen_text": proc.tokenizer.decode(new), "top5": np.argsort(-logits)[:5].tolist()})
    json.dump(meta, open(f"{a.out}/{tag}.json", "w"), indent=1, ensure_ascii=False)
    print(tag, "n", len(meta["input_ids"]), "top5", meta["top5"], f"{meta['gen_s']:.0f}s ->",
          meta["gen_text"][:400].replace("\n", " | "), flush=True)

if os.environ.get("SN_SKIP_TEXT") != "1":
    run("text", [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}])
    run("text2", [{"role": "user", "content": "Write a haiku about autumn leaves."}])
    run("code", [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}])
if a.image:
    img = Image.open(a.image).convert("RGB")
    run("image", [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in two sentences."}]}], img)
if a.ppl:
    path, n = a.ppl[0], int(a.ppl[1])
    ids = proc.tokenizer(open(path).read(), add_special_tokens=False).input_ids[:n]
    x = torch.tensor([ids]).to(dev)
    with torch.inference_mode():
        lg = model(input_ids=x).logits[0].float()
    nll = torch.nn.functional.cross_entropy(lg[:-1], x[0, 1:], reduction="mean").item()
    np.save(f"{a.out}/ppl_top1.npy", lg.argmax(-1).cpu().numpy())
    json.dump({"tokens": len(ids), "ppl": math.exp(nll)}, open(f"{a.out}/ppl.json", "w"))
    print(f"ppl tokens {len(ids)} ppl {math.exp(nll):.4f}", flush=True)
print("REF_DONE", flush=True)
