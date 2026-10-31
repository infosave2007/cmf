#!/usr/bin/env python3
"""Transformers reference for sensenova/SenseNova-Loop-9B-Preview.

Dumps the last-position logits and a greedy continuation for a text prompt
and for each image prompt (plus the processor pixels, grid and the vision
tower's merged rows). float32 on CUDA unless SN_DTYPE=bf16.

usage: sensenova_ref.py <model_dir> <out_dir> [<image> ...]
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

model_dir, out_dir, images = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
dtype = torch.bfloat16 if os.environ.get("SN_DTYPE") == "bf16" else torch.float32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype).cuda().eval()
print("loaded", type(model).__name__, dtype, "plan", len(model.model.language_model._loop_plan))
gen_n = int(os.environ.get("SN_GEN", "64"))

def run(tag, messages, image=None):
    prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    kw = dict(text=[prompt], return_tensors="pt")
    if image is not None:
        kw["images"] = [image]
    inputs = proc(**kw)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    meta = {"prompt": prompt, "input_ids": inputs["input_ids"][0].tolist()}
    with torch.inference_mode():
        if image is not None:
            pv = inputs["pixel_values"].to(dtype)
            grid = inputs["image_grid_thw"]
            vis = model.model.visual(pv, grid_thw=grid)
            rows = getattr(vis, "pooler_output", None)
            if rows is None or not torch.is_tensor(rows):
                rows = vis[0] if isinstance(vis, (tuple, list)) else vis.last_hidden_state
            np.save(f"{out_dir}/{tag}_pixels.npy", pv.float().cpu().numpy())
            np.save(f"{out_dir}/{tag}_rows.npy", rows.float().cpu().numpy())
            meta["grid_thw"] = grid.tolist()
        out = model(**inputs)
        logits = out.logits[0, -1].float().cpu().numpy()
        t0 = time.time()
        gen = model.generate(**inputs, max_new_tokens=gen_n, do_sample=False)
        meta["gen_s"] = time.time() - t0
    new = gen[0, inputs["input_ids"].shape[1]:].tolist()
    np.save(f"{out_dir}/{tag}_logits.npy", logits)
    meta.update({"gen_ids": new, "gen_text": proc.tokenizer.decode(new), "top5": np.argsort(-logits)[:5].tolist()})
    json.dump(meta, open(f"{out_dir}/{tag}.json", "w"), indent=1)
    print(tag, "n", len(meta["input_ids"]), "top5", meta["top5"], "->", meta["gen_text"][:300].replace("\n", " | "))

run("text", [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}])
run("text2", [{"role": "user", "content": "Write a haiku about autumn leaves."}])
for path in images:
    name = os.path.splitext(os.path.basename(path))[0]
    img = Image.open(path).convert("RGB")
    run(name, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in two sentences."}]}], img)
