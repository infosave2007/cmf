#!/usr/bin/env python3
"""PyTorch reference dump for MiniCPM-OCR-1.0 (transformers >= 5.21).

usage: ref_minicpm_ocr.py MODEL_DIR IMAGE OUT_DIR [PROMPT] [MAX_NEW]
Writes input ids, pixels, tower rows, last-position logits and the greedy text.
"""
import json, sys, time, os
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

path, image, out = sys.argv[1:4]
prompt = sys.argv[4] if len(sys.argv) > 4 else "Text Recognition:"
max_new = int(sys.argv[5]) if len(sys.argv) > 5 else 256
os.makedirs(out, exist_ok=True)
proc = AutoProcessor.from_pretrained(path)
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).cuda().eval()
img = Image.open(image).convert("RGB")
msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
inp = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in inp.items()}
ids = inp["input_ids"][0].tolist()
json.dump({"ids": ids, "target_sizes": inp["target_sizes"].tolist(), "image_size": img.size,
           "prompt_text": proc.tokenizer.decode(ids[:60]) }, open(f"{out}/inputs.json", "w"))
inp["pixel_values"].cpu().float().numpy().astype("<f4").tofile(f"{out}/pixels.f32")
with torch.no_grad():
    t0 = time.time()
    feats = model.model.get_image_features(inp["pixel_values"], inp["target_sizes"]).pooler_output
    print("tower", time.time() - t0, [f.shape for f in feats])
    torch.cat(feats, 0).cpu().float().numpy().astype("<f4").tofile(f"{out}/rows.f32")
    o = model(**inp)
    o.logits[0, -1].cpu().float().numpy().astype("<f4").tofile(f"{out}/logits_last.f32")
    print("top5", torch.topk(o.logits[0, -1], 5))
    t0 = time.time()
    g = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    dt = time.time() - t0
new = g[0][len(ids):].tolist()
json.dump({"token_ids": new}, open(f"{out}/gen.json", "w"))
text = proc.tokenizer.decode(new, skip_special_tokens=False)
open(f"{out}/gen.txt", "w").write(text)
print("prompt tokens", len(ids), "new", len(new), "time", dt)
print(text[:1500])
