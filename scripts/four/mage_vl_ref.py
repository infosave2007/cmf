#!/usr/bin/env python3
"""Transformers reference for microsoft/Mage-VL-Cowork (llava_onevision2).

Dumps, per image: the processor's pixel rows and grid, the input ids, the
vision tower + merger rows, the prompt's last-position logits and a greedy
continuation. The model runs in float32 on CUDA (sdpa fallback for the
tower's flash-attention calls). Output: <out>/<name>.{npy,json}.

usage: mage_vl_ref.py <model_dir> <out_dir> <image> [<image> ...]
"""
import json, sys, os, time
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

model_dir, out_dir, images = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out_dir, exist_ok=True)
proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,
                                             torch_dtype=torch.float32).cuda().eval()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def raster_positions(grid):
    pos = []
    for t, h, w in grid.tolist():
        for ti in range(t):
            for r in range(h):
                for c in range(w):
                    pos.append([ti, r, c])
    return torch.tensor(pos, dtype=torch.long)

prompts = {
    "describe": "Describe this image in two sentences.",
}
text_only = "What is the capital of France? Answer in one word."
for img_path in images:
    name = os.path.splitext(os.path.basename(img_path))[0]
    image = Image.open(img_path).convert("RGB")
    for pk, ptext in prompts.items():
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ptext}]}]
        prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[image], return_tensors="pt")
        grid = inputs["image_grid_thw"]
        pp = raster_positions(grid)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.inference_mode():
            pv = inputs["pixel_values"].float()
            t0 = time.time()
            vis = model.model.visual(pv, grid_thw=inputs["image_grid_thw"], patch_positions=pp.cuda())
            rows = vis.last_hidden_state.reshape(-1, vis.last_hidden_state.shape[-1]).float().cpu().numpy()
            tv = time.time() - t0
            out = model(**inputs, patch_positions=pp.cuda())
            logits = out.logits[0, -1].float().cpu().numpy()
            gen = model.generate(**inputs, patch_positions=pp.cuda(), max_new_tokens=96, do_sample=False)
        new = gen[0, inputs["input_ids"].shape[1]:].tolist()
        tag = f"{name}_{pk}"
        np.save(f"{out_dir}/{tag}_pixels.npy", pv.cpu().numpy())
        np.save(f"{out_dir}/{tag}_rows.npy", rows)
        np.save(f"{out_dir}/{tag}_logits.npy", logits)
        json.dump({
            "image": img_path, "size": image.size, "grid_thw": grid.tolist(),
            "prompt": prompt, "input_ids": inputs["input_ids"][0].tolist(),
            "gen_ids": new, "gen_text": proc.tokenizer.decode(new, skip_special_tokens=False),
            "top5": np.argsort(-logits)[:5].tolist(), "tower_s": tv,
        }, open(f"{out_dir}/{tag}.json", "w"), indent=1)
        print(tag, grid.tolist(), rows.shape, "top5", np.argsort(-logits)[:5].tolist())
        print("  ->", proc.tokenizer.decode(new, skip_special_tokens=False)[:400].replace("\n", " | "))
# text only
messages = [{"role": "user", "content": text_only}]
prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = proc.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
with torch.inference_mode():
    logits = model(input_ids=ids).logits[0, -1].float().cpu().numpy()
    gen = model.generate(input_ids=ids, max_new_tokens=48, do_sample=False)
np.save(f"{out_dir}/text_logits.npy", logits)
json.dump({"prompt": prompt, "input_ids": ids[0].tolist(), "gen_ids": gen[0, ids.shape[1]:].tolist(),
           "gen_text": proc.tokenizer.decode(gen[0, ids.shape[1]:])}, open(f"{out_dir}/text.json", "w"), indent=1)
print("text ->", proc.tokenizer.decode(gen[0, ids.shape[1]:]).replace("\n", " | "))
