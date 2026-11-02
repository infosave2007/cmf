#!/usr/bin/env python3
"""PyTorch reference dump for Unlimited-OCR-2 (transformers 4.52.1).

usage: ref_unlimited_ocr.py MODEL_DIR OUT_DIR MAX_NEW CROP IMAGE [IMAGE...]
CROP: auto, base or gundam. Writes prompt ids, pixels, the image rows, the routing of
layer 0 for the prompt, last-position logits and the greedy text.
"""
import json, sys, os, time
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

path, out, max_new, crop, images = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5:]
os.makedirs(out, exist_ok=True)
sys.path.insert(0, path)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# REF_DTYPE=bf16: the upstream default, to see how far the original itself moves.
bf16 = os.environ.get("REF_DTYPE") == "bf16"
model = AutoModel.from_pretrained(path, trust_remote_code=True, use_safetensors=True,
                                  torch_dtype=torch.bfloat16 if bf16 else torch.float32).eval().cuda()
crop_mode = {"auto": None, "base": False, "gundam": True}[crop]
# The processor casts pixels to bf16; keep them f32 for a float32 parity run.
mod = sys.modules[type(model).__module__]
_build = mod._uocr_build_inputs
def build_f32(*a, **k):
    ids, imgs, mask, spatial = _build(*a, **k)
    if imgs is not None:
        if not bf16:
            imgs = (imgs[0].float(), imgs[1].float())
        imgs[0].float().cpu().numpy().astype("<f4").tofile(f"{out}/pixels_crop.f32")
        imgs[1].float().cpu().numpy().astype("<f4").tofile(f"{out}/pixels_global.f32")
        print("pixels", list(imgs[0].shape), list(imgs[1].shape), "spatial", spatial.tolist())
    return ids, imgs, mask, spatial
mod._uocr_build_inputs = build_f32
rows = {}
hooks = []
def tap(name):
    def f(_m, _i, o):
        rows[name] = (o[0] if isinstance(o, tuple) else o).detach().cpu().float()
    return f
hooks.append(model.model.projector.register_forward_hook(tap("projector")))
hooks.append(model.model.sam_model.register_forward_hook(tap("sam")))
hooks.append(model.model.vision_model.register_forward_hook(tap("clip")))
calls = {"n": 0}
first = {}
def first_layer(i):
    def f(_m, _i, o):
        if i not in first:
            first[i] = (o[0] if isinstance(o, tuple) else o).detach().cpu().float()
    return f
for i, l in enumerate(model.model.layers):
    hooks.append(l.register_forward_hook(first_layer(i)))
embeds = {}
def first_in(_m, args, kwargs):
    x = args[0] if args else kwargs["hidden_states"]
    if "e" not in embeds and x.shape[1] > 1:
        embeds["e"] = x.detach().cpu().float()
hooks.append(model.model.layers[0].register_forward_pre_hook(first_in, with_kwargs=True))
t0 = time.time()
res = model.infer(tok, image_file=[Image.open(p).convert("RGB") for p in images], crop_mode=crop_mode,
                  max_new_tokens=max_new, eval_mode=True, return_dict=True)
dt = time.time() - t0
for h in hooks:
    h.remove()
json.dump({"prompt_token_ids": res["prompt_token_ids"], "token_ids": res["token_ids"],
           "finish": res["finish_reason"], "time": dt, "prefill_time": res["prefill_time"]},
          open(f"{out}/gen.json", "w"))
open(f"{out}/gen.txt", "w").write(res["text"])
for k, v in rows.items():
    v.numpy().astype("<f4").tofile(f"{out}/{k}.f32")
    print(k, list(v.shape))
if "e" in embeds:
    embeds["e"].numpy().astype("<f4").tofile(f"{out}/embeds.f32")
for i, v in first.items():
    v.numpy().astype("<f4").tofile(f"{out}/layer{i}.f32")
print("prompt", len(res["prompt_token_ids"]), "new", len(res["token_ids"]), "time", dt, "prefill", res["prefill_time"])
print(res["text"][:1500])
