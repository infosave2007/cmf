"""MNN-Pocket-0.3B-Dec -> one checkpoint `cortiq convert` reads: the LoRA adapter merged into the backbone
(W + alpha/r * B @ A, in float32), plus the pointer head and the calibrated temperature as float32 tensors
`decision.pointer.{q,k}.{weight,bias}` and `decision.pointer.temperature` [1], next to the backbone's config
and tokenizer. The converter keeps `decision.pointer.*` in float32 and `cortiq decide-pointer` reads them.

usage: python -I dec_merge.py DEC_DIR OUT_DIR"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

dec, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(dec, "decision_config.json")))
acfg = json.load(open(os.path.join(dec, cfg["adapter_dir"], "adapter_config.json")))
assert acfg["peft_type"] == "LORA" and not acfg.get("use_dora") and not acfg.get("use_rslora")
scale = acfg["lora_alpha"] / acfg["r"]
base_dir = os.path.join(dec, cfg["base_dir"])
w = {k: v.float() for k, v in load_file(os.path.join(base_dir, "model.safetensors")).items()}
lora = load_file(os.path.join(dec, cfg["adapter_dir"], "adapter_model.safetensors"))
merged = 0
for k in [k for k in lora if k.endswith(".lora_A.weight")]:
    target = k.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
    a, b = lora[k].float(), lora[k.replace(".lora_A.", ".lora_B.")].float()
    assert target in w and w[target].shape == (b.shape[0], a.shape[1]), target
    w[target] = w[target] + scale * (b @ a)
    merged += 1
assert merged * 2 == len(lora), (merged, len(lora))
head = load_file(os.path.join(dec, cfg["pointer_file"]))
for k, v in head.items():
    w[f"decision.pointer.{k}"] = v.float().contiguous()
temp = json.load(open(os.path.join(dec, cfg["temperature_file"])))["temperature"]
w["decision.pointer.temperature"] = torch.tensor([temp], dtype=torch.float32)
os.makedirs(out, exist_ok=True)
save_file(w, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(base_dir):
    if f != "model.safetensors":
        shutil.copy(os.path.join(base_dir, f), out)
print(f"merged {merged} LoRA pairs (scale {scale}), head {sorted(head)}, temperature {temp} -> {out}")
