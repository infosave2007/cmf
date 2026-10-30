"""Transformers reference for the LiquidAI LFM3-VL CMF conversion.

For every image: the processor's patch rows (`pixel_values`) and grid, the
image features after the projector (the rows that replace the `<|image|>`
placeholders), the chat prompt ids, the top-10 logits of the first answer
position and N greedy tokens — in float32 (bf16 weights upcast) unless
`bf16` is asked.

usage: python3 -I lfm3_vl_ref.py MODEL_DIR {f32|bf16} OUT_DIR IMAGE [IMAGE ...]
       (OUT_DIR gets pixels_I.f32, rows_I.f32 and ref.json)
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "Describe this image in one sentence."
MAX_NEW = 48


def main():
    model_dir, dtype, out_dir, images = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype={"bf16": torch.bfloat16, "f32": torch.float32}[dtype],
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()
    records = []
    for i, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        conv = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": QUESTION}]}]
        inputs = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to("cuda:0")
        pv = inputs["pixel_values"].float().cpu().numpy()
        pv.astype("<f4").tofile(os.path.join(out_dir, f"pixels_{i}.f32"))
        feats = model.model.get_image_features(inputs["pixel_values"].to(model.dtype),
                                               inputs["image_grid_thw"], return_dict=True)
        rows = torch.cat(feats.pooler_output, 0).float().cpu().numpy()
        rows.astype("<f4").tofile(os.path.join(out_dir, f"rows_{i}.f32"))
        logits = model(**inputs).logits[0, -1].float()
        top = logits.topk(10)
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.0)
        new = gen[0, inputs["input_ids"].shape[1]:].tolist()
        rec = {
            "image": os.path.abspath(path),
            "size": list(image.size),
            "grid_thw": inputs["image_grid_thw"][0].tolist(),
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "top10_ids": top.indices.tolist(),
            "top10_logits": top.values.tolist(),
            "generated_ids": new,
            "generated_text": proc.tokenizer.decode(new),
        }
        print(json.dumps({k: rec[k] for k in ("image", "size", "grid_thw", "generated_text")}), flush=True)
        records.append(rec)
    with open(os.path.join(out_dir, "ref.json"), "w") as f:
        json.dump({"question": QUESTION, "dtype": dtype, "records": records}, f)


if __name__ == "__main__":
    main()
