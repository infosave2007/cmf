#!/usr/bin/env python3
"""SAM3 reference fixtures for cortiq parity tests (tools/sam3_ref.py).

Produces deterministic fixtures for ONE synthetic image + text prompt:
  pixels.f32, text_embeds.f32, trunk_feats.f32, enc_hidden.f32,
  dec_hidden.f32, seg_out.f32, pred_scores/pred_boxes/pred_masks (as present),
  prompt.json, manifest.json
Run on the pod:
  HF_TOKEN=<token> python3 tools/sam3_ref.py /workspace/sam3-fx
"""
import sys, os, json, hashlib, inspect
import numpy as np
import torch
from PIL import Image

REPO = "facebook/sam3"
PROMPT = "red square"


def make_image():
    a = np.full((384, 384, 3), 255, dtype=np.uint8)
    a[64:192, 64:192] = (200, 30, 30)
    a[300:340, :, :] = (30, 60, 200)
    return Image.fromarray(a)


def dump(out_dir, name, t):
    t = t.detach().float().cpu().contiguous()
    arr = t.numpy()
    arr.tofile(os.path.join(out_dir, name + ".f32"))
    return {"shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "min": float(arr.min()), "max": float(arr.max())}


def as_ns(out):
    if isinstance(out, dict):
        from types import SimpleNamespace
        return SimpleNamespace(**out)
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/workspace/sam3-fx"
    os.makedirs(out_dir, exist_ok=True)
    tok = os.environ["HF_TOKEN"]
    from transformers import Sam3Processor, Sam3Model

    processor = Sam3Processor.from_pretrained(REPO, token=tok)
    model = Sam3Model.from_pretrained(REPO, token=tok, torch_dtype=torch.float32)
    model.eval()

    inputs = processor(images=make_image(), text=PROMPT, return_tensors="pt")
    params = set(inspect.signature(model.forward).parameters)
    inputs = {k: v for k, v in inputs.items() if k in params}
    print("forward inputs:", sorted(inputs.keys()))

    fx = {"pixels": dump(out_dir, "pixels", inputs["pixel_values"])}
    captured, hooks = {}, []

    def add(label, name):
        mod = getattr(model, name, None)
        if mod is None:
            print("skip missing module:", name)
            return

        def fn(_m, _i, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                # transformers ModelOutput: take the first tensor field
                for v in (out.values() if hasattr(out, "values") else []):
                    if torch.is_tensor(v):
                        out = v
                        break
                else:
                    return
            captured[label] = out.detach()
        hooks.append(mod.register_forward_hook(fn))

    add("text_embeds", "text_encoder")
    add("trunk_feats", "vision_encoder")
    add("enc_hidden", "detr_encoder")
    add("dec_hidden", "detr_decoder")
    add("seg_out", "mask_decoder")

    with torch.no_grad():
        out = as_ns(model(**inputs))
    for h in hooks:
        h.remove()

    for k in ("text_embeds", "trunk_feats", "enc_hidden", "dec_hidden", "seg_out"):
        if k in captured:
            fx[k] = dump(out_dir, k, captured[k])
    for k in ("pred_boxes", "pred_scores", "pred_masks", "masks", "boxes", "scores"):
        if hasattr(out, k) and getattr(out, k) is not None:
            fx[k] = dump(out_dir, k, getattr(out, k))

    json.dump({"prompt": PROMPT,
               "image": "synthetic 384x384 (white + red square + blue stripe)",
               "image_size": [384, 384],
               "forward_inputs": sorted(inputs.keys()),
               "output_attrs": [a for a in dir(out) if not a.startswith("_")]},
              open(os.path.join(out_dir, "prompt.json"), "w"), indent=1)
    json.dump(fx, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print("FIXTURES_DONE", sorted(fx.keys()))
    for k, v in sorted(fx.items()):
        print("  %-14s %s sha=%s range=[%.3f,%.3f]"
              % (k, v["shape"], v["sha256"][:16], v["min"], v["max"]))


if __name__ == "__main__":
    main()