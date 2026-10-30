"""Transformers reference for LFM3-VL video input: a directory of frames
played at FPS, sampled by the checkpoint's own video processor (2 fps by
default). Dumps the patch rows (pixels_0.f32), the tower rows after the
projector (rows_0.f32), the prompt ids and N greedy tokens (ref.json).

usage: python3 -I lfm3_vl_video_ref.py MODEL_DIR {f32|bf16} OUT_DIR FRAME_DIR FPS
"""

import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

QUESTION = "Describe what happens in this video in one sentence."
MAX_NEW = 48


def main():
    model_dir, dtype, out_dir, frame_dir, fps = sys.argv[1:6]
    fps = float(fps)
    os.makedirs(out_dir, exist_ok=True)
    torch.set_grad_enabled(False)
    names = sorted(f for f in os.listdir(frame_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    frames = np.stack([np.asarray(Image.open(os.path.join(frame_dir, n)).convert("RGB")) for n in names])
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype={"bf16": torch.bfloat16, "f32": torch.float32}[dtype],
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()
    conv = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": QUESTION}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    meta = {"total_num_frames": len(frames), "fps": fps, "frames_indices": list(range(len(frames)))}
    inputs = proc(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=True,
                  return_tensors="pt").to("cuda:0")
    pv = inputs["pixel_values_videos"]
    pv.float().cpu().numpy().astype("<f4").tofile(os.path.join(out_dir, "pixels_0.f32"))
    feats = model.model.get_video_features(pv.to(model.dtype), inputs["video_grid_thw"], return_dict=True)
    torch.cat(feats.pooler_output, 0).float().cpu().numpy().astype("<f4").tofile(os.path.join(out_dir, "rows_0.f32"))
    gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.0)
    new = gen[0, inputs["input_ids"].shape[1]:].tolist()
    rec = {
        "frame_dir": os.path.abspath(frame_dir),
        "fps": fps,
        "grid_thw": inputs["video_grid_thw"][0].tolist(),
        "prompt_ids": inputs["input_ids"][0].tolist(),
        "prompt_text": proc.tokenizer.decode(inputs["input_ids"][0]),
        "generated_ids": new,
        "generated_text": proc.tokenizer.decode(new),
    }
    print(json.dumps({k: rec[k] for k in ("grid_thw", "generated_text")}), flush=True)
    with open(os.path.join(out_dir, "ref.json"), "w") as f:
        json.dump({"question": QUESTION, "records": [rec]}, f)


if __name__ == "__main__":
    main()
