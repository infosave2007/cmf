#!/usr/bin/env python3
"""The reference implementation's time to the transcript of the benchmark recordings (the authors'
`modeling_gigaam.py`, Transformers, trust_remote_code), for the comparison line of the card.

    ref_bench.py MODEL_DIR CLIPS_DIR SECONDS... [--device cuda] [--fp16]

A recording of up to 30 s goes through `transcribe`, a longer one through `transcribe_longform` (the package's
own voice-activity cut, batches of 16 segments). Each is run twice; the second pass is printed:
    ref <device> <f32|f16> s<SECONDS>: warm <s> | <x> real time
"""
import argparse, os, sys, time
import torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("clips")
ap.add_argument("secs", nargs="+", type=int)
ap.add_argument("--device", default="cuda")
ap.add_argument("--fp16", action="store_true")
a = ap.parse_args()
model = AutoModel.from_pretrained(a.model, trust_remote_code=True, device=a.device, fp16_encoder=a.fp16).eval()
for s in a.secs:
    path = os.path.join(a.clips, f"bench_{s}s.wav")
    best = None
    for _ in range(2):
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        r = model.transcribe(path) if s <= 30 else model.transcribe_longform(path)
        if a.device == "cuda":
            torch.cuda.synchronize()
        best = time.time() - t0
    print(f"ref {a.device} {'f16' if a.fp16 else 'f32'} s{s}: warm {best:.3f} | {s / best:.1f}x real time", flush=True)
