#!/usr/bin/env python3
"""Compare round-1 outputs (out/ov1) with the reference dumps (ov/ref)."""
import json
import os

import numpy as np
import soundfile as sf

R, O = "ov/ref", "out/ov1"


def cmp_wav(a, b):
    x, _ = sf.read(a, dtype="float32")
    y = np.load(b)[0] if b.endswith(".npy") else sf.read(b, dtype="float32")[0]
    n = min(len(x), len(y))
    d = np.abs(x[:n] - y[:n])
    c = float(np.dot(x[:n], y[:n]) / (np.linalg.norm(x[:n]) * np.linalg.norm(y[:n]) + 1e-12))
    return f"len {len(x)} vs {len(y)}, max|d| {d.max():.2e}, cos {c:.6f}"


for c in ["en_auto", "ru_design", "zh_auto", "en_tag", "en_clone"]:
    print(f"== {c}")
    ref_ids = np.load(f"{R}/{c}_ids.npy")
    pf = f"{O}/{c}_prompt.json" if c != "en_clone" else f"{O}/en_clone_prompt.json"
    if os.path.exists(pf):
        p = json.load(open(pf))
        ids = np.array(p["ids"])
        same = ids.shape == ref_ids.shape and (ids == ref_ids).all()
        print(f"  prompt: ours {ids.shape} ref {ref_ids.shape} identical={same} frames {p['frames']}")
        if not same and ids.shape[1] == ref_ids.shape[1]:
            print("   first diff at", int(np.argmax((ids != ref_ids).any(0))))
        elif not same:
            print("   ours", ids[0][:40].tolist(), "\n   ref ", ref_ids[0][:40].tolist())
    rt = np.load(f"{R}/{c}_tokens.npy")
    for be in ["0", "1", "cuda", "refcodes"]:
        f = f"{O}/{c}_tok_{be}.json"
        if os.path.exists(f):
            t = np.array(json.load(open(f)))
            if t.shape == rt.shape:
                print(f"  tokens {be}: equal {float((t == rt).mean()) * 100:.1f}% ({t.shape})")
            else:
                print(f"  tokens {be}: shape {t.shape} vs ref {rt.shape}")
    if os.path.exists(f"{O}/{c}_dec_raw.wav"):
        print("  codec on ref tokens (raw):", cmp_wav(f"{O}/{c}_dec_raw.wav", f"{R}/{c}_raw.npy"))
        print("  post-processed:", cmp_wav(f"{O}/{c}_dec.wav", f"{R}/{c}.wav"))
for be in ["0", "cuda"]:
    f = f"{O}/clone_ref_{be}.json"
    if os.path.exists(f):
        a = np.array(json.load(open(f))["codes"])
        b = np.load(f"{R}/clone_ref_codes.npy")
        if a.shape == b.shape:
            print(f"clone ref codes {be}: equal {float((a == b).mean()) * 100:.1f}% per book {[round(float((a[i] == b[i]).mean()) * 100) for i in range(len(a))]}")
        else:
            print(f"clone ref codes {be}: shape {a.shape} vs {b.shape}")
