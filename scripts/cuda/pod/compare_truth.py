#!/usr/bin/env python3
"""Every path against the host with float activations (chain_truth.sh).

  compare_truth.py [--root /root/cuda]

Per file and path: over the steps both kept, how often the top-1 id is the
reference's, and the largest and mean absolute logit difference — at the
first token (after the prompt) and at the later ones separately.
"""
import glob
import os
import sys

import numpy as np

root = "/root/cuda"
if "--root" in sys.argv:
    root = sys.argv[sys.argv.index("--root") + 1]
paths = [("vk_{q}", "Vulkan, 0.8.66"), ("vkhead_{q}", "Vulkan, the first token's head on the token graph"),
         ("vk_{q}_coop0", "Vulkan 0.8.66, CMF_COOP=0"), ("cudaf_{q}_forced", "CUDA"),
         ("cudax_{q}_forced", "CUDA, exact"), ("cpu_{q}_forced", "CPU, integer activations")]
print("| file | path | top-1 as the reference | first token: max / mean | later tokens: max / mean |")
print("|---|---|---:|---:|---:|")
for q in ("q4tp", "q8_2f"):
    ref = f"{root}/out/ref/truth_{q}"
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ref}/*.gen"))
    for tag, label in paths:
        d = f"{root}/out/ref/{tag.format(q=q)}"
        same = steps = 0
        first, later = [], []
        for name in names:
            i = 0
            while True:
                fa, fb = f"{ref}/{name}.logits/step{i:05d}.f32", f"{d}/{name}.logits/step{i:05d}.f32"
                if not (os.path.exists(fa) and os.path.exists(fb)):
                    break
                a, b = np.fromfile(fa, dtype=np.float32), np.fromfile(fb, dtype=np.float32)
                diff = np.abs(a - b)
                (first if i == 0 else later).append((float(diff.max()), float(diff.mean())))
                same += int(a.argmax() == b.argmax())
                steps += 1
                i += 1
        if not steps:
            continue
        f = lambda v: f"{max(x[0] for x in v):.4f} / {sum(x[1] for x in v) / len(v):.5f}" if v else "-"
        print(f"| {q} | {label} | {same}/{steps} | {f(first)} | {f(later)} |")
