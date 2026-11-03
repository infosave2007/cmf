#!/usr/bin/env python3
"""Final latents of a cortiq render against the reference run, chunk by chunk."""
import sys
from safetensors.torch import load_file

ref = load_file(sys.argv[1])
ours = load_file(sys.argv[2])
for k in sorted(ours):
    a, b = ours[k].float(), ref[k].float()
    print(k, "rel", float((a - b).norm() / b.norm()), "cos", float((a * b).sum() / a.norm() / b.norm()))
