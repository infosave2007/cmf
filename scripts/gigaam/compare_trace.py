#!/usr/bin/env python3
"""Compare the intermediate tensors of one clip: the reference dump of
`ref_transcribe.py --dump REF` against `CMF_GIGAAM_TRACE_DIR=OURS cortiq
transcribe model.cmf clip.wav`.

    compare_trace.py REF OURS

Prints, per stage, the relative error ‖ours − ref‖ / ‖ref‖ and the worst
absolute difference. The reference `features` are [mel, frames] and the
stack outputs carry a padded frame; both are brought to our [frames, width].
"""
import sys, os
import numpy as np

ref_dir, our_dir = sys.argv[1:3]
bad = 0
for name in ("features", "frontend", "stack0", "stack1", "stack2", "encoded"):
    r = np.load(os.path.join(ref_dir, name + ".npy")).astype(np.float64)
    o = np.load(os.path.join(our_dir, name + ".npy")).astype(np.float64)
    if name == "features":
        r = r.T
    if r.ndim == 3:
        r = r[0]
    if r.shape[-1] != o.shape[-1] and r.shape[-1] % o.shape[-1] == 0:
        r = r.reshape(-1, o.shape[-1])  # a stack at half the frame rate: rows of frame pairs
    r = r[: o.shape[0]]
    if r.shape != o.shape:
        print(f"{name:9s} shape {o.shape} vs reference {r.shape}")
        bad += 1
        continue
    rel = np.linalg.norm(o - r) / max(np.linalg.norm(r), 1e-30)
    print(f"{name:9s} {o.shape}  rel {rel:.3e}  max abs {np.abs(o - r).max():.3e}  ref rms {np.sqrt((r * r).mean()):.3f}")
sys.exit(1 if bad else 0)
