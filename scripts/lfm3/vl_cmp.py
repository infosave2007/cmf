"""LFM3-VL parity of a CMF against the Transformers reference (lfm3_vl_ref.py):
per image, the patch rows the preprocessing produced, the tower + projector
rows (max |Δ| and mean row cosine), and the greedy answer of `cortiq run
--image ... --greedy` against the reference's greedy continuation.

usage: python3 -I vl_cmp.py REF_DIR MODEL.cmf [CORTIQ]
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np


def main():
    ref_dir, model = sys.argv[1], sys.argv[2]
    cortiq = sys.argv[3] if len(sys.argv) > 3 else "/root/lfm3/bin/cortiq"
    ref = json.load(open(os.path.join(ref_dir, "ref.json")))
    for i, rec in enumerate(ref["records"]):
        n = len(rec["generated_ids"])
        with tempfile.TemporaryDirectory() as dump:
            out = subprocess.run(
                [cortiq, "run", model, "--image", rec["image"], "--prompt", ref["question"],
                 "--greedy", "--max-tokens", str(n)],
                env={**os.environ, "XDG_RUNTIME_DIR": "/tmp", "CMF_LFM3_VL_DUMP": dump},
                capture_output=True, text=True, errors="replace",
            )
            px = np.fromfile(os.path.join(dump, "pixels_0.f32"), "<f4")
            rows = np.fromfile(os.path.join(dump, "rows_0.f32"), "<f4")
        rpx = np.fromfile(os.path.join(ref_dir, f"pixels_{i}.f32"), "<f4")
        rrows = np.fromfile(os.path.join(ref_dir, f"rows_{i}.f32"), "<f4")
        print(f"## {os.path.basename(rec['image'])} {rec['size']} grid {rec['grid_thw']}")
        if px.shape != rpx.shape:
            print(f"  pixels: SHAPE {px.shape} vs {rpx.shape}")
        else:
            print(f"  pixels: max |d| {np.abs(px - rpx).max():.3g}")
        if rows.shape != rrows.shape:
            print(f"  rows: SHAPE {rows.shape} vs {rrows.shape}")
        else:
            a, b = rows.reshape(-1, 2048), rrows.reshape(-1, 2048)
            cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            print(f"  rows: max |d| {np.abs(a - b).max():.3g} (ref max {np.abs(b).max():.3g}), "
                  f"cosine mean {cos.mean():.6f} min {cos.min():.6f}")
        body = out.stdout.split(f"Prompt: {ref['question']}\n\n", 1)[-1].rsplit("\n[", 1)[0].strip()
        want = rec["generated_text"].replace("<|im_end|>", "").strip()
        print(f"  greedy: {'identical' if body == want else 'DIFFERENT'}")
        if body != want:
            print(f"    ref:    {want!r}\n    cortiq: {body!r}")
            if out.returncode:
                print("    stderr:", out.stderr[-800:])


if __name__ == "__main__":
    main()
