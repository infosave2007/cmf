"""Per-position comparison of `cortiq ppl` (CMF_PPL_TRACE=1 stderr) against the
PyTorch reference (`apex_ref.py ppl` .npz) over the same ids: top-1
agreement, mean |ΔNLL|, NLL correlation and both perplexities.

usage: python3 -I ppl_cmp.py REF.npz TRACE.log [TRACE.log ...]
"""

import re
import sys

import numpy as np

LINE = re.compile(r"(?:BTRACE )?pos\s+(\d+)\s+(?:target|tgt)\s+(\d+)\s+nll\s+([-\d.]+)\D+?top1?\s+(\d+)")


def main():
    ref = np.load(sys.argv[1])
    r_nll, r_top, ids = ref["nll"], ref["argmax"], ref["ids"]
    for path in sys.argv[2:]:
        rows = {}
        for line in open(path, errors="replace"):
            m = LINE.search(line)
            if m:
                pos, tgt, nll, top = int(m[1]), int(m[2]), float(m[3]), int(m[4])
                rows[pos] = (tgt, nll, top)
        # Position p predicts ids[p + 1]; the reference arrays are indexed by p.
        pos = np.array(sorted(p for p in rows if p < len(r_nll)))
        tgt = np.array([rows[p][0] for p in pos])
        assert (tgt == ids[pos + 1]).all(), f"{path}: targets differ from the reference ids"
        c_nll = np.array([rows[p][1] for p in pos])
        c_top = np.array([rows[p][2] for p in pos])
        agree = (c_top == r_top[pos]).mean()
        d = np.abs(c_nll - r_nll[pos])
        print(
            f"{path}: {len(pos)} positions | top-1 agree {agree * 100:.2f} % | "
            f"mean |dNLL| {d.mean():.4f} (p99 {np.quantile(d, 0.99):.3f}) | "
            f"corr {np.corrcoef(c_nll, r_nll[pos])[0, 1]:.5f} | "
            f"ppl cortiq {np.exp(c_nll.mean()):.3f} vs ref {np.exp(r_nll[pos].mean()):.3f}"
        )


if __name__ == "__main__":
    main()
