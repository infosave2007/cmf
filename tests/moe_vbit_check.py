#!/usr/bin/env python3
"""C4 gate (P15 claim 12): per-expert bit allocation in vbit.

Build a tiny Qwen3-MoE, inflate the expert amplitudes in a ladder
2^{-2..+2}, convert to VBIT and check against the file (with a standalone
reader): the expert mean bits grow monotonically with amplitude,
the family's overall budget holds the target, the quiet expert is pinned to the floor, the loud one
is raised — something a per-tensor water-fill cannot provide (there ALL
experts have mean = mean_bits)."""
import subprocess
import sys
from pathlib import Path

import numpy as np

CMF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CMF / "python"))
sys.path.insert(0, str(CMF / "tests"))
from cmf_reader import CmfReader  # noqa: E402

MEAN_BITS = 4.25
N_EXPERTS = 8


def build_scaled(out: Path):
    import gen_moe_case as G
    import torch
    model = G.build("qwen3", seed=0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".mlp.experts." in name:
                e = int(name.split(".experts.")[1].split(".")[0])
                p.mul_(2.0 ** ((e - (N_EXPERTS - 1) / 2) * 0.7))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    G.tiny_tokenizer(out)


def expert_mean_bits(r: CmfReader, li: int, proj: str) -> list:
    means = []
    for e in range(N_EXPERTS):
        name = f"model.layers.{li}.mlp.experts.{e}.{proj}.weight"
        entry, _ = r.tensors[name]
        assert entry["dtype"] == "vbit", f"{name}: {entry['dtype']} ≠ vbit"
        rows = entry["shape"][0]
        bits = np.frombuffer(r.tensor_bytes(name), np.uint8, rows)
        means.append(float(bits.mean()))
    return means


def main():
    tmp = Path(sys.argv[1])
    build_scaled(tmp / "moe")
    subprocess.run(
        [sys.executable, str(CMF / "converter/convert_dtgma_to_cmf.py"),
         "--model", str(tmp / "moe"), "--quant", "VBIT",
         "--output", str(tmp / "moe-vbit.cmf")],
        check=True, capture_output=True)

    r = CmfReader(tmp / "moe-vbit.cmf")
    for li in (0, 3):
        for proj in ("gate_proj", "up_proj"):
            mb = expert_mean_bits(r, li, proj)
            fam = float(np.mean(mb))
            # 1) the family's budget holds the target
            assert abs(fam - MEAN_BITS) < 0.6, f"L{li} {proj}: budget {fam:.2f}"
            # 2) monotonic growth with amplitude (tolerate 1 inversion from
            #    level quantization)
            inversions = sum(1 for a, b in zip(mb, mb[1:]) if b < a - 1e-9)
            assert inversions <= 1, f"L{li} {proj}: not monotonic {mb}"
            # 3) real spread: the quiet one at the floor, the loud one on top
            assert mb[0] <= 3.5 and mb[-1] >= 6.0, f"L{li} {proj}: {mb}"
            print(f"  ✓ L{li} {proj}: expert bits {['%.1f' % m for m in mb]}"
                  f" | family {fam:.2f} ≈ {MEAN_BITS}")
    assert r.verify() == []
    print("  ✓ verify clean (reader)")


if __name__ == "__main__":
    main()
