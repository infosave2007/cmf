#!/usr/bin/env python3
"""Golden fixture for the Rust GatedDeltaNet core: tiny random layer +
expected per-position outputs from the validated numpy oracle
(vmfcore/gdn_layer.py). Deterministic; the fixture is checked in.

Run from cmf/: python3 tests/gen_gdn_golden.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/oleg/Documents/cortiq-bot/vmfcore")
from gdn_layer import gdn_layer_np  # noqa: E402

H, nk, nv, dk, dv, K, T = 16, 2, 4, 6, 5, 4, 7
kd, vd = nk * dk, nv * dv
rng = np.random.RandomState(7)

W = {
    "in_proj_qkv": rng.randn(2 * kd + vd, H) * 0.2,
    "in_proj_z": rng.randn(vd, H) * 0.2,
    "in_proj_a": rng.randn(nv, H) * 0.2,
    "in_proj_b": rng.randn(nv, H) * 0.2,
    "A_log": rng.rand(nv) * 2.0,
    "dt_bias": rng.randn(nv) * 0.3,
    "conv1d": rng.randn(2 * kd + vd, 1, K) * 0.3,
    "norm": (1.0 + rng.rand(dv) * 0.2),
    "out_proj": rng.randn(H, vd) * 0.2,
}
W = {k: v.astype(np.float32) for k, v in W.items()}
x = (rng.randn(T, H) * 0.5).astype(np.float32)

expect = gdn_layer_np(x, W, nk, nv, dk, dv, eps=1e-6)

fixture = {
    "cfg": {"num_v_heads": nv, "num_k_heads": nk, "key_head_dim": dk,
            "value_head_dim": dv, "conv_kernel": K, "hidden_size": H,
            "rms_eps": 1e-6},
    "weights": {k: np.asarray(v, np.float32).reshape(-1).tolist()
                for k, v in W.items()},
    "x": x.tolist(),
    "expect": expect.tolist(),
}
out = Path(__file__).resolve().parent.parent / \
    "crates/cortiq-engine/tests/fixtures/gdn_golden.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fixture))
print(f"wrote {out} | T={T} out_std={expect.std():.4f}")
