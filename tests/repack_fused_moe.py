#!/usr/bin/env python3
"""Repack tiny qwen3_next from the per-expert layout (transformers)
into fused banks AgentWorld/qwen3_5_moe: mlp.experts.gate_up_proj [E,2I,H]
(gate||up along the out axis, WITHOUT .weight) + mlp.experts.down_proj [E,H,I].
Inverse operation to MoeFusedExpertsSource — conversion from both
layouts must produce the same forward.

Usage: repack_fused_moe.py src_dir dst_dir
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

EXP = re.compile(r"^(.*\.mlp\.experts\.)(\d+)\.(gate|up|down)_proj\.weight$")


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json"):
        if (src / fn).exists():
            shutil.copy(src / fn, dst / fn)

    w = load_file(str(src / "model.safetensors"))
    groups = {}   # prefix → {e: {kind: array}}
    out = {}
    for name, t in w.items():
        m = EXP.match(name)
        if not m:
            out[name] = t
            continue
        groups.setdefault(m.group(1), {}).setdefault(
            int(m.group(2)), {})[m.group(3)] = t

    for prefix, experts in groups.items():
        E = len(experts)
        gu = np.stack([np.concatenate([experts[e]["gate"], experts[e]["up"]],
                                      axis=0) for e in range(E)])
        dn = np.stack([experts[e]["down"] for e in range(E)])
        out[prefix + "gate_up_proj"] = np.ascontiguousarray(gu)
        out[prefix + "down_proj"] = np.ascontiguousarray(dn)

    save_file(out, str(dst / "model.safetensors"))
    print(f"  fused: {len(groups)} MoE layers x E={E} -> {dst}")


if __name__ == "__main__":
    main()
