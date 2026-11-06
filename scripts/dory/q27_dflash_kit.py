#!/usr/bin/env python3
"""q27_dflash_kit.py REF OUT: the reference kit of q27_ext_draft.py (REF=DIR: ref-N.npz
and taps-N.f16 per window) as raw files `cortiq dflash-check` reads.

Per window N: wN.ids.u32 (the window's tokens), wN.own.u32 (the target's own next token
per row), wN.starts.u32 (the row before each anchor), wN.drafts.u32 / wN.plain.u32
[starts][block - 1] (the selector's tokens, each row's own top token), wN.hidden.f32
[first starts][block][hidden] (block rows after the last norm, of the first few starts),
wN.target.f32 [rows][hidden] (context rows after fc and its norm),
wN.key0.f32 / wN.value0.f32 [kv heads][rows][head dim] (layer 0's context), and
wN.taps.f16 [layers][window][hidden], linked from the reference.
"""
import json, os, sys
import numpy as np

ref, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
meta = json.load(open(f"{ref}/taps-meta.json"))
wins = sorted(int(f[4:-4]) for f in os.listdir(ref) if f.startswith("ref-") and f.endswith(".npz"))
kit = {"windows": wins, "window": meta["window"], "hidden": meta["hidden"], "layers": meta["layers"]}
for w in wins:
    z = np.load(f"{ref}/ref-{w}.npz")
    n = z["hidden"].shape[0]
    z["ids"].astype(np.uint32).tofile(f"{out}/w{w}.ids.u32")
    z["own"].astype(np.uint32).tofile(f"{out}/w{w}.own.u32")
    z["starts"].astype(np.uint32).tofile(f"{out}/w{w}.starts.u32")
    z["hidden"].astype(np.float32).tofile(f"{out}/w{w}.hidden.f32")
    z["drafts"].astype(np.uint32).tofile(f"{out}/w{w}.drafts.u32")
    z["plain"].astype(np.uint32).tofile(f"{out}/w{w}.plain.u32")
    for k in ("target", "key0", "value0"):
        z[k].astype(np.float32).tofile(f"{out}/w{w}.{k}.f32")
    link = f"{out}/w{w}.taps.f16"
    if not os.path.exists(link):
        os.symlink(os.path.abspath(f"{ref}/taps-{w}.f16"), link)
    kit.update(starts=int(n), block=int(z["hidden"].shape[1]), rows=int(z["target"].shape[0]), length=kit.get("length", {}) | {str(w): int(z["length"])})
    print(f"window {w}: {len(z['starts'])} starts, block rows of the first {n}, drafts {z['drafts'].shape}", flush=True)
json.dump(kit, open(f"{out}/kit.json", "w"), indent=1)
