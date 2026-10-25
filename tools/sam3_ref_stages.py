#!/usr/bin/env python3
"""SAM3 trunk/text sub-stage reference dumps (bisection helper).

Mirrors the Rust debug dumps: trunk_patch (conv, no pos), trunk_pre
(pos + layer_norm), trunk_b{l} (layer l output), text_embeds/text_lh.
Run:
  /home/hermes/venvs/sam3/bin/python tools/sam3_ref_stages.py <out_dir>
"""
import os, sys, json
import numpy as np
import torch

FX = "/home/hermes/cmf/crates/cortiq-engine/tests/fixtures/sam3"
MODEL = "/home/hermes/models/sam3-hf"
LAYERS = {int(x) for x in os.environ.get("REF_LAYERS", "0,1,2,3,4,8,16,31").split(",")}


def dump(out, name, t):
    a = t.detach().float().cpu().contiguous().numpy()
    a.tofile(os.path.join(out, name + ".f32"))
    print("dump", name, a.shape, "absmax", float(np.abs(a).max()), flush=True)
    return a


def rel_cos(got, want):
    got = got.astype(np.float64).ravel()
    want = want.astype(np.float64).ravel()
    assert got.size == want.size, (got.size, want.size)
    d = np.abs(got - want)
    mx = np.abs(want).max()
    rel = d.max() / mx if mx > 0 else d.max()
    denom = np.linalg.norm(got) * np.linalg.norm(want)
    cos = float(got @ want / denom) if denom > 0 else 1.0
    return rel, cos


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/home/hermes/.hermes/cache/scratch/fx_ref"
    os.makedirs(out, exist_ok=True)
    from transformers import Sam3Model

    pix = np.fromfile(os.path.join(FX, "pixels.f32"), dtype=np.float32).reshape(1, 3, 1008, 1008)
    pv = torch.from_numpy(pix.copy())
    ids = [49406, 736, 3999] + [49407] * 29
    attn = [1, 1, 1, 1] + [0] * 28

    print("loading model...", flush=True)
    m = Sam3Model.from_pretrained(MODEL, torch_dtype=torch.float32)
    m.eval()

    with torch.no_grad():
        # ---- text ----
        ii = torch.tensor([ids])
        am = torch.tensor([attn])
        to = m.text_encoder(input_ids=ii, attention_mask=am)
        te = to.text_embeds if hasattr(to, "text_embeds") and to.text_embeds is not None else to[0]
        dump(out, "text_embeds", te)
        dump(out, "text_lh", to.last_hidden_state)

        # ---- trunk ----
        bb = m.vision_encoder.backbone
        patch = bb.embeddings.patch_embeddings(pv)
        dump(out, "trunk_patch", patch)
        e = bb.embeddings(pv)
        hs = e.view(1, 72, 72, -1)
        hs = bb.layer_norm(hs)
        dump(out, "trunk_pre", hs.view(1, 5184, -1))
        for l, layer in enumerate(bb.layers):
            hs = layer(hs)
            if l in LAYERS:
                dump(out, f"trunk_b{l}", hs.view(1, 5184, -1))
        final = hs.view(1, 5184, -1)
        dump(out, "trunk_final", final)

    # compare against the committed fixtures
    for name in ("trunk_patch", "trunk_pre", "text_embeds", "trunk_final"):
        ref = os.path.join(FX, {"trunk_final": "trunk_feats"}.get(name, name) + ".f32")
        if not os.path.exists(ref):
            continue
        want = np.fromfile(ref, dtype=np.float32)
        got = np.fromfile(os.path.join(out, name + ".f32"), dtype=np.float32)
        if got.size != want.size:
            print("SIZE", name, got.size, want.size, flush=True)
            continue
        rel, cos = rel_cos(got, want)
        print(f"vs-fixture {name:12} rel={rel:.3e} cos={cos:.6}", flush=True)
    print("REF_DONE", flush=True)


if __name__ == "__main__":
    main()