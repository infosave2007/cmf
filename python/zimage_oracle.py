#!/usr/bin/env python3
"""Z-Image / Z-Image-Turbo oracle dumper + diffusers bf16 baseline for the cortiq port.

Both models share the text encoder (Qwen3-4B), the tokenizer and the Flux VAE; they
differ in the DiT weights (Turbo: fp32 checkpoint, base: bf16) and the recipe:

  turbo: scheduler shift 3, 8 steps, guidance 0 (one DiT forward per step)
  base : scheduler shift 6, 28-50 steps, guidance 3-5 with a negative prompt (CFG:
         pred = pos + g*(pos-neg), optional cfg_normalization / cfg_truncation)

Usage (RunPod 3090; the GPU is shared, so every GPU phase is run under
`flock /root/gpu.lock`; the fp32 phases run on the CPU and take no lock):

  PY=/root/zimage/venv/bin/python
  $PY zimage_oracle.py --model turbo tok te32 run32 dit32 vae          # CPU, fp32
  flock /root/gpu.lock $PY zimage_oracle.py --model turbo te16 run16 dit16 bench
  $PY zimage_oracle.py --model base run32 dit32                        # CPU (TE from turbo dir)
  flock /root/gpu.lock $PY zimage_oracle.py --model base run16 dit16 full bench

Layout: /root/zimage/oracles/v1/{turbo,base}/ ; safetensors, f32 (bf16 results upcast
exactly), i64 ids, u8 images. Sequence tensors are in diffusers order [img..., cap...]
with padded lengths. Noise is drawn on the CPU (torch.Generator("cpu").manual_seed(42))
and injected; the raw `noise_<res>_s42.f32` file is the CMF_INIT_LATENT format.

fp32 = fp32 compute on the checkpoint weights (bf16 TE / base DiT upcast), on the CPU,
TF32 off. bf16 = the user path: bf16 weights and compute on CUDA, SDPA default.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file, load_file

ROOTS = {"turbo": "/root/zimage/Z-Image-Turbo", "base": "/root/zimage/Z-Image"}
OUT = "/root/zimage/oracles/v1"

PROMPTS = {
    "p0": "A cat sitting on a windowsill at sunset, photorealistic.",
    "p1": ("A cozy mountain cabin interior at dusk, warm light from a stone fireplace, a wooden table "
           "with a steaming cup of tea and an open notebook, snow falling outside the large window, "
           "pine forest in the background, highly detailed, 35mm photograph, soft shadows, "
           "cinematic color grading."),
    "p2": "一只橘猫坐在窗台上，夕阳，写实摄影，窗上贴着一张写有“你好”的便签。",
    "p3": "",
    # negative prompt used by the base-model CFG cases
    "n0": "blurry, low quality, distorted, watermark",
}
QSET = {
    "q0": PROMPTS["p0"],
    "q1": PROMPTS["p1"],
    "q2": PROMPTS["p2"],
    "q3": "Close-up portrait of an elderly fisherman with weathered hands holding a rope, natural light.",
    "q4": "A neon sign on a brick wall that says \"CORTIQ\", night, rain, reflections.",
    "q5": "Aerial view of a winding river through autumn forest, morning fog, ultra detailed.",
}
RES = {"r512": (512, 512), "r1024": (1024, 1024), "r400x592": (400, 592)}  # (H, W)
SEED = 42
SPEC_SIGMAS_TURBO_N8 = [1.0, 0.954545438, 0.899999976, 0.833333313, 0.75,
                        0.642857134, 0.5, 0.300000012, 0.0]
TE_TAP_LAYERS = [0, 17, 34]
DIT_LAYER_TAPS = [0, 1, 14, 29]

# Recipes. A "case" is (res, prompt_key, recipe_key).
RECIPES = {
    # Turbo: 8 steps, no CFG
    "t8": dict(steps=8, guidance=0.0, neg=None, cfg_norm=False, cfg_trunc=1.0),
    # Base: short CFG runs for tensor-level parity
    "c3": dict(steps=3, guidance=4.0, neg="n0", cfg_norm=False, cfg_trunc=1.0),
    # Base: CFG + normalization (clip to 1.0*||pos||) + truncation after t_norm > 0.1
    # (3 steps shift 6: t_norm = [0, .077, .25] -> the last step runs without CFG)
    "c3nt": dict(steps=3, guidance=4.0, neg="n0", cfg_norm=True, cfg_trunc=0.1),
    # Base: the full recipe, 28 steps, guidance 4, negative prompt
    "c28": dict(steps=28, guidance=4.0, neg="n0", cfg_norm=False, cfg_trunc=1.0),
}

# ------------------------------------------------------------------ utils


def f32(t):
    return t.detach().to(torch.float32).cpu().contiguous().clone()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Out:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.manifest_path = os.path.join(root, "manifest.json")

    def _rw(self, fn):
        # manifest is shared by concurrent CPU and GPU phases: re-read, update, write.
        import fcntl
        with open(self.manifest_path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            m = {}
            if os.path.exists(self.manifest_path):
                with open(self.manifest_path) as f:
                    m = json.load(f)
            m.setdefault("files", {})
            m.setdefault("checks", {})
            fn(m)
            tmp = self.manifest_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(m, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.manifest_path)

    def save(self, name, tensors, meta=None):
        path = os.path.join(self.root, name + ".safetensors")
        md = {k: json.dumps(v, ensure_ascii=False) for k, v in (meta or {}).items()}
        save_file({k: v.contiguous() for k, v in tensors.items()}, path, metadata=md)
        ent = {"sha256": sha256(path), "tensors": {k: [str(v.dtype), list(v.shape)] for k, v in tensors.items()},
               "meta": meta or {}}
        self._rw(lambda m: m["files"].__setitem__(name + ".safetensors", ent))
        print("wrote", path, flush=True)
        return path

    def raw_f32(self, name, t):
        path = os.path.join(self.root, name)
        t.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4").tofile(path)
        ent = {"sha256": sha256(path), "shape": list(t.shape), "dtype": "f32le"}
        self._rw(lambda m: m["files"].__setitem__(name, ent))

    def check(self, key, ok, detail):
        self._rw(lambda m: m["checks"].__setitem__(key, {"ok": bool(ok), "detail": detail}))
        print(("CHECK OK  " if ok else "CHECK FAIL"), key, detail, flush=True)

    def put(self, key, val):
        self._rw(lambda m: m.__setitem__(key, val))

    def load(self, name):
        return load_file(os.path.join(self.root, name + ".safetensors"))


def env_info(args):
    import diffusers
    import transformers
    return {
        "torch": torch.__version__, "diffusers": diffusers.__version__,
        "transformers": transformers.__version__, "python": sys.version.split()[0],
        "cuda": torch.version.cuda, "root": args.root, "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_threads": torch.get_num_threads(),
    }


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

# ------------------------------------------------------------------ loaders


def load_tok(root):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(root, subfolder="tokenizer")


def load_te(root, dtype, device):
    from transformers import AutoModel
    return AutoModel.from_pretrained(root, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()


def load_dit(root, dtype, device):
    from diffusers import ZImageTransformer2DModel
    return ZImageTransformer2DModel.from_pretrained(root, subfolder="transformer", torch_dtype=dtype).to(device).eval()


def load_vae(root, dtype, device):
    from diffusers import AutoencoderKL
    return AutoencoderKL.from_pretrained(root, subfolder="vae", torch_dtype=dtype).to(device).eval()


def load_sched(root):
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_pretrained(root, subfolder="scheduler")


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()

# ------------------------------------------------------------------ tokenizer


def templated(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def tokenize(tok, prompt):
    s = templated(tok, prompt)
    enc = tok([s], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    mask = enc.attention_mask.bool()
    L = int(mask[0].sum())
    return s, enc.input_ids, mask, L


def cmd_tok(args, out):
    tok = load_tok(args.root)
    T, meta = {}, {}
    long_prompt = " ".join(["word%d" % i for i in range(700)])   # truncation case
    for k, p in {**PROMPTS, **QSET, "long": long_prompt}.items():
        s, ids, mask, L = tokenize(tok, p)
        assert bool(mask[0, :L].all()) and not bool(mask[0, L:].any()), "padding is not on the right"
        T["ids_" + k] = ids[0, :L].to(torch.int64).clone()
        meta[k] = {"prompt": p, "templated": s, "L": L, "L_p": (L + 31) // 32 * 32}
    out.check("tok.p0_len22", meta["p0"]["L"] == 22, meta["p0"]["L"])
    out.check("tok.p3_len8", meta["p3"]["L"] == 8, meta["p3"]["L"])
    out.check("tok.long_len512", meta["long"]["L"] == 512, meta["long"]["L"])
    out.check("tok.template_head", T["ids_p0"][:3].tolist() == [151644, 872, 198], T["ids_p0"][:3].tolist())
    out.check("tok.template_tail", T["ids_p0"][-5:].tolist() == [151645, 198, 151644, 77091, 198],
              T["ids_p0"][-5:].tolist())
    out.save("tok", T, {"prompts": meta})

# ------------------------------------------------------------------ text encoder


def te_forward(te, ids, mask, L, dev):
    taps, hooks = {}, []
    for li in TE_TAP_LAYERS:
        def mk(li):
            def f(m, a, o):
                taps[li] = (o[0] if isinstance(o, (tuple, list)) else o).detach()
            return f
        hooks.append(te.layers[li].register_forward_hook(mk(li)))
    hooks.append(te.embed_tokens.register_forward_hook(lambda m, a, o: taps.__setitem__("emb", o.detach())))
    with torch.no_grad():
        o = te(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
    for h in hooks:
        h.remove()
    hs = o.hidden_states
    r = {"h_m2": f32(hs[-2][0, :L]), "h_last": f32(hs[-1][0, :L]), "h_emb": f32(taps["emb"][0, :L])}
    for li in TE_TAP_LAYERS:
        r["h_l%d" % li] = f32(taps[li][0, :L])
    return r, len(hs)


def cmd_te(args, out, prec):
    tok = load_tok(args.root)
    dev = torch.device("cpu") if prec == "fp32" else torch.device("cuda")
    if prec == "fp32":
        strict_fp32()
    te = load_te(args.root, torch.float32 if prec == "fp32" else torch.bfloat16, dev)
    keys = list(PROMPTS) + (list(QSET) if args.qset else [])
    for k in keys:
        p = PROMPTS.get(k, QSET.get(k))
        s, ids, mask, L = tokenize(tok, p)
        t0 = time.time()
        r, nhs = te_forward(te, ids, mask, L, dev)
        r["ids"] = ids[0, :L].to(torch.int64).clone()
        same = torch.equal(r["h_m2"], r["h_l34"])
        out.check("te.%s.%s.hm2_is_layer34" % (k, prec), same,
                  {"n_hidden_states": nhs, "maxdiff": float((r["h_m2"] - r["h_l34"]).abs().max())})
        out.save("te_%s_%s" % (k, prec), r, {"L": L, "n_hidden_states": nhs, "wall_s": time.time() - t0})
    del te

# ------------------------------------------------------------------ noise + schedule


def noise_for(res, seed=SEED):
    H, W = RES[res]
    g = torch.Generator("cpu").manual_seed(seed)
    return torch.randn((1, 16, H // 8, W // 8), generator=g, dtype=torch.float32)


def cmd_sweep(args, out, prec):
    """Seed sweep for the E2E quality gate: one prompt, several injected noises,
    fp32 (CPU) or bf16 (CUDA) images. Single-image PSNR of an 8-step distilled
    trajectory is chaotic, so the gate compares distributions over seeds."""
    dev = torch.device("cpu") if prec == "fp32" else torch.device("cuda")
    dtype = torch.float32 if prec == "fp32" else torch.bfloat16
    if prec == "fp32":
        strict_fp32()
    tr = load_dit(args.root, dtype, dev)
    vae32 = load_vae(args.root, torch.float32, torch.device("cpu"))
    ted = te_dir(args)
    rk = "t8" if args.model == "turbo" else "c3"
    rc = RECIPES[rk]
    res, pk = "r512", args.sweep_prompt
    cap = ted.load("te_%s_%s" % (pk, prec))["h_m2"]
    ncap = ted.load("te_%s_%s" % (rc["neg"], prec))["h_m2"] if rc["neg"] else None
    for s in [int(x) for x in args.seeds.split(",")]:
        n = noise_for(res, s)
        out.raw_f32("noise_%s_s%d.f32" % (res, s), n)
        _, rec = manual_run(tr, args.root, cap, ncap, n, dtype, dev, rc)
        _, img = decode(vae32, rec["lat"][-1].unsqueeze(0), torch.device("cpu"))
        save_png(os.path.join(out.root, "sweep_%s_%s_%s_s%d_%s.png" % (res, pk, rk, s, prec)), to_u8(img))
        print("sweep seed", s, prec, flush=True)
    del tr, vae32


def schedule(root, steps, dev):
    sch = load_sched(root)
    sig = torch.linspace(1.0, 1.0 / steps, steps).tolist()   # get_default_z_image_sigmas
    sch.set_timesteps(sigmas=sig, device=dev)
    sch.set_begin_index(0)
    return sch

# ------------------------------------------------------------------ full run (manual loop == pipeline math)


def manual_run(tr, root, cap, ncap, noise, dtype, dev, rc):
    """Replicates ZImagePipeline.__call__ (diffusers 0.40, pipeline_z_image.py) for batch 1:
    guidance 0 -> one forward; guidance > 0 -> batched [pos, neg] forward, pred = pos + g(pos-neg),
    optional whole-tensor norm clip, cfg truncation on t_norm = (1000 - t)/1000."""
    sch = schedule(root, rc["steps"], dev)
    lat = noise.to(dev, torch.float32)
    g = rc["guidance"]
    do_cfg = g > 0
    t_norms = ((1000 - sch.timesteps.float()) / 1000).tolist() if (do_cfg and rc["cfg_trunc"] <= 1) else None
    rec = {"lat": [], "v": [], "vpos": [], "vneg": [], "cfg_on": [], "t_model": []}
    for i, t in enumerate(sch.timesteps):
        tm = (1000 - t.expand(1)) / 1000
        rec["t_model"].append(float(tm[0]))
        cg = g
        if t_norms is not None and t_norms[i] > rc["cfg_trunc"]:
            cg = 0.0
        apply_cfg = do_cfg and cg > 0
        x = lat.to(dtype)
        if apply_cfg:
            xin = x.repeat(2, 1, 1, 1).unsqueeze(2)
            caps = [cap.to(dev, dtype), ncap.to(dev, dtype)]
            tin = tm.repeat(2)
        else:
            xin = x.unsqueeze(2)
            caps = [cap.to(dev, dtype)]
            tin = tm
        with torch.no_grad():
            out = tr(list(xin.unbind(0)), tin, caps, return_dict=False)[0]
        if apply_cfg:
            pos = out[0].float()
            neg = out[1].float()
            pred = pos + cg * (pos - neg)
            if rc["cfg_norm"] and float(rc["cfg_norm"]) > 0.0:
                opn = torch.linalg.vector_norm(pos)
                npn = torch.linalg.vector_norm(pred)
                mx = opn * float(rc["cfg_norm"])
                if npn > mx:
                    pred = pred * (mx / npn)
            rec["vpos"].append(f32(pos.squeeze(1)))
            rec["vneg"].append(f32(neg.squeeze(1)))
        else:
            pred = out[0].float()
        pred = pred.squeeze(1).unsqueeze(0)               # [1,16,h,w]
        rec["v"].append(f32(pred[0]))
        rec["cfg_on"].append(1 if apply_cfg else 0)
        lat = sch.step(-pred.to(torch.float32), t, lat, return_dict=False)[0]
        rec["lat"].append(f32(lat[0]))
        print("  step %d/%d cfg=%d" % (i + 1, rc["steps"], int(apply_cfg)), flush=True)
    return sch, rec


def decode(vae, lat, dev, taps=None):
    z = lat.to(dev, vae.dtype) / vae.config.scaling_factor + vae.config.shift_factor
    hooks = []
    if taps is not None:
        hooks.append(vae.decoder.mid_block.register_forward_hook(lambda m, a, o: taps.__setitem__("mid", f32(o[0]))))
        for i, ub in enumerate(vae.decoder.up_blocks):
            hooks.append(ub.register_forward_hook(
                (lambda i: lambda m, a, o: taps.__setitem__("up%d" % i, f32(o[0])))(i)))
    with torch.no_grad():
        img = vae.decode(z, return_dict=False)[0]
    for h in hooks:
        h.remove()
    return f32(z[0]), f32(img[0])


def to_u8(img):  # img [3,H,W] raw decoder output
    x = (img / 2 + 0.5).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255).round().astype("uint8")


def save_png(path, u8):
    from PIL import Image
    Image.fromarray(u8).save(path)


def te_dir(args):
    # The text encoder is byte-identical in both repos: the base model reuses the Turbo TE oracles.
    return Out(os.path.join(OUT, "turbo"))


def cmd_run(args, out, prec, cases):
    for res in sorted({c[0] for c in cases}):
        n = noise_for(res)
        out.save("noise_%s_s%d" % (res, SEED), {"noise": n})
        out.raw_f32("noise_%s_s%d.f32" % (res, SEED), n)
    dev = torch.device("cpu") if prec == "fp32" else torch.device("cuda")
    dtype = torch.float32 if prec == "fp32" else torch.bfloat16
    if prec == "fp32":
        strict_fp32()
    tr = load_dit(args.root, dtype, dev)
    vae32 = load_vae(args.root, torch.float32, torch.device("cpu"))   # fp32 CPU decode of every final latent
    ted = te_dir(args)
    for res, pk, rk in cases:
        rc = RECIPES[rk]
        cap = ted.load("te_%s_%s" % (pk, prec))["h_m2"]
        ncap = ted.load("te_%s_%s" % (rc["neg"], prec))["h_m2"] if rc["neg"] else None
        n = noise_for(res)
        t0 = time.time()
        sch, rec = manual_run(tr, args.root, cap, ncap, n, dtype, dev, rc)
        wall = time.time() - t0
        sig = [float(s) for s in sch.sigmas]
        if args.model == "turbo" and rc["steps"] == 8:
            ok = len(sig) == 9 and all(abs(a - b) <= 1e-7 * max(1.0, abs(b)) for a, b in zip(sig, SPEC_SIGMAS_TURBO_N8))
            out.check("sched.%s.%s.%s.%s" % (res, pk, rk, prec), ok, sig)
        z, img = decode(vae32, rec["lat"][-1].unsqueeze(0), torch.device("cpu"))
        T = {"noise": n[0], "sigmas": torch.tensor(sig, dtype=torch.float32),
             "timesteps": f32(sch.timesteps), "t_model": torch.tensor(rec["t_model"], dtype=torch.float32),
             "cfg_on": torch.tensor(rec["cfg_on"], dtype=torch.int64),
             "img": img, "img_u8": torch.from_numpy(to_u8(img))}
        for i, v in enumerate(rec["v"]):
            T["v_%d" % i] = v
        for i, v in enumerate(rec["vpos"]):
            T["vpos_%d" % i] = v
        for i, v in enumerate(rec["vneg"]):
            T["vneg_%d" % i] = v
        for i, l in enumerate(rec["lat"]):
            T["lat_%d" % (i + 1)] = l
        name = "run_%s_%s_%s_%s" % (res, pk, rk, prec)
        meta = {"H": RES[res][0], "W": RES[res][1], "wall_s": wall, "recipe": rc, "prompt": PROMPTS[pk],
                "negative": PROMPTS[rc["neg"]] if rc["neg"] else None}
        out.save(name, T, meta)
        save_png(os.path.join(out.root, name + ".png"), to_u8(img))
        print(name, "wall %.1f s" % wall, flush=True)
    del tr, vae32

# ------------------------------------------------------------------ single forward with taps


def dit_taps(tr, T, stats):
    hs = []

    def fwd(name, pick=None):
        def f(m, a, o):
            x = o if pick is None else pick(o)
            T[name] = f32(x[0] if x.dim() >= 2 and x.shape[0] == 1 else x)
        return f

    def pre(name, idx=0, kw=None):
        def f(m, a, k):
            x = k[kw] if kw is not None else a[idx]
            if torch.is_complex(x):
                T[name + "_cos"] = f32(x.real[0])
                T[name + "_sin"] = f32(x.imag[0])
            else:
                T[name] = f32(x[0])
        return f

    def amax(key):
        def f(m, a, o=None):
            x = a[0] if o is None else (o[0] if isinstance(o, (tuple, list)) else o)
            stats[key] = float(x.detach().float().abs().max())
        return f

    def R(h):
        hs.append(h)

    def PH(mod, name, idx=0):
        # block forward(x, attn_mask, freqs_cis, adaln_input, ...) is called positionally
        R(mod.register_forward_pre_hook(pre(name, idx), with_kwargs=True))

    R(tr.t_embedder.register_forward_hook(fwd("temb")))
    R(tr.all_x_embedder["2-1"].register_forward_hook(fwd("x_embed_raw")))
    PH(tr.noise_refiner[0], "x_seq", 0)
    PH(tr.noise_refiner[0], "rope_img", 2)
    R(tr.noise_refiner[0].adaLN_modulation.register_forward_hook(fwd("mod_nr0")))
    R(tr.cap_embedder.register_forward_hook(fwd("cap_embed_raw")))
    PH(tr.context_refiner[0], "cap_seq", 0)
    PH(tr.context_refiner[0], "rope_cap", 2)
    PH(tr.layers[0], "u_in", 0)
    PH(tr.layers[0], "rope_joint", 2)
    R(tr.layers[0].adaLN_modulation.register_forward_hook(fwd("mod_l0")))
    R(tr.layers[29].adaLN_modulation.register_forward_hook(fwd("mod_l29")))
    for i in range(len(tr.noise_refiner)):
        R(tr.noise_refiner[i].register_forward_hook(fwd("nr%d_out" % i)))
    for i in range(len(tr.context_refiner)):
        R(tr.context_refiner[i].register_forward_hook(fwd("cr%d_out" % i)))
    for i in DIT_LAYER_TAPS:
        R(tr.layers[i].register_forward_hook(fwd("l%d_out" % i)))
    b0 = tr.layers[0]
    PH(b0.attention.to_q, "l0_attn_in", 0)
    R(b0.attention.to_q.register_forward_hook(fwd("l0_q")))
    R(b0.attention.norm_q.register_forward_hook(fwd("l0_qn")))
    R(b0.attention.norm_k.register_forward_hook(fwd("l0_kn")))
    R(b0.attention.register_forward_hook(fwd("l0_attn_out")))
    PH(b0.feed_forward, "l0_ffn_in", 0)
    PH(b0.feed_forward.w2, "l0_ffn_h", 0)
    R(b0.feed_forward.register_forward_hook(fwd("l0_ffn_out")))
    fl = tr.all_final_layer["2-1"]
    R(fl.adaLN_modulation.register_forward_hook(fwd("final_mod")))
    R(fl.register_forward_hook(fwd("final_out")))
    for nm, blocks in (("nr", tr.noise_refiner), ("l", tr.layers), ("cr", tr.context_refiner)):
        for i, b in enumerate(blocks):
            R(b.attention.to_q.register_forward_pre_hook(amax("%s%d.attn_in" % (nm, i))))
            R(b.feed_forward.w2.register_forward_pre_hook(amax("%s%d.ffn_h" % (nm, i))))
            R(b.attention.to_out[0].register_forward_pre_hook(amax("%s%d.attn_o_in" % (nm, i))))
            R(b.register_forward_hook(lambda m, a, o, k="%s%d.resid" % (nm, i): stats.__setitem__(
                k, float(o.detach().float().abs().max()))))
    return hs


def cmd_dit(args, out, prec, cases):
    """cases: (res, prompt_key, recipe_key, step_index). Inputs come from the fp32 run, so both
    precisions see identical x_in / t / caption features."""
    dev = torch.device("cpu") if prec == "fp32" else torch.device("cuda")
    dtype = torch.float32 if prec == "fp32" else torch.bfloat16
    if prec == "fp32":
        strict_fp32()
    tr = load_dit(args.root, dtype, dev)
    ted = te_dir(args)
    for res, pk, rk, i in cases:
        run = out.load("run_%s_%s_%s_fp32" % (res, pk, rk))
        x_in = run["noise"] if i == 0 else run["lat_%d" % i]
        tm = run["t_model"][i:i + 1].clone()
        cap = ted.load("te_%s_fp32" % pk)["h_m2"]
        T, stats = {}, {}
        hs = dit_taps(tr, T, stats)
        t0 = time.time()
        with torch.no_grad():
            v = tr([x_in.unsqueeze(1).to(dev, dtype)], tm.to(dev), [cap.to(dev, dtype)], return_dict=False)[0]
        wall = time.time() - t0
        for h in hs:
            h.remove()
        T["v"] = f32(v[0].squeeze(1))
        T["x_in"] = x_in.clone()
        T["t_model"] = tm
        T["cap"] = cap.clone()
        H, W = RES[res]
        n_img = (H // 16) * (W // 16)
        L = cap.shape[0]
        meta = {"H": H, "W": W, "n_img": n_img, "n_img_p": (n_img + 31) // 32 * 32, "L": L,
                "L_p": (L + 31) // 32 * 32, "order": "img_then_cap", "step": i, "wall_s": wall}
        meta["S"] = meta["n_img_p"] + meta["L_p"]
        out.save("dit_%s_%s_%s_i%d_%s" % (res, pk, rk, i, prec), T, meta)
        with open(os.path.join(out.root, "stats_%s_%s_%s_i%d_%s.json" % (res, pk, rk, i, prec)), "w") as f:
            json.dump(stats, f, indent=1)
        ref = run["vpos_%d" % i] if ("vpos_%d" % i) in run else run["v_%d" % i]
        d = float((T["v"] - ref).norm() / ref.norm())
        out.check("dit.%s.%s.%s.i%d.%s.vs_run" % (res, pk, rk, i, prec), d < (1e-5 if prec == "fp32" else 5e-2), d)
    del tr

# ------------------------------------------------------------------ VAE


def cmd_vae(args, out, cases):
    strict_fp32()
    dev = torch.device("cpu")
    vae = load_vae(args.root, torch.float32, dev)
    for res, pk, rk in cases:
        lat = out.load("run_%s_%s_%s_fp32" % (res, pk, rk))["lat_%d" % RECIPES[rk]["steps"]]
        taps = {}
        z_in, img = decode(vae, lat.unsqueeze(0), dev, taps)
        T = {"z": lat.clone(), "z_in": z_in, "img": img}
        T.update(taps)
        out.save("vae_%s_%s_%s" % (res, pk, rk), T,
                 {"scaling": vae.config.scaling_factor, "shift": vae.config.shift_factor})

# ------------------------------------------------------------------ bf16 pipeline: full image + baseline timing


def build_pipe(args, dev):
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(args.root, torch_dtype=torch.bfloat16)
    mode = "all_resident"
    try:
        pipe.to(dev)
    except torch.cuda.OutOfMemoryError:
        pipe.to("cpu")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        mode = "model_cpu_offload"
    return pipe, mode


def run_pipe(pipe, st, kw):
    """One pipeline call; on a CUDA OOM switch the pipe to model_cpu_offload once and retry."""
    try:
        return pipe(**kw)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pipe.to("cpu")
        pipe.enable_model_cpu_offload()
        st["mode"] = "model_cpu_offload(after OOM)"
        kw["latents"] = kw["latents"].to("cpu")
        return pipe(**kw)


def pipe_kwargs(args, pk, res, rk):
    H, W = RES[res]
    rc = RECIPES[rk]
    kw = dict(prompt=PROMPTS[pk], height=H, width=W, num_inference_steps=rc["steps"], guidance_scale=rc["guidance"],
              cfg_normalization=rc["cfg_norm"], cfg_truncation=rc["cfg_trunc"],
              latents=noise_for(res).to("cuda"), max_sequence_length=512)
    if rc["neg"]:
        kw["negative_prompt"] = PROMPTS[rc["neg"]]
    return kw


def cmd_full(args, out):
    """The real bf16 pipeline on CUDA with the injected latent: the image a diffusers user gets."""
    dev = torch.device("cuda")
    pipe, mode = build_pipe(args, dev)
    cases = [("r1024", "p0", "t8"), ("r512", "p0", "t8"), ("r1024", "p1", "t8")] if args.model == "turbo" else \
        [("r1024", "p0", "c28"), ("r512", "p0", "c3"), ("r512", "p0", "c3nt")]
    for res, pk, rk in cases:
        t0 = time.time()
        st = {"mode": mode}
        kw = pipe_kwargs(args, pk, res, rk)
        kw["output_type"] = "latent"
        lat = run_pipe(pipe, st, kw).images
        mode = st["mode"]
        sync(dev)
        wall = time.time() - t0
        vae32 = load_vae(args.root, torch.float32, torch.device("cpu"))
        _, img = decode(vae32, lat.float().cpu(), torch.device("cpu"))
        name = "pipe_%s_%s_%s_bf16" % (res, pk, rk)
        out.save(name, {"lat": f32(lat[0]), "img": img, "img_u8": torch.from_numpy(to_u8(img))},
                 {"mode": mode, "wall_s": wall, "recipe": RECIPES[rk]})
        save_png(os.path.join(out.root, name + ".png"), to_u8(img))
        del vae32


def cmd_bench(args, out):
    dev = torch.device("cuda")
    pipe, mode = build_pipe(args, dev)
    nfe = {"n": 0}
    ev = {"te": [], "dit": [], "vae": []}

    def timer(key):
        state = {}

        def pre(m, a):
            sync(dev)
            state["t"] = time.perf_counter()

        def post(m, a, o):
            sync(dev)
            ev[key].append(time.perf_counter() - state["t"])
            if key == "dit":
                nfe["n"] += 1
        return pre, post
    for key, mod in (("te", pipe.text_encoder), ("dit", pipe.transformer)):
        p, q = timer(key)
        mod.register_forward_pre_hook(p)
        mod.register_forward_hook(q)
    orig_decode = pipe.vae.decode

    def timed_decode(*a, **k):
        sync(dev)
        t = time.perf_counter()
        r = orig_decode(*a, **k)
        sync(dev)
        ev["vae"].append(time.perf_counter() - t)
        return r
    pipe.vae.decode = timed_decode
    results = {"mode": mode, "env": env_info(args), "runs": {}}
    rk = "t8" if args.model == "turbo" else "c28"
    for res in ("r512", "r1024"):
        for pk in ("p0", "p1"):
            key = "%s_%s_%s" % (res, pk, rk)
            walls = []
            for rep in range(args.reps + 1):              # rep 0 = warm-up
                for v in ev.values():
                    v.clear()
                nfe["n"] = 0
                torch.cuda.reset_peak_memory_stats()
                sync(dev)
                t0 = time.perf_counter()
                st = {"mode": mode}
                img = run_pipe(pipe, st, pipe_kwargs(args, pk, res, rk)).images[0]
                if st["mode"] != mode:
                    mode = results["mode"] = st["mode"]
                    continue
                sync(dev)
                wall = time.perf_counter() - t0
                if rep == 0:
                    img.save(os.path.join(out.root, "bench_%s.png" % key))
                    continue
                d = sorted(ev["dit"])
                walls.append({"wall": wall, "te": sum(ev["te"]), "dit_steps": list(ev["dit"]),
                              "dit_median": d[len(d) // 2], "dit_total": sum(ev["dit"]),
                              "vae": sum(ev["vae"]), "nfe": nfe["n"],
                              "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
                print(key, "rep", rep, json.dumps({k: v for k, v in walls[-1].items() if k != "dit_steps"}), flush=True)
            results["runs"][key] = walls
            ws = sorted(w["wall"] for w in walls)
            ds = sorted(w["dit_median"] for w in walls)
            if not walls:
                continue
            results["runs"][key + "_summary"] = {"wall_median": ws[len(ws) // 2], "dit_step_median": ds[len(ds) // 2],
                                                 "mode": mode}
    gname = torch.cuda.get_device_name(0).replace(" ", "_")
    with open(os.path.join(out.root, "bench_%s.json" % gname), "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results["runs"].items() if k.endswith("summary")}, indent=1), flush=True)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["turbo", "base"], required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--qset", action="store_true", help="also TE-encode the WP4 quality prompts")
    ap.add_argument("--seeds", default="1,2,3,4,5,6", help="sweep32/sweep16 noise seeds")
    ap.add_argument("--sweep-prompt", default="p0")
    ap.add_argument("what", nargs="+",
                    choices=["tok", "te32", "te16", "run32", "run16", "dit32", "dit16", "vae", "full", "bench", "qrun",
                             "sweep32", "sweep16"])
    args = ap.parse_args()
    args.root = ROOTS[args.model]
    torch.set_num_threads(args.threads)
    out = Out(os.path.join(OUT, args.model))
    out.put("env_" + ("gpu" if any(w in ("te16", "run16", "dit16", "full", "bench") for w in args.what) else "cpu"),
            env_info(args))
    if args.model == "turbo":
        run32 = [("r512", "p0", "t8"), ("r400x592", "p0", "t8"), ("r512", "p1", "t8"), ("r1024", "p0", "t8")]
        run16 = [("r512", "p0", "t8"), ("r512", "p1", "t8"), ("r512", "p2", "t8"), ("r1024", "p0", "t8"),
                 ("r1024", "p1", "t8"), ("r400x592", "p0", "t8")]
        dits = [("r512", "p0", "t8", 0), ("r512", "p0", "t8", 5), ("r512", "p1", "t8", 0),
                ("r400x592", "p0", "t8", 3), ("r1024", "p0", "t8", 0)]
        vaes = [("r512", "p0", "t8"), ("r1024", "p0", "t8"), ("r400x592", "p0", "t8")]
    else:
        run32 = [("r512", "p0", "c3"), ("r512", "p0", "c3nt"), ("r400x592", "p1", "c3")]
        run16 = [("r512", "p0", "c3"), ("r512", "p0", "c3nt"), ("r1024", "p0", "c28")]
        dits = [("r512", "p0", "c3", 0), ("r512", "p0", "c3", 2)]
        vaes = []
    for w in args.what:
        if w == "tok":
            cmd_tok(args, out)
        elif w == "te32":
            cmd_te(args, out, "fp32")
        elif w == "te16":
            cmd_te(args, out, "bf16")
        elif w == "run32":
            cmd_run(args, out, "fp32", run32)
        elif w == "run16":
            cmd_run(args, out, "bf16", run16)
        elif w == "dit32":
            cmd_dit(args, out, "fp32", dits)
        elif w == "dit16":
            cmd_dit(args, out, "bf16", dits)
        elif w == "vae":
            cmd_vae(args, out, vaes)
        elif w == "full":
            cmd_full(args, out)
        elif w == "bench":
            cmd_bench(args, out)
        elif w == "sweep32":
            cmd_sweep(args, out, "fp32")
        elif w == "sweep16":
            cmd_sweep(args, out, "bf16")
        elif w == "qrun":
            qs = [("r512", q, "t8") for q in QSET] if args.model == "turbo" else [("r512", q, "c28") for q in QSET]
            for q in QSET:
                PROMPTS[q] = QSET[q]
            cmd_run(args, out, "bf16", qs)


if __name__ == "__main__":
    main()
