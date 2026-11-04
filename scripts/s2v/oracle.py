#!/usr/bin/env python3
"""Reference dumps for the Kandinsky S2V port, one stage per process.

The whole pipeline does not fit the pod at once (DiT 42.7 GB, Thinker 17.8 GB,
62 GB cgroup, 32 GB card), so every stage loads only what it needs and leaves a
safetensors file the next stage and `cortiq s2v-check` read.

    oracle.py enc   --root HF --image I --audio A --chunks N --out enc.safetensors
    oracle.py dit   --root HF --enc enc.safetensors --lh 16 --lw 16 --lt 8 --dtype f32 --out dit.safetensors
    oracle.py run   --root HF --enc enc.safetensors --chunks N --seed 5 --dtype bf16 --out run.safetensors
    oracle.py vae   --root HF --latents run.safetensors --out frames.safetensors
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_meta_dit(root):
    from diffusers.models import KandinskyS2VTransformer3DModel as M

    cfg = json.load(open(f"{root}/transformer/config.json"))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = M(**cfg)
    return model, cfg


class LazyDit:
    """Holds every module on `meta` and materializes one block at a time."""

    def __init__(self, root, dtype, device):
        self.model, self.cfg = load_meta_dit(root)
        self.path = f"{root}/transformer/diffusion_pytorch_model.safetensors"
        self.f = safe_open(self.path, "pt", device="cpu")
        self.dtype, self.device = dtype, device
        self.taps = {}
        self.tap_on = False
        self.load_top()
        for name, mod in self.blocks():
            mod.register_forward_pre_hook(self._pre(name))
            mod.register_forward_hook(self._post(name))

    def blocks(self):
        for prefix in ("text_transformer_blocks", "audio_transformer_blocks", "visual_transformer_blocks"):
            for i, b in enumerate(getattr(self.model, prefix)):
                yield f"{prefix}.{i}", b

    def _dt(self, name, t):
        # The checkpoint keeps norms and the time embedding in fp32; the reference
        # never flattens those, so neither does the bf16 arm.
        keep = any(k in name for k in ("norm", "time_embeddings", "modulation"))
        if t.dtype == torch.float32 and keep:
            return t
        return t.to(self.dtype)

    def _load(self, module, prefix):
        keys = [k for k in self.f.keys() if k.startswith(prefix + ".")]
        for k in keys:
            rel = k[len(prefix) + 1 :]
            mod = module
            parts = rel.split(".")
            for p in parts[:-1]:
                mod = getattr(mod, p)
            t = self._dt(k, self.f.get_tensor(k)).to(self.device)
            setattr(mod, parts[-1], torch.nn.Parameter(t, requires_grad=False))

    def _offload(self, module):
        for m in module.modules():
            for n, p in list(m._parameters.items()):
                if p is not None:
                    m._parameters[n] = torch.nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)

    def _pre(self, name):
        def hook(mod, args):
            self._load(mod, name)

        return hook

    def _post(self, name):
        def hook(mod, args, out):
            if self.tap_on:
                self.taps[name] = out.detach().float().cpu()
            self._offload(mod)
            return out

        return hook

    def load_top(self):
        tops = [k for k in self.f.keys() if "transformer_blocks" not in k]
        for k in tops:
            mod = self.model
            parts = k.split(".")
            for p in parts[:-1]:
                mod = getattr(mod, p)
            t = self._dt(k, self.f.get_tensor(k)).to(self.device)
            setattr(mod, parts[-1], torch.nn.Parameter(t, requires_grad=False))

def stage_enc(a):
    import librosa
    from PIL import Image
    from torchvision import transforms
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from diffusers.models import AutoencoderKLHunyuanVideo
    from diffusers.modular_pipelines.kandinsky_s2v.modular_pipeline import (
        DEFAULT_PROMPT,
        SYSTEM_PROMPT,
        select_resolution,
    )

    root = a.root
    out = {}
    prompt = a.prompt or DEFAULT_PROMPT
    # chunks → seconds of speech, as the input step computes them.
    chunk_seconds = ((32 - 1) * 4 + 1) / 25
    target = a.chunks * chunk_seconds
    wav, _ = librosa.load(a.audio, sr=16000, offset=0.0, duration=None)
    pad = int(target * 16000) - len(wav)
    if pad > 0:
        wav = np.concatenate([wav, np.zeros(pad, dtype=wav.dtype)])
    wav = wav[: int(round(target * 16000))]
    out["wav"] = torch.from_numpy(wav.copy())

    proc = Qwen2_5OmniProcessor.from_pretrained(f"{root}/tokenizer")
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "audio", "audio": "x"}]},
    ]
    text = proc.apply_chat_template([conv], add_generation_prompt=False, tokenize=False)
    inputs = proc(text=text, audio=[wav], return_tensors="pt", padding=True, use_audio_in_video=False)
    ids = inputs["input_ids"]
    out["input_ids"] = ids[0].to(torch.int32)
    feat_len = inputs["feature_attention_mask"].sum(-1)
    out["mel"] = inputs["input_features"][0, :, : int(feat_len[0])].float()
    out["feat_len"] = feat_len.to(torch.int32)

    im_start = proc.tokenizer.convert_tokens_to_ids("<|im_start|>")
    astart = 151647
    role = len(proc.tokenizer("user\n", add_special_tokens=False).input_ids)
    pos = torch.arange(ids.shape[1])
    last = pos[ids[0] == im_start].max()
    start = int(last) + 1 + role
    end = int(pos[ids[0] == astart].min())
    out["span"] = torch.tensor([start, end], dtype=torch.int32)

    dev = a.device
    for tag, dt in (("bf16", torch.bfloat16), ("f32", torch.float32)):
        if tag == "f32" and a.skip_f32:
            continue
        d = "cpu" if tag == "f32" else dev
        t0 = time.time()
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            f"{root}/text_encoder", dtype=dt, attn_implementation="sdpa"
        ).to(d).eval()
        print(f"thinker {tag} load {time.time() - t0:.1f}s", flush=True)
        cap = {}
        tower = thinker.audio_tower if hasattr(thinker, "audio_tower") else thinker.model.audio_tower
        h = tower.register_forward_hook(
            lambda _m, _i, o: cap.__setitem__("audio", getattr(o, "last_hidden_state", o))
        )
        # Tap the per-layer states only in the exact arm.
        layer_caps = {}
        hooks = [h]
        if tag == "f32":
            for i, lyr in enumerate(tower.layers):
                if i in (0, 31):
                    hooks.append(
                        lyr.register_forward_hook(
                            lambda _m, _i, o, i=i: layer_caps.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().float().cpu())
                        )
                    )
        with torch.no_grad():
            t0 = time.time()
            inp = inputs.to(d).to(dt)
            o = thinker(**inp, output_hidden_states=True, return_dict=True)
            print(f"thinker {tag} forward {time.time() - t0:.1f}s", flush=True)
        for hk in hooks:
            hk.remove()
        out[f"text_{tag}"] = o.hidden_states[-1][0, start:end].float().cpu()
        out[f"audio_{tag}"] = cap["audio"].float().cpu()
        for i, v in layer_caps.items():
            out[f"audio_layer{i}_{tag}"] = v
        if tag == "f32":
            out["text_h0_f32"] = o.hidden_states[0][0, : end].float().cpu()
            out["text_h1_f32"] = o.hidden_states[1][0, : end].float().cpu()
        del thinker, o, inp
        torch.cuda.empty_cache()

    clip_tok = CLIPTokenizer.from_pretrained(f"{root}/tokenizer_2")
    clip = CLIPTextModel.from_pretrained(f"{root}/text_encoder_2", dtype=torch.float32).eval()
    ci = clip_tok([prompt], max_length=77, truncation=True, add_special_tokens=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        po = clip(**ci)
    out["clip_ids"] = ci["input_ids"][0].to(torch.int32)
    out["pooled"] = po["pooler_output"].float()
    out["clip_last"] = po["last_hidden_state"][0].float()

    image = Image.open(a.image).convert("RGB")
    height, width = select_resolution(image)
    scale = max(width / image.width, height / image.height)
    resized = transforms.functional.resize(
        image, (round(image.height * scale), round(image.width * scale)), interpolation=transforms.InterpolationMode.BILINEAR
    )
    cropped = transforms.functional.center_crop(resized, (height, width))
    tensor = transforms.ToTensor()(cropped).unsqueeze(0) * 2.0 - 1.0
    out["img"] = tensor[0].float()
    out["hw"] = torch.tensor([height, width], dtype=torch.int32)
    vae = AutoencoderKLHunyuanVideo.from_pretrained(f"{root}/vae", torch_dtype=torch.float32).to(dev).eval()
    with torch.no_grad():
        t0 = time.time()
        dist = vae.encode(tensor.to(dev).unsqueeze(2)).latent_dist
        print(f"vae encode {time.time() - t0:.1f}s", flush=True)
    out["vae_mean"] = dist.mean[0, :, 0].float().cpu()
    out["vae_logvar"] = dist.logvar[0, :, 0].float().cpu()
    save_file({k: v.contiguous().cpu() for k, v in out.items()}, a.out, metadata={"prompt": prompt})
    print("enc ok", {k: list(v.shape) for k, v in out.items()}, flush=True)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-20))


def dit_inputs(cfg, lt, lh, lw, device, dtype, gen, text, audio_all, pooled, chunk_idx=0, memory=None):
    from diffusers.modular_pipelines.kandinsky_s2v.before_denoise import init_memory, memory_rope_pos
    from diffusers.modular_pipelines.kandinsky_s2v.modular_pipeline import audio_rope_pos

    class C:  # attribute view of the plain dict
        pass

    c = C()
    c.__dict__.update(cfg)
    ps = cfg["patch_size"]
    if memory is None:
        memory = init_memory(c, lh, lw, torch.float32, device)
    vis_pos = [torch.arange(lt, device=device), torch.arange(lh // ps[1], device=device), torch.arange(lw // ps[2], device=device)]
    ref_pos = [torch.tensor([cfg["index_rope_ref_image"]], device=device), torch.arange(lh // ps[1], device=device), torch.arange(lw // ps[2], device=device)]
    mem_pos = memory_rope_pos(c, memory, device)
    audio = audio_all.to(device)
    return dict(
        encoder_hidden_states=text.to(device).to(dtype).unsqueeze(0),
        audio_hidden_states=audio.to(dtype).unsqueeze(0),
        pooled_projections=pooled.to(device).to(dtype),
        memory_hidden_states=tuple(m.unsqueeze(0).to(dtype) for m in reversed(memory)),
        visual_rope_pos=vis_pos,
        reference_rope_pos=ref_pos,
        memory_rope_pos=tuple(reversed(mem_pos)),
        text_rope_pos=torch.arange(text.shape[0], device=device),
        audio_rope_pos=audio_rope_pos(audio.shape[0], lt, device),
        scale_factor=(1.0, 2.0, 2.0),
        return_dict=False,
    )


def stage_dit(a):
    """One forward on a small latent, every block tapped, in the asked dtype."""
    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    dev = a.device
    enc = {}
    with safe_open(a.enc, "pt") as f:
        for k in ("text_f32", "text_bf16", "audio_f32", "audio_bf16", "pooled"):
            if k in f.keys():
                enc[k] = f.get_tensor(k)
    text = enc.get("text_f32", enc["text_bf16"])
    audio = enc.get("audio_f32", enc["audio_bf16"])[: a.audio_tokens]
    pooled = enc["pooled"]
    d = LazyDit(a.root, dtype, dev)
    d.tap_on = True
    gen = torch.Generator("cpu").manual_seed(1234)
    lt, lh, lw = a.lt, a.lh, a.lw
    x = torch.randn(1, lt, lh, lw, 16, generator=gen)
    ref = torch.randn(1, 1, lh, lw, 16, generator=gen)
    args = dit_inputs(d.cfg, lt, lh, lw, dev, dtype, gen, text, audio, pooled)
    # Non-zero memory exercises the three FramePack projections.
    mem = []
    for m in args["memory_hidden_states"]:
        mem.append(torch.randn(*m.shape, generator=gen).to(dev).to(dtype))
    args["memory_hidden_states"] = tuple(mem)
    t = torch.tensor([float(a.t)], device=dev, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        o = d.model(hidden_states=x.to(dev).to(dtype), reference_hidden_states=ref.to(dev).to(dtype), timestep=t, **args)[0]
    print(f"dit forward {time.time() - t0:.1f}s", flush=True)
    out = {
        "x": x[0],
        "ref": ref[0],
        "mem4": mem[0][0].float().cpu(),
        "mem2": mem[1][0].float().cpu(),
        "mem1": mem[2][0].float().cpu(),
        "t": t.cpu(),
        "text": text,
        "audio": audio,
        "pooled": pooled,
        "out": o[0].float().cpu(),
    }
    for k, v in d.taps.items():
        out[f"tap.{k}"] = v
    save_file({k: v.contiguous().cpu() for k, v in out.items()}, a.out)
    print("dit ok", {k: list(v.shape) for k, v in out.items() if not k.startswith("tap.")}, flush=True)


def stage_run(a):
    """The chunk loop of the distilled pipeline with the real blocks' helpers."""
    from diffusers.modular_pipelines.kandinsky_s2v.before_denoise import roll_memory
    from diffusers.modular_pipelines.kandinsky_s2v.modular_pipeline import split_audio_embeds
    from diffusers.schedulers.scheduling_piflow import PiflowScheduler as S

    dtype = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    dev = a.device
    with safe_open(a.enc, "pt") as f:
        text = f.get_tensor("text_bf16")
        audio = f.get_tensor("audio_bf16")
        pooled = f.get_tensor("pooled")
        mean, logvar = f.get_tensor("vae_mean"), f.get_tensor("vae_logvar")
        hw = f.get_tensor("hw").tolist()
    gen = torch.Generator(a.gen_device).manual_seed(a.seed)
    # The pipeline draws the reference posterior first, then per-chunk noise.
    eps = torch.randn(1, 16, 1, *mean.shape[-2:], generator=gen, device=a.gen_device).to(dev)
    ref_latent = (mean.to(dev) + (0.5 * logvar.to(dev)).exp() * eps[0, :, 0]).permute(1, 2, 0) * 0.476986
    lh, lw = ref_latent.shape[0], ref_latent.shape[1]
    d = LazyDit(a.root, dtype, dev)
    cfg = d.cfg
    sched = S.from_pretrained(f"{a.root}/scheduler")
    chunks = split_audio_embeds(audio.to(dtype), a.chunks)
    memory = None
    out = {"ref_latent": ref_latent.float().cpu(), "ref_eps": eps[0, :, 0].cpu()}
    for k in range(a.chunks):
        noise = torch.randn(32, lh, lw, 16, generator=gen, device=a.gen_device, dtype=torch.float32).to(dev)
        latents = noise.clone()
        out[f"noise.{k}"] = noise.cpu()
        sched.set_timesteps(a.steps, device=dev)
        args = dit_inputs(cfg, 32, lh, lw, dev, dtype, gen, text, chunks[k], pooled, memory=memory)
        tot = 0.0
        for i, t in enumerate(sched.timesteps):
            t0 = time.time()
            with torch.no_grad():
                pred = d.model(
                    hidden_states=latents.unsqueeze(0).to(dtype),
                    reference_hidden_states=ref_latent.unsqueeze(0).unsqueeze(0).to(dtype),
                    timestep=t.expand(1).to(latents.dtype),
                    **args,
                )[0]
            latents = sched.step(pred.squeeze(0).float(), t, latents, return_dict=False)[0]
            dt = time.time() - t0
            tot += dt
            if i in (0, 1, a.steps - 1):
                out[f"pred.{k}.{i}"] = pred[0].float().cpu()
            out[f"lat.{k}.{i}"] = latents.float().cpu() if i in (0, 1, a.steps - 1) else latents.float().cpu()[:1]
            print(f"chunk {k} step {i} t={float(t):.3f} {dt:.1f}s |lat|={latents.norm():.2f}", flush=True)
            if a.max_steps and i + 1 >= a.max_steps:
                break
        print(f"chunk {k}: {tot:.1f}s", flush=True)
        out[f"final.{k}"] = latents.float().cpu()
        save_file({n: v.contiguous().cpu() for n, v in out.items()}, a.out)
        if k + 1 < a.chunks:
            class C:
                pass

            c = C()
            c.__dict__.update(cfg)
            memory = roll_memory(c, latents)


def stage_vae(a):
    from diffusers.models import AutoencoderKLHunyuanVideo

    dev = a.device
    with safe_open(a.latents, "pt") as f:
        lats = [f.get_tensor(k) for k in sorted((k for k in f.keys() if k.startswith("final.")), key=lambda s: int(s.split(".")[1]))]
    out = {}
    # As the pipeline loads it: bf16 weights, the decode under bf16 autocast. An
    # f32 decode does not fit 32 GB with the causal attention mask of a tile.
    vae = AutoencoderKLHunyuanVideo.from_pretrained(f"{a.root}/vae", torch_dtype=torch.bfloat16).to(dev).eval()
    frames = []
    t0 = time.time()
    for lat in lats:
        z = (lat.to(dev) / 0.476986).permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            frames.append(vae.decode(z.to(vae.dtype)).sample.float().cpu())
        print(f"vae decode chunk {time.time() - t0:.1f}s", flush=True)
    out["frames_f32"] = torch.cat(frames, dim=2)[0]
    save_file({k: v.contiguous() for k, v in out.items()}, a.out)
    print("vae ok", {k: list(v.shape) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["enc", "dit", "run", "vae"])
    p.add_argument("--root", default="/workspace/s2v/hf")
    p.add_argument("--image")
    p.add_argument("--audio")
    p.add_argument("--prompt")
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--enc")
    p.add_argument("--latents")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--gen-device", default="cpu")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lt", type=int, default=8)
    p.add_argument("--lh", type=int, default=16)
    p.add_argument("--lw", type=int, default=16)
    p.add_argument("--t", type=float, default=900.0)
    p.add_argument("--audio-tokens", type=int, default=29)
    p.add_argument("--skip-f32", action="store_true")
    p.add_argument("--skip-bf16", action="store_true")
    a = p.parse_args()
    {"enc": stage_enc, "dit": stage_dit, "run": stage_run, "vae": stage_vae}[a.stage](a)
