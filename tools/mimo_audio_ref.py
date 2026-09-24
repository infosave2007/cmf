#!/usr/bin/env python3
"""MiMo-V2.6 audio-input oracle (spec milestones M8, M9, M10 tower part).

Runs on a box with torch + numpy + safetensors + transformers (no
torchaudio / soundfile / PIL needed): the frontend is torchaudio's code
copied (resample kernel, melscale_fbanks, the Spectrogram call), and the
towers are the HF modules from the checkpoint's modeling_mimo_v2.py in fp32.

Commands (python tools/mimo_audio_ref.py <cmd> --help):
  wavs      --in-dir D --out D      build the WAV test set: format variants of
                                    one clip, a stereo 44.1 kHz clip, exact 5 s
                                    and 65 s clips (from `say` WAVs in --in-dir)
  decode    --wav F --out F.npy     numpy/stdlib decode to float32 [C, N]
  frontend  --wav F --out DIR       24 kHz mono (torchaudio Resample port) and
                                    log-mel, fp32 as torchaudio computes it
                                    and an fp64 twin of the same math
  tokenizer --mel F.npy --out DIR   HF MiMoAudioTokenizer encoder, fp32:
                                    per-segment pre-RVQ features and codes
                                    (exact and bf16-rounded codebooks), the
                                    bf16 serving path, and HF's own batched run
  encoder   --codes F.npy --out DIR HF MiMoAudioEncoder (speech embeddings +
                                    local Qwen2 + projection), fp32, checked
                                    against a hand-written bidirectional Qwen2
  cmp       --ref DIR --eng DIR     gate numbers (G8.x, G9.x, G10.1)

Every array is a .npy file; the engine side (examples/mimo_audio_dump.rs)
writes the same names.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

DEFAULT_SRC = os.environ.get("MIMO_SRC", "/root/mimo/src")
SEGMENT = 6000


# --------------------------------------------------------------------------
# WAV: numpy/stdlib decode and the variant writer
# --------------------------------------------------------------------------
def read_wav(path):
    """float32 [C, N] + rate. Scaling = ffmpeg's (u8: (x-128)/128,
    sN: x/2^(N-1), float passthrough)."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "not RIFF/WAVE"
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:min(pos + 8 + size, len(raw))]
        if cid == b"fmt ":
            tag, ch, rate, _, block, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, rate, block // ch)
        elif cid == b"data":
            data = body
            if fmt:
                break
        pos += 8 + size + (size & 1)
    tag, ch, rate, cb = fmt
    n = len(data) // (ch * cb)
    data = data[: n * ch * cb]
    if tag == 1 and cb == 1:
        x = (np.frombuffer(data, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and cb == 2:
        x = np.frombuffer(data, "<i2").astype(np.float32) / 32768.0
    elif tag == 1 and cb == 3:
        b = np.frombuffer(data, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / 8388608.0
    elif tag == 1 and cb == 4:
        x = (np.frombuffer(data, "<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    elif tag == 3 and cb == 4:
        x = np.frombuffer(data, "<f4").astype(np.float32)
    elif tag == 3 and cb == 8:
        x = np.frombuffer(data, "<f8").astype(np.float32)
    else:
        raise SystemExit(f"{path}: unsupported tag {tag} width {cb}")
    return np.ascontiguousarray(x.reshape(n, ch).T), rate


def write_wav(path, x, rate, kind, extensible=False):
    """x float [C, N] in [-1, 1]; kind in u8, s16, s24, s32, f32, f64."""
    ch, n = x.shape
    inter = x.T.reshape(-1)
    if kind == "u8":
        payload = np.clip(np.round(inter * 128.0 + 128.0), 0, 255).astype(np.uint8).tobytes()
        tag, bits = 1, 8
    elif kind == "s16":
        payload = np.clip(np.round(inter * 32768.0), -32768, 32767).astype("<i2").tobytes()
        tag, bits = 1, 16
    elif kind == "s24":
        v = np.clip(np.round(inter * 8388608.0), -8388608, 8388607).astype(np.int32)
        payload = np.stack([(v & 255), (v >> 8) & 255, (v >> 16) & 255], 1).astype(np.uint8).tobytes()
        tag, bits = 1, 24
    elif kind == "s32":
        v = np.clip(np.round(inter.astype(np.float64) * 2147483648.0), -2147483648, 2147483647)
        payload = v.astype("<i4").tobytes()
        tag, bits = 1, 32
    elif kind == "f32":
        payload, tag, bits = inter.astype("<f4").tobytes(), 3, 32
    elif kind == "f64":
        payload, tag, bits = inter.astype("<f8").tobytes(), 3, 64
    else:
        raise ValueError(kind)
    block = ch * bits // 8
    fmt = struct.pack("<HHIIHH", 0xFFFE if extensible else tag, ch, rate, rate * block, block, bits)
    if extensible:
        guid = struct.pack("<H", tag) + bytes([0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xAA, 0, 0x38, 0x9B, 0x71])
        fmt += struct.pack("<HHI", 22, bits, 0) + guid
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        body += b"\0"
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def cmd_wavs(a):
    os.makedirs(a.out, exist_ok=True)
    base, rate = read_wav(os.path.join(a.in_dir, "say1_24k.wav"))
    assert rate == 24000, rate
    mono = base[:1]
    # Format variants of the same 24 kHz clip (G8.1).
    for kind, ext in [("u8", False), ("s16", False), ("s24", False), ("s32", False),
                      ("f32", False), ("f64", False), ("s24", True), ("f32", True), ("s16", True)]:
        name = f"fmt_{kind}{'_ext' if ext else ''}.wav"
        write_wav(os.path.join(a.out, name), mono, 24000, kind, ext)
    # A true stereo clip at 44.1 kHz: right = half-level, shifted left.
    x44, r44 = read_wav(os.path.join(a.in_dir, "say2_44k.wav"))
    left = x44[0]
    right = 0.5 * np.roll(left, 1000)
    write_wav(os.path.join(a.out, "stereo_44k.wav"), np.stack([left, right]), r44, "s16")
    # Exactly 5 s and exactly 65 s at 24 kHz (spec worked examples).
    clips = [read_wav(os.path.join(a.in_dir, f))[0][0] for f in sorted(os.listdir(a.in_dir))
             if f.startswith("say") and f.endswith("_24k.wav")]
    first = clips[0]
    five = np.zeros(120000, np.float32)
    five[: min(120000, len(first))] = first[:120000]
    write_wav(os.path.join(a.out, "clip5_24k.wav"), five[None], 24000, "s16")
    long = []
    while sum(len(c) for c in long) < 65 * 24000:
        for c in clips:
            long.append(c)
            long.append(np.zeros(4800, np.float32))
    long = np.concatenate(long)[: 65 * 24000]
    write_wav(os.path.join(a.out, "clip65_24k.wav"), long[None], 24000, "s16")
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".wav"):
            x, r = read_wav(os.path.join(a.out, f))
            print(f"{f}: rate {r} channels {x.shape[0]} frames {x.shape[1]}")


def cmd_decode(a):
    x, rate = read_wav(a.wav)
    np.save(a.out, x)
    print(f"{a.wav}: rate {rate} shape {x.shape}")


# --------------------------------------------------------------------------
# torchaudio frontend, copied (TA:1305-1391 resample, TA:518-575 fbanks)
# --------------------------------------------------------------------------
def sinc_resample_kernel(orig_freq, new_freq, gcd, lowpass_filter_width=6, rolloff=0.99):
    import torch
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    base_freq = min(orig_freq, new_freq)
    base_freq *= rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx = torch.arange(-width, width + orig_freq, dtype=torch.float64)[None, None] / orig_freq
    t = torch.arange(0, -new_freq, -1, dtype=None)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clamp_(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t *= math.pi
    scale = base_freq / orig_freq
    kernels = torch.where(t == 0, torch.tensor(1.0).to(t), t.sin() / t)
    kernels *= window * scale
    kernels = kernels.to(dtype=torch.float32)
    return kernels, width


def apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width):
    import torch
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    num_wavs, length = waveform.shape
    waveform = torch.nn.functional.pad(waveform, (width, width + orig_freq))
    resampled = torch.nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
    target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
    resampled = resampled[..., :target_length]
    return resampled.view(shape[:-1] + resampled.shape[-1:])


def resample(x, orig, new, dtype):
    import torch
    if orig == new:
        return x.to(dtype)
    g = math.gcd(int(orig), int(new))
    k, w = sinc_resample_kernel(orig, new, g)
    return apply_sinc_resample_kernel(x.to(dtype), orig, new, g, k.to(dtype), w)


def melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, dtype):
    import torch
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs, dtype=dtype)
    hz_to_mel = lambda f: 2595.0 * math.log10(1.0 + (f / 700.0))
    m_min, m_max = hz_to_mel(f_min), hz_to_mel(f_max)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=dtype)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    zero = torch.zeros(1, dtype=dtype)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]
    up_slopes = slopes[:, 2:] / f_diff[1:]
    return torch.max(zero, torch.min(down_slopes, up_slopes))


def log_mel(wave, dtype):
    """torchaudio MelSpectrogram(24000, 960, win 960, hop 240, f_min 0,
    f_max None, 128 mels, power 1, center reflect) + log(clip(1e-7)).T

    The window and the filterbank are what torchaudio stores (float32
    buffers); `dtype` is the precision of the arithmetic."""
    import torch
    window = torch.hann_window(960, dtype=torch.float32).to(dtype)
    spec = torch.stft(wave.to(dtype)[None], n_fft=960, hop_length=240, win_length=960,
                      window=window, center=True, pad_mode="reflect", normalized=False,
                      onesided=True, return_complex=True).abs()
    fb = melscale_fbanks(481, 0.0, float(24000 // 2), 128, 24000, torch.float32).to(dtype)
    mel = torch.matmul(spec.transpose(-1, -2), fb).transpose(-1, -2)
    return torch.log(torch.clip(mel, min=1e-7)).squeeze(0).transpose(0, 1)


def cmd_frontend(a):
    """Two runs of torchaudio's computation. `""`: as torchaudio runs it
    (float32 arithmetic). `_f64`: every arithmetic op in float64, every
    stored tensor at torchaudio's float32 (kernel, resampled channels, the
    channel mean, window, filterbank) — the exact-math reference."""
    import torch
    os.makedirs(a.out, exist_ok=True)
    x, rate = read_wav(a.wav)
    xt = torch.from_numpy(x)
    out = {}
    for tag, dt in (("", torch.float32), ("_f64", torch.float64)):
        chans = resample(xt, rate, 24000, dt).float()  # [C, N'] each channel separately
        mono = chans.mean(dim=0) if chans.shape[0] > 1 else chans[0]
        np.save(os.path.join(a.out, f"chan24k{tag}.npy"), chans.numpy())
        np.save(os.path.join(a.out, f"wave24k{tag}.npy"), mono.numpy())
        mel = log_mel(mono, dt)
        np.save(os.path.join(a.out, f"mel{tag}.npy"), mel.float().numpy())
        out[tag or "_f32"] = dict(samples=int(mono.shape[0]), mel_frames=int(mel.shape[0]))
    meta = dict(wav=a.wav, rate=rate, channels=int(x.shape[0]), frames=int(x.shape[1]), **out)
    json.dump(meta, open(os.path.join(a.out, "frontend.json"), "w"), indent=1)
    print(json.dumps(meta))


# --------------------------------------------------------------------------
# HF towers
# --------------------------------------------------------------------------
def import_hf(hf_dir):
    import importlib
    import shutil
    import tempfile
    root = tempfile.mkdtemp(prefix="mimo_hf_")
    pkg = os.path.join(root, "mimo_hf_pkg")
    os.makedirs(pkg)
    for fn in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        shutil.copy(os.path.join(hf_dir, fn), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, root)
    return importlib.import_module("mimo_hf_pkg.modeling_mimo_v2")


def load_tokenizer(src, dtype):
    import torch
    from safetensors.torch import load_file
    M = import_hf(src)
    cfg = json.load(open(os.path.join(src, "audio_tokenizer", "config.json")))
    model = M.MiMoAudioTokenizer(M.MiMoAudioTokenizerConfig(**cfg))
    sd = load_file(os.path.join(src, "audio_tokenizer", "model.safetensors"))
    sd = {k: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith("decoder.")]
    assert not missing, missing
    model = model.to(dtype).eval()
    return M, model


def segment_features(enc, mel):
    """pre-RVQ features of ONE segment [T,128] -> [L2, d] (HF get_features)."""
    import torch
    from importlib import import_module
    lens = torch.tensor([mel.shape[0]], dtype=torch.long)
    out_len = enc.get_output_length(lens)
    M = sys.modules["mimo_hf_pkg.modeling_mimo_v2"]
    feats = M._at_unpack_hidden_states(mel, lens)
    h, *_ = enc.get_features(input_features=feats.transpose(1, 2), output_length=out_len)
    return h


def cmd_tokenizer(a):
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    mel = torch.from_numpy(np.load(a.mel)).float()
    T = mel.shape[0]
    segs = [SEGMENT] * (T // SEGMENT) + ([T % SEGMENT] if T % SEGMENT else [])
    M, model = load_tokenizer(a.src, torch.float32)
    enc = model.encoder
    q = enc.quantizer
    feats, codes, codes_b = [], [], []
    t0 = time.time()
    with torch.no_grad():
        s0 = 0
        for s in segs:
            h = segment_features(enc, mel[s0:s0 + s]).float()
            feats.append(h)
            codes.append(q.encode(h).T)
            s0 += s
        # bf16-rounded codebooks on the fp32 features (the engine's default flag)
        saved = [l._codebook.embed.clone() for l in q.vq.layers]
        for l in q.vq.layers:
            l._codebook.embed.copy_(l._codebook.embed.to(torch.bfloat16).float())
        codes_b = [q.encode(h).T for h in feats]
        for l, e in zip(q.vq.layers, saved):
            l._codebook.embed.copy_(e)
        # HF's own batched call (C11: request-dependent codes, reported only)
        batched = M.tokenize_audio_batch([mel], enc, segment_size=SEGMENT)[0]
    t_fp32 = time.time() - t0
    feats = torch.cat(feats).numpy()
    codes = torch.cat(codes).to(torch.int32).numpy()
    codes_b = torch.cat(codes_b).to(torch.int32).numpy()
    np.save(os.path.join(a.out, "feats.npy"), feats)
    np.save(os.path.join(a.out, "codes_exact.npy"), codes)
    np.save(os.path.join(a.out, "codes_bf16books.npy"), codes_b)
    np.save(os.path.join(a.out, "codes_hf_batched.npy"), batched.to(torch.int32).numpy())
    meta = dict(mel_frames=T, segments=segs, codes=int(codes.shape[0]), fp32_s=round(t_fp32, 2))
    if not a.no_bf16:
        del model
        _, mb = load_tokenizer(a.src, torch.bfloat16)
        with torch.no_grad():
            cb = M.tokenize_audio_batch([mel.to(torch.bfloat16)], mb.encoder, segment_size=SEGMENT)[0]
            # per-segment too (the engine's segmentation), bf16 serving numerics
            per = []
            s0 = 0
            for s in segs:
                hb = segment_features(mb.encoder, mel[s0:s0 + s].to(torch.bfloat16))
                mb.encoder.quantizer.float()
                per.append(mb.encoder.quantizer.encode(hb.float()).T)
                s0 += s
        np.save(os.path.join(a.out, "codes_bf16_serving.npy"), cb.to(torch.int32).numpy())
        np.save(os.path.join(a.out, "codes_bf16_segments.npy"), torch.cat(per).to(torch.int32).numpy())
    json.dump(meta, open(os.path.join(a.out, "tokenizer.json"), "w"), indent=1)
    print(json.dumps(meta))


def load_audio_encoder(src):
    import torch
    from safetensors import safe_open
    from types import SimpleNamespace
    M = import_hf(src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    acfg = SimpleNamespace(**cfg["audio_config"])
    speech = M._build_speech_embeddings(acfg)
    enc = M.MiMoAudioEncoder(acfg)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    want = [k for k in idx if k.startswith("audio_encoder.") or k.startswith("speech_embeddings.")]
    sd_enc, sd_sp = {}, {}
    for fn in sorted({idx[k] for k in want}):
        with safe_open(os.path.join(src, fn), "pt") as f:
            for k in f.keys():
                if k.startswith("audio_encoder."):
                    sd_enc[k[len("audio_encoder."):]] = f.get_tensor(k).float()
                elif k.startswith("speech_embeddings."):
                    sd_sp[k[len("speech_embeddings."):]] = f.get_tensor(k).float()
    miss, unexp = enc.load_state_dict(sd_enc, strict=False)
    assert miss == ["input_local_transformer.embed_tokens.weight"], miss
    assert not unexp, unexp
    miss, unexp = speech.load_state_dict(sd_sp, strict=True)
    return M, acfg, speech.float().eval(), enc.float().eval()


def handwritten_encoder(acfg, speech, enc, codes):
    """Bidirectional Qwen2 + projection written out (the G10.1 oracle)."""
    import torch
    import torch.nn.functional as F
    M = sys.modules["mimo_hf_pkg.modeling_mimo_v2"]
    g = M._pad_and_group_audio_codes(codes, acfg.audio_channels, acfg.group_size)  # [G,4,C]
    x = torch.zeros(g.shape[0], acfg.group_size, acfg.input_local_dim)
    for c in range(acfg.audio_channels):
        x = x + speech[c].weight[g[:, :, c].long()]
    lt = enc.input_local_transformer
    H = acfg.input_local_attn_heads
    hd = acfg.input_local_dim // H
    inv = 1.0 / (float(acfg.rope_theta) ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
    pos = torch.arange(acfg.group_size).float()
    fr = pos[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    cos, sin = emb.cos(), emb.sin()

    def rms(t, w, eps=1e-6):
        return w * (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps))

    def rot(t):
        return torch.cat([-t[..., hd // 2:], t[..., : hd // 2]], -1)

    G, S, D = x.shape
    for L in lt.layers:
        n = rms(x, L.input_layernorm.weight)
        q = L.self_attn.q_proj(n).view(G, S, H, hd).transpose(1, 2)
        k = L.self_attn.k_proj(n).view(G, S, H, hd).transpose(1, 2)
        v = L.self_attn.v_proj(n).view(G, S, H, hd).transpose(1, 2)
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        p = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd), -1)
        o = (p @ v).transpose(1, 2).reshape(G, S, H * hd)
        x = x + L.self_attn.o_proj(o)
        n = rms(x, L.post_attention_layernorm.weight)
        x = x + L.mlp.down_proj(F.silu(L.mlp.gate_proj(n)) * L.mlp.up_proj(n))
    x = rms(x, lt.norm.weight)
    return enc.projection(x.reshape(G, -1))


def cmd_encoder(a):
    import torch
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    codes = torch.from_numpy(np.load(a.codes).astype(np.int64))
    M, acfg, speech, enc = load_audio_encoder(a.src)
    with torch.no_grad():
        hf = enc(speech_embeddings=speech, audio_codes=codes)
        hw = handwritten_encoder(acfg, speech, enc, codes)
        # the same local transformer run causally, to show the flag matters
        g = M._pad_and_group_audio_codes(codes, acfg.audio_channels, acfg.group_size)
        e = enc._apply_speech_embeddings(g, speech)
        causal = enc.input_local_transformer(inputs_embeds=e, use_cache=False, is_causal=True).last_hidden_state
        causal = enc.projection(causal.reshape(causal.shape[0], -1))
    d_hw = (hf - hw).abs().max().item() / hf.abs().max().item()
    d_causal = (hf - causal).abs().max().item() / hf.abs().max().item()
    np.save(os.path.join(a.out, "embeds.npy"), hf.numpy())
    meta = dict(groups=int(hf.shape[0]), dim=int(hf.shape[1]), frames=int(codes.shape[0]),
                handwritten_vs_hf_rel=d_hw, causal_vs_bidirectional_rel=d_causal)
    json.dump(meta, open(os.path.join(a.out, "encoder.json"), "w"), indent=1)
    print(json.dumps(meta))
    assert d_hw < 1e-5, "hand-written bidirectional Qwen2 disagrees with HF"


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------
def compute_audio_token_len(mel_len, kernel=3, stride=2, avg_pooler=2, group=4):
    """The processor's placeholder count (PA:159-163), verbatim."""
    n = mel_len + 3 - kernel
    n = (n + 2 - kernel) // stride + 1
    n = n // avg_pooler + int(n % avg_pooler != 0)
    return math.ceil(n / group)


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def row_cos(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)


def code_agree(a, b):
    n = min(len(a), len(b))
    eq = a[:n] == b[:n]
    return dict(rows=n, same_len=len(a) == len(b), all=float(eq.mean()),
                l0=float(eq[:, 0].mean()), l1=float(eq[:, 1].mean()), l2=float(eq[:, 2].mean()),
                flips=int((~eq).sum()), first_levels_identical=bool(eq[:, :3].all()))


def cmd_cmp(a):
    ref, eng = a.ref, a.eng
    have = lambda d, f: os.path.exists(os.path.join(d, f))
    L = lambda d, f: np.load(os.path.join(d, f))
    res = {}
    if have(ref, "wave24k.npy") and have(eng, "wave24k.npy"):
        r, e = L(ref, "wave24k.npy"), L(eng, "wave24k.npy")
        res["resample"] = dict(len_ref=len(r), len_eng=len(e),
                               max_abs_over_peak=float(np.abs(r - e).max() / np.abs(r).max()) if len(r) == len(e) else None)
        if have(ref, "wave24k_f64.npy"):
            r64 = L(ref, "wave24k_f64.npy")
            res["resample"]["vs_f64_over_peak"] = float(np.abs(r64 - e).max() / np.abs(r64).max())
    if have(ref, "mel.npy") and have(eng, "mel.npy"):
        r, e = L(ref, "mel.npy"), L(eng, "mel.npy")
        res["mel"] = dict(shape_ref=list(r.shape), shape_eng=list(e.shape))
        if r.shape == e.shape:
            res["mel"]["max_abs_vs_torch_f32"] = float(np.abs(r - e).max())
            if have(ref, "mel_f64.npy"):
                r64 = L(ref, "mel_f64.npy")
                res["mel"]["max_abs_vs_f64"] = float(np.abs(r64 - e).max())
                res["mel"]["torch_f32_vs_f64"] = float(np.abs(r64 - r).max())
    if have(ref, "feats.npy") and have(eng, "feats.npy"):
        r, e = L(ref, "feats.npy"), L(eng, "feats.npy")
        c = row_cos(e, r)
        res["feats"] = dict(rows=len(r), rel=rel(e, r), min_row_cos=float(c.min()), mean_row_cos=float(c.mean()))
    for rf, ef in [("codes_exact.npy", "codes_exact.npy"), ("codes_bf16books.npy", "codes_bf16books.npy"),
                   ("codes_bf16_serving.npy", "codes_bf16books.npy"), ("codes_bf16_segments.npy", "codes_bf16books.npy"),
                   ("codes_hf_batched.npy", "codes_exact.npy")]:
        if have(ref, rf) and have(eng, ef):
            res[f"{rf[:-4]}~eng.{ef[:-4]}"] = code_agree(L(eng, ef), L(ref, rf))
    if have(ref, "codes_bf16_serving.npy") and have(ref, "codes_exact.npy"):
        res["hf_bf16_serving~hf_fp32"] = code_agree(L(ref, "codes_bf16_serving.npy"), L(ref, "codes_exact.npy"))
    if have(ref, "embeds.npy") and have(eng, "embeds.npy"):
        r, e = L(ref, "embeds.npy"), L(eng, "embeds.npy")
        c = row_cos(e, r)
        res["embeds"] = dict(shape_ref=list(r.shape), shape_eng=list(e.shape), rel=rel(e, r),
                             min_row_cos=float(c.min()), mean_row_cos=float(c.mean()))
    if have(eng, "tower.json"):
        t = json.load(open(os.path.join(eng, "tower.json")))
        k = compute_audio_token_len(t["mel_frames"])
        res["placeholders"] = dict(processor_K=k, engine_K=t["placeholder_count_K"],
                                   engine_rows=t["embed_rows_own"], equal=k == t["placeholder_count_K"] == t["embed_rows_own"])
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("wavs"); s.add_argument("--in-dir", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("decode"); s.add_argument("--wav", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("frontend"); s.add_argument("--wav", required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("tokenizer"); s.add_argument("--mel", required=True); s.add_argument("--out", required=True)
    s.add_argument("--src", default=DEFAULT_SRC); s.add_argument("--no-bf16", action="store_true")
    s.add_argument("--threads", type=int, default=8)
    s = sub.add_parser("encoder"); s.add_argument("--codes", required=True); s.add_argument("--out", required=True)
    s.add_argument("--src", default=DEFAULT_SRC); s.add_argument("--threads", type=int, default=8)
    s = sub.add_parser("cmp"); s.add_argument("--ref", required=True); s.add_argument("--eng", required=True)
    s.add_argument("--json")
    a = p.parse_args()
    {"wavs": cmd_wavs, "decode": cmd_decode, "frontend": cmd_frontend, "tokenizer": cmd_tokenizer,
     "encoder": cmd_encoder, "cmp": cmd_cmp}[a.cmd](a)


if __name__ == "__main__":
    main()
