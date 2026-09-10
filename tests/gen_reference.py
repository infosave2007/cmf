#!/usr/bin/env python3
"""Golden-parity reference: numpy forward pass over the tiny checkpoint.

Weights are passed through the SAME encoders the converter writes with
(encode → decode), so the reference sees bit-identical dequantized
values to what the Rust engine reads from the .cmf file. The forward
math mirrors the Rust engine op for op: RMS-norm with f64 mean (Qwen
style x̂·w), half-split RoPE, GQA attention with stable softmax, SwiGLU.

Output JSON: prompt_ids, greedy_ids (N steps), first_logits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "converter"))
import convert_dtgma_to_cmf as conv  # noqa: E402


def roundtrip(t: np.ndarray, dtype: str) -> np.ndarray:
    """Encode with the converter's encoder, decode back — exactly what
    the Rust engine dequantizes from the file."""
    t = np.asarray(t, dtype=np.float32)
    if dtype == "f32":
        return t
    if dtype == "f16":
        return t.astype(np.float16).astype(np.float32)
    if dtype == "q8_2f":
        raw = conv.encode_q8_2f(t)
        out_dim, in_dim = t.shape
        n = out_dim * in_dim
        q = np.frombuffer(raw[:n], dtype=np.int8).reshape(out_dim, in_dim)
        scale = np.frombuffer(raw[n:n + out_dim * 2], dtype=np.float16).astype(np.float32)
        col = np.frombuffer(raw[n + out_dim * 2:], dtype=np.float16).astype(np.float32)
        return q.astype(np.float32) * scale[:, None] * col[None, :]
    if dtype == "q8_row":
        raw = conv.encode_q8_row(t)
        out_dim, in_dim = t.shape
        q = np.frombuffer(raw[: out_dim * in_dim], dtype=np.int8).reshape(out_dim, in_dim)
        scales = np.frombuffer(raw[out_dim * in_dim:], dtype=np.float16).astype(np.float32)
        return q.astype(np.float32) * scales[:, None]
    if dtype == "q4_block":
        raw = conv.encode_q4_block(t)
        n = t.size
        n_groups = (n + 31) // 32
        packed = np.frombuffer(raw[: n_groups * 16], dtype=np.uint8)
        scales = np.frombuffer(raw[n_groups * 16:], dtype=np.float16).astype(np.float32)
        lo = (packed & 0x0F).astype(np.int32) - 8
        hi = ((packed >> 4) & 0x0F).astype(np.int32) - 8
        vals = np.empty(n_groups * 32, dtype=np.float32)
        vals[0::2] = lo
        vals[1::2] = hi
        vals = vals.reshape(n_groups, 32) * scales[:, None]
        return vals.flatten()[:n].reshape(t.shape)
    raise ValueError(dtype)


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x.astype(np.float64) ** 2)
    inv = np.float32(1.0 / np.sqrt(ms + eps))
    return (x * inv * w).astype(np.float32)


def rope(x: np.ndarray, pos: int, base: float) -> np.ndarray:
    hd = x.shape[-1]
    half = hd // 2
    i = np.arange(half, dtype=np.float32)
    freq = (1.0 / np.float32(base) ** (2.0 * i / np.float32(hd))).astype(np.float32)
    angle = (np.float32(pos) * freq).astype(np.float32)
    cos, sin = np.cos(angle), np.sin(angle)
    out = x.copy()
    x0, x1 = x[..., :half], x[..., half:]
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:] = x0 * sin + x1 * cos
    return out


class Ref:
    def __init__(self, model_dir: Path, quant: str):
        cfg = json.loads((model_dir / "config.json").read_text())
        tc = cfg.get("text_config", cfg)
        self.hid = tc["hidden_size"]
        self.inter = tc["intermediate_size"]
        self.layers = tc["num_hidden_layers"]
        self.heads = tc["num_attention_heads"]
        self.kv = tc["num_key_value_heads"]
        self.hd = tc["head_dim"]
        self.eps = tc.get("rms_norm_eps", 1e-6)
        self.base = tc.get("rope_theta", 10000.0)

        default = conv.QUANT_CHOICES[quant]
        raw = {}
        with np.load(model_dir / "weights.npz") as z:
            for k in z.files:
                raw[k] = z[k]
        self.w = {}
        for name, t in raw.items():
            dtype = conv.pick_dtype(name, np.asarray(t), default)
            self.w[name] = roundtrip(np.asarray(t), dtype)

    def forward(self, ids: list[int]) -> np.ndarray:
        """Returns logits after the last position."""
        # Head-major KV per layer.
        kcache = [[[] for _ in range(self.kv)] for _ in range(self.layers)]
        vcache = [[[] for _ in range(self.kv)] for _ in range(self.layers)]
        hpk = self.heads // self.kv
        hidden = None

        for pos, tid in enumerate(ids):
            h = self.w["model.embed_tokens.weight"][tid].astype(np.float32).copy()
            for li in range(self.layers):
                p = f"model.layers.{li}."
                normed = rms_norm(h, self.w[p + "input_layernorm.weight"], self.eps)

                q = (self.w[p + "self_attn.q_proj.weight"] @ normed).reshape(self.heads, self.hd)
                k = (self.w[p + "self_attn.k_proj.weight"] @ normed).reshape(self.kv, self.hd)
                v = (self.w[p + "self_attn.v_proj.weight"] @ normed).reshape(self.kv, self.hd)
                q = np.stack([rope(q[hh], pos, self.base) for hh in range(self.heads)])
                k = np.stack([rope(k[g], pos, self.base) for g in range(self.kv)])
                for g in range(self.kv):
                    kcache[li][g].append(k[g])
                    vcache[li][g].append(v[g])

                attn = np.zeros(self.heads * self.hd, dtype=np.float32)
                for hh in range(self.heads):
                    g = hh // hpk
                    K = np.stack(kcache[li][g])  # [seq, hd]
                    V = np.stack(vcache[li][g])
                    scores = (K @ q[hh]) / np.float32(np.sqrt(self.hd))
                    scores = scores - scores.max()
                    e = np.exp(scores)
                    probs = e / e.sum()
                    attn[hh * self.hd:(hh + 1) * self.hd] = probs @ V
                h = h + self.w[p + "self_attn.o_proj.weight"] @ attn

                post = rms_norm(h, self.w[p + "post_attention_layernorm.weight"], self.eps)
                gate = self.w[p + "mlp.gate_proj.weight"] @ post
                up = self.w[p + "mlp.up_proj.weight"] @ post
                act = gate / (1.0 + np.exp(-gate)) * up
                h = h + self.w[p + "mlp.down_proj.weight"] @ act
            hidden = h

        final = rms_norm(hidden, self.w["model.norm.weight"], self.eps)
        return self.w["lm_head.weight"] @ final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="Q8_ROW", choices=list(conv.QUANT_CHOICES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()

    ref = Ref(Path(args.model), args.quant)
    prompt_ids = [3, 17, 29]
    ids = list(prompt_ids)
    first_logits = None
    greedy = []
    for _ in range(args.steps):
        logits = ref.forward(ids)
        if first_logits is None:
            first_logits = logits.tolist()
        nxt = int(np.argmax(logits))
        greedy.append(nxt)
        ids.append(nxt)

    Path(args.out).write_text(json.dumps({
        "quant": args.quant,
        "prompt_ids": prompt_ids,
        "greedy_ids": greedy,
        "first_logits": first_logits,
    }))
    print(f"reference written: {args.out} (greedy: {greedy})")


if __name__ == "__main__":
    main()
