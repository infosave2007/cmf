#!/usr/bin/env python3
"""GGUF → HF directory (config.json + model.safetensors + tokenizer.json),
then the usual convert_dtgma_to_cmf.py. Opens the GGUF model library
with a single command (roadmap D1).

v1: llama/qwen2/qwen3 architectures (RMSNorm+RoPE+SwiGLU, no bias);
F32/F16/Q8_0 tensors (K-quants come later). Tokenizer: only
tokenizer.huggingface.json from metadata or a neighboring tokenizer.json —
reconstruction from tokenizer.ggml.* is not done (honest refusal).

Usage: import_gguf.py model.gguf out_dir/
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
# kv value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8
GGML_Q4_K, GGML_Q5_K, GGML_Q6_K = 12, 13, 14
QK_K = 256

NAME_MAP = [
    ("token_embd.weight", "model.embed_tokens.weight"),
    ("output_norm.weight", "model.norm.weight"),
    ("output.weight", "lm_head.weight"),
]
BLK_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def read_kv(f):
    def s():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode()

    def val(t):
        fmt = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I",
               T_I32: "<i", T_F32: "<f", T_BOOL: "<B", T_U64: "<Q",
               T_I64: "<q", T_F64: "<d"}
        if t == T_STR:
            return s()
        if t == T_ARR:
            at, n = struct.unpack("<IQ", f.read(12))
            return [val(at) for _ in range(n)]
        v = struct.unpack(fmt[t], f.read(struct.calcsize(fmt[t])))[0]
        return bool(v) if t == T_BOOL else v

    key = s()
    t = struct.unpack("<I", f.read(4))[0]
    return key, val(t)


def dequant(dt, raw, n):
    if dt == GGML_F32:
        return np.frombuffer(raw, np.float32)[:n]
    if dt == GGML_F16:
        return np.frombuffer(raw, np.float16)[:n].astype(np.float32)
    if dt == GGML_Q8_0:  # blocks of 32: f16 scale + 32×i8
        blocks = np.frombuffer(raw, np.uint8).reshape(-1, 34)
        sc = blocks[:, :2].copy().view(np.float16).astype(np.float32)
        q = blocks[:, 2:].copy().view(np.int8).astype(np.float32)
        return (q * sc).reshape(-1)[:n]
    if dt == GGML_Q4_K:  # 144 B / 256 weights: d, dmin, 6-bit scales, nibbles
        b = np.frombuffer(raw, np.uint8).reshape(-1, 144)
        d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
        dm = b[:, 2:4].copy().view(np.float16).astype(np.float32)
        sc8, mn8 = _kq_scale_min(b[:, 4:16])
        qs = b[:, 16:144]
        y = np.empty((b.shape[0], QK_K), np.float32)
        for i in range(4):
            chunk = qs[:, i * 32:(i + 1) * 32]
            y[:, i*64:i*64+32] = d * sc8[:, [2*i]] * (chunk & 0xF) - dm * mn8[:, [2*i]]
            y[:, i*64+32:i*64+64] = d * sc8[:, [2*i+1]] * (chunk >> 4) - dm * mn8[:, [2*i+1]]
        return y.reshape(-1)[:n]
    if dt == GGML_Q5_K:  # 176 B: Q4_K + high bit from qh
        b = np.frombuffer(raw, np.uint8).reshape(-1, 176)
        d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
        dm = b[:, 2:4].copy().view(np.float16).astype(np.float32)
        sc8, mn8 = _kq_scale_min(b[:, 4:16])
        qh = b[:, 16:48]
        qs = b[:, 48:176]
        y = np.empty((b.shape[0], QK_K), np.float32)
        for i in range(4):
            chunk = qs[:, i * 32:(i + 1) * 32]
            h1 = ((qh >> (2 * i)) & 1) * 16
            h2 = ((qh >> (2 * i + 1)) & 1) * 16
            y[:, i*64:i*64+32] = d * sc8[:, [2*i]] * ((chunk & 0xF) + h1) - dm * mn8[:, [2*i]]
            y[:, i*64+32:i*64+64] = d * sc8[:, [2*i+1]] * ((chunk >> 4) + h2) - dm * mn8[:, [2*i+1]]
        return y.reshape(-1)[:n]
    if dt == GGML_Q6_K:  # 210 B: 6-bit values −32, 16 int8 scales
        b = np.frombuffer(raw, np.uint8).reshape(-1, 210)
        ql = b[:, :128]
        qh = b[:, 128:192]
        sc = b[:, 192:208].copy().view(np.int8).astype(np.float32)
        d = b[:, 208:210].copy().view(np.float16).astype(np.float32)
        y = np.empty((b.shape[0], QK_K), np.float32)
        for half in range(2):  # ggml: ql+=64, qh+=32, sc+=8 per half
            A = ql[:, half*64:half*64+32]
            B = ql[:, half*64+32:half*64+64]
            hi = qh[:, half*32:half*32+32]
            base, sb = half * 128, half * 8
            q = [((A & 0xF) | ((hi & 3) << 4)),
                 ((B & 0xF) | (((hi >> 2) & 3) << 4)),
                 ((A >> 4) | (((hi >> 4) & 3) << 4)),
                 ((B >> 4) | (((hi >> 6) & 3) << 4))]
            for k in range(4):  # quarters: scales sb+0/+2/+4/+6 (pairs of 16)
                y[:, base+32*k:base+32*(k+1)] = \
                    (q[k].astype(np.int16) - 32) * _sc16(sc, sb + 2*k, d)
        return y.reshape(-1)[:n]
    raise SystemExit(f"GGML type {dt} not supported (F32/F16/Q8_0/Q4_K/Q5_K/Q6_K)")


def _sc16(sc, j, d):
    """d * int8 scale of sub-block j, broadcast over 16-weight pairs."""
    out = np.repeat(sc[:, j:j+2], 16, axis=1)  # [B, 32]
    return d * out


def _kq_scale_min(scales):
    """6-bit scales/mins for Q4_K/Q5_K (get_scale_min_k4), [B,8]+[B,8]."""
    B = scales.shape[0]
    sc = np.empty((B, 8), np.float32)
    mn = np.empty((B, 8), np.float32)
    for j in range(4):
        sc[:, j] = scales[:, j] & 63
        mn[:, j] = scales[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
        mn[:, j] = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, mn


def unpermute(w: np.ndarray, n_head: int) -> np.ndarray:
    """llama.cpp permutes attn_q/attn_k rows (HF→GGUF, interleaved-rope
    convention); our engine expects HF half-split — undo it back."""
    out, inn = w.shape
    return (w.reshape(n_head, out // n_head // 2, 2, inn)
             .swapaxes(1, 2)
             .reshape(out, inn))


def canon(name: str):
    for a, b in NAME_MAP:
        if name == a:
            return b
    if name.startswith("blk."):
        _, li, rest = name.split(".", 2)
        mapped = BLK_MAP.get(rest)
        if mapped:
            return f"model.layers.{li}.{mapped}"
    return None


def main():
    src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open(src, "rb")
    assert f.read(4) == GGUF_MAGIC, "not GGUF"
    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
    assert version in (2, 3), f"GGUF v{version} not supported"
    kv = dict(read_kv(f) for _ in range(n_kv))

    infos = []
    for _ in range(n_tensors):
        ln = struct.unpack("<Q", f.read(8))[0]
        name = f.read(ln).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        dt, off = struct.unpack("<IQ", f.read(12))
        infos.append((name, dims, dt, off))
    align = kv.get("general.alignment", 32)
    data_start = (f.tell() + align - 1) // align * align

    arch = kv["general.architecture"]
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)
    n_layers = g("block_count")
    config = {
        "model_type": arch,
        "hidden_size": g("embedding_length"),
        "intermediate_size": g("feed_forward_length"),
        "num_hidden_layers": n_layers,
        "num_attention_heads": g("attention.head_count"),
        "num_key_value_heads": g("attention.head_count_kv",
                                 g("attention.head_count")),
        "rms_norm_eps": g("attention.layer_norm_rms_epsilon", 1e-5),
        "rope_theta": g("rope.freq_base", 10000.0),
        "max_position_embeddings": g("context_length", 4096),
        "vocab_size": len(kv.get("tokenizer.ggml.tokens", [])) or None,
        "tie_word_embeddings": not any(n == "output.weight"
                                       for n, *_ in infos),
        "_imported_from_gguf": str(src.name),
    }
    hd = g("attention.key_length")
    if hd:
        config["head_dim"] = hd

    # Tokenizer: full HF JSON from metadata or a neighboring file.
    hf_tok = kv.get("tokenizer.huggingface.json")
    sidecar = src.with_name("tokenizer.json")
    if hf_tok:
        (out_dir / "tokenizer.json").write_text(hf_tok)
    elif sidecar.exists():
        (out_dir / "tokenizer.json").write_bytes(sidecar.read_bytes())
    else:
        print("  ! no tokenizer.huggingface.json and no neighboring tokenizer.json —"
              " place it into out_dir yourself (reconstruction from the ggml vocab"
              " is not implemented)")
    if "tokenizer.chat_template" in kv:
        (out_dir / "chat_template.jinja").write_text(kv["tokenizer.chat_template"])

    # Tensors → a single safetensors (f32; our converter quantizes later).
    st_meta, blobs, cursor = {}, [], 0
    for name, dims, dt, off in infos:
        cname = canon(name)
        if cname is None:
            print(f"  skip {name} (no mapping)")
            continue
        shape = list(dims)[::-1]  # GGUF stores dims in reverse order
        n = int(np.prod(shape))
        f.seek(data_start + off)
        nb = {GGML_F32: n * 4, GGML_F16: n * 2,
              GGML_Q8_0: (n // 32) * 34,
              GGML_Q4_K: (n // QK_K) * 144,
              GGML_Q5_K: (n // QK_K) * 176,
              GGML_Q6_K: (n // QK_K) * 210}[dt]
        w = dequant(dt, f.read(nb), n).astype(np.float32).reshape(shape)
        if arch == "llama" and cname.endswith("self_attn.q_proj.weight"):
            w = unpermute(w, config["num_attention_heads"])
        elif arch == "llama" and cname.endswith("self_attn.k_proj.weight"):
            w = unpermute(w, config["num_key_value_heads"])
        raw = w.tobytes()
        st_meta[cname] = {"dtype": "F32", "shape": shape,
                          "data_offsets": [cursor, cursor + len(raw)]}
        blobs.append(raw)
        cursor += len(raw)
    hdr = json.dumps(st_meta).encode()
    pad = (-len(hdr)) % 8
    with open(out_dir / "model.safetensors", "wb") as o:
        o.write(struct.pack("<Q", len(hdr) + pad))
        o.write(hdr + b" " * pad)
        for b in blobs:
            o.write(b)
    if config["vocab_size"] is None:
        config["vocab_size"] = st_meta["model.embed_tokens.weight"]["shape"][0]
    json.dump(config, open(out_dir / "config.json", "w"), indent=1)
    print(f"  ✓ {out_dir}: {len(st_meta)} tensors | arch={arch} | "
          f"{n_layers} layers | next: convert_dtgma_to_cmf.py --model {out_dir}")


if __name__ == "__main__":
    main()
