#!/usr/bin/env python3
"""
Convert DTG-MA training output (model + masks) → Cortiq Model Format v2 (.cmf)

Spec: cmf/docs/CMF_V2_SPEC.md. The writer is symmetric to the Rust reader
(crates/cortiq-core/src/format.rs): binary tensor directory (56-byte
records, .vmfc v2 canon), canonical quant layouts ("quants first, then
scales"), bit-packed masks with zeroed tail bits, embedded tokenizer,
binary sparse index, per-tensor hash64 (bit-for-bit vmfcore-compatible).

Usage:
    python convert_dtgma_to_cmf.py \
        --model ./checkpoint --masks ./masks \
        --quant Q8_ROW --output model.cmf

Input model dir: config.json + weights (*.safetensors | *.bin/*.pt | weights.npz)
Input masks dir: *.pt / *.json, each with ffn_masks / head_masks /
layer_gates / metadata {task_name, sparsity, description, quality:{...}}.
Quality is written ONLY if measured — no default declarations.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

CMF_MAGIC = b"CMF\x01"
CMF_VERSION = 2
ENVELOPE_LEN = 128
DATA_ALIGNMENT = 4096
TENSOR_ALIGNMENT = 64
DIR_RECORD_LEN = 56
DIR_MAX_NDIM = 6
GROUP_SIZE = 32

# required_features bits
FEAT_TENSOR_DIR = 1 << 0
FEAT_BINARY_MASKS = 1 << 1
FEAT_QUANT_2F = 1 << 2

# dtype ids — shared with .vmfc, never reuse
DTYPE_ID = {"f32": 0, "f16": 1, "bf16": 2, "q8_row": 3, "q4_block": 4,
            "mix8_4": 5, "u8": 6, "q4_col": 7, "vbit": 8, "q8_2f": 9}

QUANT_CHOICES = {
    "VBIT": "vbit",     # variable-bit 3..8 (P13; validated grouped layout)
    "Q8_2F": "q8_2f",   # two-field 𝒲×θ (validated: recovers ~75% of the q8→f16 gap)
    "Q8_ROW": "q8_row",
    "Q4_BLOCK": "q4_block",
    "F16": "f16",
    "F32": "f32",
}


def align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# ───────────────────── hash64 (≡ vmfcore.hash64, ≡ Rust) ─────────────────────

def _fmix64(x):
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> np.uint64(33))
    x = x * np.uint64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> np.uint64(33))
    return x


def hash64(buf) -> int:
    a = np.frombuffer(bytes(buf), dtype=np.uint8)
    n = a.size
    pad = (-n) % 8
    if pad:
        a = np.concatenate([a, np.zeros(pad, np.uint8)])
    w = a.view(np.uint64)
    with np.errstate(over="ignore"):
        m = _fmix64(w)
        idx = np.arange(w.size, dtype=np.uint64)
        m = m ^ (idx * np.uint64(0x9E3779B97F4A7C15))
        h = np.bitwise_xor.reduce(m) if w.size else np.uint64(0)
        h = h ^ np.uint64(n)
        h = _fmix64(np.array([h], dtype=np.uint64))[0]
    return int(h)


# ───────────────────── canonical quant encoders ─────────────────────

def encode_q8_row(t: np.ndarray) -> bytes:
    """[int8: out·in][f16: out] — scale per row, w = q·scale[o].
    Row-chunked above _CHUNK_ELEMS (identical per-element math)."""
    assert t.ndim == 2, "q8_row is for 2-D matrices"
    out_dim, in_dim = t.shape
    if t.size <= 64_000_000:
        m = t.astype(np.float32)
        absmax = np.abs(m).max(axis=1)
        scales = np.maximum((absmax / 127.0).astype(np.float16).astype(np.float32),
                            _F16_TINY)  # quantize vs f16-rounded
        q = np.clip(np.round(m / scales[:, None]), -128, 127).astype(np.int8)
        return q.tobytes() + scales.astype(np.float16).tobytes()

    step = max(1, 64_000_000 // in_dim)
    q = np.empty((out_dim, in_dim), dtype=np.int8)
    scales = np.empty(out_dim, dtype=np.float32)
    for r0 in range(0, out_dim, step):
        m = t[r0:r0 + step].astype(np.float32)
        absmax = np.abs(m).max(axis=1)
        s_chunk = np.maximum((absmax / 127.0).astype(np.float16).astype(np.float32),
                             _F16_TINY)  # quantize vs f16-rounded
        scales[r0:r0 + len(s_chunk)] = s_chunk
        q[r0:r0 + len(s_chunk)] = np.clip(np.round(m / s_chunk[:, None]), -128, 127).astype(np.int8)
        del m
    return q.tobytes() + scales.astype(np.float16).tobytes()


_CHUNK_ELEMS = 64_000_000  # above this, encode row-chunked (memory-lean)


# Smallest normal f16: post-rounding clamp so degenerate (all-zero) rows
# never divide by an f16-flushed-to-zero scale.
_F16_TINY = np.float32(6.104e-5)


def encode_q8_2f(t: np.ndarray) -> bytes:
    """Two-field 𝒲×θ int8 (Madelung split, ≡ vmfcore _enc_q8_2f):
    column field col[i] = RMS over rows (absorbs outlier input channels),
    then per-row int8 over the normalized residual.
    Layout: [int8: out·in][f16 row_scale: out][f16 col: in].
    Dequant: w[o,i] = q[o,i]·row_scale[o]·col[i].

    Tensors above _CHUNK_ELEMS take a row-chunked path: identical math
    per element, f64 column accumulation is sequential-chunked (peak RAM
    stays ~one chunk instead of ~3 full copies of the matrix)."""
    assert t.ndim == 2, "q8_2f is for 2-D matrices"
    out_dim, in_dim = t.shape
    if t.size <= _CHUNK_ELEMS:
        rows = t.astype(np.float32)
        # Quantize against the f16-ROUNDED fields — the decoder will
        # multiply by exactly these values (bounce lesson: targeting the
        # unrounded scale quantizes toward reconstruction points that
        # don't exist).
        col = np.maximum(np.sqrt((rows.astype(np.float64) ** 2).mean(axis=0)), 1e-12)
        col = np.maximum(col.astype(np.float16).astype(np.float32), _F16_TINY)
        wn = rows / col[None, :]
        scale = (np.maximum(np.abs(wn).max(axis=1), 1e-12) / 127.0)
        scale = np.maximum(scale.astype(np.float16).astype(np.float32), _F16_TINY)
        q = np.round(wn / scale[:, None]).clip(-127, 127).astype(np.int8)
        return q.tobytes() + scale.astype(np.float16).tobytes() + col.astype(np.float16).tobytes()

    step = max(1, _CHUNK_ELEMS // in_dim)
    acc = np.zeros(in_dim, dtype=np.float64)
    for r0 in range(0, out_dim, step):
        chunk = t[r0:r0 + step].astype(np.float64)
        acc += (chunk ** 2).sum(axis=0)
        del chunk
    # f16-rounded fields (see the small-tensor path for why).
    col = np.maximum(np.sqrt(acc / out_dim), 1e-12).astype(np.float16).astype(np.float32)
    col = np.maximum(col, _F16_TINY)

    q = np.empty((out_dim, in_dim), dtype=np.int8)
    scale = np.empty(out_dim, dtype=np.float32)
    for r0 in range(0, out_dim, step):
        wn = t[r0:r0 + step].astype(np.float32) / col[None, :]
        s_chunk = np.maximum((np.maximum(np.abs(wn).max(axis=1), 1e-12) / 127.0)
                             .astype(np.float16).astype(np.float32), _F16_TINY)
        scale[r0:r0 + len(s_chunk)] = s_chunk
        q[r0:r0 + len(s_chunk)] = np.round(wn / s_chunk[:, None]).clip(-127, 127).astype(np.int8)
        del wn
    return q.tobytes() + scale.astype(np.float16).tobytes() + col.astype(np.float16).tobytes()


def encode_q4_block(t: np.ndarray) -> bytes:
    """[u8: ceil(n/32)·16][f16: ceil(n/32)] — groups of 32, low nibble first,
    w = (q − 8)·scale."""
    flat = t.astype(np.float32).flatten()
    pad = (-len(flat)) % GROUP_SIZE
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, np.float32)])
    groups = flat.reshape(-1, GROUP_SIZE)
    absmax = np.abs(groups).max(axis=1)
    scales = np.maximum((absmax / 7.0).astype(np.float16).astype(np.float32),
                        _F16_TINY)  # quantize vs f16-rounded
    q = (np.clip(np.round(groups / scales[:, None]), -8, 7).astype(np.int8) + 8).astype(np.uint8)
    packed = (q[:, 0::2] & 0x0F) | (q[:, 1::2] << 4)
    return packed.tobytes() + scales.astype(np.float16).tobytes()


VBIT_LEVELS = (3, 4, 5, 6, 8)  # safe-floor 3 (P13 claim 13)

# name → bit-budget shift (C4: experts of one layer share a common
# budget; a hot/loud expert gets more bits, a quiet one — fewer).
VBIT_BIAS: dict = {}
VBIT_MEAN_BITS = [4.25]  # water-filling target; --mean-bits
# Allocation-curve shape: "log2" (water-filling, clipped at floor) or
# "cubic" — a causal soft cutoff from the VMF-2026 recomputation (a hard
# step excluded by theory at ~65σ): b(I) = 3 + 5/(1+(x_c/I)³), x_c by
# bisection to the tensor's exact budget. VMF-queue experiment #3.
VBIT_SHAPE = ["log2"]


def vbit_bits(m: np.ndarray, mean_bits: float | None = None,
              bias: float = 0.0) -> np.ndarray:
    """Water-filling by log2 of the row amplitude → VBIT_LEVELS levels.
    Deterministic: both the layout (phase 1) and encoder (phase 2) call it.
    `bias` shifts the tensor's target budget (per-expert allocation,
    P15 claim 12): mean_bits + bias + (a − ā) ≡ joint water-filling
    over the rows of the whole expert family when bias = ā_tensor − ā_family."""
    if mean_bits is None:
        mean_bits = VBIT_MEAN_BITS[0]
    a = np.log2(np.maximum(np.abs(m).max(axis=1), 1e-12))
    rel = a - a.mean() + bias           # log2 of relative importance
    levels = np.asarray(VBIT_LEVELS, np.float64)

    def to_levels(raw):
        return levels[np.abs(np.asarray(raw, np.float64)[:, None] - levels)
                      .argmin(1)]

    if VBIT_SHAPE[0] == "cubic":
        # b(I) = 3 + 5/(1+(x_c/I)³) in the linear importance domain I=2^rel;
        # bisection of log2(x_c) to the exact mean budget (monotone ↓).
        lo, hi = -24.0, 24.0
        for _ in range(48):
            mid = (lo + hi) / 2
            raw = 3.0 + 5.0 / (1.0 + np.exp2(3.0 * (mid - rel)))
            if float(to_levels(raw).mean()) < mean_bits:
                hi = mid
            else:
                lo = mid
        xc = (lo + hi) / 2
        raw = 3.0 + 5.0 / (1.0 + np.exp2(3.0 * (xc - rel)))
        return to_levels(raw).astype(np.uint8)

    raw = mean_bits + rel
    bits = to_levels(raw)
    return np.maximum(bits, 3).astype(np.uint8)


def vbit_nbytes(bits: np.ndarray, cols: int) -> int:
    ng = cols // GROUP_SIZE
    rows = len(bits)
    return rows + rows * ng * 2 + int(sum((cols * int(b) + 7) // 8 for b in bits))


def _gptq_vbit_indices(W, H, scale2d, L, percdamp=0.01, act_order=True):
    """GPTQ error-feedback under vbit (vmfcore port, validated: gptq-vbit4
    15.61 vs uniform q4 17.02 PPL). scale2d[out,in] per-(row,group),
    L[out] — max level of the variable bit-width. Returns Q ∈ [-L, L]."""
    out, inn = W.shape
    W = W.astype(np.float64).copy()
    H = np.asarray(H, dtype=np.float64).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H[np.diag_indices(inn)] += percdamp * np.mean(np.diag(H))
    perm = np.argsort(np.diag(H))[::-1] if act_order else np.arange(inn)
    R = np.linalg.cholesky(np.linalg.inv(H[perm][:, perm])).T
    Wp = W[:, perm]
    s2 = scale2d.astype(np.float64)[:, perm]
    Lr = L.reshape(-1).astype(np.float64)
    Qp = np.zeros((out, inn), dtype=np.int32)
    for j in range(inn):
        w = Wp[:, j]
        sc = s2[:, j]
        qi = np.clip(np.round(w / sc), -Lr, Lr)
        Qp[:, j] = qi
        err = (w - qi * sc) / R[j, j]
        if j + 1 < inn:
            Wp[:, j + 1:] -= np.outer(err, R[j, j + 1:])
    Q = np.zeros((out, inn), dtype=np.int32)
    Q[:, perm] = Qp
    return Q


# name → npz file with "hess" [in,in] (Σ x·xᵀ of the calibration); set via --hessians
HESSIANS: dict = {}


def encode_vbit(t: np.ndarray, mean_bits: float | None = None, hess=None,
                bias: float = 0.0) -> bytes:
    """Grouped variable-bit (P13 FIG.3 + validated vmfcore layout):
    [u8 bits: rows][f16 scales: rows·cols/32][bit-packed rows, MSB-first,
    each row padded to a byte]. w = (u − L)·scale, L = 2^{b−1}−1.
    Allocation: water-filling by log2 of the row energy (the amplitude
    field A_r; the product A·B with Fisher — once calibration appears),
    rounding into VBIT_LEVELS, floor = 3 bits."""
    assert t.ndim == 2 and t.shape[1] % GROUP_SIZE == 0
    m = t.astype(np.float32)
    rows, cols = m.shape
    bits = vbit_bits(m, mean_bits, bias)

    g = m.reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    L = (2.0 ** (bits - 1) - 1).astype(np.float32)
    sc = np.abs(g).max(axis=2) / L[:, None]
    sc = np.maximum(sc.astype(np.float16).astype(np.float32), _F16_TINY)
    if hess is not None:
        scale2d = sc[:, np.arange(cols) // GROUP_SIZE]
        qsym = _gptq_vbit_indices(m, hess, scale2d, L.astype(np.float64))
        q = (qsym + L[:, None].astype(np.int32)).astype(np.uint32) \
            .reshape(rows, cols // GROUP_SIZE, GROUP_SIZE)
    else:
        q = np.clip(np.round(g / sc[:, :, None]) + L[:, None, None],
                    0, (2.0 ** bits - 1)[:, None, None]).astype(np.uint32)

    out = bytearray(bits.tobytes())
    out += sc.astype(np.float16).tobytes()
    # MSB-first packing, vectorized: value bits → packbits (the zero
    # tail of the last byte = the old semantics (acc << (8-nb)) & 0xFF).
    # Byte-for-byte with the per-byte loop — gated in the tests.
    for r in range(rows):
        b = int(bits[r])
        vals = q[r].reshape(-1)
        vbits = ((vals[:, None] >> np.arange(b - 1, -1, -1, dtype=np.uint32))
                 & 1).astype(np.uint8)
        out += np.packbits(vbits.reshape(-1)).tobytes()
    return bytes(out)


def encode_tensor(t: np.ndarray, dtype: str, name: str = "") -> bytes:
    if dtype == "vbit":
        h = None
        if name in HESSIANS:
            h = np.load(HESSIANS[name])["hess"]
        return encode_vbit(t, hess=h, bias=VBIT_BIAS.get(name, 0.0))
    if dtype == "q8_2f":
        return encode_q8_2f(t)
    if dtype == "q8_row":
        return encode_q8_row(t)
    if dtype == "q4_block":
        return encode_q4_block(t)
    if dtype == "f16":
        return t.astype(np.float16).tobytes()
    if dtype == "f32":
        return t.astype(np.float32).tobytes()
    raise ValueError(f"unsupported write dtype: {dtype}")


def pick_dtype(name: str, t: np.ndarray, default: str) -> str:
    # Norms / 1-D / tiny tensors are always f16 — precision of
    # normalizations at maximal matrix compression.
    if t.ndim < 2 or t.size < GROUP_SIZE:
        return "f16" if default not in ("f32",) else "f32"
    # GDN conv taps [c_dim, 1, K]: tiny and gate-critical — keep f16.
    if name.endswith("linear_attn.conv1d.weight"):
        return "f16" if default not in ("f32",) else "f32"
    # Noise-sensitive (vmfcore lesson on 35B): the MoE router (a bit-flip
    # changes top-k), the shared-expert sigmoid gate, the scalar a/b GDN
    # projections — always f16, they cost pennies.
    if (name.endswith("mlp.gate.weight")
            or name.endswith("shared_expert_gate.weight")
            or name.endswith("linear_attn.in_proj_a.weight")
            or name.endswith("linear_attn.in_proj_b.weight")):
        return "f16" if default not in ("f32",) else "f32"
    if default in ("q8_row", "q8_2f") and t.ndim != 2:
        return "q4_block"  # row-wise dtypes need 2-D; fall back for >2-D
    if default == "vbit" and (t.ndim != 2 or t.shape[1] % GROUP_SIZE != 0):
        return "q8_2f" if t.ndim == 2 else "q4_block"
    return default
# ───────────────────── weight sources (lazy, streaming-friendly) ─────────────────────

# Canonical renames: HF Qwen3.5-MTP names → CMF spec §2.1 names.
# Multimodal wrappers (Qwen3.5 VL-style): text tensors live under
# model.language_model.*; non-text towers are not executed by this
# runtime and are skipped LOUDLY (count printed at load).
CANON_PREFIXES = [("model.language_model.", "model.")]
SKIP_PREFIXES = ("model.visual.",)


def canon_name(raw: str):
    """Canonical CMF name for a source tensor; None = skip (non-text)."""
    if raw.startswith(SKIP_PREFIXES):
        return None
    for pre, rep in CANON_PREFIXES:
        if raw.startswith(pre):
            raw = rep + raw[len(pre):]
            break
    return RENAMES.get(raw, raw)


RENAMES = {
    "model.mtp.fc.weight": "model.mtp.eh_proj.weight",
    "model.mtp.pre_fc_norm_embedding.weight": "model.mtp.enorm.weight",
    "model.mtp.pre_fc_norm_hidden.weight": "model.mtp.hnorm.weight",
}

_ST_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
              "I8": np.int8, "U8": np.uint8}


def _st_header(path: Path):
    """Minimal safetensors parser: (meta {name: (dtype, shape, (b0, b1))}, data_start)."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    meta = {k: (v["dtype"], v["shape"], tuple(v["data_offsets"])) for k, v in header.items()}
    return meta, 8 + hlen


class SafetensorsSource:
    """Lazy per-tensor reads — one tensor in RAM at a time (numpy only,
    no safetensors dependency; bf16 decoded via bit shift)."""

    def __init__(self, files):
        self.entries = {}  # canonical name → (path, dtype, shape, byte-range)
        for path in files:
            meta, data_start = _st_header(path)
            for raw_name, (dt, shape, (b0, b1)) in meta.items():
                name = canon_name(raw_name)
                if name is None:
                    continue  # non-text tower (vision …) — counted below
                self.entries[name] = (path, dt, shape, (data_start + b0, data_start + b1))
        skipped = sum(1 for path in files for raw in _st_header(path)[0]
                      if canon_name(raw) is None)
        note = f", {skipped} non-text skipped" if skipped else ""
        print(f"  Source: {len(self.entries)} tensors in {len(files)} safetensors (lazy{note})")

    def names(self):
        return list(self.entries.keys())

    def shape(self, name):
        return list(self.entries[name][2])

    def load(self, name) -> np.ndarray:
        path, dt, shape, (b0, b1) = self.entries[name]
        with open(path, "rb") as f:
            f.seek(b0)
            raw = np.frombuffer(f.read(b1 - b0), dtype=_ST_DTYPES[dt])
        return _to_f32(raw, dt).reshape(shape)

    def load_rows(self, name, r0: int, r1: int) -> np.ndarray:
        """Rows [r0, r1) of the 2-D unfold [prod(shape[:-1]), shape[-1]] —
        a contiguous byte range (slice-pushdown for fused experts)."""
        path, dt, shape, (b0, _) = self.entries[name]
        cols = shape[-1]
        item = np.dtype(_ST_DTYPES[dt]).itemsize
        with open(path, "rb") as f:
            f.seek(b0 + r0 * cols * item)
            raw = np.frombuffer(f.read((r1 - r0) * cols * item),
                                dtype=_ST_DTYPES[dt])
        return _to_f32(raw, dt).reshape(r1 - r0, cols)


def _to_f32(raw: np.ndarray, dt: str) -> np.ndarray:
    if dt == "BF16":
        return (raw.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return raw.astype(np.float32, copy=False)


def _http_get(url: str, headers=None, tries: int = 8) -> bytes:
    import urllib.request
    import urllib.error
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            return urllib.request.urlopen(req, timeout=120).read()
        except urllib.error.HTTPError as e:
            # 4xx (except 429) are definitive — e.g. a probe for an optional
            # index.json that doesn't exist. Don't burn ~56 s of backoff on them.
            if 400 <= e.code < 500 and e.code != 429:
                raise
            if t == tries - 1:
                raise
            time.sleep(min(2 * (t + 1), 15))
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(min(2 * (t + 1), 15))
    raise RuntimeError("unreachable")


def _http_range(url: str, a: int, b: int, chunk: int = 16 * 1024 * 1024) -> bytes:
    """Chunked range download with per-chunk retry (vmfcore fetch port)."""
    out = bytearray()
    pos = a
    while pos <= b:
        hi = min(pos + chunk - 1, b)
        out += _http_get(url, {"Range": f"bytes={pos}-{hi}"})
        pos = hi + 1
    return bytes(out)


class HfStreamSource:
    """Weights STREAMED from HuggingFace (vmfcore convert_qwen35moe port):
    safetensors-shard headers are read with range requests, each tensor
    (or a row slice of a fused bank) — a separate ranged GET. On disk only
    the output .cmf lives; a 70GB checkpoint converts with no room for the
    source."""

    def __init__(self, repo: str):
        self.base = repo if repo.startswith("http") \
            else f"https://huggingface.co/{repo}/resolve/main"
        try:
            wm = json.loads(_http_get(
                f"{self.base}/model.safetensors.index.json"))["weight_map"]
            shards = sorted(set(wm.values()))
        except Exception:
            shards = ["model.safetensors"]
        self.entries = {}  # canon name → (url, dtype, shape, abs_b0)
        skipped = 0
        for sh in shards:
            url = f"{self.base}/{sh}"
            hlen = struct.unpack("<Q", _http_range(url, 0, 7))[0]
            hdr = json.loads(_http_range(url, 8, 8 + hlen - 1))
            hdr.pop("__metadata__", None)
            data0 = 8 + hlen
            for raw, info in hdr.items():
                name = canon_name(raw)
                if name is None:
                    skipped += 1
                    continue
                b0, _ = info["data_offsets"]
                self.entries[name] = (url, info["dtype"], info["shape"],
                                      data0 + b0)
            print(f"  Stream: {sh} (+{len(hdr)} tensors)", flush=True)
        note = f", {skipped} non-text skipped" if skipped else ""
        print(f"  Source: {len(self.entries)} tensors, STREAM from {self.base}{note}")

    def names(self):
        return list(self.entries.keys())

    def shape(self, name):
        return list(self.entries[name][2])

    def load(self, name) -> np.ndarray:
        url, dt, shape, b0 = self.entries[name]
        n = int(np.prod(shape)) * np.dtype(_ST_DTYPES[dt]).itemsize
        raw = np.frombuffer(_http_range(url, b0, b0 + n - 1), _ST_DTYPES[dt])
        return _to_f32(raw, dt).reshape(shape)

    def load_rows(self, name, r0: int, r1: int) -> np.ndarray:
        url, dt, shape, b0 = self.entries[name]
        cols = shape[-1]
        item = np.dtype(_ST_DTYPES[dt]).itemsize
        a = b0 + r0 * cols * item
        raw = np.frombuffer(
            _http_range(url, a, a + (r1 - r0) * cols * item - 1),
            _ST_DTYPES[dt])
        return _to_f32(raw, dt).reshape(r1 - r0, cols)


class MoeFusedExpertsSource:
    """qwen3_5_moe/AgentWorld: expert banks are packed into fused tensors
    `mlp.experts.gate_up_proj` [E, 2I, H] (gate||up along the out axis) and
    `mlp.experts.down_proj` [E, H, I] — already in nn.Linear orientation,
    without a transpose (validated by vmfcore). CMF keeps experts as
    SEPARATE records (carrier of a per-expert dtype, P15 claim 12) — here
    the fuse expands into `experts.{e}.{gate,up,down}_proj.weight`; each
    expert is a contiguous row slice → slice-pushdown into the source's
    load_rows (ranged GET for the stream, seek for local files)."""

    def __init__(self, inner):
        self.inner = inner
        self._map = {}   # per-expert name → (fused name, kind, e)
        self._names = []
        self._cache = {}  # fused name → f32 [rows, cols] (LRU, capacity 2)
        for n in inner.names():
            base = n[:-len(".weight")] if n.endswith(".weight") else n
            if base.endswith(".mlp.experts.gate_up_proj"):
                p = base[:-len("gate_up_proj")]
                E = inner.shape(n)[0]
                for e in range(E):
                    for kind in ("gate", "up"):
                        nn = f"{p}{e}.{kind}_proj.weight"
                        self._map[nn] = (n, kind, e)
                        self._names.append(nn)
            elif base.endswith(".mlp.experts.down_proj"):
                p = base[:-len("down_proj")]
                E = inner.shape(n)[0]
                for e in range(E):
                    nn = f"{p}{e}.down_proj.weight"
                    self._map[nn] = (n, "down", e)
                    self._names.append(nn)
            else:
                self._names.append(n)

    def names(self):
        return list(self._names)

    def shape(self, name):
        if name not in self._map:
            return self.inner.shape(name)
        fused, kind, _ = self._map[name]
        E, a, b = self.inner.shape(fused)
        return [a // 2, b] if kind in ("gate", "up") else [a, b]

    def _rows(self, name):
        fused, kind, e = self._map[name]
        E, a, b = self.inner.shape(fused)
        if kind == "down":
            return fused, e * a, (e + 1) * a
        inter = a // 2
        r0 = e * a + (0 if kind == "gate" else inter)
        return fused, r0, r0 + inter

    def load(self, name) -> np.ndarray:
        if name not in self._map:
            return self.inner.load(name)
        fused, r0, r1 = self._rows(name)
        # A layer's experts run in canonical order back-to-back → load the
        # bank WHOLE once (bulk in 16MB — cheap even for streaming), slice
        # from cache; capacity 2 = gate_up + down of this layer (~3GB f32 35B).
        if fused not in self._cache:
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            flat = self.inner.load(fused)
            self._cache[fused] = flat.reshape(-1, flat.shape[-1])
        return np.ascontiguousarray(self._cache[fused][r0:r1])


class CmfSource:
    """Source = an existing .cmf (CMF→CMF requantization): tensors are
    dequantized by the standalone reader (python/cmf_reader), arch is taken
    from the header as-is. The main case — vbit-A/B on large models without
    re-streaming 70GB: the q8_2f master already sits locally."""

    def __init__(self, path: Path):
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
        from cmf_reader import CmfReader
        self.r = CmfReader(path)
        print(f"  Source: requant from {path} ({len(self.r.tensors)} tensors, "
              f"quant {self.r.header.get('quant_type')})")

    def names(self):
        return list(self.r.tensors.keys())

    def shape(self, name):
        e, _ = self.r.tensors[name]
        return list(e["shape"])

    def load(self, name) -> np.ndarray:
        return np.asarray(self.r.tensor(name), dtype=np.float32)


class DropPrefixSource:
    """Drops tensors by prefix (e.g. model.mtp.* for checkpoints whose
    MTP block the runtime does not execute)."""

    def __init__(self, inner, prefixes):
        self.inner = inner
        self.prefixes = tuple(prefixes)

    def names(self):
        return [n for n in self.inner.names()
                if not n.startswith(self.prefixes)]

    def shape(self, name):
        return self.inner.shape(name)

    def load(self, name):
        return self.inner.load(name)

    def load_rows(self, name, r0, r1):
        return self.inner.load_rows(name, r0, r1)


class DictSource:
    """In-RAM source for npz / torch checkpoints (small models, tests)."""

    def __init__(self, weights: dict):
        self.weights = {RENAMES.get(k, k): np.asarray(v) for k, v in weights.items()}

    def names(self):
        return list(self.weights.keys())

    def shape(self, name):
        return list(self.weights[name].shape)

    def load(self, name) -> np.ndarray:
        return self.weights[name].astype(np.float32, copy=False)


class GdnFusedSplitSource:
    """qwen3_next checkpoints (HF Qwen3Next / AgentWorld) pack the
    GDN projections fused: in_proj_qkvz and in_proj_ba, rows interleaved
    by GROUPS of k-heads — group g carries [q_g(dk), k_g(dk),
    v-block(r·dv), z-block(r·dv)], r = nv/nk (fix_query_key_value_ordering
    in transformers). The CMF canon is the Qwen3.5-hub layout: separate
    in_proj_{qkv,z,a,b} with plain concatenation over heads. This is a pure
    row permutation, not a single changed value; conv1d is already in
    canonical order (HF applies the convolution AFTER reordering)."""

    def __init__(self, inner, arch: dict):
        self.inner = inner
        self.nk = arch["linear_num_key_heads"]
        self.dk = arch["linear_key_head_dim"]
        self.nv = arch["linear_num_value_heads"]
        self.dv = arch["linear_value_head_dim"]
        self._map = {}  # new name → (original fused name, field)
        self._names = []
        for n in inner.names():
            if n.endswith(".linear_attn.in_proj_qkvz.weight"):
                p = n[:-len("in_proj_qkvz.weight")]
                for t in ("qkv", "z"):
                    nn = f"{p}in_proj_{t}.weight"
                    self._map[nn] = (n, t)
                    self._names.append(nn)
            elif n.endswith(".linear_attn.in_proj_ba.weight"):
                p = n[:-len("in_proj_ba.weight")]
                for t in ("b", "a"):
                    nn = f"{p}in_proj_{t}.weight"
                    self._map[nn] = (n, t)
                    self._names.append(nn)
            else:
                self._names.append(n)

    def names(self):
        return list(self._names)

    def shape(self, name):
        if name not in self._map:
            return self.inner.shape(name)
        hid = self.inner.shape(self._map[name][0])[1]
        rows = {"qkv": 2 * self.nk * self.dk + self.nv * self.dv,
                "z": self.nv * self.dv,
                "b": self.nv, "a": self.nv}[self._map[name][1]]
        return [rows, hid]

    def load(self, name) -> np.ndarray:
        if name not in self._map:
            return self.inner.load(name)
        orig, kind = self._map[name]
        w = np.asarray(self.inner.load(orig), dtype=np.float32)
        nk, dk, nv, dv = self.nk, self.dk, self.nv, self.dv
        r = nv // nk
        hid = w.shape[1]
        if kind in ("qkv", "z"):
            g = w.reshape(nk, 2 * dk + 2 * r * dv, hid)
            if kind == "z":
                return np.ascontiguousarray(
                    g[:, 2 * dk + r * dv:].reshape(nv * dv, hid))
            return np.concatenate([
                g[:, :dk].reshape(nk * dk, hid),
                g[:, dk:2 * dk].reshape(nk * dk, hid),
                g[:, 2 * dk:2 * dk + r * dv].reshape(nv * dv, hid),
            ], axis=0)
        g = w.reshape(nk, 2 * r, hid)
        blk = g[:, :r] if kind == "b" else g[:, r:]
        return np.ascontiguousarray(blk.reshape(nv, hid))


class VmfPhaseFoldSource:
    """Fold GatedDeltaNet linear layers onto the canonical vmf_phase core
    at CONVERT time (the runtime never executes vendor recurrences):

      v_proj   ← v-rows of linear_attn.in_proj_qkv      (real weights)
      out_proj ← linear_attn.out_proj                    (real weights)
      thq/thk  ← uniformly subsampled q/k rows           (init; heal pending)
      A_log    ← per-v-head GDN A_log tiled to 2·nphase  (decay carried)

    Quality needs an offline FCD-heal pass (vmf_swap_heal); throughput
    measurements are valid immediately — dims and weight traffic are the
    canonical core's own. `linear_attn.*` tensors are not written.
    """

    def __init__(self, inner, arch: dict, nphase: int, heal_dir=None):
        self.inner = inner
        self.nph = nphase
        self.heal_dir = Path(heal_dir) if heal_dir else None
        self.healed_count = 0
        self.nk = arch["linear_num_key_heads"]
        self.nv = arch["linear_num_value_heads"]
        self.dk = arch.get("linear_key_head_dim") or arch["head_dim"]
        self.dv = arch.get("linear_value_head_dim") or arch["head_dim"]
        self.hidden = arch["hidden_size"]
        self.rep = self.nv // self.nk
        self.kd = self.nk * self.dk

        self._names = []
        self.synth = {}  # synthesized name → (layer prefix, kind)
        for n in inner.names():
            if ".linear_attn." not in n:
                self._names.append(n)
                continue
            prefix = n.split("linear_attn.")[0]  # "model.layers.{i}."
            if prefix in {p for p, _ in self.synth.values()}:
                continue
            for kind, name in [
                ("thq", f"{prefix}vmf_attn.thq.weight"),
                ("thk", f"{prefix}vmf_attn.thk.weight"),
                ("v", f"{prefix}vmf_attn.v_proj.weight"),
                ("out", f"{prefix}vmf_attn.out_proj.weight"),
                ("alog", f"{prefix}vmf_attn.A_log"),
            ]:
                self.synth[name] = (prefix, kind)
                self._names.append(name)

    def names(self):
        return list(self._names)

    def shape(self, name):
        if name not in self.synth:
            return self.inner.shape(name)
        _, kind = self.synth[name]
        return {
            "thq": [self.nv * self.nph, self.hidden],
            "thk": [self.nv * self.nph, self.hidden],
            "v": [self.nv * self.dv, self.hidden],
            "out": self.inner.shape(self.synth[name][0] + "linear_attn.out_proj.weight"),
            "alog": [self.nv * 2 * self.nph],
        }[kind]

    def load(self, name) -> np.ndarray:
        if name not in self.synth:
            return self.inner.load(name)
        prefix, kind = self.synth[name]
        # FCD-healed weights (converter/heal_vmf_phase.py) take priority
        # over the fold init.
        if self.heal_dir is not None:
            li = int(prefix.split(".")[2])
            f = self.heal_dir / f"heal_L{li}.npz"
            if f.exists():
                key = {"thq": "thq", "thk": "thk", "v": "v_proj",
                       "out": "out_proj", "alog": "a_log"}[kind]
                self.healed_count += 1
                return np.load(f)[key].astype(np.float32)
        if kind == "out":
            return self.inner.load(prefix + "linear_attn.out_proj.weight")
        if kind == "alog":
            a = self.inner.load(prefix + "linear_attn.A_log").reshape(-1)  # [nv]
            return np.repeat(a[:, None], 2 * self.nph, axis=1).flatten().astype(np.float32)
        qkv = self.inner.load(prefix + "linear_attn.in_proj_qkv.weight")  # [2·kd+vd, hidden]
        if kind == "v":
            return qkv[2 * self.kd:2 * self.kd + self.nv * self.dv]
        # thq/thk: per v-head, subsample nphase rows from its source q/k head.
        base = 0 if kind == "thq" else self.kd
        rows = np.empty((self.nv * self.nph, self.hidden), dtype=np.float32)
        for h in range(self.nv):
            ko = h // self.rep
            for i in range(self.nph):
                src = base + ko * self.dk + (i * self.dk) // self.nph
                rows[h * self.nph + i] = qkv[src]
        return rows


def open_weight_source(model_path: Path):
    st_files = sorted(model_path.glob("*.safetensors"))
    if st_files:
        return SafetensorsSource(st_files)

    pt_files = sorted(model_path.glob("*.bin")) + sorted(model_path.glob("*.pt"))
    if pt_files:
        import torch
        weights = {}
        for pf in pt_files:
            state = torch.load(str(pf), map_location="cpu", weights_only=True)
            for k, v in state.items():
                weights[k] = v.float().numpy()
        print(f"  Loaded {len(weights)} tensors from {len(pt_files)} PyTorch files")
        return DictSource(weights)

    npz_files = sorted(model_path.glob("*.npz"))
    if npz_files:
        weights = {}
        for nf in npz_files:
            with np.load(str(nf)) as z:
                for k in z.files:
                    weights[k] = z[k]
        print(f"  Loaded {len(weights)} tensors from {len(npz_files)} npz files")
        return DictSource(weights)

    raise FileNotFoundError(f"No model weights found in {model_path}")


def load_arch(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found — arch is required in v2")
    with open(config_path) as f:
        config = json.load(f)
    tc = config.get("text_config", config)

    n_layers = tc["num_hidden_layers"]
    lt_raw = tc.get("layer_types", [])
    layer_types = ["FullAttention" if t == "full_attention" else "LinearAttention"
                   for t in lt_raw]
    if not layer_types:
        layer_types = ["FullAttention"] * n_layers

    model_type = config.get("model_type", "unknown")
    # Zero-centered RMSNorm x̂·(1+w): Gemma family AND Qwen3.5/Qwen3-Next
    # (validated oracle: vmfcore/qwen35_oracle.py — "GEMMA norms (1+w)",
    # including per-head q/k-norm). Classic Qwen/Llama stay x̂·w.
    mt = model_type.lower()
    norm_style = "gemma" if ("gemma" in mt or mt.startswith("qwen3_5") or "qwen3_next" in mt) else "qwen"

    hidden = tc["hidden_size"]
    n_heads = tc["num_attention_heads"]

    # MoE (Qwen2-MoE / Qwen3-MoE): presence of num_experts in the config.
    # Which layers are MoE — decided by the presence of the mlp.gate.weight
    # router in the directory (mlp_only_layers stay dense on their own).
    moe = None
    if tc.get("num_experts"):
        # norm_topk_prob is absent from the qwen3_5/next-family config —
        # the model code's default is True (Qwen3NextConfig); qwen2_moe — False.
        ntp_default = mt.startswith("qwen3_5") or "qwen3_next" in mt
        moe = {
            "num_experts": tc["num_experts"],
            "top_k": tc.get("num_experts_per_tok", 2),
            "moe_intermediate_size": tc["moe_intermediate_size"],
            "norm_topk_prob": bool(tc.get("norm_topk_prob", ntp_default)),
        }
        if tc.get("shared_expert_intermediate_size"):
            moe["shared_expert_intermediate_size"] = \
                tc["shared_expert_intermediate_size"]

    return {
        "arch_name": model_type,
        "moe": moe,
        "hidden_size": hidden,
        # A pure MoE (qwen3_5_moe) has no dense intermediate_size —
        # substitute moe_intermediate_size (the dense path is unused).
        "intermediate_size": tc.get("intermediate_size",
                                    tc.get("moe_intermediate_size", 0)),
        "num_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_kv_heads": tc.get("num_key_value_heads", n_heads),
        "head_dim": tc.get("head_dim", hidden // n_heads),
        "vocab_size": tc["vocab_size"],
        "layer_types": layer_types,
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "norm_style": norm_style,
        "rope_theta": tc.get("rope_theta", tc.get("rope_parameters", {}).get("rope_theta", 10000.0)),
        "partial_rotary_factor": tc.get(
            "partial_rotary_factor",
            tc.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)),
        "tie_word_embeddings": bool(config.get("tie_word_embeddings", False)),
        "max_position_embeddings": tc.get("max_position_embeddings", 4096),
        "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim"),
        "linear_num_key_heads": tc.get("linear_num_key_heads"),
        "linear_num_value_heads": tc.get("linear_num_value_heads"),
        "linear_key_head_dim": tc.get("linear_key_head_dim"),
        "linear_value_head_dim": tc.get("linear_value_head_dim"),
    }


# ───────────────────── masks ─────────────────────

def load_masks(masks_dir: Path) -> list:
    masks = []
    for mask_file in sorted(masks_dir.glob("*.pt")) + sorted(masks_dir.glob("*.json")):
        if mask_file.suffix == ".json":
            with open(mask_file) as f:
                md = json.load(f)
        else:
            import torch
            md = torch.load(str(mask_file), map_location="cpu", weights_only=True)
            for key in ("ffn_masks", "head_masks"):
                if key in md:
                    md[key] = [m.numpy().tolist() if hasattr(m, "numpy") else m
                               for m in md[key]]
            if hasattr(md.get("layer_gates"), "tolist"):
                md["layer_gates"] = md["layer_gates"].tolist()
        name = md.get("metadata", {}).get("task_name", mask_file.stem)
        print(f"  Mask: {name}")
        masks.append(md)
    return masks


def bitfield(values, n_bits: int) -> bytearray:
    """Float/bool list → LSB-first bitfield with zeroed tail bits.
    None → all active."""
    out = bytearray((n_bits + 7) // 8)
    if values is None:
        for i in range(n_bits):
            out[i // 8] |= 1 << (i % 8)
        return out
    for i, v in enumerate(values[:n_bits]):
        if float(v) > 0.5:
            out[i // 8] |= 1 << (i % 8)
    return out


def build_mask_entries(masks_data: list, arch: dict) -> list:
    """→ list of dicts: meta + 'blob' bytes (spec §5)."""
    n_layers = arch["num_layers"]
    inter = arch["intermediate_size"]
    n_heads = arch["num_attention_heads"]
    entries = []
    for task_id, md in enumerate(masks_data):
        meta = md.get("metadata", {})
        blob = bytearray()
        ffn = md.get("ffn_masks", [])
        for li in range(n_layers):
            blob += bitfield(ffn[li] if li < len(ffn) else None, inter)
        heads = md.get("head_masks", [])
        for li in range(n_layers):
            blob += bitfield(heads[li] if li < len(heads) else None, n_heads)
        gates = list(md.get("layer_gates", []))
        gates += [True] * (n_layers - len(gates))
        gate_bits = bytearray((n_layers + 7) // 8)
        for li in range(n_layers):
            if gates[li]:
                gate_bits[li // 8] |= 1 << (li % 8)
        blob += gate_bits

        quality = meta.get("quality")  # dict {metric, value, ...} or None
        if quality is not None and "metric" not in quality:
            raise ValueError(
                f"mask '{meta.get('task_name')}': quality must be a measured "
                "contract {metric, value, ...}, not a bare score")

        entries.append({
            "task_id": task_id,
            "name": meta.get("task_name", f"task_{task_id}"),
            "description": meta.get("description"),
            "sparsity": float(meta.get("sparsity", 0.0)),
            "quality": quality,
            "parent": None,
            "priority": "Fallback" if task_id == 0 else "Normal",
            "has_hot_pack": False,
            "blob": bytes(blob),
        })
    return entries


def encode_masks_section(entries: list, default_task: str) -> bytes:
    """[u32 n][u32 meta_len][meta JSON][blobs 8-aligned] — symmetric to
    cortiq-core mask::encode_masks_section (offsets stabilized by loop)."""
    def build_meta(blobs_start: int):
        metas, off = [], blobs_start
        for e in entries:
            off = align(off, 8)
            metas.append({
                "task_id": e["task_id"], "name": e["name"],
                "description": e["description"], "sparsity": e["sparsity"],
                "quality": e["quality"], "parent": e["parent"],
                "priority": e["priority"], "has_hot_pack": e["has_hot_pack"],
                "blob_off": off, "blob_len": len(e["blob"]),
            })
            off += len(e["blob"])
        return {"default_task": default_task, "masks": metas}

    meta_len = 0
    while True:
        meta_json = json.dumps(build_meta(8 + meta_len)).encode()
        if len(meta_json) == meta_len:
            break
        meta_len = len(meta_json)

    meta = build_meta(8 + meta_len)
    out = bytearray()
    out += struct.pack("<I", len(entries))
    out += struct.pack("<I", meta_len)
    out += meta_json
    for mm, e in zip(meta["masks"], entries):
        out += b"\x00" * (mm["blob_off"] - len(out))
        out += e["blob"]
    return bytes(out)


# ───────────────────── sparse index ─────────────────────

def build_sparse_index(entries: list, arch: dict) -> bytes:
    n_layers = arch["num_layers"]
    inter = arch["intermediate_size"]
    n_heads = arch["num_attention_heads"]
    ffn_b = (inter + 7) // 8
    head_b = (n_heads + 7) // 8
    n_groups = (inter + 31) // 32

    records = []
    for e in entries:
        blob = e["blob"]
        gates_base = n_layers * (ffn_b + head_b)
        for li in range(n_layers):
            if not (blob[gates_base + li // 8] >> (li % 8)) & 1:
                continue
            row = blob[li * ffn_b:(li + 1) * ffn_b]
            groups = [g for g in range(n_groups)
                      if any(row[g * 4:min(g * 4 + 4, len(row))])]
            hrow = blob[n_layers * ffn_b + li * head_b:
                        n_layers * ffn_b + (li + 1) * head_b]
            heads = [h for h in range(n_heads) if (hrow[h // 8] >> (h % 8)) & 1]
            records.append((e["task_id"], li, groups, heads))

    out = bytearray()
    out += struct.pack("<II", len(records), 0)
    for task_id, li, groups, heads in records:
        out += struct.pack("<IIII", task_id, li, len(groups), len(heads))
        out += struct.pack(f"<{len(groups)}H", *groups) if groups else b""
        out += bytes(heads)
        out += b"\x00" * ((-len(out)) % 4)
    return bytes(out)


# ───────────────────── tensor directory ─────────────────────

def encode_directory(entries: list) -> bytes:
    """[u64 count][u64 pool_off][count × 56-byte records][name pool]."""
    pool = bytearray()
    noffs = []
    for e in entries:
        nb = e["name"].encode("utf-8")
        noffs.append((len(pool), len(nb)))
        pool += nb
    pool_off = 16 + len(entries) * DIR_RECORD_LEN

    out = bytearray()
    out += struct.pack("<QQ", len(entries), pool_off)
    for e, (noff, nlen) in zip(entries, noffs):
        sh = list(e["shape"])[:DIR_MAX_NDIM]
        shp = sh + [0] * (DIR_MAX_NDIM - len(sh))
        out += struct.pack("<IHBB6IQQQ", noff, nlen, DTYPE_ID[e["dtype"]],
                           len(sh), *shp, e["off"], e["nbytes"], e["hash"])
    out += pool
    return bytes(out)


# ───────────────────── main conversion ─────────────────────

def canonical_tensor_order(names) -> list:
    """embed → final norm → lm_head → layers by (index, name) → mtp → rest."""
    def key(name: str):
        if name == "model.embed_tokens.weight":
            return (0, 0, name)
        if name == "model.norm.weight":
            return (1, 0, name)
        if name == "lm_head.weight":
            return (2, 0, name)
        if name.startswith("model.layers."):
            li = int(name.split(".")[2])
            return (3, li, name)
        if name.startswith("model.mtp."):
            return (4, 0, name)
        return (5, 0, name)
    return sorted(names, key=key)


def detect_mtp(names, config: dict) -> dict | None:
    """MTP head: `model.mtp.*` tensors (spec §2.1) or a config declaration
    (num_nextn_predict_layers / mtp_num_layers). Shared embed + lm_head."""
    names = set(names)
    mtp_layers = {int(n.split(".")[3]) for n in names
                  if n.startswith("model.mtp.layers.")}
    n = (max(mtp_layers) + 1) if mtp_layers else 0
    if n == 0:
        tc = config.get("text_config", config)
        n = int(tc.get("num_nextn_predict_layers", tc.get("mtp_num_layers", 0)))
        if n and not any(k.startswith("model.mtp.") for k in names):
            print(f"  MTP: config declares {n} block(s) but no model.mtp.* tensors — skipping")
            return None
    if n == 0:
        return None
    required = ["model.mtp.enorm.weight", "model.mtp.hnorm.weight",
                "model.mtp.eh_proj.weight", "model.mtp.norm.weight"]
    missing = [r for r in required if r not in names]
    if missing:
        raise ValueError(f"MTP tensors present but incomplete, missing: {missing}")
    print(f"  MTP: {n} block(s), shared embed + lm_head")
    return {"num_layers": n, "share_lm_head": True, "share_embed": True}


def encoded_nbytes(dtype: str, shape) -> int:
    """Encoded size, computable from shape alone (mirrors Rust
    expected_nbytes) — this is what makes single-pass streaming possible."""
    n = int(np.prod(shape))
    if dtype == "f32":
        return n * 4
    if dtype == "f16":
        return n * 2
    if dtype == "q8_row":
        return n + int(shape[0]) * 2
    if dtype == "q8_2f":
        return n + int(shape[0]) * 2 + (n // int(shape[0])) * 2
    if dtype == "q4_block":
        groups = (n + GROUP_SIZE - 1) // GROUP_SIZE
        return groups * 16 + groups * 2
    raise ValueError(dtype)


class SkillsSource:
    """Wraps a source, adding skill.{id}.{orig} replacement tensors from
    make_skill.py output dirs (spec §9)."""

    def __init__(self, inner, skill_dirs):
        self.inner = inner
        self.extra = {}   # canonical skill name → npy path
        self.records = []  # header.skills entries
        for d in skill_dirs:
            d = Path(d)
            meta = json.load(open(d / "skill.json"))
            sid = meta["id"]
            for f in sorted((d / "tensors").glob("*.npy")):
                self.extra[f"skill.{sid}.{f.stem}"] = f
            self.records.append(meta)
            print(f"  Skill '{sid}': {len(list((d / 'tensors').glob('*.npy')))} "
                  f"replacement tensors, quality={meta.get('quality')}")

    def names(self):
        return self.inner.names() + list(self.extra.keys())

    def shape(self, name):
        if name in self.extra:
            return list(np.load(self.extra[name], mmap_mode="r").shape)
        return self.inner.shape(name)

    def load(self, name):
        if name in self.extra:
            return np.load(self.extra[name]).astype(np.float32)
        return self.inner.load(name)


def load_tokenizer_bundle(model_dir: Path):
    """Chat/eos bundle (spec §6.1): chat_template.jinja (preferred) or
    tokenizer_config.json's chat_template; stop ids from
    generation_config.json + the chat end token. The FILE — not the
    runtime binary — defines chat behavior."""
    template = None
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        template = jinja.read_text()
    tc_path = model_dir / "tokenizer_config.json"
    tc = json.load(open(tc_path)) if tc_path.exists() else {}
    if template is None:
        template = tc.get("chat_template")

    eos_ids = set()
    bos_id = pad_id = None
    gc_path = model_dir / "generation_config.json"
    if gc_path.exists():
        gc = json.load(open(gc_path))
        e = gc.get("eos_token_id")
        eos_ids.update([e] if isinstance(e, int) else (e or []))
        bos_id, pad_id = gc.get("bos_token_id"), gc.get("pad_token_id")
    # The chat end token stops generation too (Qwen: <|im_end|>).
    tok_path = model_dir / "tokenizer.json"
    if tc.get("eos_token") and tok_path.exists():
        tok = json.load(open(tok_path))
        for at in tok.get("added_tokens", []):
            if at["content"] == tc["eos_token"]:
                eos_ids.add(at["id"])

    if template is None and not eos_ids:
        return None
    bundle = {"eos_token_ids": sorted(eos_ids)}
    if template is not None:
        bundle["chat_template"] = template
    if bos_id is not None:
        bundle["bos_token_id"] = bos_id
    if pad_id is not None:
        bundle["pad_token_id"] = pad_id
    print(f"  Chat bundle: template {len(template) if template else 0} chars, "
          f"stop ids {sorted(eos_ids)}")
    return bundle


def _progress(frac: float, phase: str = "") -> None:
    """Machine-readable progress marker parsed by the gateway import job.
    Format: ``@PROGRESS <0..1> <phase text>`` on its own stdout line."""
    try:
        f = max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        f = 0.0
    print(f"@PROGRESS {f:.4f} {phase}", flush=True)


def convert(model_path: str, masks_dir: str | None, quant: str,
            output_path: str, tokenizer_path: str | None = None,
            linear_core: str = "vmf_phase", nphase: int = 64,
            heal_dir: str | None = None, skill_dirs: list | None = None,
            shard_max_gb: float | None = None, vbit_flat: bool = False,
            skip_mtp: bool = False, route_stats: str | None = None):
    # A non-local path = HF repo-id / URL → streaming conversion:
    # the meta (config/tokenizer) goes into a staging dir, weights — ranged GET.
    stream = None
    cmf_src = None
    _progress(0.003, "fetching metadata")
    if Path(model_path).is_file() and str(model_path).endswith(".cmf"):
        cmf_src = Path(model_path)
        model_path = cmf_src.parent  # arch/vocab taken from the header
    elif not Path(model_path).exists():
        stream = str(model_path)
        base = stream if stream.startswith("http") \
            else f"https://huggingface.co/{stream}/resolve/main"
        staging = Path("models") / (stream.rstrip("/").split("/")[-1] + "-meta")
        staging.mkdir(parents=True, exist_ok=True)
        for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
                   "generation_config.json"):
            tgt = staging / fn
            if not tgt.exists():
                try:
                    tgt.write_bytes(_http_get(f"{base}/{fn}"))
                except Exception:
                    if fn == "config.json":
                        raise
        model_path = staging
        print(f"  Stream-meta: {staging}")
    model_path = Path(model_path)
    default_dtype = QUANT_CHOICES[quant]

    print(f"\n  Converting → CMF v2 (streaming)")
    print(f"  Model:  {model_path}")
    print(f"  Masks:  {masks_dir or 'none'}")
    print(f"  Quant:  {quant} ({default_dtype})")
    print(f"  Output: {output_path}\n")
    _progress(0.01, "reading model")

    if cmf_src:
        source = CmfSource(cmf_src)
        arch = dict(source.r.header["arch"])
    else:
        arch = load_arch(model_path)
    print(f"  Arch: {arch['arch_name']} | {arch['num_layers']}L | "
          f"hidden={arch['hidden_size']} | FFN={arch['intermediate_size']} | "
          f"norm={arch['norm_style']}")
    _progress(0.02, "opening weights")

    if not cmf_src:
        source = HfStreamSource(stream) if stream else open_weight_source(model_path)
    names = source.names()

    if skip_mtp and any(n.startswith("model.mtp.") for n in names):
        source = DropPrefixSource(source, ["model.mtp."])
        names = source.names()
        print("  MTP: skipped (--skip-mtp)")

    # AgentWorld/qwen3_5_moe: fused expert banks → separate records
    # (per-expert dtype = claim 12); row slices go by slice-pushdown.
    if any(".mlp.experts.gate_up_proj" in n for n in names):
        source = MoeFusedExpertsSource(source)
        names = source.names()
        print(f"  MoE: fused expert banks expanded → {len(names)} tensors")

    if cmf_src:
        raw_config = {}          # arch is already canonical, from the header
    else:
        with open(model_path / "config.json") as f:
            raw_config = json.load(f)
        arch["mtp"] = detect_mtp(names, raw_config)
        if arch["mtp"] is None:
            del arch["mtp"]

    # qwen3_next-style: fused GDN projections → canonical split
    # (pure row permutation, BEFORE fold/faithful logic).
    if any(n.endswith(".linear_attn.in_proj_qkvz.weight") for n in names):
        source = GdnFusedSplitSource(source, arch)
        names = source.names()
        print("  GDN: fused in_proj_qkvz/ba → canonical split "
              "(qwen3_next checkpoint)")

    # Linear layers: faithful vendor operator by default (carried 1:1,
    # no training); the canonical-core fold is the research option.
    has_linear = any(t == "LinearAttention" for t in arch["layer_types"])
    has_gdn = any(".linear_attn." in n for n in names)
    if has_linear and has_gdn and linear_core == "vmf_phase":
        source = VmfPhaseFoldSource(source, arch, nphase, heal_dir)
        names = source.names()
        if heal_dir:
            n_healed = sum(1 for f in Path(heal_dir).glob("heal_L*.npz"))
            print(f"  Heal: {n_healed} healed layer(s) from {heal_dir}")
        arch["linear_core"] = {
            "kind": "vmf_phase",
            "num_heads": source.nv,
            "nphase": nphase,
            "value_head_dim": source.dv,
        }
        print(f"  Linear core: GDN → vmf_phase fold "
              f"({source.nv} heads × {nphase} phases, dv={source.dv}; "
              f"v/out/A_log carried, thq/thk init — heal pending)")
    elif has_linear and has_gdn:
        nv = arch.get("linear_num_value_heads")
        dv = arch.get("linear_value_head_dim")
        nk = arch.get("linear_num_key_heads")
        dk = arch.get("linear_key_head_dim")
        kk = arch.get("linear_conv_kernel_dim")
        if not all((nv, dv, nk, dk, kk)):
            raise SystemExit("GDN model without linear_* dims in config.json")
        arch["linear_core"] = {
            "kind": "gated_delta_net",
            "num_heads": nv,
            "value_head_dim": dv,
        }
        print(f"  Linear core: gated_delta_net carried 1:1 (faithful; "
              f"nv={nv} nk={nk} dk={dk} dv={dv} conv={kk})")

    skills_meta = []
    if cmf_src and not skill_dirs:
        skills_meta = source.r.skills   # the swarm registry is inherited as-is
    if skill_dirs:
        source = SkillsSource(source, skill_dirs)
        names = source.names()
        skills_meta = source.records

    # C4 (P15 claim 12): per-expert bit allocation. Experts of one
    # layer and one projection share a COMMON bit budget: the shift
    # ā_expert − ā_family turns per-tensor water-filling into a joint one
    # over the rows of the whole family (a loud expert gets more bits).
    VBIT_BIAS.clear()
    if default_dtype == "vbit" and arch.get("moe") and not vbit_flat:
        import re as _re
        _exp = _re.compile(
            r"^model\.layers\.(\d+)\.mlp\.experts\.\d+\.(\w+_proj)\.weight$")
        fams: dict = {}
        for n in names:
            mm = _exp.match(n)
            if mm and int(source.shape(n)[1]) % GROUP_SIZE == 0:
                fams.setdefault((mm.group(1), mm.group(2)), []).append(n)
        # B-field (claim 12, b ∝ log2(A·B)): how often the router selects
        # the expert on calibration (CMF_MOE_STATS in cortiq ppl). Laplace
        # +1: a dead expert → a large negative → floor of 3 bits.
        rstats = None
        if route_stats:
            rstats = {int(k): np.asarray(v, np.float64)
                      for k, v in json.load(open(route_stats)).items()}
            print(f"  MoE vbit: B-field from {route_stats} "
                  f"({len(rstats)} layers, {int(sum(v.sum() for v in rstats.values()))} selections)")
        levels = np.asarray(VBIT_LEVELS, np.float64)

        def _fam_mean_bits(rowdev, delta):
            raw = VBIT_MEAN_BITS[0] + rowdev + delta
            return float(levels[np.abs(raw[:, None] - levels).argmin(1)].mean())

        for (li, proj), members in sorted(fams.items()):
            am, avec = {}, {}
            for n in members:
                t = source.load(n).astype(np.float32)
                a = np.log2(np.maximum(np.abs(t).max(axis=1), 1e-12))
                am[n] = float(a.mean())
                avec[n] = a - a.mean()   # the tensor's row deviations
                del t
            fam_mean = sum(am.values()) / len(am)
            bfield = {}
            if rstats is not None and int(li) in rstats:
                counts = rstats[int(li)]
                lg = np.log2(counts + 1.0)
                lg -= lg.mean()
                _exp_e = _re.compile(r"\.experts\.(\d+)\.")
                bfield = {n: float(lg[int(_exp_e.search(n).group(1))])
                          for n in members}
            # FAMILY budget neutrality (not global): the raw shift is
            # clipped (±2.5 bits — within the dynamics of levels
            # 3..8) and re-centered; otherwise the floor clamp
            # asymmetrically inflates the family, and a global mean_bits
            # compensation starves NON-expert tensors (measured on 35B:
            # mean 3.62 → EN PPL 18.7, worse than flat).
            raw_b = {n: am[n] - fam_mean + bfield.get(n, 0.0)
                     for n in members}
            clipped = {n: float(np.clip(v, -2.5, 2.5))
                       for n, v in raw_b.items()}
            center = sum(clipped.values()) / len(clipped)
            # EXACT family budget neutrality: levels {3,4,5,6,8} are
            # discrete and the clamp is asymmetric, so centering the
            # raw shifts does not hold the budget (+7% measured on 35B).
            # Bisection of a common shift δ over the bits ACTUALLY assigned
            # to all rows of the family: f(δ) is monotone.
            rowdev = np.concatenate(
                [avec[n] + clipped[n] - center for n in members])
            lo, hi = -3.0, 3.0
            for _ in range(40):
                mid = (lo + hi) / 2
                if _fam_mean_bits(rowdev, mid) < VBIT_MEAN_BITS[0]:
                    lo = mid
                else:
                    hi = mid
            delta = (lo + hi) / 2
            for n in members:
                VBIT_BIAS[n] = clipped[n] - center + delta
        if fams:
            spread = max(abs(v) for v in VBIT_BIAS.values())
            print(f"  MoE vbit: joint budget across {len(fams)} expert "
                  f"families (max|shift| {spread:.2f} bits"
                  f"{', A·B' if rstats else ', A only'})")

    # Sharding (spec §10): greedy split of the tensor list into groups
    # ≤ shard_max_gb; each shard is written as a standalone .cmf via a
    # recursive convert() call with its own tensor subset.
    if shard_max_gb:
        budget = shard_max_gb * 1e9
        groups, cur, size = [], [], 0
        for name in canonical_tensor_order(names):
            nb = int(np.prod(source.shape(name))) * 4  # rough f32 estimate
            if cur and size + nb > budget:
                groups.append(cur)
                cur, size = [], 0
            cur.append(name)
            size += nb
        if cur:
            groups.append(cur)
        count = len(groups)
        base = Path(output_path)
        stem = base.name[:-4] if base.name.endswith(".cmf") else base.name
        print(f"  Sharding: {count} files ≤ {shard_max_gb} GB (rough f32)")

        class SubsetSource:
            def __init__(self, inner, keep):
                self.inner, self.keep = inner, set(keep)
            def names(self):
                return [n for n in self.inner.names() if n in self.keep]
            def shape(self, n):
                return self.inner.shape(n)
            def load(self, n):
                return self.inner.load(n)

        for no, group in enumerate(groups, 1):
            out = base.with_name(f"{stem}-{no:05}-of-{count:05}.cmf")
            _write_cmf(SubsetSource(source, group), dict(arch), quant,
                       str(out), model_path,
                       skills_meta if no == 1 else [],
                       masks_dir if no == 1 else None,
                       tokenizer_path, heal_dir,
                       shard={"no": no, "count": count},
                       with_extras=(no == 1))
        _progress(1.0, "done")
        return

    _write_cmf(source, arch, quant, output_path, model_path, skills_meta,
               masks_dir, tokenizer_path, heal_dir,
               shard=None, with_extras=True)
    _progress(1.0, "done")


def _write_cmf(source, arch, quant, output_path, model_path, skills_meta,
               masks_dir, tokenizer_path, heal_dir, shard, with_extras):
    names = source.names()
    default_dtype = QUANT_CHOICES[quant]

    # Shard-aware progress: map a local 0..1 into this shard's global slice.
    sh_no = shard["no"] if shard else 1
    sh_ct = shard["count"] if shard else 1

    def _emit(local: float, phase: str = "") -> None:
        overall = ((sh_no - 1) + max(0.0, min(1.0, local))) / sh_ct
        _progress(overall, phase)

    # ── Phase 1: full layout from shapes alone (no tensor data read) ──
    order = canonical_tensor_order(names)
    _emit(0.02, "analyzing tensors")
    dir_entries = []
    cursor = 0
    n_order = len(order)
    for idx, name in enumerate(order):
        if (idx + 1) % 100 == 0 or idx + 1 == n_order:
            _emit(0.02 + 0.10 * ((idx + 1) / max(n_order, 1)), "planning layout")
        shape = source.shape(name)
        # pick_dtype needs ndim/size only — probe with an empty array.
        probe = np.empty(shape, dtype=np.uint8) if int(np.prod(shape)) < 1 \
            else np.lib.stride_tricks.as_strided(np.zeros(1, np.float32), shape, [0] * len(shape))
        dtype = pick_dtype(name, probe, default_dtype)
        if dtype == "vbit":
            # vbit size depends on the data: a cheap pre-pass over the rows
            # (streaming, one tensor in RAM; phase 2 recomputes the same bits).
            nbytes = vbit_nbytes(vbit_bits(source.load(name).astype(np.float32),
                                           bias=VBIT_BIAS.get(name, 0.0)),
                                 int(shape[1]))
        else:
            nbytes = encoded_nbytes(dtype, shape)
        cursor = align(cursor, TENSOR_ALIGNMENT)
        dir_entries.append({
            "name": name, "dtype": dtype, "shape": list(shape),
            "off": cursor, "nbytes": nbytes, "hash": 0,
        })
        cursor += nbytes
    data_len = cursor

    # Masks + sparse index.
    masks_bytes = b""
    index_bytes = b""
    n_masks = 0
    if masks_dir and Path(masks_dir).exists():
        masks_data = load_masks(Path(masks_dir))
        if masks_data:
            entries = build_mask_entries(masks_data, arch)
            default_task = entries[0]["name"]
            masks_bytes = encode_masks_section(entries, default_task)
            index_bytes = build_sparse_index(entries, arch)
            n_masks = len(entries)

    # Tokenizer (only shard 1 carries vocab/bundle).
    vocab_bytes = b""
    if with_extras:
        tok = Path(tokenizer_path) if tokenizer_path else model_path / "tokenizer.json"
        if tok.exists():
            vocab_bytes = tok.read_bytes()
            print(f"  Tokenizer: embedded ({len(vocab_bytes) / 1e3:.0f} KB)")
        elif hasattr(source, "r"):
            vocab_bytes = bytes(source.r.vocab)   # requant: vocab from the .cmf
            print(f"  Tokenizer: inherited from source ({len(vocab_bytes) / 1e3:.0f} KB)")
        else:
            print("  Tokenizer: not found — model will need a sidecar tokenizer.json")

    # Header JSON (provenance carries runtime-relevant HF extras verbatim).
    raw_config = json.load(open(model_path / "config.json")) \
        if (model_path / "config.json").exists() else {}
    tc = raw_config.get("text_config", raw_config)
    extras = {k: tc[k] for k in
              ("attn_output_gate", "partial_rotary_factor", "full_attention_interval",
               "rope_parameters", "output_gate_type", "linear_key_head_dim",
               "linear_value_head_dim") if k in tc}
    header = {
        "format": "cmf",
        "version": CMF_VERSION,
        "arch": arch,
        "quant_type": quant if quant in ("F16", "F32") else quant,
        "tokenizer_config": ((source.r.header.get("tokenizer_config")
                               if hasattr(source, "r")
                               else load_tokenizer_bundle(Path(model_path)))
                              if with_extras else None),
        "skills": skills_meta,
        "shard": shard,
        "provenance": {
            "tool": "convert_dtgma_to_cmf.py",
            "source_model": str(model_path),
            "hf_config_extras": extras,
            "linear_fold": ({
                "from": "gated_delta_net",
                "v_proj/out_proj/A_log": "carried from source",
                "thq/thk": ("FCD-healed (converter/heal_vmf_phase.py)"
                            if heal_dir else
                            "subsampled q/k rows — FCD-heal pending"),
            } if arch.get("linear_core", {}).get("kind") == "vmf_phase"
                else ({"carried": "gated_delta_net 1:1 (faithful, no training)"}
                      if "linear_core" in arch else None)),
        },
    }
    # Section hashes go INTO the header (spec §8.1): the envelope's
    # header hash then transitively covers masks/vocab/index integrity.
    if masks_bytes or vocab_bytes or index_bytes:
        header["section_hashes"] = {
            k: f"{hash64(b):016x}"
            for k, b in (("masks", masks_bytes), ("vocab", vocab_bytes),
                         ("index", index_bytes)) if b
        }
    header_json = json.dumps(header).encode()
    dir_bytes = encode_directory(dir_entries)

    required = FEAT_TENSOR_DIR
    if masks_bytes:
        required |= FEAT_BINARY_MASKS
    if any(e["dtype"] in ("q8_2f", "vbit") for e in dir_entries):
        required |= FEAT_QUANT_2F

    header_off = ENVELOPE_LEN
    dir_off = header_off + len(header_json)
    data_off = align(dir_off + len(dir_bytes), DATA_ALIGNMENT)
    masks_off = data_off + data_len if masks_bytes else 0
    vocab_off = (data_off + data_len + len(masks_bytes)) if vocab_bytes else 0
    index_off = (data_off + data_len + len(masks_bytes) + len(vocab_bytes)) if index_bytes else 0

    env = bytearray()
    env += CMF_MAGIC
    env += struct.pack("<III", CMF_VERSION, 0, required)
    for off, ln in ((header_off, len(header_json)), (dir_off, len(dir_bytes)),
                    (data_off, data_len), (masks_off, len(masks_bytes)),
                    (vocab_off, len(vocab_bytes)), (index_off, len(index_bytes))):
        env += struct.pack("<QQ", off, ln)
    # Reserved bytes: header/dir integrity (spec §8.1). The dir hash is
    # patched again after phase 2 (per-tensor hashes land in the dir).
    env += struct.pack("<QQ", hash64(header_json), 0)
    env += b"\x00" * (ENVELOPE_LEN - len(env))
    assert len(env) == ENVELOPE_LEN

    # ── Phase 2: stream tensors (one in RAM at a time), then patch hashes ──
    total_gb = data_len / 1e9
    print(f"  Writing {output_path} ({len(dir_entries)} tensors, {total_gb:.2f} GB data) ...")
    _emit(0.14, f"writing {len(dir_entries)} tensors")
    t_start = time.time()
    with open(output_path, "w+b") as f:
        f.write(env)
        f.write(header_json)
        f.write(dir_bytes)
        f.write(b"\x00" * (data_off - dir_off - len(dir_bytes)))
        pos = data_off
        done = 0
        for i, e in enumerate(dir_entries):
            target = data_off + e["off"]
            f.write(b"\x00" * (target - pos))
            t = source.load(e["name"])
            data = encode_tensor(t, e["dtype"], e["name"])
            del t
            assert len(data) == e["nbytes"], f"{e['name']}: {len(data)} != {e['nbytes']}"
            e["hash"] = hash64(data)
            f.write(data)
            done += len(data)
            pos = target + len(data)
            del data
            n_dir = len(dir_entries)
            step = max(1, n_dir // 200)  # ~200 updates regardless of model size
            if (i + 1) % step == 0 or i + 1 == n_dir:
                _emit(0.14 + 0.84 * ((i + 1) / max(n_dir, 1)), f"[{i + 1}/{n_dir}]")
            if (i + 1) % 50 == 0 or i + 1 == n_dir:
                rate = done / 1e6 / max(time.time() - t_start, 1e-9)
                print(f"    [{i + 1}/{n_dir}] {done / 1e9:.2f}/{total_gb:.2f} GB "
                      f"({rate:.0f} MB/s)", flush=True)
        assert pos == data_off + data_len
        f.write(masks_bytes)
        f.write(vocab_bytes)
        f.write(index_bytes)

        # Patch the real hashes into the directory records
        # (record layout: ... off:u64 @+32, nbytes:u64 @+40, hash:u64 @+48).
        for i, e in enumerate(dir_entries):
            f.seek(dir_off + 16 + i * DIR_RECORD_LEN + 48)
            f.write(struct.pack("<Q", e["hash"]))
        # Final dir bytes are now on disk — seal the envelope dir hash.
        f.seek(dir_off)
        final_dir = f.read(len(dir_bytes))
        f.seek(0x78)
        f.write(struct.pack("<Q", hash64(final_dir)))

    total = Path(output_path).stat().st_size
    print(f"  ✓ {output_path}: {total / 1e9:.2f} GB | {len(dir_entries)} tensors | "
          f"{n_masks} masks | tokenizer {'yes' if vocab_bytes else 'no'} | "
          f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert DTG-MA model to CMF v2")
    p.add_argument("--model", required=True, help="Model checkpoint directory")
    p.add_argument("--masks", default=None, help="Masks directory")
    p.add_argument("--quant", default="Q8_ROW", choices=list(QUANT_CHOICES.keys()))
    p.add_argument("--output", required=True, help="Output .cmf path")
    p.add_argument("--tokenizer", default=None, help="tokenizer.json path override")
    p.add_argument("--linear-core", default="gated_delta_net",
                   choices=["gated_delta_net", "vmf_phase"],
                   help="gated_delta_net = faithful vendor operator, no "
                        "training (default); vmf_phase = fold onto the "
                        "canonical core (+offline heal)")
    p.add_argument("--nphase", type=int, default=64, help="vmf_phase phases per head")
    p.add_argument("--skip-mtp", action="store_true",
                   help="do not carry model.mtp.* (checkpoints with an MTP "
                        "block the runtime does not execute)")
    p.add_argument("--vbit-shape", choices=["log2", "cubic"], default="log2",
                   help="vbit allocation-curve shape (cubic — VMF experiment "
                        "#3: a soft causal cutoff instead of a step)")
    p.add_argument("--mean-bits", type=float, default=4.25,
                   help="target mean bit of vbit water-filling (equal size "
                        "in A/B comparisons)")
    p.add_argument("--route-stats", default=None,
                   help="JSON of expert-selection frequencies (CMF_MOE_STATS) "
                        "— water-filling B-field, full claim 12: b ∝ log2(A·B)")
    p.add_argument("--vbit-flat", action="store_true",
                   help="A/B arm: per-tensor water-filling WITHOUT the expert "
                        "family budget (claim 12 disabled)")
    p.add_argument("--shard-max-gb", type=float, default=None,
                   help="split into standalone shards ≤ N GB (spec §10)")
    p.add_argument("--skills", nargs="*", default=None,
                   help="skill dirs from make_skill.py (spec §9 swarm)")
    p.add_argument("--hessians", default=None,
                   help="dir with {tensor_name}.npz carrying 'hess' [in,in] "
                        "(GPTQ error-feedback for vbit; P13 claim 7)")
    p.add_argument("--heal-dir", default=None,
                   help="dir with heal_L{i}.npz (FCD-healed linear cores)")
    a = p.parse_args()
    if a.hessians:
        for f in Path(a.hessians).glob("*.npz"):
            HESSIANS[f.stem] = f
        print(f"  GPTQ: {len(HESSIANS)} hessians from {a.hessians}")
    VBIT_MEAN_BITS[0] = a.mean_bits
    VBIT_SHAPE[0] = a.vbit_shape
    convert(a.model, a.masks, a.quant, a.output, a.tokenizer, a.linear_core,
            a.nphase, a.heal_dir, a.skills, a.shard_max_gb, a.vbit_flat,
            a.skip_mtp, a.route_stats)
