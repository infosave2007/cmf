#!/usr/bin/env python3
"""Standalone Python reader for CMF v2 (D4): stdlib + numpy, none of the
CMF-runtime code. Source of truth — spec (docs/CMF_V2_SPEC.md) and the Rust
reader (crates/cortiq-core/src/format.rs); byte layouts are duplicated here
deliberately: the point of this file is to read the format WITHOUT our code.

    from cmf_reader import CmfReader
    r = CmfReader("model.cmf")            # shards are picked up automatically
    r.header["arch"]["hidden_size"]
    w = r.tensor("model.embed_tokens.weight")      # np.ndarray (dequantized)
    w = r.tensor("model.layers.0.mlp.gate_proj.weight", skill="ru")
    problems = r.verify()                 # [] = integrity confirmed

CLI: python3 cmf_reader.py model.cmf [--verify] [--tensor NAME]
"""
from __future__ import annotations

import json
import mmap
import re
import struct
from pathlib import Path

import numpy as np

CMF_MAGIC = b"CMF\x01"
ENVELOPE_LEN = 128
DIR_RECORD_LEN = 56
GROUP_SIZE = 32
KNOWN_FEATURES = 0b111  # tensor dir | binary masks | 2f quant

DTYPE_NAME = {0: "f32", 1: "f16", 2: "bf16", 3: "q8_row", 4: "q4_block",
              5: "mix8_4", 6: "u8", 7: "q4_col", 8: "vbit", 9: "q8_2f",
              10: "vbit_ro", 11: "q4_tiled"}

_SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.cmf$")


# ───────────────────── hash64 (≡ Rust hash64, ≡ converter) ─────────────────────

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


# ───────────────────── dequant (spec §4) ─────────────────────

def _dq_q8_row(raw: bytes, shape) -> np.ndarray:
    out, inn = shape
    q = np.frombuffer(raw, np.int8, out * inn).reshape(out, inn)
    sc = np.frombuffer(raw, np.float16, out, out * inn).astype(np.float32)
    return q.astype(np.float32) * sc[:, None]


def _dq_q8_2f(raw: bytes, shape) -> np.ndarray:
    out, inn = shape
    q = np.frombuffer(raw, np.int8, out * inn).reshape(out, inn)
    row = np.frombuffer(raw, np.float16, out, out * inn).astype(np.float32)
    col = np.frombuffer(raw, np.float16, inn, out * inn + out * 2).astype(np.float32)
    return q.astype(np.float32) * row[:, None] * col[None, :]


def _dq_q4_block(raw: bytes, shape) -> np.ndarray:
    n = int(np.prod(shape))
    ng = (n + GROUP_SIZE - 1) // GROUP_SIZE
    packed = np.frombuffer(raw, np.uint8, ng * 16).reshape(ng, 16)
    sc = np.frombuffer(raw, np.float16, ng, ng * 16).astype(np.float32)
    q = np.empty((ng, GROUP_SIZE), np.uint8)
    q[:, 0::2] = packed & 0x0F
    q[:, 1::2] = packed >> 4
    w = (q.astype(np.float32) - 8.0) * sc[:, None]
    return w.reshape(-1)[:n].reshape(shape)


def _dq_vbit(raw: bytes, shape) -> np.ndarray:
    rows, cols = shape
    ng = cols // GROUP_SIZE
    bits = np.frombuffer(raw, np.uint8, rows)
    sc = np.frombuffer(raw, np.float16, rows * ng, rows).astype(np.float32) \
        .reshape(rows, ng)
    w = np.empty((rows, cols), np.float32)
    pos = rows + rows * ng * 2
    for r in range(rows):
        b = int(bits[r])
        nbytes = (cols * b + 7) // 8
        rb = np.frombuffer(raw, np.uint8, nbytes, pos)
        pos += nbytes
        u = np.unpackbits(rb)[:cols * b].reshape(cols, b)
        vals = u.astype(np.uint32) @ (1 << np.arange(b - 1, -1, -1, dtype=np.uint32))
        L = float(2 ** (b - 1) - 1)
        w[r] = (vals.astype(np.float32) - L) * sc[r, np.arange(cols) // GROUP_SIZE]
    return w


def _dq_vbit_ro(raw: bytes, shape) -> np.ndarray:
    """vbit_ro (§4.2): vbit plus a u32 row-offset table between the
    scales and the packed rows; reconstruction is identical to vbit."""
    rows, cols = shape
    ng = cols // GROUP_SIZE
    bits = np.frombuffer(raw, np.uint8, rows)
    sc = np.frombuffer(raw, np.float16, rows * ng, rows).astype(np.float32) \
        .reshape(rows, ng)
    off_off = rows + rows * ng * 2
    offsets = np.frombuffer(raw, np.uint32, rows + 1, off_off)
    packed_off = off_off + (rows + 1) * 4
    w = np.empty((rows, cols), np.float32)
    for r in range(rows):
        b = int(bits[r])
        start = packed_off + int(offsets[r])
        nbytes = int(offsets[r + 1]) - int(offsets[r])
        rb = np.frombuffer(raw, np.uint8, nbytes, start)
        u = np.unpackbits(rb)[:cols * b].reshape(cols, b)
        vals = u.astype(np.uint32) @ (1 << np.arange(b - 1, -1, -1, dtype=np.uint32))
        L = float(2 ** (b - 1) - 1)
        w[r] = (vals.astype(np.float32) - L) * sc[r, np.arange(cols) // GROUP_SIZE]
    return w


def _dq_q4_tiled(raw: bytes, shape) -> np.ndarray:
    """q4_tiled (§4.3): 18-byte tiles [f16 scale][16B nibbles] per
    32-group; values identical to q4_block."""
    n = int(np.prod(shape))
    ng = n // GROUP_SIZE
    tiles = np.frombuffer(raw, np.uint8, ng * 18).reshape(ng, 18)
    sc = tiles[:, :2].copy().view(np.float16).astype(np.float32).reshape(ng)
    pk = tiles[:, 2:]
    lo = (pk & 0x0F).astype(np.int32) - 8
    hi = ((pk >> 4) & 0x0F).astype(np.int32) - 8
    vals = np.empty((ng, GROUP_SIZE), np.float32)
    vals[:, 0::2] = lo
    vals[:, 1::2] = hi
    vals *= sc[:, None]
    return vals.reshape(shape).copy()


def dequant(raw: bytes, dtype: str, shape) -> np.ndarray:
    if dtype == "f32":
        return np.frombuffer(raw, np.float32, int(np.prod(shape))).reshape(shape).copy()
    if dtype == "f16":
        return np.frombuffer(raw, np.float16, int(np.prod(shape))).reshape(shape).copy()
    if dtype == "bf16":
        u = np.frombuffer(raw, np.uint16, int(np.prod(shape))).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape).copy()
    if dtype == "u8":
        return np.frombuffer(raw, np.uint8, int(np.prod(shape))).reshape(shape).copy()
    if dtype == "q8_row":
        return _dq_q8_row(raw, shape)
    if dtype == "q8_2f":
        return _dq_q8_2f(raw, shape)
    if dtype == "q4_block":
        return _dq_q4_block(raw, shape)
    if dtype == "vbit":
        return _dq_vbit(raw, shape)
    if dtype == "vbit_ro":
        return _dq_vbit_ro(raw, shape)
    if dtype == "q4_tiled":
        return _dq_q4_tiled(raw, shape)
    raise ValueError(f"dtype {dtype}: dequant not implemented in the reader")


# ───────────────────── container ─────────────────────

class _Shard:
    """One open .cmf: envelope, header JSON, directory, mmap."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = open(self.path, "rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:4] != CMF_MAGIC:
            raise ValueError(f"{path}: not CMF (magic {self.mm[:4]!r})")
        version, _, required = struct.unpack_from("<III", self.mm, 4)
        if version != 2:
            raise ValueError(f"{path}: version {version}, reader knows 2")
        if required & ~KNOWN_FEATURES:
            raise ValueError(f"{path}: unknown required_features "
                             f"{required & ~KNOWN_FEATURES:#x}")
        offs = struct.unpack_from("<12Q", self.mm, 16)
        (self.header_off, self.header_len, self.dir_off, self.dir_len,
         self.data_off, self.data_len, self.masks_off, self.masks_len,
         self.vocab_off, self.vocab_len, self.index_off, self.index_len) = offs
        self.header_hash, self.dir_hash = struct.unpack_from("<QQ", self.mm, 0x70)

        self.header = json.loads(
            self.mm[self.header_off:self.header_off + self.header_len])

        # Directory: [u64 count][u64 pool_off][count×56B][name pool].
        count, pool_off = struct.unpack_from("<QQ", self.mm, self.dir_off)
        pool_base = self.dir_off + pool_off
        self.entries = {}
        for i in range(count):
            rec = self.dir_off + 16 + i * DIR_RECORD_LEN
            noff, nlen, dt, ndim = struct.unpack_from("<IHBB", self.mm, rec)
            shape = struct.unpack_from("<6I", self.mm, rec + 8)[:ndim]
            off, nbytes, thash = struct.unpack_from("<QQQ", self.mm, rec + 32)
            name = self.mm[pool_base + noff:pool_base + noff + nlen].decode()
            self.entries[name] = {
                "dtype": DTYPE_NAME.get(dt, f"?{dt}"), "shape": tuple(shape),
                "off": off, "nbytes": nbytes, "hash": thash,
            }

    def tensor_bytes(self, e) -> bytes:
        a = self.data_off + e["off"]
        return self.mm[a:a + e["nbytes"]]

    def close(self):
        self.mm.close()
        self.file.close()


class CmfReader:
    """The whole model: shard 1 + siblings (spec §10), merged directory."""

    def __init__(self, path, sharded: bool = True):
        path = Path(path)
        m = _SHARD_RE.match(path.name)
        if m and sharded:
            path = path.with_name(f"{m.group(1)}-00001-of-{m.group(3)}.cmf")
        first = _Shard(path)
        self.shards = [first]
        self.header = first.header
        info = self.header.get("shard")
        if sharded and info:
            base = _SHARD_RE.match(path.name).group(1)
            for no in range(2, info["count"] + 1):
                sib = path.with_name(f"{base}-{no:05}-of-{info['count']:05}.cmf")
                sh = _Shard(sib)
                s = sh.header.get("shard") or {}
                if s.get("no") != no or s.get("count") != info["count"]:
                    raise ValueError(f"{sib}: header.shard {s} ≠ expected")
                self.shards.append(sh)
        # Merged directory: name → (record, shard index).
        self.tensors = {}
        for si, sh in enumerate(self.shards):
            for name, e in sh.entries.items():
                self.tensors[name] = (e, si)

    # ── tensors ──

    def resolve(self, name: str, skill: str | None = None) -> str:
        """Tensor-source indirection (P15 claim 1): the skill substitutes."""
        if skill is not None:
            s = f"skill.{skill}.{name}"
            if s in self.tensors:
                return s
        return name

    def tensor_bytes(self, name: str) -> bytes:
        e, si = self.tensors[name]
        return self.shards[si].tensor_bytes(e)

    def tensor(self, name: str, skill: str | None = None) -> np.ndarray:
        name = self.resolve(name, skill)
        e, _ = self.tensors[name]
        return dequant(self.tensor_bytes(name), e["dtype"], e["shape"])

    # ── sections ──

    @property
    def arch(self) -> dict:
        return self.header["arch"]

    @property
    def skills(self) -> list:
        return self.header.get("skills") or []

    @property
    def vocab(self) -> bytes:
        sh = self.shards[0]
        return bytes(sh.mm[sh.vocab_off:sh.vocab_off + sh.vocab_len])

    @property
    def masks_meta(self) -> dict | None:
        sh = self.shards[0]
        if not sh.masks_len:
            return None
        n, meta_len = struct.unpack_from("<II", sh.mm, sh.masks_off)
        return json.loads(sh.mm[sh.masks_off + 8:sh.masks_off + 8 + meta_len])

    # ── integrity (spec §8.1) ──

    def verify(self) -> list:
        problems = []
        for sh in self.shards:
            hdr = sh.mm[sh.header_off:sh.header_off + sh.header_len]
            if hash64(hdr) != sh.header_hash:
                problems.append(f"{sh.path.name}: header hash mismatch")
            d = sh.mm[sh.dir_off:sh.dir_off + sh.dir_len]
            if sh.dir_hash and hash64(d) != sh.dir_hash:
                problems.append(f"{sh.path.name}: dir hash mismatch")
            for name, e in sh.entries.items():
                if hash64(sh.tensor_bytes(e)) != e["hash"]:
                    problems.append(f"{sh.path.name}: tensor {name} hash mismatch")
        return problems

    def close(self):
        for sh in self.shards:
            sh.close()


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Standalone CMF v2 reader")
    ap.add_argument("model")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tensor")
    a = ap.parse_args()
    r = CmfReader(a.model)
    arch = r.arch
    print(f"{a.model}: {arch['arch_name']} | {arch['num_layers']}L | "
          f"hidden {arch['hidden_size']} | vocab {arch['vocab_size']} | "
          f"{len(r.tensors)} tensors | {len(r.shards)} shard(s) | "
          f"quant {r.header.get('quant_type')}")
    for s in r.skills:
        print(f"  skill: {s['id']}")
    if a.tensor:
        w = r.tensor(a.tensor)
        print(f"  {a.tensor}: {w.shape} {w.dtype} | mean {w.mean():.6f} "
              f"| absmax {np.abs(w).max():.6f}")
    if a.verify:
        problems = r.verify()
        for p in problems:
            print(f"  ✗ {p}")
        print("  ✓ integrity confirmed" if not problems else
              f"  {len(problems)} problems")
        raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    _main()
