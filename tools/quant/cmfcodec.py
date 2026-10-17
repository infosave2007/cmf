"""Byte-exact numpy emulation of the CMF payload layouts the refit tools read and rewrite
(crates/cortiq-core/src/quant.rs).

q4tp  [nibbles rows*gpr*16][row params rows*(f16 lo, f16 step)][5-bit rung codes rows*stride]
      w[r, j] = (nib - 8) * L[r, code[r, j // 32]],  L[r, c] = 2^lo * (2^step)^c  (c = 0..31)
q2tp  [2-bit codes rows*gpr*8][row params, as q4tp][rung codes, as q4tp]
      w[r, j] = (c - 1.5) * L2[r, code],  L2[r, 0] = 0, L2[r, c] = L[r, c - 1]  (c = 1..31)
q8_2f [int8 rows*cols][f16 row scale rows][f16 column scale cols]
      w[r, j] = q[r, j] * rs[r] * cs[j]

The ladder is built exactly like the Rust decoder: L[:, 0] = exp2(lo) in f32 and every next
rung is the previous one times exp2(step), rounded to f32 at each product. A refit changes
only the f16 row fields (lo, step) or the f16 row scales; `with_row_params` /
`with_row_scales` splice new fields into a copy of the original payload, so every code byte
stays identical by construction.
"""
import numpy as np

G = 32          # group size of the tiled codecs
LMAX = 31       # highest rung index


def f16_bits(x):
    """float -> f16 bits (round to nearest even), as uint16."""
    return np.asarray(x, dtype=np.float32).astype(np.float16).view(np.uint16)


def f16_val(bits):
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float32)


def bf16_val(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_bits(x):
    """float -> bf16 bits, round to nearest even (NaN kept quiet)."""
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(np.asarray(x, dtype=np.float32)), np.uint16(0x7FC0), r)


def stride_of(gpr):
    """bytes per row of the packed 5-bit rung codes"""
    return (gpr * 5 + 7) // 8


def code_bytes_per_row(dtype, cols):
    gpr = cols // G
    return gpr * (16 if dtype == "Q4TiledP" else 8)


def tiled_bytes(dtype, rows, cols):
    gpr = cols // G
    return rows * code_bytes_per_row(dtype, cols) + rows * 4 + rows * stride_of(gpr)


def q8_2f_bytes(rows, cols):
    return rows * cols + 2 * rows + 2 * cols


def payload_bytes(dtype, shape):
    """encoded size of a dtype the tools can write (None for the others)"""
    if dtype in ("Q4TiledP", "Q2TiledP"):
        return tiled_bytes(dtype, shape[0], shape[1])
    if dtype == "Q8_2f":
        return q8_2f_bytes(shape[0], int(np.prod(shape[1:])))
    n = int(np.prod(shape))
    return {"F32": 4 * n, "F16": 2 * n, "Bf16": 2 * n}.get(dtype)


def ladder(lo_bits, st_bits):
    """[rows, 32] f32 rung scales, sequential f32 products like the Rust decoder."""
    lo = f16_val(lo_bits).astype(np.float32)
    st = f16_val(st_bits).astype(np.float32)
    t = np.empty((lo.shape[0], 32), dtype=np.float32)
    t[:, 0] = np.exp2(lo).astype(np.float32)
    ratio = np.exp2(st).astype(np.float32)
    for c in range(1, 32):
        t[:, c] = (t[:, c - 1] * ratio).astype(np.float32)
    return t


def ladder2(lo_bits, st_bits):
    """q2tp ladder: rung 0 is the exact-zero group, rung c >= 1 is the q4tp rung c - 1."""
    base = ladder(lo_bits, st_bits)
    t = np.zeros_like(base)
    t[:, 1:] = base[:, :31]
    return t


class Tiled:
    """A decoded q4tp / q2tp payload: the raw code section (still packed), the f16 row
    fields and the unpacked rung index per group. Nothing is re-encoded later: payloads are
    rebuilt by splicing new row fields into `raw`."""

    def __init__(self, dtype, payload, rows, cols):
        assert dtype in ("Q4TiledP", "Q2TiledP"), dtype
        if cols % G:
            raise ValueError(f"{dtype}: cols {cols} not a multiple of {G}")
        self.dtype, self.rows, self.cols = dtype, rows, cols
        gpr = self.gpr = cols // G
        b = np.ascontiguousarray(np.asarray(payload, dtype=np.uint8))
        nb = self.nb = rows * code_bytes_per_row(dtype, cols)
        stride = stride_of(gpr)
        if b.size != nb + rows * 4 + rows * stride:
            raise ValueError(f"{dtype}: payload {b.size} B, want {nb + rows * 4 + rows * stride} for {rows}x{cols}")
        self.raw = b
        self.codes = b[:nb].reshape(rows, nb // rows)       # packed nibbles / 2-bit codes
        par = b[nb:nb + rows * 4].reshape(rows, 4).copy().view(np.uint16)
        self.lo, self.st = par[:, 0].copy(), par[:, 1].copy()
        cb = b[nb + rows * 4:].reshape(rows, stride)
        bits = np.unpackbits(cb, axis=1, bitorder="little")[:, :gpr * 5].reshape(rows, gpr, 5)
        self.code = (bits * (1 << np.arange(5, dtype=np.uint8))).sum(-1).astype(np.uint8)

    def ladder(self):
        return ladder(self.lo, self.st) if self.dtype == "Q4TiledP" else ladder2(self.lo, self.st)

    def scales(self):
        """[rows, gpr] f32 group scales, exactly the decoder's."""
        return np.take_along_axis(self.ladder(), self.code.astype(np.int64), 1)

    def rung_power(self):
        """[rows, gpr] the power k the step is raised to in each group's scale (q4tp: the
        rung; q2tp: rung - 1, and 0 for the exact-zero rung whose scale stays 0)."""
        if self.dtype == "Q4TiledP":
            return self.code.copy()
        return np.maximum(self.code.astype(np.int16) - 1, 0).astype(np.uint8)

    def levels(self):
        """[rows, cols] f32 code multipliers: nib - 8 (q4tp) or c - 1.5 (q2tp)."""
        return unpack_levels(self.dtype, self.codes, self.cols)

    def dequant(self):
        s = self.scales()
        return (self.levels().reshape(self.rows, self.gpr, G) * s[:, :, None]).reshape(self.rows, self.cols)

    def with_row_params(self, lo_bits, st_bits):
        """the original payload with the f16 (lo, step) row block replaced"""
        out = self.raw.copy()
        par = np.stack([np.asarray(lo_bits, np.uint16), np.asarray(st_bits, np.uint16)], 1)
        out[self.nb:self.nb + self.rows * 4] = par.view(np.uint8).reshape(-1)
        return out.tobytes()


def unpack_levels(dtype, codes, cols):
    """packed code bytes [rows, bytes_per_row] -> f32 multipliers [rows, cols]"""
    rows = codes.shape[0]
    out = np.empty((rows, cols), dtype=np.float32)
    if dtype == "Q4TiledP":
        out[:, 0::2] = (codes & 15).astype(np.float32) - 8.0
        out[:, 1::2] = (codes >> 4).astype(np.float32) - 8.0
    else:
        for k in range(4):
            out[:, k::4] = ((codes >> (2 * k)) & 3).astype(np.float32) - 1.5
    return out


def encode_tiled(dtype, q, lo, st, code):
    """Full encode from levels (q4tp: ints -8..7, q2tp: ints 0..3) — used by tests and by
    tools that build new payloads; the refit itself only splices row fields."""
    rows, cols = q.shape
    gpr = cols // G
    stride = stride_of(gpr)
    if dtype == "Q4TiledP":
        qq = (q.astype(np.int16) + 8).astype(np.uint8)
        assert qq.max() <= 15
        ch = (qq[:, 0::2] | (qq[:, 1::2] << 4)).astype(np.uint8)
    else:
        qq = q.astype(np.uint8)
        assert qq.max() <= 3
        ch = (qq[:, 0::4] | (qq[:, 1::4] << 2) | (qq[:, 2::4] << 4) | (qq[:, 3::4] << 6)).astype(np.uint8)
    par = np.stack([lo.astype(np.uint16), st.astype(np.uint16)], 1).view(np.uint8).reshape(rows, 4)
    bits = ((code[:, :, None].astype(np.uint8) >> np.arange(5, dtype=np.uint8)) & 1).reshape(rows, gpr * 5)
    pad = stride * 8 - gpr * 5
    if pad:
        bits = np.concatenate([bits, np.zeros((rows, pad), np.uint8)], 1)
    cb = np.packbits(bits, axis=1, bitorder="little")
    return ch.tobytes() + par.tobytes() + cb.tobytes()


class Q8_2f:
    """q8_2f payload: int8 codes + f16 row scales + f16 column scales."""

    def __init__(self, payload, rows, cols):
        b = np.ascontiguousarray(np.asarray(payload, dtype=np.uint8))
        if b.size != q8_2f_bytes(rows, cols):
            raise ValueError(f"Q8_2f: payload {b.size} B, want {q8_2f_bytes(rows, cols)} for {rows}x{cols}")
        self.rows, self.cols, self.raw = rows, cols, b
        self.q = b[:rows * cols].view(np.int8).reshape(rows, cols)
        self.rs_bits = b[rows * cols:rows * cols + 2 * rows].copy().view(np.uint16)
        self.cs_bits = b[rows * cols + 2 * rows:].copy().view(np.uint16)

    @property
    def rs(self):
        return f16_val(self.rs_bits)

    @property
    def cs(self):
        return f16_val(self.cs_bits)

    def dequant(self):
        return self.q.astype(np.float32) * self.rs[:, None] * self.cs[None, :]

    def with_row_scales(self, rs_bits):
        out = self.raw.copy()
        n = self.rows * self.cols
        out[n:n + 2 * self.rows] = np.asarray(rs_bits, np.uint16).view(np.uint8)
        return out.tobytes()


def decode_float(dtype, payload, shape):
    """F32 / F16 / Bf16 payload -> f32 array of `shape`"""
    b = np.asarray(payload, dtype=np.uint8)
    if dtype == "F32":
        v = b.view("<f4").astype(np.float32)
    elif dtype == "F16":
        v = b.view("<f2").astype(np.float32)
    elif dtype == "Bf16":
        v = bf16_val(b.view("<u2"))
    else:
        raise ValueError(f"not a float dtype: {dtype}")
    return v.reshape(shape)


def encode_float(dtype, vals):
    v = np.asarray(vals, dtype=np.float32)
    if dtype == "F32":
        return v.astype("<f4").tobytes()
    if dtype == "F16":
        return f16_bits(v).astype("<u2").tobytes()
    if dtype == "Bf16":
        return bf16_bits(v).astype("<u2").tobytes()
    raise ValueError(f"not a float dtype: {dtype}")


def round_float(dtype, vals):
    """the value a float field of this dtype will hold after encode_float"""
    return decode_float(dtype, np.frombuffer(encode_float(dtype, vals), dtype=np.uint8), np.shape(vals))


def dequant(dtype, payload, shape):
    """f32 decode of the payload dtypes the tools understand"""
    if dtype in ("Q4TiledP", "Q2TiledP"):
        return Tiled(dtype, payload, shape[0], shape[1]).dequant()
    if dtype == "Q8_2f":
        return Q8_2f(payload, shape[0], int(np.prod(shape[1:]))).dequant().reshape(shape)
    return decode_float(dtype, payload, shape)
