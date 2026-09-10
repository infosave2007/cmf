#!/usr/bin/env python3
"""Independent bounded QA for a Qwen Image GGUF transformer converted to CMF.

The converter is deliberately outside this script. This checker parses the GGUF
header and tensor records, parses CMF's envelope and directory, validates bounds,
names, semantic shapes, embedded configuration/provenance, and CMF hash fields,
then compares sampled source Q6_K/F32 values with decoded native CMF values.

It uses only Python's standard library and never imports the Rust workspace.
Full payload hashing is intentionally opt-in; pair this checker with
cortiq verify for the complete CMF hash pass.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


# GGUF metadata scalar tags.
T_U8 = 0
T_I8 = 1
T_U16 = 2
T_I16 = 3
T_U32 = 4
T_I32 = 5
T_F32 = 6
T_BOOL = 7
T_STR = 8
T_ARR = 9
T_U64 = 10
T_I64 = 11
T_F64 = 12

# GGML type IDs and their block lengths. The checker inventories every known
# source type, while value sampling only needs Q6_K/F32/F16/BF16.
GGML_BYTES = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    20: (32, 18),    # IQ4_NL
    23: (256, 136),  # IQ4_XS
    30: (1, 2),      # BF16
}
GGML_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    20: "IQ4_NL",
    23: "IQ4_XS",
    30: "BF16",
}

# CMF TensorDtype IDs used by the public format.
CMF_NAMES = {
    0: "f32",
    1: "f16",
    2: "bf16",
    3: "q8_row",
    4: "q4_block",
    5: "mix8_4",
    6: "u8",
    7: "q4_col",
    8: "vbit",
    9: "q8_2f",
    10: "vbit_ro",
    11: "q4_tiled",
    12: "q1",
    13: "q1s",
    14: "q1t",
    15: "q4tp",
    16: "q2tp",
}
CMF_SUPPORTED_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6)
CMF_DATA_ALIGNMENT = 4096
CMF_TENSOR_ALIGNMENT = 64
CMF_DIR_RECORD_LEN = 56
CMF_MAX_NDIM = 6
QGROUP = 32


class CheckError(Exception):
    """A malformed input or failed contract check."""


class Reader:
    def __init__(self, data: memoryview, pos: int = 0):
        self.data = data
        self.pos = pos

    def need(self, n: int) -> memoryview:
        if n < 0 or self.pos < 0 or self.pos + n > len(self.data):
            raise CheckError("truncated input while reading at offset %d" % self.pos)
        start = self.pos
        self.pos += n
        return self.data[start:self.pos]

    def u8(self) -> int:
        return self.need(1)[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.need(1))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.need(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.need(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.need(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.need(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.need(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.need(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.need(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.need(8))[0]

    def string(self, limit: int = 64 * 1024 * 1024) -> str:
        n = self.u64()
        if n > limit:
            raise CheckError("GGUF string length %d exceeds limit %d" % (n, limit))
        return bytes(self.need(n)).decode("utf-8", errors="strict")


def read_gguf_scalar(r: Reader, tag: int) -> Any:
    if tag == T_U8 or tag == T_BOOL:
        return r.u8()
    if tag == T_I8:
        return r.i8()
    if tag == T_U16:
        return r.u16()
    if tag == T_I16:
        return r.i16()
    if tag == T_U32:
        return r.u32()
    if tag == T_I32:
        return r.i32()
    if tag == T_F32:
        return r.f32()
    if tag == T_STR:
        return r.string()
    if tag == T_U64:
        return r.u64()
    if tag == T_I64:
        return r.i64()
    if tag == T_F64:
        return r.f64()
    raise CheckError("unsupported GGUF metadata scalar tag %d" % tag)


def read_gguf_value(r: Reader, tag: int, array_limit: int = 250_000) -> Any:
    if tag != T_ARR:
        return read_gguf_scalar(r, tag)
    element_tag = r.u32()
    count = r.u64()
    if count > 20_000_000:
        raise CheckError("GGUF metadata array length %d is unreasonable" % count)
    values = []
    keep = count <= array_limit
    for _ in range(count):
        value = read_gguf_scalar(r, element_tag)
        if keep:
            values.append(value)
    if keep:
        return values
    return {"array_len": count, "element_tag": element_tag}


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        alignment = 1
    return ((value + alignment - 1) // alignment) * alignment


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def product(shape: Iterable[int]) -> int:
    result = 1
    for dim in shape:
        if dim < 0:
            raise CheckError("negative tensor dimension %d" % dim)
        result *= dim
    return result


@dataclass(frozen=True)
class GgufTensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int

    @property
    def elements(self) -> int:
        return product(self.dims)

    @property
    def type_name(self) -> str:
        return GGML_NAMES.get(self.ggml_type, "GGML_%d" % self.ggml_type)


class GgufFile:
    def __init__(self, path: Path):
        self.path = path
        self.fd = path.open("rb")
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.buf = memoryview(self.mm)
        self.metadata: dict[str, Any] = {}
        self.tensors: list[GgufTensor] = []
        self.version = 0
        self.data_start = 0
        self.alignment = 32

    def close(self) -> None:
        try:
            self.buf.release()
        except Exception:
            pass
        try:
            self.mm.close()
        finally:
            self.fd.close()

    def parse(self) -> None:
        if len(self.buf) < 24 or bytes(self.buf[:4]) != b"GGUF":
            raise CheckError("%s is not a GGUF file" % self.path)
        r = Reader(self.buf)
        r.need(4)
        self.version = r.u32()
        if self.version not in (2, 3):
            raise CheckError("unsupported GGUF version %d" % self.version)
        tensor_count = r.u64()
        metadata_count = r.u64()
        if tensor_count > 2_000_000 or metadata_count > 2_000_000:
            raise CheckError("GGUF counts are unreasonable: tensors=%d metadata=%d"
                             % (tensor_count, metadata_count))
        for _ in range(metadata_count):
            key = r.string()
            tag = r.u32()
            self.metadata[key] = read_gguf_value(r, tag)
        seen: set[str] = set()
        for _ in range(tensor_count):
            name = r.string()
            if name in seen:
                raise CheckError("duplicate GGUF tensor name %r" % name)
            seen.add(name)
            ndim = r.u32()
            if ndim > CMF_MAX_NDIM:
                raise CheckError("tensor %r has ndim %d > %d" % (name, ndim, CMF_MAX_NDIM))
            dims = tuple(r.u64() for _ in range(ndim))
            ggml_type = r.u32()
            offset = r.u64()
            n = product(dims)
            if ggml_type not in GGML_BYTES:
                raise CheckError("tensor %r has unsupported GGML type id %d"
                                 % (name, ggml_type))
            block_elems, block_bytes = GGML_BYTES[ggml_type]
            nbytes = ceil_div(n, block_elems) * block_bytes if n else 0
            self.tensors.append(GgufTensor(name, dims, ggml_type, offset, nbytes))
        raw_alignment = self.metadata.get("general.alignment", 32)
        if isinstance(raw_alignment, (int, float)) and int(raw_alignment) > 0:
            self.alignment = int(raw_alignment)
        self.data_start = align_up(r.pos, self.alignment)
        if self.data_start > len(self.buf):
            raise CheckError("GGUF data start %d exceeds file length %d"
                             % (self.data_start, len(self.buf)))
        ranges = []
        for index, tensor in enumerate(self.tensors):
            if tensor.offset % self.alignment != 0:
                raise CheckError("GGUF tensor %r offset %d is not %d-aligned"
                                 % (tensor.name, tensor.offset, self.alignment))
            start = self.data_start + tensor.offset
            end = start + tensor.nbytes
            if end < start or end > len(self.buf):
                raise CheckError("GGUF tensor %r span [%d,%d) exceeds file length %d"
                                 % (tensor.name, start, end, len(self.buf)))
            ranges.append((start, end, index, tensor.name))
        ranges.sort()
        previous_end = self.data_start
        previous_name = None
        for start, end, _, name in ranges:
            if start < previous_end and end > start:
                raise CheckError("overlapping GGUF tensor spans: %r and %r" %
                                 (previous_name, name))
            previous_end = max(previous_end, end)
            previous_name = name

    def raw(self, tensor: GgufTensor) -> memoryview:
        start = self.data_start + tensor.offset
        return self.buf[start:start + tensor.nbytes]


@dataclass(frozen=True)
class CmfEntry:
    name: str
    dtype: int
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    stored_hash: int

    @property
    def dtype_name(self) -> str:
        return CMF_NAMES.get(self.dtype, "dtype_%d" % self.dtype)

    @property
    def elements(self) -> int:
        return product(self.shape)


class CmfFile:
    def __init__(self, path: Path):
        self.path = path
        self.fd = path.open("rb")
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.buf = memoryview(self.mm)
        self.envelope: dict[str, Any] = {}
        self.header: dict[str, Any] = {}
        self.entries: list[CmfEntry] = []

    def close(self) -> None:
        try:
            self.buf.release()
        except Exception:
            pass
        try:
            self.mm.close()
        finally:
            self.fd.close()

    def section(self, name: str) -> memoryview:
        off, length = self.envelope[name]
        return self.buf[off:off + length]

    def payload(self, entry: CmfEntry) -> memoryview:
        data_off, data_len = self.envelope["data"]
        start = data_off + entry.offset
        return self.buf[start:start + entry.nbytes]

    def parse(self, errors: list[str], warnings: list[str]) -> None:
        if len(self.buf) < 128 or bytes(self.buf[:4]) != b"CMF\x01":
            raise CheckError("%s is not a CMF v2 file" % self.path)
        u32 = lambda off: struct.unpack_from("<I", self.buf, off)[0]
        u64 = lambda off: struct.unpack_from("<Q", self.buf, off)[0]
        version = u32(4)
        if version != 2:
            errors.append("CMF version is %d, expected 2" % version)
        required = u32(12)
        unknown = required & ~CMF_SUPPORTED_FEATURES
        if unknown:
            errors.append("CMF required_features has unsupported bits 0x%x" % unknown)
        names = ("header", "dir", "data", "masks", "vocab", "index")
        offsets = ((0x10, 0x18), (0x20, 0x28), (0x30, 0x38),
                   (0x40, 0x48), (0x50, 0x58), (0x60, 0x68))
        spans: list[tuple[int, int, str]] = []
        for name, (oo, ll) in zip(names, offsets):
            off, length = u64(oo), u64(ll)
            self.envelope[name] = (off, length)
            if name in ("header", "dir") and length == 0:
                errors.append("required CMF section %s is empty" % name)
            if length:
                end = off + length
                if off < 128 or end < off or end > len(self.buf):
                    errors.append("CMF section %s span [%d,%d) is out of bounds"
                                  % (name, off, end))
                else:
                    spans.append((off, end, name))
        spans.sort()
        for (a0, a1, an), (b0, b1, bn) in zip(spans, spans[1:]):
            if b0 < a1:
                errors.append("CMF sections %s and %s overlap" % (an, bn))
        data_off, data_len = self.envelope["data"]
        if data_len and data_off % CMF_DATA_ALIGNMENT:
            errors.append("CMF data offset %d is not %d-aligned"
                          % (data_off, CMF_DATA_ALIGNMENT))
        if self.envelope["header"][1]:
            raw_header = bytes(self.section("header"))
            self.header = json.loads(raw_header.decode("utf-8"))
            stored = u64(0x70)
            actual = hash64(raw_header)
            if stored and stored != actual:
                errors.append("CMF header hash mismatch: stored %016x actual %016x"
                              % (stored, actual))
            if not stored:
                warnings.append("CMF header hash field is zero")
        if not self.envelope["dir"][1]:
            return
        raw_dir = self.section("dir")
        stored_dir_hash = u64(0x78)
        actual_dir_hash = hash64(raw_dir)
        if stored_dir_hash and stored_dir_hash != actual_dir_hash:
            errors.append("CMF directory hash mismatch: stored %016x actual %016x"
                          % (stored_dir_hash, actual_dir_hash))
        if not stored_dir_hash:
            warnings.append("CMF directory hash field is zero")
        if len(raw_dir) < 16:
            errors.append("CMF directory is shorter than its preamble")
            return
        count, pool_off = struct.unpack_from("<QQ", raw_dir, 0)
        records_end = 16 + count * CMF_DIR_RECORD_LEN
        if count > 2_000_000 or records_end > len(raw_dir) or pool_off < records_end or pool_off > len(raw_dir):
            errors.append("malformed CMF directory count=%d pool_off=%d len=%d"
                          % (count, pool_off, len(raw_dir)))
            return
        pool = raw_dir[pool_off:]
        names_seen: set[str] = set()
        ranges: list[tuple[int, int, str]] = []
        for i in range(count):
            base = 16 + i * CMF_DIR_RECORD_LEN
            name_off, name_len, dtype, ndim = struct.unpack_from("<IHBB", raw_dir, base)
            if ndim > CMF_MAX_NDIM:
                errors.append("directory entry %d ndim=%d is invalid" % (i, ndim))
                continue
            dims = tuple(struct.unpack_from("<I", raw_dir, base + 8 + j * 4)[0]
                         for j in range(ndim))
            offset = struct.unpack_from("<Q", raw_dir, base + 32)[0]
            nbytes = struct.unpack_from("<Q", raw_dir, base + 40)[0]
            stored_hash = struct.unpack_from("<Q", raw_dir, base + 48)[0]
            if name_off + name_len > len(pool):
                errors.append("directory entry %d name span is out of bounds" % i)
                continue
            try:
                name = bytes(pool[name_off:name_off + name_len]).decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                errors.append("directory entry %d name is not UTF-8" % i)
                continue
            if name in names_seen:
                errors.append("duplicate CMF tensor name %r" % name)
            names_seen.add(name)
            if offset % CMF_TENSOR_ALIGNMENT:
                errors.append("CMF tensor %r offset %d is not %d-aligned"
                              % (name, offset, CMF_TENSOR_ALIGNMENT))
            end = offset + nbytes
            if end < offset or end > data_len:
                errors.append("CMF tensor %r span [%d,%d) exceeds data section %d"
                              % (name, offset, end, data_len))
            else:
                ranges.append((offset, end, name))
            expected = cmf_expected_nbytes(dtype, dims)
            if expected is not None and expected != nbytes:
                errors.append("CMF tensor %r has %d bytes, expected %d for %s%s"
                              % (name, nbytes, expected, CMF_NAMES.get(dtype, str(dtype)), dims))
            if dtype not in CMF_NAMES:
                warnings.append("CMF tensor %r uses unknown/reserved dtype id %d" % (name, dtype))
            self.entries.append(CmfEntry(name, dtype, dims, offset, nbytes, stored_hash))
        ranges.sort()
        for (a0, a1, an), (b0, b1, bn) in zip(ranges, ranges[1:]):
            if b0 < a1:
                errors.append("CMF tensor spans %r and %r overlap" % (an, bn))
        self._validate_section_hashes(errors, warnings)

    def _validate_section_hashes(self, errors: list[str], warnings: list[str]) -> None:
        hashes = self.header.get("section_hashes")
        if not isinstance(hashes, dict):
            return
        for name in ("masks", "vocab", "index"):
            text = hashes.get(name)
            if text is None:
                continue
            if not isinstance(text, str):
                errors.append("section_hashes.%s is not hexadecimal text" % name)
                continue
            try:
                stored = int(text, 16)
            except ValueError:
                errors.append("section_hashes.%s is malformed" % name)
                continue
            off, length = self.envelope[name]
            actual = hash64(self.section(name)) if length else 0
            if stored != actual:
                errors.append("CMF %s section hash mismatch: stored %016x actual %016x"
                              % (name, stored, actual))


def cmf_expected_nbytes(dtype: int, shape: tuple[int, ...]) -> Optional[int]:
    n = product(shape)
    if dtype == 0 or dtype == 1 or dtype == 2:
        return n * 4 if dtype == 0 else n * 2
    if dtype == 3:
        if len(shape) != 2:
            return None
        return n + shape[0] * 2
    if dtype in (4, 11):
        return ceil_div(n, QGROUP) * 18
    if dtype == 6:
        return n
    if dtype == 9:
        if len(shape) != 2:
            return None
        rows, cols = shape
        return n + rows * 2 + cols * 2
    if dtype == 12:
        return ceil_div(n, QGROUP) * 6
    if dtype == 15:
        if len(shape) != 2 or shape[1] == 0 or shape[1] % QGROUP:
            return None
        rows, cols = shape
        groups = cols // QGROUP
        code_stride = ceil_div(groups * 5, 8)
        return rows * groups * 16 + rows * 4 + rows * code_stride
    if dtype == 16:
        if len(shape) != 2 or shape[1] == 0 or shape[1] % QGROUP:
            return None
        rows, cols = shape
        groups = cols // QGROUP
        code_stride = ceil_div(groups * 5, 8)
        return rows * groups * 8 + rows * 4 + rows * code_stride
    return None


def fmix64(value: int) -> int:
    value &= 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    value = (value * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 33
    return value & 0xFFFFFFFFFFFFFFFF


def hash64(data: bytes | memoryview) -> int:
    full = len(data) // 8
    acc = 0
    salt = 0x9E3779B97F4A7C15
    for i in range(full):
        word = int.from_bytes(data[i * 8:i * 8 + 8], "little")
        acc ^= fmix64(word) ^ ((i * salt) & 0xFFFFFFFFFFFFFFFF)
    rem = len(data) % 8
    if rem:
        word = int.from_bytes(data[full * 8:], "little")
        acc ^= fmix64(word) ^ ((full * salt) & 0xFFFFFFFFFFFFFFFF)
    acc ^= len(data)
    return fmix64(acc)


def f16_to_float(bits: int) -> float:
    sign = -1.0 if bits & 0x8000 else 1.0
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    if exponent == 0:
        if mantissa == 0:
            return -0.0 if sign < 0 else 0.0
        return sign * (mantissa / 1024.0) * (2.0 ** -14)
    if exponent == 0x1F:
        if mantissa == 0:
            return math.copysign(math.inf, sign)
        return math.nan
    return sign * (1.0 + mantissa / 1024.0) * (2.0 ** (exponent - 15))


def bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def signed_u8(value: int) -> int:
    return value - 256 if value >= 128 else value


def q6k_value(raw: memoryview, index: int) -> float:
    block = index // 256
    local = index % 256
    base = block * 210
    ql = raw[base:base + 128]
    qh = raw[base + 128:base + 192]
    scales = raw[base + 192:base + 208]
    d = f16_to_float(struct.unpack_from("<H", raw, base + 208)[0])
    half = 0 if local < 128 else 1
    within = local - half * 128
    group = within // 32
    l = within % 32
    ql_off = half * 64
    qh_off = half * 32
    scale_off = half * 8 + group * 2
    ql0 = ql[ql_off + l]
    ql32 = ql[ql_off + l + 32]
    qhb = qh[qh_off + l]
    if group == 0:
        q = (ql0 & 0xF) | ((qhb & 3) << 4)
    elif group == 1:
        q = (ql32 & 0xF) | (((qhb >> 2) & 3) << 4)
    elif group == 2:
        q = (ql0 >> 4) | (((qhb >> 4) & 3) << 4)
    else:
        q = (ql32 >> 4) | (((qhb >> 6) & 3) << 4)
    q -= 32
    return d * signed_u8(scales[scale_off]) * q


def source_value_at(tensor: GgufTensor, raw: memoryview, index: int) -> float:
    if tensor.ggml_type == 0:
        return struct.unpack_from("<f", raw, index * 4)[0]
    if tensor.ggml_type == 1:
        return f16_to_float(struct.unpack_from("<H", raw, index * 2)[0])
    if tensor.ggml_type == 30:
        return bf16_to_float(struct.unpack_from("<H", raw, index * 2)[0])
    if tensor.ggml_type == 14:
        return q6k_value(raw, index)
    raise CheckError("sampling source type %s is not implemented" % tensor.type_name)


def q4tp_code(raw: memoryview, code_start: int, stride: int, group: int) -> int:
    bit = group * 5
    byte = bit // 8
    shift = bit % 8
    value = raw[code_start + byte]
    if shift > 3:
        value |= raw[code_start + byte + 1] << 8
    return (value >> shift) & 0x1F


def cmf_value_at(entry: CmfEntry, raw: memoryview, index: int) -> float:
    dtype = entry.dtype
    shape = entry.shape
    n = entry.elements
    if index < 0 or index >= n:
        raise CheckError("sample index %d outside %d-element CMF tensor %r"
                         % (index, n, entry.name))
    if dtype == 0:
        return struct.unpack_from("<f", raw, index * 4)[0]
    if dtype == 1:
        return f16_to_float(struct.unpack_from("<H", raw, index * 2)[0])
    if dtype == 2:
        return bf16_to_float(struct.unpack_from("<H", raw, index * 2)[0])
    if dtype == 3:
        if len(shape) != 2:
            raise CheckError("q8_row tensor %r is not 2-D" % entry.name)
        rows, cols = shape
        row, col = divmod(index, cols)
        scale = f16_to_float(struct.unpack_from("<H", raw, rows * cols + row * 2)[0])
        return signed_u8(raw[index]) * scale
    if dtype == 9:
        if len(shape) != 2:
            raise CheckError("q8_2f tensor %r is not 2-D" % entry.name)
        rows, cols = shape
        row, col = divmod(index, cols)
        row_scale = f16_to_float(struct.unpack_from("<H", raw, n + row * 2)[0])
        col_scale = f16_to_float(struct.unpack_from("<H", raw, n + rows * 2 + col * 2)[0])
        return signed_u8(raw[index]) * row_scale * col_scale
    if dtype == 4:
        group = index // QGROUP
        in_group = index % QGROUP
        byte = raw[group * 16 + in_group // 2]
        scale = f16_to_float(struct.unpack_from("<H", raw, ceil_div(n, QGROUP) * 16 + group * 2)[0])
        code = (byte >> (4 * (in_group % 2))) & 0xF
        return (code - 8.0) * scale
    if dtype == 11:
        group = index // QGROUP
        in_group = index % QGROUP
        tile = group * 18
        scale = f16_to_float(struct.unpack_from("<H", raw, tile)[0])
        byte = raw[tile + 2 + in_group // 2]
        code = (byte >> (4 * (in_group % 2))) & 0xF
        return (code - 8.0) * scale
    if dtype == 12:
        group = index // QGROUP
        in_group = index % QGROUP
        tile = group * 6
        scale = f16_to_float(struct.unpack_from("<H", raw, tile)[0])
        bit = (raw[tile + 2 + in_group // 8] >> (in_group % 8)) & 1
        return (2.0 * bit - 1.0) * scale
    if dtype in (15, 16):
        if len(shape) != 2:
            raise CheckError("%s tensor %r is not 2-D" % (CMF_NAMES[dtype], entry.name))
        rows, cols = shape
        row, col = divmod(index, cols)
        groups = cols // QGROUP
        group, in_group = divmod(col, QGROUP)
        if dtype == 15:
            chunk = 16
            nibble_base = 0
        else:
            chunk = 8
            nibble_base = 1
        data_plane = rows * groups * chunk
        params = data_plane + row * 4
        lo = f16_to_float(struct.unpack_from("<H", raw, params)[0])
        step = f16_to_float(struct.unpack_from("<H", raw, params + 2)[0])
        stride = ceil_div(groups * 5, 8)
        codes_start = data_plane + rows * 4 + row * stride
        rung = q4tp_code(raw, codes_start, stride, group)
        if dtype == 15:
            scale = 2.0 ** (lo + rung * step)
            byte = raw[(row * groups + group) * 16 + in_group // 2]
            code = (byte >> (4 * (in_group % 2))) & 0xF
            return (code - 8.0) * scale
        if rung == 0:
            scale = 0.0
        else:
            scale = 2.0 ** (lo + (rung - 1) * step)
        byte = raw[(row * groups + group) * 8 + in_group // 4]
        code = (byte >> (2 * (in_group % 4))) & 0x3
        return (code - 1.5) * scale
    raise CheckError("CMF dtype %d is not decoded by this checker" % dtype)


def sample_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    limit = max(1, limit)
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [(i * (count - 1)) // (limit - 1) for i in range(limit)]


def evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if not items or count <= 0:
        return []
    count = min(max(1, count), len(items))
    if count == 1:
        return [items[0]]
    return [items[(i * (len(items) - 1)) // (count - 1)] for i in range(count)]


def recursive_strings(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            out.append(str(key))
            out.extend(recursive_strings(item))
    elif isinstance(value, list):
        for item in value:
            out.extend(recursive_strings(item))
    return out


def find_all_key_values(value: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, dict):
        for k, item in value.items():
            if k == key:
                out.append(item)
            out.extend(find_all_key_values(item, key))
    elif isinstance(value, list):
        for item in value:
            out.extend(find_all_key_values(item, key))
    return out


def load_name_map(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("source_to_cmf"), dict):
        value = value["source_to_cmf"]
    if not isinstance(value, dict):
        raise CheckError("name map must be an object or an object with source_to_cmf")
    result = {}
    for source, target in value.items():
        if not isinstance(source, str) or not isinstance(target, str):
            raise CheckError("name map keys and values must be strings")
        result[source] = target
    return result


def manifest_identity(path: Optional[Path], source_path: Path) -> tuple[Optional[int], Optional[str]]:
    if path is None:
        return None, None
    value = json.loads(path.read_text(encoding="utf-8"))
    base = source_path.name
    candidates: list[dict[str, Any]] = []
    if isinstance(value, dict) and isinstance(value.get("file"), list):
        candidates = [item for item in value["file"] if isinstance(item, dict)]
        exact = [item for item in candidates
                 if item.get("rfilename") == base or Path(str(item.get("rfilename", ""))).name == base]
        if exact:
            candidates = exact
    if not candidates:
        def visit(item: Any) -> None:
            if isinstance(item, dict):
                if "sha256" in item or "size" in item:
                    candidates.append(item)
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)
        visit(value)
    if not candidates:
        raise CheckError("source manifest contains no file identity for %s" % base)
    item = candidates[0]
    lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
    sha = lfs.get("sha256") or item.get("sha256")
    size = lfs.get("size") or item.get("size")
    return (int(size) if size is not None else None,
            str(sha).lower() if sha else None)


def verify_source_manifest(source: GgufFile, manifest: Optional[Path],
                            verify_sha: bool, errors: list[str]) -> dict[str, Any]:
    expected_size, expected_sha = manifest_identity(manifest, source.path)
    receipt = {
        "manifest": str(manifest) if manifest else None,
        "expected_size": expected_size,
        "expected_sha256": expected_sha,
        "size_matches": expected_size is None or expected_size == source.path.stat().st_size,
        "sha256_verified": False,
    }
    if expected_size is not None and expected_size != source.path.stat().st_size:
        errors.append("source size %d does not match manifest size %d"
                      % (source.path.stat().st_size, expected_size))
    if verify_sha:
        import hashlib
        digest = hashlib.sha256()
        with source.path.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        actual = digest.hexdigest()
        receipt["actual_sha256"] = actual
        receipt["sha256_verified"] = expected_sha is None or actual == expected_sha
        if expected_sha and actual != expected_sha:
            errors.append("source SHA-256 mismatch: expected %s actual %s" %
                          (expected_sha, actual))
    return receipt


def config_tensor_candidates(entries: list[CmfEntry], explicit: Optional[str]) -> list[CmfEntry]:
    if explicit:
        return [entry for entry in entries if entry.name == explicit]
    exact_names = {
        "image.config_json",
        "qwen_image.transformer.config_json",
        "qwen_image.transformer_config_json",
        "transformer.config_json",
        "dit.config_json",
    }
    exact = [entry for entry in entries if entry.name in exact_names]
    if exact:
        return exact
    return [entry for entry in entries
            if entry.name.endswith("config_json")
            and any(token in entry.name.lower() for token in ("qwen", "image", "transformer", "dit"))]


def is_metadata_entry(name: str) -> bool:
    lower = name.lower()
    return lower.endswith("config_json") or lower.endswith("model_index_json") \
        or lower.endswith("scheduler_config_json") or lower.endswith("provenance_json")


def config_bytes(cmf: CmfFile, entry: CmfEntry) -> bytes:
    if entry.dtype != 6:
        raise CheckError("embedded config %r must use CMF U8 dtype, got %s"
                         % (entry.name, entry.dtype_name))
    raw = cmf.payload(entry)
    if len(raw) != entry.elements:
        raise CheckError("embedded config %r has shape/payload mismatch" % entry.name)
    return bytes(raw)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tensor_mapping(source: GgufFile, prefix: str, name_map: dict[str, str],
                   errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for tensor in source.tensors:
        target = name_map.get(tensor.name, prefix + tensor.name)
        if target in reverse and reverse[target] != tensor.name:
            errors.append("two source tensors map to CMF name %r: %r and %r"
                          % (target, reverse[target], tensor.name))
        reverse[target] = tensor.name
        result[tensor.name] = target
    if name_map:
        missing_map = sorted(set(t.name for t in source.tensors) - set(name_map))
        if missing_map:
            errors.append("name map omits %d source tensors; first: %s"
                          % (len(missing_map), ", ".join(missing_map[:5])))
        extra_map = sorted(set(name_map) - set(t.name for t in source.tensors))
        if extra_map:
            errors.append("name map contains %d unknown source names; first: %s"
                          % (len(extra_map), ", ".join(extra_map[:5])))
    return result


def compare_inventory(source: GgufFile, cmf: CmfFile, shape_order: str,
                      prefix: str, name_map: dict[str, str],
                      expected_count: int, errors: list[str],
                      warnings: list[str]) -> dict[str, Any]:
    mapping = tensor_mapping(source, prefix, name_map, errors)
    by_name = {entry.name: entry for entry in cmf.entries}
    missing = []
    shape_mismatches = []
    dtype_metadata = []
    matched = 0
    for tensor in source.tensors:
        target = mapping[tensor.name]
        entry = by_name.get(target)
        if entry is None:
            missing.append({"source": tensor.name, "expected_cmf": target})
            continue
        matched += 1
        expected_shape = tuple(reversed(tensor.dims)) if shape_order == "semantic" else tensor.dims
        if entry.shape != expected_shape:
            shape_mismatches.append({
                "source": tensor.name,
                "cmf": target,
                "source_ggml_dims": list(tensor.dims),
                "expected_cmf_shape": list(expected_shape),
                "actual_cmf_shape": list(entry.shape),
            })
        if entry.dtype == 6:
            dtype_metadata.append({"source": tensor.name, "cmf": target})
    source_names = set(mapping.values())
    metadata_names = {entry.name for entry in cmf.entries if is_metadata_entry(entry.name)}
    extras = sorted(set(by_name) - source_names - metadata_names)
    if expected_count and len(source.tensors) != expected_count:
        errors.append("source tensor count %d != expected %d"
                      % (len(source.tensors), expected_count))
    if missing:
        errors.append("%d source tensors are missing from CMF" % len(missing))
    if shape_mismatches:
        errors.append("%d source/CMF semantic shape mismatches" % len(shape_mismatches))
    if dtype_metadata:
        errors.append("%d mapped source weights use CMF U8 metadata dtype" % len(dtype_metadata))
    if extras:
        warnings.append("%d CMF entries are not source tensors or recognized metadata"
                        % len(extras))
    inventory_text = "\n".join(
        "%s\t%s\t%d" % (tensor.name, ",".join(str(dim) for dim in tensor.dims), tensor.ggml_type)
        for tensor in source.tensors
    )
    return {
        "expected_source_count": expected_count or None,
        "source_tensor_count": len(source.tensors),
        "cmf_entry_count": len(cmf.entries),
        "matched_count": matched,
        "missing": missing,
        "shape_mismatches": shape_mismatches,
        "mapped_u8_weights": dtype_metadata,
        "extra_entries": extras,
        "name_map_supplied": bool(name_map),
        "shape_order": shape_order,
        "prefix": prefix,
        "source_name_sha256": __import__("hashlib").sha256(
            "\n".join(t.name for t in source.tensors).encode("utf-8")).hexdigest(),
        "source_inventory_sha256": __import__("hashlib").sha256(
            inventory_text.encode("utf-8")).hexdigest(),
    }


def check_header_contract(source: GgufFile, cmf: CmfFile,
                          source_receipt: dict[str, Any],
                          config_path: Optional[Path], config_name: Optional[str],
                          errors: list[str], warnings: list[str]) -> dict[str, Any]:
    header = cmf.header
    arch_values = find_all_key_values(header, "arch_name")
    arch_texts = [str(item).lower() for item in arch_values]
    arch_text = arch_texts[0] if arch_texts else ""
    if not any("qwen" in item and "image" in item for item in arch_texts):
        errors.append("CMF header arch_name does not identify qwen image: %s"
                      % arch_values)
    provenance = header.get("provenance")
    if not isinstance(provenance, (dict, list, str)):
        errors.append("CMF header has no provenance object/value")
    provenance_strings = [item.lower() for item in recursive_strings(provenance)]
    has_image_pipeline = any(
        ("qwen" in item and "image" in item)
        or ("image" in item and ("transformer" in item or "dit" in item))
        for item in provenance_strings
    )
    if not has_image_pipeline:
        errors.append("CMF provenance does not identify the Qwen Image transformer component")
    expected_sha = source_receipt.get("expected_sha256")
    sha_present = bool(expected_sha and
                       any(expected_sha.lower() in item for item in provenance_strings))
    if expected_sha and not sha_present:
        warnings.append("CMF provenance omits source GGUF SHA-256 %s; source manifest identity is recorded separately" % expected_sha)
    source_counts = find_all_key_values(provenance, "source_tensor_count")
    numeric_counts = [item for item in source_counts if isinstance(item, (int, float))]
    if numeric_counts and any(int(item) != len(source.tensors) for item in numeric_counts):
        errors.append("CMF provenance source_tensor_count does not match GGUF inventory")
    if not source_counts:
        warnings.append("CMF provenance has no source_tensor_count field")
    config_entries = config_tensor_candidates(cmf.entries, config_name)
    if config_name and not config_entries:
        errors.append("requested embedded config tensor %r was not found" % config_name)
    if len(config_entries) > 1:
        warnings.append("multiple candidate config tensors: %s"
                        % ", ".join(entry.name for entry in config_entries))
    embedded_config = None
    config_entry_name = None
    if config_entries:
        entry = config_entries[0]
        config_entry_name = entry.name
        try:
            embedded_config = json.loads(config_bytes(cmf, entry).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, CheckError) as exc:
            errors.append("embedded config %r is not valid JSON/U8: %s" % (entry.name, exc))
    config_match = None
    if config_path is not None:
        try:
            external = json.loads(config_path.read_text(encoding="utf-8"))
            config_match = embedded_config is not None and canonical_json(external) == canonical_json(embedded_config)
            if embedded_config is None:
                errors.append("external transformer config was supplied but no valid embedded config exists")
            elif not config_match:
                errors.append("embedded transformer config differs from %s" % config_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append("cannot read external config %s: %s" % (config_path, exc))
    source_arch = source.metadata.get("general.architecture")
    if str(source_arch).lower() != "qwen_image":
        errors.append("GGUF general.architecture=%r, expected qwen_image" % source_arch)
    return {
        "arch_name": arch_values,
        "source_architecture": source_arch,
        "provenance_has_qwen_image": has_image_pipeline,
        "provenance_has_source_sha256": sha_present,
        "embedded_config_name": config_entry_name,
        "embedded_config_matches_external": config_match,
    }


def compute_metrics(reference: list[float], decoded: list[float]) -> dict[str, Any]:
    if len(reference) != len(decoded) or not reference:
        raise CheckError("metric inputs have different/empty lengths")
    if any(not math.isfinite(value) for value in reference + decoded):
        raise CheckError("metric inputs contain non-finite values")
    count = len(reference)
    sum_ref2 = math.fsum(value * value for value in reference)
    sum_diff2 = math.fsum((actual - expected) ** 2
                          for expected, actual in zip(reference, decoded))
    ref_rms = math.sqrt(sum_ref2 / count)
    diff_rms = math.sqrt(sum_diff2 / count)
    dot = math.fsum(expected * actual for expected, actual in zip(reference, decoded))
    norm_ref = math.sqrt(sum_ref2)
    norm_actual = math.sqrt(math.fsum(value * value for value in decoded))
    cosine = dot / (norm_ref * norm_actual) if norm_ref and norm_actual else None
    max_abs = max(abs(actual - expected) for expected, actual in zip(reference, decoded))
    relative = diff_rms / ref_rms if ref_rms > 1e-30 else diff_rms
    return {
        "sample_count": count,
        "reference_rms": ref_rms,
        "rms_error": diff_rms,
        "relative_rms": relative,
        "max_abs_error": max_abs,
        "cosine": cosine,
    }


def check_payload_hashes(cmf: CmfFile, entries: list[CmfEntry],
                         full: bool, errors: list[str],
                         warnings: list[str], max_bytes: int) -> dict[str, Any]:
    if max_bytes <= 0:
        raise CheckError("--payload-hash-max-bytes must be positive")
    selected = entries if full else []
    if not full:
        # Always protect config/provenance-adjacent entries and deterministic
        # first/last weight entries; sampled metrics add their own entries.
        metadata = [entry for entry in entries if is_metadata_entry(entry.name)]
        weights = [entry for entry in entries if not is_metadata_entry(entry.name)]
        selected = metadata + evenly_spaced(weights, min(8, len(weights)))
    selected_unique = []
    seen = set()
    for entry in selected:
        if entry.name not in seen:
            selected_unique.append(entry)
            seen.add(entry.name)
    mismatches = []
    receipts = []
    partial_count = 0
    skipped_bytes = 0
    for entry in selected_unique:
        payload = cmf.payload(entry)
        hashed_bytes = len(payload) if full else min(len(payload), max_bytes)
        actual = hash64(payload[:hashed_bytes])
        complete = hashed_bytes == len(payload)
        verified = None
        if complete:
            verified = actual == entry.stored_hash
        else:
            partial_count += 1
            skipped_bytes += len(payload) - hashed_bytes
        receipts.append({
            "name": entry.name,
            "stored_hash": "%016x" % entry.stored_hash,
            "computed_hash": "%016x" % actual,
            "payload_bytes": len(payload),
            "hashed_bytes": hashed_bytes,
            "skipped_bytes": len(payload) - hashed_bytes,
            "complete": complete,
            "verified": verified,
        })
        if verified is False:
            mismatches.append({
                "name": entry.name,
                "stored": "%016x" % entry.stored_hash,
                "actual": "%016x" % actual,
            })
    if mismatches:
        errors.append("%d CMF payload hash mismatches" % len(mismatches))
    if not full:
        warnings.append("Python payload pass hashed %d selected entries with a %d-byte cap; run native cortiq verify for all full payload hashes"
                        % (len(selected_unique), max_bytes))
    return {
        "full": full,
        "checked_count": len(selected_unique),
        "entry_count": len(entries),
        "fully_verified_count": len(selected_unique) - partial_count,
        "partial_count": partial_count,
        "skipped_entry_count": partial_count,
        "skipped_bytes": skipped_bytes,
        "max_bytes_per_entry": None if full else max_bytes,
        "entries": receipts,
        "mismatches": mismatches,
    }


def same_bytes(left: memoryview, right: memoryview, chunk: int = 8 * 1024 * 1024) -> bool:
    if len(left) != len(right):
        return False
    for start in range(0, len(left), chunk):
        end = min(start + chunk, len(left))
        if left[start:end] != right[start:end]:
            return False
    return True


def check_float_byte_identity(source: GgufFile, cmf: CmfFile,
                              mapping: dict[str, str], errors: list[str]) -> dict[str, Any]:
    expected_dtype = {0: 0, 1: 1, 30: 2}
    by_name = {entry.name: entry for entry in cmf.entries}
    checked = 0
    mismatches = []
    for tensor in source.tensors:
        if tensor.ggml_type not in expected_dtype:
            continue
        target = mapping[tensor.name]
        entry = by_name.get(target)
        if entry is None:
            continue
        checked += 1
        want = expected_dtype[tensor.ggml_type]
        if entry.dtype != want:
            mismatches.append({
                "source_name": tensor.name,
                "cmf_name": target,
                "source_type": tensor.type_name,
                "expected_cmf_dtype": CMF_NAMES[want],
                "actual_cmf_dtype": entry.dtype_name,
                "reason": "dtype_changed",
            })
            continue
        source_raw = source.raw(tensor)
        cmf_raw = cmf.payload(entry)
        if not same_bytes(source_raw, cmf_raw):
            mismatches.append({
                "source_name": tensor.name,
                "cmf_name": target,
                "source_type": tensor.type_name,
                "expected_cmf_dtype": CMF_NAMES[want],
                "actual_cmf_dtype": entry.dtype_name,
                "reason": "payload_bytes_changed",
            })
    if mismatches:
        errors.append("%d native float/control tensors are not byte-identical" % len(mismatches))
    return {
        "checked_count": checked,
        "mismatches": mismatches,
        "policy": "GGML F32/F16/BF16 controls must remain CMF F32/F16/BF16 bytes",
    }


def run_samples(source: GgufFile, cmf: CmfFile, mapping: dict[str, str],
                sample_per_kind: int, max_sample_elements: int,
                errors: list[str], warnings: list[str]) -> dict[str, Any]:
    selected: dict[str, list[GgufTensor]] = {}
    for type_id, label in ((14, "Q6_K"), (0, "F32")):
        selected[label] = evenly_spaced(
            [tensor for tensor in source.tensors if tensor.ggml_type == type_id],
            sample_per_kind,
        )
        if not selected[label]:
            warnings.append("source contains no %s tensors; no %s metric was produced"
                            % (label, label))
    by_name = {entry.name: entry for entry in cmf.entries}
    records = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for label, tensors in selected.items():
        for tensor in tensors:
            target = mapping[tensor.name]
            entry = by_name.get(target)
            if entry is None:
                continue
            indices = sample_indices(tensor.elements, max_sample_elements)
            try:
                source_raw = source.raw(tensor)
                reference = [source_value_at(tensor, source_raw, index) for index in indices]
                cmf_raw = cmf.payload(entry)
                decoded = [cmf_value_at(entry, cmf_raw, index) for index in indices]
                metric = compute_metrics(reference, decoded)
            except (CheckError, struct.error, ValueError, OverflowError) as exc:
                errors.append("sample %s -> %s failed: %s" % (tensor.name, target, exc))
                continue
            item = {
                "source_name": tensor.name,
                "cmf_name": target,
                "source_type": label,
                "source_ggml_type_id": tensor.ggml_type,
                "cmf_dtype": entry.dtype_name,
                "elements": tensor.elements,
                "indices": {
                    "count": len(indices),
                    "first": indices[0] if indices else None,
                    "last": indices[-1] if indices else None,
                },
                "metrics": metric,
            }
            records.append(item)
            grouped.setdefault(label, []).append(metric)
    aggregate = {}
    for label, metrics in grouped.items():
        keys = ("reference_rms", "rms_error", "max_abs_error")
        aggregate[label] = {
            "tensor_count": len(metrics),
            "sample_count": sum(int(item["sample_count"]) for item in metrics),
            "max_relative_rms": max(item["relative_rms"] for item in metrics),
            "mean_relative_rms": sum(item["relative_rms"] for item in metrics) / len(metrics),
            "max_abs_error": max(item["max_abs_error"] for item in metrics),
        }
    return {
        "sample_per_source_type": sample_per_kind,
        "max_sample_elements_per_tensor": max_sample_elements,
        "tensors": records,
        "aggregate": aggregate,
    }


def write_report(path: Optional[Path], report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent bounded Qwen Image GGUF-to-CMF inventory, integrity, and quant QA")
    parser.add_argument("--gguf", type=Path, required=True,
                        help="real source GGUF file")
    parser.add_argument("--cmf", type=Path, required=True,
                        help="converted CMF file")
    parser.add_argument("--prefix", default="",
                        help="CMF prefix used when no name map is supplied (default: source names unchanged)")
    parser.add_argument("--name-map", type=Path,
                        help="JSON object source_to_cmf for canonicalized names")
    parser.add_argument("--config-json", type=Path,
                        help="external transformer config to compare with embedded U8 config")
    parser.add_argument("--config-name",
                        help="exact CMF tensor name containing transformer config JSON")
    parser.add_argument("--source-manifest", type=Path,
                        help="GGUF source manifest containing expected size/SHA-256")
    parser.add_argument("--verify-source-sha256", action="store_true",
                        help="stream and verify the full source SHA-256; slow for a 16.8 GB file")
    parser.add_argument("--expected-source-count", type=int, default=1933,
                        help="expected source tensor count; set 0 to disable")
    parser.add_argument("--shape-order", choices=("semantic", "raw"), default="semantic",
                        help="compare CMF shapes to reversed GGML dims (semantic) or raw dims")
    parser.add_argument("--sample-per-kind", type=int, default=4,
                        help="number of evenly spaced Q6_K and F32 tensors to sample")
    parser.add_argument("--max-sample-elements", type=int, default=65536,
                        help="maximum values sampled from each selected tensor")
    parser.add_argument("--max-relative-rms", type=float,
                        help="optional explicit per-tensor relative RMS acceptance limit")
    parser.add_argument("--full-payload-hash", action="store_true",
                        help="explicitly override the per-entry cap and recompute every CMF directory payload hash")
    parser.add_argument("--payload-hash-max-bytes", type=int, default=1_048_576,
                        help="maximum bytes hashed per selected entry in the fast Python pass (default: 1048576)")
    parser.add_argument("--report", type=Path,
                        help="write a JSON receipt to this path")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "schema": "qwen-image-component-qa-1",
        "policy": {
            "component_scope": "Qwen Image transformer/DiT only",
            "quantization": "existing native CMF F16/Q8/Q4TP/Q2TP codecs after faithful GGUF dequantization",
            "raw_q6k_preservation": False,
            "metric_definition": "relative_rms = sqrt(mean((decoded - source)^2)) / sqrt(mean(source^2)); also RMS error, reference RMS, max absolute error, cosine",
            "end_to_end_runtime_claim": False,
        },
        "command": " ".join(sys.argv),
        "errors": errors,
        "warnings": warnings,
    }
    source: Optional[GgufFile] = None
    cmf: Optional[CmfFile] = None
    try:
        source = GgufFile(args.gguf)
        source.parse()
        cmf = CmfFile(args.cmf)
        cmf.parse(errors, warnings)
        receipt = verify_source_manifest(source, args.source_manifest,
                                         args.verify_source_sha256, errors)
        mapping = tensor_mapping(source, args.prefix, load_name_map(args.name_map), errors)
        name_map = load_name_map(args.name_map)
        inventory = compare_inventory(
            source, cmf, args.shape_order, args.prefix, name_map,
            args.expected_source_count, errors, warnings)
        header = check_header_contract(
            source, cmf, receipt, args.config_json, args.config_name, errors, warnings)
        payload = check_payload_hashes(cmf, cmf.entries, args.full_payload_hash,
                                       errors, warnings, args.payload_hash_max_bytes)
        float_identity = check_float_byte_identity(source, cmf, mapping, errors)
        samples = run_samples(
            source, cmf, mapping, max(0, args.sample_per_kind),
            max(1, args.max_sample_elements), errors, warnings)
        if args.max_relative_rms is not None:
            for item in samples["tensors"]:
                value = item["metrics"]["relative_rms"]
                if value > args.max_relative_rms:
                    errors.append("sample %s relative RMS %.9g exceeds %.9g"
                                  % (item["source_name"], value, args.max_relative_rms))
        report.update({
            "source": {
                "path": str(args.gguf),
                "size": args.gguf.stat().st_size,
                "gguf_version": source.version,
                "data_start": source.data_start,
                "alignment": source.alignment,
                "metadata": {
                    key: source.metadata[key]
                    for key in ("general.architecture", "general.quantization_version",
                                "general.file_type")
                    if key in source.metadata
                },
                "type_counts": dict(sorted(
                    Counter(t.type_name for t in source.tensors).items())),
                "identity": receipt,
            },
            "cmf": {
                "path": str(args.cmf),
                "size": args.cmf.stat().st_size,
                "envelope": cmf.envelope,
                "header_contract": header,
                "dtype_counts": dict(sorted(
                    Counter(entry.dtype_name for entry in cmf.entries).items())),
            },
            "inventory": inventory,
            "float_byte_identity": float_identity,
            "payload_hashes": payload,
            "samples": samples,
        })
    except (OSError, CheckError, json.JSONDecodeError, struct.error, ValueError) as exc:
        errors.append(str(exc))
    finally:
        if source is not None:
            source.close()
        if cmf is not None:
            cmf.close()
    report["status"] = "PASS" if not errors else "FAIL"
    write_report(args.report, report)
    print(json.dumps({
        "status": report["status"],
        "source_tensors": report.get("inventory", {}).get("source_tensor_count"),
        "cmf_entries": report.get("inventory", {}).get("cmf_entry_count"),
        "matched": report.get("inventory", {}).get("matched_count"),
        "sampled": len(report.get("samples", {}).get("tensors", [])),
        "errors": errors,
        "warnings": warnings,
        "report": str(args.report) if args.report else None,
    }, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
