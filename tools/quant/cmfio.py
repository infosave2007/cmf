"""Read-only access to CMF containers for the offline quantization tools, and the
write-back path through `cortiq patch-tensor --raw`.

The reader parses exactly what `CmfModel::open` reads (crates/cortiq-core/src/format.rs):
the 128-byte envelope, the header JSON, the 56-byte directory records and the 64-byte
aligned payloads (sibling shards `-0000N-of-0000M.cmf` merged like `open_sharded`).
Payloads are returned as zero-copy uint8 views of a read-only mmap, so a 27B file costs
no RAM until a tensor is touched. Hashes are not checked here (that is `cortiq verify`,
which every write path below runs). Writing is never done in Python: re-encoded payloads
go back through cortiq, which recomputes the record and directory hashes.
"""
import json
import mmap
import os
import re
import shutil
import struct
import subprocess

import numpy as np

CMF_MAGIC = b"CMF\x01"
CMF_VERSION = 2
ENVELOPE_LEN = 128
DIR_RECORD_LEN = 56

# TensorDtype ids (crates/cortiq-core/src/types.rs `from_id`), spelled as cortiq's
# `{:?}` prints them (the listing of `cortiq dequant --all`).
DTYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Bf16", 3: "Q8Row", 4: "Q4Block", 5: "Mix84", 6: "U8",
    7: "Q4Col", 8: "Vbit", 9: "Q8_2f", 10: "VbitRo", 11: "Q4Tiled", 12: "Q1", 13: "Q1S",
    14: "Q1T", 15: "Q4TiledP", 16: "Q2TiledP", 17: "U32", 18: "U64", 20: "Q1G",
}
# the `--dtype` / `--tensor-quant` spelling of the dtypes the tools emit
QUANT_NAME = {"F16": "f16", "Q8_2f": "q8_2f", "Q4TiledP": "q4tp", "Q2TiledP": "q2tp",
              "Q4Tiled": "q4t", "Q8Row": "q8"}

_SHARD_RE = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<count>\d{5})\.cmf$")


class Tensor:
    __slots__ = ("name", "dtype", "shape", "off", "nbytes", "shard", "hash")

    def __init__(self, name, dtype, shape, off, nbytes, shard, hsh):
        self.name, self.dtype, self.shape = name, dtype, tuple(shape)
        self.off, self.nbytes, self.shard, self.hash = off, nbytes, shard, hsh

    @property
    def n_elems(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def __repr__(self):
        return f"Tensor({self.name!r}, {self.dtype}, {list(self.shape)}, {self.nbytes} B)"


class _Part:
    """One mapped file: envelope, header, directory."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        size = os.fstat(self.f.fileno()).st_size
        if size < ENVELOPE_LEN:
            raise ValueError(f"{path}: too small for a CMF envelope ({size} bytes)")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        env = self.mm[:ENVELOPE_LEN]
        if env[:4] != CMF_MAGIC:
            raise ValueError(f"{path}: not a CMF file (bad magic)")
        version, _flags, self.required_features = struct.unpack_from("<III", env, 4)
        if version != CMF_VERSION:
            raise ValueError(f"{path}: unsupported CMF version {version}")
        sec = struct.unpack_from("<12Q", env, 0x10)
        self.header_span, self.dir_span, self.data_span = sec[0:2], sec[2:4], sec[4:6]
        for nm, (off, ln) in (("header", self.header_span), ("dir", self.dir_span)):
            if ln == 0 or off + ln > size:
                raise ValueError(f"{path}: section {nm} [{off}, {off + ln}) out of file ({size})")
        h0, hl = self.header_span
        self.header = json.loads(bytes(self.mm[h0:h0 + hl]))
        d0, dl = self.dir_span
        self.records = self._directory(bytes(self.mm[d0:d0 + dl]))

    @staticmethod
    def _directory(b):
        count, pool_off = struct.unpack_from("<QQ", b, 0)
        if 16 + count * DIR_RECORD_LEN > len(b) or pool_off > len(b):
            raise ValueError("tensor directory malformed")
        pool = b[pool_off:]
        out = []
        for i in range(count):
            r = 16 + i * DIR_RECORD_LEN
            name_off, name_len, dt, ndim = struct.unpack_from("<IHBB", b, r)
            if ndim > 6:
                raise ValueError(f"tensor #{i}: ndim {ndim} > 6")
            shape = struct.unpack_from(f"<{ndim}I", b, r + 8)
            off, nbytes, hsh = struct.unpack_from("<QQQ", b, r + 32)
            if dt not in DTYPE_NAMES:
                raise ValueError(f"tensor #{i}: unknown dtype id {dt}")
            name = pool[name_off:name_off + name_len].decode("utf-8")
            out.append((name, DTYPE_NAMES[dt], shape, off, nbytes, hsh))
        return out

    def close(self):
        self.mm.close()
        self.f.close()


class CmfFile:
    """A CMF model (single file or the first shard of a set). `tensors` keeps directory order."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        first = _Part(self.path)
        self.parts = [first]
        shard = first.header.get("shard")
        if shard:
            m = _SHARD_RE.match(os.path.basename(self.path))
            if int(shard.get("no", 0)) != 1 or not m:
                raise ValueError(f"{path}: open shard 1 of a sharded set (got {shard})")
            count = int(shard["count"])
            for no in range(2, count + 1):
                sib = os.path.join(os.path.dirname(self.path), f"{m.group('stem')}-{no:05d}-of-{count:05d}.cmf")
                self.parts.append(_Part(sib))
        self.header = first.header
        self.arch = first.header.get("arch", {})
        self.tensors = {}
        for si, part in enumerate(self.parts):
            for name, dt, shape, off, nbytes, hsh in part.records:
                if name in self.tensors:
                    raise ValueError(f"duplicate tensor name {name!r} across shards")
                self.tensors[name] = Tensor(name, dt, shape, off, nbytes, si, hsh)

    @property
    def sharded(self):
        return len(self.parts) > 1

    def __contains__(self, name):
        return name in self.tensors

    def tensor(self, name):
        t = self.tensors.get(name)
        if t is None:
            raise KeyError(f"{self.path}: no tensor {name!r}")
        return t

    def payload(self, name):
        """Zero-copy uint8 view of the encoded payload (read-only)."""
        t = self.tensor(name)
        part = self.parts[t.shard]
        start = part.data_span[0] + t.off
        return np.frombuffer(part.mm, dtype=np.uint8, count=t.nbytes, offset=start)

    def close(self):
        for p in self.parts:
            p.close()


# ----------------------------------------------------------------------------- token ids
def load_ids(path):
    """Token ids of a corpus: `.npy` (any int dtype), `.u32`/`.i32` raw little-endian int32,
    or text with whitespace-separated integers (what CMF_PPL_IDS_OUT writes)."""
    if path.endswith(".npy"):
        return np.load(path).astype(np.int64).ravel()
    if path.endswith(".u32") or path.endswith(".i32"):
        return np.fromfile(path, dtype="<i4").astype(np.int64)
    with open(path) as f:
        return np.array(f.read().split(), dtype=np.int64)


def tokenize_with_cortiq(cortiq, model, text_path, out_ids):
    """Tokenize a text file with the model's own tokenizer: `cortiq ppl` writes the ids it
    scores to CMF_PPL_IDS_OUT (the whole file when --windows is given, no BOS); one short
    window is scored so the call stays cheap."""
    env = dict(os.environ, CMF_PPL_IDS_OUT=out_ids, CMF_GPU=os.environ.get("CMF_GPU", "0"))
    subprocess.run([cortiq, "ppl", model, "--file", text_path, "--windows", "1", "--window-len", "8"],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    return load_ids(out_ids)


# ----------------------------------------------------------------------------- write-back
def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])} ... failed (rc {r.returncode}):\n{r.stderr[-4000:]}")
    return r


def patch_raw(cortiq, src, out, payload_paths, dtype=None, verify=True, batch=256):
    """Store already ENCODED payloads into a copy of `src` at `out` through
    `cortiq patch-tensor --raw` (hashes recomputed by cortiq), then `cortiq verify`.

    payload_paths: {tensor name: path of the payload file}. Same-size payloads of a
    single-file model are written in place into a byte copy of `src` (no tensor is held in
    RAM, which matters at 27B); anything else (a sharded source, a dtype change via
    `dtype`) goes through `--output`, which rewrites the container."""
    if os.path.abspath(src) == os.path.abspath(out):
        raise ValueError("refusing to patch the source file itself; give a new output path")
    items = sorted(payload_paths.items())
    m = CmfFile(src)
    same_size = dtype is None and not m.sharded and all(
        os.path.getsize(p) == m.tensor(n).nbytes for n, p in items)
    m.close()
    if same_size:
        shutil.copyfile(src, out)
        for i in range(0, len(items), batch):
            cmd = [cortiq, "patch-tensor", out, "--raw"]
            for n, p in items[i:i + batch]:
                cmd += ["--set", f"{n}={p}"]
            _run(cmd)
    else:
        cmd = [cortiq, "patch-tensor", src, "--raw", "--output", out]
        if dtype is not None:
            cmd += ["--dtype", dtype]
        for n, p in items:
            cmd += ["--set", f"{n}={p}"]
        _run(cmd)
    if verify:
        _run([cortiq, "verify", out])
    return out
