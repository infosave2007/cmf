#!/usr/bin/env python3
"""D4 reader gate: dequant of each dtype against the source weights
(tolerance = lattice step of the given quant), merged shards == whole file,
verify() catches corruption. Run from tests/py_reader.sh."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from cmf_reader import CmfReader  # noqa: E402


def check_quant(cmf_path: str, npz_path: str, quant: str):
    r = CmfReader(cmf_path)
    src = np.load(npz_path)
    checked = 0
    for name in src.files:
        if name not in r.tensors:
            continue
        orig = src[name].astype(np.float32)
        e, _ = r.tensors[name]
        dq = np.asarray(r.tensor(name), dtype=np.float32)
        assert dq.shape == orig.shape, f"{name}: shape {dq.shape} != {orig.shape}"
        err = float(np.abs(dq - orig).max())
        amax = float(np.abs(orig).max()) or 1.0
        dt = e["dtype"]
        if dt == "f32":
            bound = 0.0
        elif dt == "f16":
            bound = amax * 2 ** -10
        elif dt in ("q8_row", "q8_2f"):
            bound = amax / 127.0 * 1.6 + 1e-5
        elif dt == "q4_block":
            bound = amax / 7.0 * 1.6 + 1e-5
        elif dt == "vbit":
            bound = amax / 3.0 * 1.6 + 1e-5  # worst-case row: 3 bits, L=3
        else:
            raise AssertionError(f"{name}: unexpected dtype {dt}")
        assert err <= bound, f"{name} [{dt}]: err {err:.3e} > bound {bound:.3e}"
        checked += 1
    assert checked >= 10, f"only {checked} tensors checked"
    assert r.verify() == [], "verify() found problems on a clean file"
    r.close()
    print(f"  ✓ {quant}: {checked} tensors within tolerance, verify clean")


def check_sharded(whole_path: str, shard1_path: str):
    w = CmfReader(whole_path)
    s = CmfReader(shard1_path)
    assert len(s.shards) > 1, "shards were not picked up"
    assert set(w.tensors) == set(s.tensors), "directories differ"
    for name in w.tensors:
        assert bytes(w.tensor_bytes(name)) == bytes(s.tensor_bytes(name)), \
            f"{name}: bytes in shards ≠ whole file"
    assert s.verify() == []
    print(f"  ✓ sharding: {len(s.shards)} shards, {len(s.tensors)} tensors "
          f"byte-for-byte with the whole file")
    w.close()
    s.close()


def check_corruption(cmf_path: str):
    import struct
    data = bytearray(open(cmf_path, "rb").read())
    data_off = struct.unpack("<Q", data[0x30:0x38])[0]
    data[data_off + 64] ^= 0xFF
    open(cmf_path, "wb").write(bytes(data))
    r = CmfReader(cmf_path)
    problems = r.verify()
    assert problems, "data corruption not detected"
    r.close()
    print(f"  ✓ corruption: verify() honestly fails ({problems[0]})")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "quant":
        check_quant(sys.argv[2], sys.argv[3], sys.argv[4])
    elif mode == "sharded":
        check_sharded(sys.argv[2], sys.argv[3])
    elif mode == "corruption":
        check_corruption(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {mode}")
