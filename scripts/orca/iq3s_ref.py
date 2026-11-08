#!/usr/bin/env python3
# One real IQ3_S block of the GGUF + gguf-py's dequantization of it (the test reference). iq3s_ref.py GGUF
import sys, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(sys.argv[1])
for t in r.tensors:
    if t.tensor_type.name == "IQ3_S":
        raw = np.asarray(t.data).reshape(-1).view(np.uint8)
        blk = raw[110 * 7:110 * 8]  # the 8th block of the tensor
        y = dequantize(blk.reshape(1, 110), t.tensor_type).reshape(-1)
        print(t.name, "block 7")
        print("HEX", blk.tobytes().hex())
        print("REF", [(i, float(f"{y[i]:.7g}")) for i in (0, 1, 2, 3, 4, 7, 8, 31, 32, 63, 64, 100, 128, 200, 255)])
        print("NZ", int((y != 0).sum()), "MAXABS", float(np.abs(y).max()))
        break
