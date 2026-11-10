"""Raw f32 tensor dumps (cortiq dequant --all) + the dequant log's `name dtype [shape]` lines -> one
safetensors file (f32), next to a copy of the HF config/tokenizer, so `cortiq convert` can take the
authors' QAT weights (their GGUF Q4_1, imported) with the usual flags.
usage: python -I raw_to_st.py RAW_DIR DEQUANT_LOG HF_SRC_DIR OUT_DIR"""
import json, os, re, shutil, sys
import numpy as np
from safetensors.numpy import save_file
raw, log, src, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
t = {}
for line in open(log):
    m = re.match(r"^(\S+)\t(\S+)\t\[([0-9, ]+)\]$", line.strip())
    if m:
        shape = [int(x) for x in m.group(3).split(",")]
        a = np.fromfile(os.path.join(raw, m.group(1) + ".bin"), dtype="<f4")
        assert a.size == int(np.prod(shape)), m.group(1)
        t[m.group(1)] = a.reshape(shape)
save_file(t, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
for f in os.listdir(src):
    if f.endswith((".json", ".jinja", ".txt")) and f not in ("model.safetensors.index.json",):
        shutil.copy(os.path.join(src, f), out)
print(len(t), "tensors ->", out)
