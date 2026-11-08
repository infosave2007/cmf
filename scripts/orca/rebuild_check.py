#!/usr/bin/env python3
# Run rebuild.work on chosen tensors and print the reports (V-order and convention check). rebuild_check.py GGUF QWEN NAME...
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebuild as R
g, q = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(q, "model.safetensors.index.json")))["weight_map"]
R.init(g)
tmp = tempfile.mkdtemp(dir="/root/o")
for n in sys.argv[3:]:
    sp = os.path.join(q, idx[n]); d0, h = R.st_header(sp)
    hf, p, rep = R.work((n, sp, d0, h[n], tmp, 32, 2.0))
    os.remove(p)
    print(json.dumps(rep), flush=True)
