"""diffusers ZImageTransformer2DModel forward time on MPS (bf16), the Mac baseline for WP3.
usage: zimage_mps_bench.py <transformer dir> <res,res> [reps]"""
import sys
import time

import torch
from diffusers import ZImageTransformer2DModel

d = sys.argv[1]
res = [int(r) for r in sys.argv[2].split(',')]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
t0 = time.time()
m = ZImageTransformer2DModel.from_pretrained(d, torch_dtype=torch.bfloat16).to('mps').eval()
torch.mps.synchronize()
print(f"load {time.time() - t0:.1f}s", flush=True)
g = torch.Generator(device='cpu').manual_seed(0)
cap = torch.randn(22, 2560, generator=g).to(torch.bfloat16).to('mps')
for r in res:
    for b in (1, 2):
        x = [torch.randn(16, 1, r // 8, r // 8, generator=g).to(torch.bfloat16).to('mps') for _ in range(b)]
        t = torch.full((b,), 0.5, dtype=torch.float32, device='mps')
        caps = [cap] * b
        ts = []
        with torch.no_grad():
            for i in range(reps + 1):
                torch.mps.synchronize()
                t1 = time.time()
                out = m(x, t, caps, return_dict=False)[0]
                torch.mps.synchronize()
                ts.append(time.time() - t1)
                print(f"res {r} batch {b} rep {i}: {ts[-1]:.3f}s", flush=True)
        w = sorted(ts[1:])
        print(f"RESULT res {r} batch {b}: median {w[len(w) // 2]:.3f}s (first {ts[0]:.3f}s) finite {all(bool(torch.isfinite(o).all()) for o in out)}",
              flush=True)
        del out
        torch.mps.empty_cache()
