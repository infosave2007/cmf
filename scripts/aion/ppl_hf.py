"""HF perplexity over the same windows `cortiq ppl --windows N --window-len L` scores.

python3 -I ppl_hf.py MODEL_DIR TEXT N L
Windows: the text is tokenized whole (no special tokens); N windows of L tokens evenly
at k*stride, stride=(T-L-1)//N; every prediction inside a window is scored.
"""
import math, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
md, path, N, L = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(md, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
ids = tok(open(path, encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
T = len(ids)
stride = (T - L - 1) // N  # cortiq PplWindows::offsets
starts = [k * stride for k in range(N)]
nll, n = 0.0, 0
for s in starts:
    x = torch.tensor([ids[s:s + L]]).cuda()
    with torch.no_grad():
        lg = m(x).logits[0].float()
    lp = torch.log_softmax(lg[:-1], -1).gather(1, x[0, 1:, None])[:, 0]
    nll -= lp.sum().item(); n += L - 1
print(f"{path}: tokens {T}, windows {len(starts)}x{L}, ppl {math.exp(nll / n):.4f}")
