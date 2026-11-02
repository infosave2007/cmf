#!/usr/bin/env python3
"""Reference perplexity of a causal LM over the first N tokens of a text file
(no BOS, one window), float32 on CUDA. usage: ref_ppl.py <model_dir> <text> <N> [auto-class]"""
import sys, math, torch
from transformers import AutoTokenizer
import transformers
model_dir, path, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cls = getattr(transformers, sys.argv[4] if len(sys.argv) > 4 else "AutoModelForCausalLM")
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
ids = tok(open(path).read(), add_special_tokens=False).input_ids[:n]
m = cls.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32).cuda().eval()
torch.backends.cuda.matmul.allow_tf32 = False
x = torch.tensor([ids]).cuda()
with torch.inference_mode():
    lg = m(input_ids=x).logits[0].float()
nll = torch.nn.functional.cross_entropy(lg[:-1], x[0, 1:], reduction="mean").item()
print(f"tokens {len(ids)} ppl {math.exp(nll):.4f}")
