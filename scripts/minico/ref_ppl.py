"""Transformers reference for the Minico-M1-1105 CMF conversion (Mac CPU).

  ppl  IDS_FILE OUT.npz   teacher-forced NLL over exactly the ids `cortiq ppl`
                          scored (CMF_PPL_IDS_OUT=IDS_FILE): per-position NLL,
                          argmax and the perplexity, bf16 weights upcast to f32.
  tok  CORPUS IDS_FILE    does the HF tokenizer give the same ids (BOS + text)?

usage: python -I ref_ppl.py MODEL_DIR ppl IDS_FILE OUT.npz
       python -I ref_ppl.py MODEL_DIR tok CORPUS IDS_FILE
"""
import math
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, mode = sys.argv[1], sys.argv[2]
if mode == "tok":
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = [int(t) for t in open(sys.argv[4]).read().split()]
    text = open(sys.argv[3], encoding="utf-8").read()
    hf = [tok.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
    n = len(ids)
    diff = next((i for i in range(n) if ids[i] != hf[i]), None)
    print(f"tokenizer: {n} cortiq ids, first difference at {diff}")
    sys.exit(0 if diff is None else 1)

ids = [int(t) for t in open(sys.argv[3]).read().split()]
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
t0 = time.time()
with torch.no_grad():
    x = torch.tensor([ids])
    logits = model(x).logits[0, :-1].double()
    lp = torch.log_softmax(logits, dim=-1)
    tgt = x[0, 1:]
    nll = -lp[torch.arange(len(tgt)), tgt]
    top = logits.argmax(-1)
np.savez(sys.argv[4], nll=nll.numpy(), argmax=top.numpy(), ids=np.array(ids))
print(f"{len(tgt)} positions in {time.time() - t0:.1f} s; "
      f"REF PPL (transformers f32) = {math.exp(nll.mean().item()):.3f}")
