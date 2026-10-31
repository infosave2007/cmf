"""Transformers reference for the LiquidAI LFM3-1.7B CMF conversion.

The checkpoint's own modeling code (trust_remote_code) on one GPU, in bf16
(the published dtype) or float32 (bf16 weights upcast: the exact target of
an f32/f16 CMF, free of bf16 activation rounding).

  ppl  IDS_FILE OUT.npz  teacher-forced NLL over exactly the ids `cortiq ppl`
                         scored (CMF_PPL_IDS_OUT=IDS_FILE): per-position NLL
                         and argmax, and the perplexity.
  chat OUT.json [N]      chat prompts through the checkpoint's own template:
                         the rendered text, prompt ids, the top-10 logits of
                         the first answer position and N greedy tokens.

usage: python3 -I lfm3_ref.py MODEL_DIR {bf16|f32} ppl IDS_FILE OUT.npz
       python3 -I lfm3_ref.py MODEL_DIR {bf16|f32} chat OUT.json [MAX_NEW_TOKENS]
"""

import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Compute 17 * 23. Reply with just the number.",
    "Write one sentence about a lighthouse keeper.",
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Переведи на английский: «Мы скоро вернёмся».",
]


def load(model_dir, dtype):
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype={"bf16": torch.bfloat16, "f32": torch.float32}[dtype],
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda:0").eval()
    print(f"loaded {dtype} in {time.time() - t0:.0f} s", flush=True)
    return model


def ppl(model, ids_path, out_path):
    ids = [int(t) for t in open(ids_path).read().split()]
    x = torch.tensor([ids], device="cuda:0")
    t0 = time.time()
    hidden = model.model(input_ids=x, use_cache=False).last_hidden_state
    nll, top = [], []
    for s in range(0, len(ids) - 1, 512):
        e = min(s + 512, len(ids) - 1)
        logits = model.lm_head(hidden[0, s:e]).float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(ids[s + 1 : e + 1], device=logits.device)
        nll.append((-lp.gather(1, tgt[:, None])[:, 0]).cpu())
        top.append(logits.argmax(-1).cpu())
    nll = torch.cat(nll).double().numpy()
    top = torch.cat(top).numpy().astype(np.int64)
    p = float(np.exp(nll.mean()))
    print(f"PPL = {p:.4f} over {len(ids)} tokens ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out_path, nll=nll, argmax=top, ids=np.array(ids, dtype=np.int64), ppl=p)


def chat(model, model_dir, out_path, max_new):
    tok = AutoTokenizer.from_pretrained(model_dir)
    records = []
    for text in PROMPTS:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda:0")
        t0 = time.time()
        logits = model(input_ids=ids).logits[0, -1].float()
        top = logits.topk(10)
        gen = model.generate(
            ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0
        )
        new = gen[0, ids.shape[1] :].tolist()
        rec = {
            "prompt": text,
            "rendered": rendered,
            "prompt_ids": ids[0].tolist(),
            "top10_ids": top.indices.tolist(),
            "top10_logits": top.values.tolist(),
            "generated_ids": new,
            "generated_text": tok.decode(new),
            "seconds": time.time() - t0,
        }
        print(json.dumps({k: rec[k] for k in ("prompt", "generated_text")}, ensure_ascii=False), flush=True)
        records.append(rec)
        with open(out_path, "w") as f:
            json.dump(records, f, ensure_ascii=False)


def main():
    model_dir, dtype, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    torch.set_grad_enabled(False)
    model = load(model_dir, dtype)
    if mode == "ppl":
        ppl(model, sys.argv[4], sys.argv[5])
    elif mode == "chat":
        chat(model, model_dir, sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 48)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
