"""Reference for the Apex-35B-V1.0 CMF conversion (PyTorch, bf16).

The bf16 checkpoint (~67 GB of text weights) does not fit one 48 GB card,
so the layers that do not fit run from host memory (accelerate offload).

  ppl  IDS_FILE OUT.npz  teacher-forced NLL over exactly the ids `cortiq ppl`
                         scored (CMF_PPL_IDS_OUT=IDS_FILE): per-position NLL
                         and argmax, and the perplexity.
  chat OUT.json [N]      three chat prompts through the checkpoint's own
                         template: prompt ids, the top-10 logits of the first
                         answer position and N greedy tokens.

usage: python3 -I apex_ref.py MODEL_DIR ppl IDS_FILE OUT.npz
       python3 -I apex_ref.py MODEL_DIR chat OUT.json [MAX_NEW_TOKENS]
"""

import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Compute 17 * 23. Reply with just the number.",
    "Write one sentence about a lighthouse keeper.",
]


def load(model_dir):
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "38GiB", "cpu": "34GiB"},
    ).eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    return model


def ppl(model, ids_path, out_path):
    ids = [int(t) for t in open(ids_path).read().split()]
    x = torch.tensor([ids])
    lm = model.model.language_model
    head = model.lm_head
    t0 = time.time()
    # Offloaded modules hold `meta` parameters: inputs go to cuda:0 and the
    # accelerate hooks move them to each module's execution device.
    hidden = lm(input_ids=x.to("cuda:0"), use_cache=False).last_hidden_state
    nll, top = [], []
    for s in range(0, len(ids) - 1, 512):
        e = min(s + 512, len(ids) - 1)
        logits = head(hidden[0, s:e]).float()
        lp = torch.log_softmax(logits, dim=-1)
        tgt = torch.tensor(ids[s + 1 : e + 1], device=logits.device)
        nll.append((-lp.gather(1, tgt[:, None])[:, 0]).cpu())
        top.append(logits.argmax(-1).cpu())
    nll = torch.cat(nll).double().numpy()
    top = torch.cat(top).numpy().astype(np.int64)
    p = float(np.exp(nll.mean()))
    print(f"PPL = {p:.3f} over {len(ids)} tokens ({time.time() - t0:.0f} s)", flush=True)
    np.savez(out_path, nll=nll, argmax=top, ids=np.array(ids, dtype=np.int64), ppl=p)


def chat(model, model_dir, out_path, max_new):
    tok = AutoTokenizer.from_pretrained(model_dir)
    records = []
    for text in PROMPTS:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(rendered, add_special_tokens=False, return_tensors="pt").input_ids
        dev = "cuda:0"
        t0 = time.time()
        logits = model(input_ids=ids.to(dev)).logits[0, -1].float()
        top = logits.topk(10)
        gen = model.generate(ids.to(dev), max_new_tokens=max_new, do_sample=False)
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
        print(json.dumps({k: rec[k] for k in ("prompt", "generated_text", "seconds")}), flush=True)
        records.append(rec)
        with open(out_path, "w") as f:
            json.dump(records, f)


def main():
    model_dir, mode = sys.argv[1], sys.argv[2]
    torch.set_grad_enabled(False)
    model = load(model_dir)
    if mode == "ppl":
        ppl(model, sys.argv[3], sys.argv[4])
    elif mode == "chat":
        chat(model, model_dir, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 48)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
