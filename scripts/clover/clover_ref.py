"""Reference dump for the Clover CMF port (PyTorch, CPU, bf16).

For each prompt: the prompt ids, every row's top-10 logits of the first
denoising pass over an all-mask block (`model(block, preview=True)`), and
the greedy block generation with the vendor defaults. The engine's
`CMF_CLOVER_DUMP` writes the same first-pass record.

usage: python3 -I clover_ref.py MODEL_DIR OUT.json [MAX_NEW_TOKENS]
"""

import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Compute 17 * 23. Reply with just the number.",
    "Write one sentence about a lighthouse keeper.",
]


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    cfg = model.config
    records = []
    for text in PROMPTS:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(chat, add_special_tokens=False, return_tensors="pt").input_ids
        t0 = time.time()
        cache = model.model(ids, use_cache=True).past_key_values
        block = torch.full((1, cfg.block_size), cfg.mask_token_id, dtype=ids.dtype)
        out = model(
            block,
            attention_mask=torch.ones_like(block, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
            preview=True,
        )
        logits = out.logits[0].float()
        top = logits.topk(10, dim=-1)
        rows = [
            {"ids": i.tolist(), "logits": v.tolist()}
            for i, v in zip(top.indices, top.values)
        ]
        t1 = time.time()
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False)
        new = gen[0, ids.shape[1]:].tolist()
        t2 = time.time()
        rec = {
            "prompt": text,
            "prompt_ids": ids[0].tolist(),
            "rows": rows,
            "generated_ids": new,
            "generated_text": tok.decode(new),
            "preview_s": t1 - t0,
            "generate_s": t2 - t1,
        }
        print(json.dumps({k: rec[k] for k in ("prompt", "generated_text", "preview_s", "generate_s")}), flush=True)
        records.append(rec)
        with open(out_path, "w") as f:
            json.dump(records, f)


if __name__ == "__main__":
    main()
