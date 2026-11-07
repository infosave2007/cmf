"""Transformers greedy reference for chat prompts (Minico-M1-1105, f32 CPU).

Prints, per prompt and effort, the rendered prompt ids and the greedy answer,
as JSON lines, for comparison with `cortiq run --greedy` (CMF_PROMPT_DUMP=1).

usage: python -I chat_ref.py MODEL_DIR OUT.jsonl MAX_NEW EFFORT PROMPT [PROMPT ...]
       EFFORT: medium | off | low | high | max | default (no kwarg)
"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out, max_new, effort = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
kw = {} if effort == "default" else {"reasoning_effort": effort}
with open(out, "w") as f:
    for p in sys.argv[5:]:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      tokenize=True, return_dict=True, **kw)["input_ids"]
        with torch.no_grad():
            g = model.generate(torch.tensor([ids]), max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0)
        ans = g[0, len(ids):].tolist()
        f.write(json.dumps({"prompt": p, "effort": effort, "ids": ids, "answer_ids": ans,
                            "answer": tok.decode(ans, skip_special_tokens=False)}) + "\n")
        print(f"[{effort}] {p!r}: {len(ids)} prompt ids, {len(ans)} answer ids")
