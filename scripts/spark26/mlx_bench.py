#!/usr/bin/env python3
"""Stock mlx-lm on a published MLX build of Spark-X2.6, with the build's own model file
(trust_remote_code), on the same prompts as `cortiq run`.
mlx_bench.py MODEL_DIR PROMPT_FILE [MAX_TOKENS]"""
import sys
from mlx_lm import load, stream_generate

model_dir, prompt_file = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 128
model, tok = load(model_dir, trust_remote_code=True)
text = open(prompt_file, encoding="utf-8").read()
prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, enable_thinking=False)
last, out = None, []
for r in stream_generate(model, tok, prompt, max_tokens=n):
    last = r
    out.append(r.text)
print(f"mlx {model_dir}: prompt {last.prompt_tokens} tok {last.prompt_tps:.1f} tok/s | decode {last.generation_tokens} tok {last.generation_tps:.1f} tok/s | peak {last.peak_memory:.2f} GB", flush=True)
print("text:", "".join(out)[:160].replace("\n", " "), flush=True)
