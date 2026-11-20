"""Transformers reference for Aion-1.0-Instruct-2.3B.

python3 -I ref_hf.py MODEL_DIR OUT_DIR
- logits of fixed prompts (f32 weights, eager attention): last-position top-10 and the
  whole last-row logits as .npy, per prompt
- greedy 48 tokens of a chat prompt (current_date pinned)
- the rendered chat prompts and ids (for cortiq's template parity)
"""
import json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

md, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
dtype = torch.float32 if os.environ.get("REF_DTYPE", "f32") == "f32" else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(md, trust_remote_code=True, dtype=dtype,
                                             attn_implementation="eager").cuda().eval()
prompts = {
    "fox": "The quick brown fox jumps over the lazy dog. The capital of France is",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "zh": "人工智能是",
}
res = {}
for name, text in prompts.items():
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        lg = model(torch.tensor([ids]).cuda()).logits[0].float().cpu().numpy()
    np.save(f"{out}/logits_{name}.npy", lg[-1])
    top = np.argsort(-lg[-1])[:10]
    res[name] = {"ids": ids, "top10": top.tolist(), "top10_logit": lg[-1][top].tolist(),
                 "top10_tok": [tok.decode([t]) for t in top]}
    print(name, len(ids), res[name]["top10_tok"][:5], flush=True)

chats = {
    "capital": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "system": [{"role": "system", "content": "You are a terse assistant."},
               {"role": "user", "content": "Name three prime numbers."}],
    "code": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    "two_turns": [{"role": "user", "content": "Hi! Who are you?"},
                  {"role": "assistant", "content": "I am Aion Instruct, a language model by Microsoft."},
                  {"role": "user", "content": "Summarize in one line: the sun is a star at the center of the solar system."}],
    "unicode": [{"role": "user", "content": "Переведи на английский: «Привет, мир!» 你好"}],
}
weather = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
cases = []
for name, msgs in list(chats.items()) + [("tools", [{"role": "user", "content": "What's the weather in Paris?"}])]:
    tools = [weather] if name == "tools" else None
    text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False,
                                   current_date="2026-11-20")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]).cuda(), max_new_tokens=48, do_sample=False)
    ans = g[0, len(ids):].tolist()
    cases.append({"name": name, "messages": msgs, "tools": tools, "text": text, "ids": ids,
                  "greedy_ids": ans, "greedy": tok.decode(ans)})
    print("==", name, len(ids), repr(tok.decode(ans))[:300], flush=True)
json.dump(res, open(f"{out}/logits.json", "w"), ensure_ascii=False, indent=1)
json.dump(cases, open(f"{out}/chat.json", "w"), ensure_ascii=False, indent=1)
