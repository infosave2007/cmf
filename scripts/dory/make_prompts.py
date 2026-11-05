#!/usr/bin/env python3
"""Token ids of the reference prompts, by the vendor's tokenizer and template."""
import json, sys
from transformers import PreTrainedTokenizerFast

model, out = sys.argv[1], sys.argv[2]
# AutoTokenizer validates config.json, and Transformers does not know the "mlp" layer type.
tok = PreTrainedTokenizerFast(tokenizer_file=model + "/tokenizer.json", bos_token="<s>", eos_token="<|im_end|>", unk_token="<unk>", pad_token="<|im_end|>")
tok.chat_template = open(model + "/chat_template.jinja").read()


def chat(msg, **kw):
    return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False, **kw)


P = [
    ("raw_hello", "The capital of France is"),
    ("chat_think", chat("What is 17 * 23? Answer briefly.")),
    ("chat_nothink", chat("Write a haiku about the sea.", enable_thinking=False)),
]
res = [{"name": n, "text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for n, t in P]
if len(sys.argv) > 3:
    text = open(sys.argv[3], encoding="utf-8").read().replace("\r", "")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    res.append({"name": "long1500", "text": "", "ids": ids[:1500]})
json.dump(res, open(out, "w"))
for r in res:
    print(r["name"], len(r["ids"]), repr(r["text"][:300]))
print("specials", tok.bos_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id, tok.vocab_size, len(tok))
