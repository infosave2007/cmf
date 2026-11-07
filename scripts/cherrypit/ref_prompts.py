"""Reference chat prompts for CherryPit-1.2-2B rendered by transformers.

Usage: python -I ref_prompts.py MODEL_DIR OUT_JSON
Writes, for each test conversation, the rendered text and the token ids
(the HF tokenizer and chat_template.jinja), so cortiq's own rendering can be
checked against them.
"""
import json
import sys

from transformers import AutoTokenizer

model_dir, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

cases = [
    ("plain", [{"role": "user", "content": "What is the capital of France?"}], None, None),
    ("no_think", [{"role": "user", "content": "What is the capital of France?"}], None, False),
    ("think", [{"role": "user", "content": "What is 17 * 23?"}], None, True),
    (
        "system",
        [
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Name three prime numbers."},
        ],
        None,
        None,
    ),
    (
        "two_turns",
        [
            {"role": "user", "content": "Hi! Who are you?"},
            {"role": "assistant", "content": "I'm CherryPit 1.2, an AI assistant made by the team at DRUPEs."},
            {"role": "user", "content": "Write a haiku about cherries."},
        ],
        None,
        None,
    ),
    ("tools", [{"role": "user", "content": "What's the weather in Paris?"}], [weather_tool], False),
    ("unicode", [{"role": "user", "content": "Переведи на английский: «Привет, мир!» 你好"}], None, False),
]

out = []
for name, msgs, tools, think in cases:
    kw = {}
    if think is not None:
        kw["enable_thinking"] = think
    text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, **kw)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out.append({"name": name, "messages": msgs, "tools": tools, "enable_thinking": think, "text": text, "ids": ids})
    print(name, len(ids))

with open(out_path, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
