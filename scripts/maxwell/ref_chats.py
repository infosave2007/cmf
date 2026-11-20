"""The chat prompts of the checks, rendered by the reference template (tokenizer only).

python3 -I ref_chats.py MODEL_DIR OUT.json
"""
import json, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
chats = {
    "capital": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    "system": [{"role": "system", "content": "You are a terse assistant."}, {"role": "user", "content": "Name three prime numbers."}],
    "code": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome."}],
    "two_turns": [{"role": "user", "content": "Hi! Who are you?"}, {"role": "assistant", "content": "I am a language model."},
                  {"role": "user", "content": "Summarize in one line: the sun is a star at the center of the solar system."}],
    "unicode": [{"role": "user", "content": "Переведи на английский: «Привет, мир!» 你好"}],
}
weather = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
out = []
for name, msgs in list(chats.items()) + [("tools", [{"role": "user", "content": "What's the weather in Paris?"}])]:
    tools = [weather] if name == "tools" else None
    for think in (True, False):
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        out.append({"name": name + ("" if think else "_nothink"), "messages": msgs, "tools": tools, "think": think, "text": text,
                    "ids": tok(text, add_special_tokens=False)["input_ids"]})
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("CHATS_DONE", len(out))
