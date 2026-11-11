#!/usr/bin/env python3
"""A three-turn greedy chat against a running `cortiq serve` (no-think), each
turn carrying the model's own previous answers. Writes the answers to
OUT/turnN.txt; the server prints each turn's prompt / decode time.
    serve_chat.py PORT PROMPT_FILE OUT"""
import json, sys, urllib.request

port, prompt_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
first = open(prompt_file).read()
follow = [
    first,
    "Now list every function you changed, one line each, with what changed in it.",
    "Write a short commit message for these changes.",
]
msgs = []
for i, text in enumerate(follow, 1):
    msgs.append({"role": "user", "content": text})
    body = json.dumps({"model": "q", "messages": msgs, "max_tokens": 300, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    ans = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]
    open(f"{out}/turn{i}.txt", "w").write(ans)
    msgs.append({"role": "assistant", "content": ans})
print("CHAT_DONE")
