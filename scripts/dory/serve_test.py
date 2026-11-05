#!/usr/bin/env python3
"""Multi-turn chat, reasoning and tool calls over `cortiq serve`. serve_test.py PORT"""
import json, sys, time, urllib.request

port = sys.argv[1]


def call(body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=300))
    return d, time.time() - t


msgs = [{"role": "user", "content": "What is the capital of Japan? One word."}]
for turn in range(3):
    d, dt = call({"model": "dory", "messages": msgs, "max_tokens": 200, "temperature": 0})
    m = d["choices"][0]["message"]
    print(f"turn {turn}: {dt:.2f}s usage {d.get('usage')}\n  reasoning: {str(m.get('reasoning_content'))[:120]!r}\n  content: {str(m.get('content'))[:200]!r}")
    msgs.append({"role": "assistant", "content": m.get("content") or ""})
    msgs.append({"role": "user", "content": ["And of France? One word.", "And of Italy? One word."][turn] if turn < 2 else "Thanks"})
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]
d, dt = call({"model": "dory", "messages": [{"role": "user", "content": "What is the weather in Paris right now, in celsius?"}], "tools": tools, "max_tokens": 400, "temperature": 0})
m = d["choices"][0]["message"]
print(f"tools: {dt:.2f}s finish {d['choices'][0].get('finish_reason')} tool_calls {json.dumps(m.get('tool_calls'))} content {str(m.get('content'))[:200]!r}")
