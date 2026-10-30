"""Tool-call smoke test against `cortiq serve` (OpenAI API): one weather
tool, two turns — the model must call it, then answer from its result —
with thinking on and off.

usage: python3 -I tool_smoke.py [http://127.0.0.1:8080]
"""

import json
import sys
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080") + "/v1/chat/completions"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }
]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def main():
    for thinking in (True, False):
        msgs = [{"role": "user", "content": "What is the weather in Paris right now, in celsius?"}]
        base = {"model": "lfm3", "tools": TOOLS, "max_tokens": 2048, "enable_thinking": thinking}
        r = post({**base, "messages": msgs})
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        print(f"thinking={thinking} turn 1: finish={r['choices'][0]['finish_reason']} calls="
              f"{[(c['function']['name'], c['function']['arguments']) for c in calls]}")
        if not calls:
            print("  content:", (m.get("content") or "")[:300])
            continue
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        msgs.append({"role": "tool", "tool_call_id": calls[0].get("id", "0"),
                     "content": json.dumps({"city": "Paris", "temperature": 17, "unit": "celsius",
                                            "condition": "light rain"})})
        r = post({**base, "messages": msgs})
        print(f"  turn 2: finish={r['choices'][0]['finish_reason']} answer="
              f"{(r['choices'][0]['message'].get('content') or '')[:300]!r}")


if __name__ == "__main__":
    main()
