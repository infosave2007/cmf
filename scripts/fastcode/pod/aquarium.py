# aquarium.py URL MODEL OUT.html [EXTRA_JSON]: the owner's 3D aquarium prompt, greedy, no thinking, streamed;
# writes the page (the first ```html block, or the whole answer) and a .json with timings.
import json, sys, time, re, urllib.request
url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
prompt = open("/root/fc/data/3d-aquarium-prompt.md").read()
body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
        "max_tokens": 16000, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}]}
body.update(extra)
req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time(); t_first = t_last = None; parts = []; usage = {}
with urllib.request.urlopen(req, timeout=7200) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        d = json.loads(data)
        if d.get("usage"): usage = d["usage"]
        for ch in d.get("choices", []):
            c = (ch.get("delta") or {}).get("content")
            if c:
                now = time.time(); t_first = t_first or now; t_last = now; parts.append(c)
text = "".join(parts)
m = re.findall(r"```html\s*\n(.*?)```", text, re.S)
page = max(m, key=len) if m else text
open(out, "w").write(page)
ct = usage.get("completion_tokens", 0)
info = {"model": model, "completion_tokens": ct, "prompt_tokens": usage.get("prompt_tokens"), "ttft_s": round((t_first or t0) - t0, 2),
        "total_s": round(time.time() - t0, 1), "decode_tok_s": round((ct - 1) / (t_last - t_first), 1) if t_first and t_last and t_last > t_first else 0,
        "page_bytes": len(page), "had_html_block": bool(m)}
json.dump(info, open(out + ".json", "w"), indent=1); open(out + ".answer.md", "w").write(text)
print("AQUARIUM", json.dumps(info))
