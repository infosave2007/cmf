# speedtest.py URL MODEL OUT.json CTX[,CTX...] [EXTRA_JSON]: streaming chat requests with a code context of about CTX
# tokens (real code from the held-out eval set, a fresh tag in front so no cache helps), greedy, no thinking.
# prefill tok/s = prompt tokens / TTFT; decode tok/s = (completion tokens - 1) / (t_last - t_first).
import json, sys, time, urllib.request, random
url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
ctxs = [int(c) for c in sys.argv[4].split(",")]
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
code = open("/root/fc/data/eval_code.txt").read()
# one untimed request first: shader compilation and first-touch uploads stay out of the numbers
_w = {"model": model, "temperature": 0, "max_tokens": 8, "enable_thinking": False,
      "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "Say hi."}]}
_w.update(extra)
urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(_w).encode(),
                       headers={"Content-Type": "application/json"}), timeout=3600).read()
res = []
for ctx in ctxs:
    nchar = int(ctx * 3.4)
    body_txt = (code * (1 + nchar // len(code)))[:nchar]
    tag = f"[session {random.randrange(1<<30):08x}]\n"
    msg = (tag + "Here is a code base:\n\n" + body_txt + "\n\nWrite a new Python function `summarize_repo(files: dict[str, str]) -> str` "
           "that lists every function and class defined in these files, with a short docstring. Answer with code only.")
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": 256, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": msg}]}
    body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; t_last = None; n_chunks = 0; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
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
                    now = time.time()
                    if t_first is None: t_first = now
                    t_last = now; n_chunks += 1; text.append(c)
    pt = usage.get("prompt_tokens", 0); ct = usage.get("completion_tokens", n_chunks)
    ttft = (t_first or time.time()) - t0
    dec = (ct - 1) / (t_last - t_first) if t_first and t_last and t_last > t_first else 0
    row = {"ctx": ctx, "prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 3),
           "prefill_tok_s": round(pt / ttft, 1) if ttft > 0 else 0, "decode_tok_s": round(dec, 1), "text": "".join(text)[:400]}
    res.append(row)
    print(f"ctx {ctx}: prompt {pt} tok, TTFT {ttft:.2f} s = {row['prefill_tok_s']} tok/s, decode {ct} tok at {row['decode_tok_s']} tok/s", flush=True)
json.dump(res, open(out, "w"), indent=1)
