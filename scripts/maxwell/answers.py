"""Greedy answers of a `cortiq serve` to the reference's rendered chat prompts, for `ref_layers.py`'s check.

python3 answers.py CORTIQ MODEL.cmf CHATS.json OUT_JOBS.json [MAX_TOKENS=40] [ENV=VAL ...]

CHATS.json is `ref_chats.py`'s. Every prompt goes twice: its rendered text through /v1/completions (the model
alone) and its messages through /v1/chat/completions (cortiq's own rendering of the template); the two answers
must be the same text. OUT_JOBS.json: the `greedy` jobs of ref_layers.py.
"""
import json, os, subprocess, sys, time, urllib.request

cortiq, model, chats, out = sys.argv[1:5]
rest = sys.argv[5:]
max_tokens = int(rest[0]) if rest and "=" not in rest[0] else 40
env = dict(os.environ, **dict(a.split("=", 1) for a in rest if "=" in a))
port = 8123
srv = subprocess.Popen([cortiq, "serve", model, "--port", str(port)], env=env, stdout=subprocess.DEVNULL, stderr=open("serve_answers.log", "w"))


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


try:
    for _ in range(1200):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            break
        except Exception:
            if srv.poll() is not None:
                raise SystemExit("the server exited: see serve_answers.log")
            time.sleep(1)
    greedy = {"temperature": 0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": max_tokens}
    jobs, same = [], 0
    for c in json.load(open(chats)):
        t0 = time.time()
        raw = post("/v1/completions", {"model": "m", "prompt": c["text"], **greedy})["choices"][0]["text"]
        body = {"model": "m", "messages": c["messages"], "enable_thinking": c["think"], "reasoning_format": "raw", **greedy}
        if c["tools"]:
            body["tools"] = c["tools"]
        try:
            msg = post("/v1/chat/completions", body)["choices"][0]["message"]
            chat = (msg.get("reasoning_content") or "", msg.get("content") or "", msg.get("tool_calls"))
        except Exception as e:
            chat = (f"ERROR {e}", "", None)
        # the chat answer's visible parts must all be in the raw continuation, in order
        parts = [p.strip() for p in chat[:2] if p and p.strip()]
        at, ok = 0, True
        for p in parts:
            i = raw.find(p, at)
            ok, at = ok and i >= 0, i + len(p)
        same += ok
        jobs.append({"name": c["name"], "kind": "greedy", "prompt": c["text"], "answer": raw})
        print(f"== {c['name']} ({time.time() - t0:.1f} s) template {'same' if ok else 'DIFFERS'}\n  raw : {raw!r}\n  chat: {chat!r}", flush=True)
    json.dump(jobs, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"ANSWERS {len(jobs)} prompts, chat = raw on {same}")
finally:
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except Exception:
        srv.kill()
