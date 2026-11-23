#!/usr/bin/env python3
"""engine_eval.py PORT TAG [chat]: the ode_repro.py test sets (seed 0) against a cortiq server -> accuracy
in-distribution and on held-out coefficients. Default: raw /v1/completions, 32 greedy tokens (the skill's
format); `chat`: /v1/chat/completions with the chat template, thinking off, 300 tokens (the fair base)."""
import importlib.util, json, os, random, sys, time, urllib.request

port, tag = sys.argv[1], sys.argv[2]
chat = len(sys.argv) > 3 and sys.argv[3] == "chat"
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r", os.path.join(HERE, "ode_repro.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
rng = random.Random(1000)  # the same draw order as ode_repro.py: in-distribution first, then held-out
sets = {"in_dist": r.make_set(r.TRAIN, 100, rng), "held_out": r.make_set(r.HELD, 100, rng)}
res = {"tag": tag}
t0 = time.time()
for name, items in sets.items():
    ok_n, rows = 0, []
    for kind, prompt, sol, ok in items:
        if chat:
            body = json.dumps({"model": "cmf", "messages": [{"role": "user", "content": prompt}], "max_tokens": 300,
                               "temperature": 0.0, "repetition_penalty": 1.0, "enable_thinking": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
            txt = (json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or "").strip()
        else:
            body = json.dumps({"model": "cmf", "prompt": prompt, "max_tokens": 32, "temperature": 0.0,
                               "repetition_penalty": 1.0}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["text"].strip()
        good = bool(ok(txt))
        ok_n += good
        rows.append({"kind": kind, "prompt": prompt.splitlines()[0], "expected": sol, "got": txt[:160], "ok": good})
    res[name] = ok_n / len(items)
    agg = {}
    for x in rows:
        a = agg.setdefault(x["kind"], [0, 0])
        a[0] += x["ok"]
        a[1] += 1
    res[name + "_kinds"] = {k: f"{v[0]}/{v[1]}" for k, v in sorted(agg.items())}
    res[name + "_examples"] = rows[:4] + [x for x in rows if not x["ok"]][:6]
res["seconds"] = round(time.time() - t0, 1)
json.dump(res, open(f"engine_{tag}.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if not k.endswith("_examples")}))
