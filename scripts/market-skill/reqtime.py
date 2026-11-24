# reqtime.py PORT EVAL.jsonl: time single requests in several shapes against a running serve
import json, sys, time, urllib.request
port, path = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(path)][:6]
sys.path.insert(0, ".")
from market_skill import system_prompt, CLASSES
def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return time.time() - t, out
for name, extra in [("class, max_tokens 0", {"max_tokens": 0, "cortiq": {"class_tokens": ["DOWN", "UP"]}}),
                    ("class, max_tokens 1", {"max_tokens": 1, "cortiq": {"class_tokens": ["DOWN", "UP"]}}),
                    ("plain, max_tokens 1", {"max_tokens": 1})]:
    ts = []
    for r in recs:
        p = {"model": "m", "temperature": 0.0, "enable_thinking": False,
             "messages": [{"role": "system", "content": system_prompt("dir", r["symbol"])}, {"role": "user", "content": r["prompt"]}]}
        p.update(extra)
        dt, out = post(p)
        ts.append(dt)
    u = out.get("usage", {})
    print(f"{name}: {', '.join(f'{x:.2f}' for x in ts)} s | usage {u} | cortiq keys {list(out.get('cortiq', {}).keys())}")
