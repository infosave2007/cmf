#!/usr/bin/env python3
"""The planted-fact prompts of facts_ref.py through `cortiq run` (greedy, no thinking).
usage: facts_engine.py CORTIQ MODEL.cmf FACTS.json [ENV=VAL ...]"""
import json, os, re, subprocess, sys, time

cortiq, model, facts = sys.argv[1:4]
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", **dict(a.split("=", 1) for a in sys.argv[4:]))
items = json.load(open(facts))
ok = 0
for i, it in enumerate(items):
    pf = f"/tmp/facts_prompt_{os.getpid()}.txt"
    open(pf, "w").write(it["msg"])
    t0 = time.time()
    r = subprocess.run([cortiq, "run", model, "--prompt-file", pf, "--greedy", "--no-think", "--max-tokens", "40"],
                       capture_output=True, text=True, env=env, timeout=3600)
    out = r.stdout
    stats = re.findall(r"\[\d+ tokens ·[^\]]*\]", out)
    # `run` echoes the prompt, fact included: judge only what follows the question
    q = it["msg"].rsplit("\n", 1)[-1]
    answer = out[out.rfind(q) + len(q):] if q in out else ""
    answer = answer.split("\n[")[0].strip()
    hit = it["fact"] in answer
    ok += hit
    print(f"  {i} tokens {it.get('tokens')} depth {it['depth']:.2f}: {'OK' if hit else 'miss'} {it['fact']} "
          f"wall {time.time()-t0:.0f}s {stats[-1] if stats else ''} ref_ok {it.get('ref_ok')} "
          f"answer {answer[:80]!r}", flush=True)
print(f"FACTS {ok}/{len(items)}", flush=True)
