"""Prompt parity: cortiq serve vs the transformers rendering.

Usage: python3 -I parity.py REF_JSON PORT CORTIQ MODEL
For every case: the server's prompt_tokens must equal len(ref ids), and the
greedy answer through /v1/chat/completions must equal the greedy answer of
`cortiq run` fed the reference ids (CMF_PROMPT_IDS).
"""
import json
import os
import subprocess
import sys
import urllib.request

ref_path, port, cortiq, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
cases = json.load(open(ref_path))
N = 40


def chat(case):
    body = {"model": "m", "messages": case["messages"], "temperature": 0, "max_tokens": N}
    if case["tools"]:
        body["tools"] = case["tools"]
    if case["enable_thinking"] is not None:
        body["enable_thinking"] = case["enable_thinking"]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


ok_all = True
for case in cases:
    resp = chat(case)
    msg = resp["choices"][0]["message"]
    usage = resp.get("usage", {})
    served = (msg.get("reasoning_content") or "", msg.get("content") or "", msg.get("tool_calls"))
    ids_path = os.path.join(os.path.dirname(os.path.abspath(ref_path)), f"ids_{case['name']}.json")
    with open(ids_path, "w") as f:
        json.dump(case["ids"], f)
    env = dict(os.environ, CMF_PROMPT_IDS=ids_path)
    run = subprocess.run(
        [cortiq, "run", model, "--greedy", "-n", str(N), "--prompt", "x"],
        env=env, capture_output=True, text=True,
    )
    pt = usage.get("prompt_tokens")
    same_len = pt == len(case["ids"])
    print(f"== {case['name']}: prompt_tokens server={pt} ref={len(case['ids'])} {'OK' if same_len else 'DIFF'}")
    print("   served:", json.dumps(served, ensure_ascii=False)[:400])
    body = run.stdout.split("Prompt: x", 1)[-1].strip()
    print("   ref-ids run:", json.dumps(body, ensure_ascii=False)[:400])
    ok_all &= same_len
print("ALL PROMPT LENGTHS MATCH" if ok_all else "SOME PROMPT LENGTHS DIFFER")
