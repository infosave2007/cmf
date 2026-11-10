# humaneval.py URL MODEL OUT.jsonl [N] [EXTRA_JSON]: HumanEval pass@1 through an OpenAI-compatible chat API,
# greedy, no thinking; every answer's program runs in a subprocess with a timeout.
import json, sys, time, re, subprocess, tempfile, os, urllib.request
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
lim = int(sys.argv[4]) if len(sys.argv) > 4 else 164
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
p = hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
rows = pq.read_table(p).to_pylist()[:lim]
SYS = "You are a careful Python programmer."
def ask(prompt):
    msg = ("Complete the following Python function. Return the whole function, with any imports it needs, "
           "in one ```python code block and nothing else.\n\n```python\n" + prompt + "```")
    body = {"model": model, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": msg}],
            "temperature": 0, "max_tokens": 1024, "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    body.update(extra)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    text = d["choices"][0]["message"].get("content") or ""
    return text, dt, d.get("usage", {})
def program(prompt, text, entry):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    code = max(m, key=len) if m else text
    if f"def {entry}" not in code:
        code = prompt + code
    head = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
    return head + "\n" + code
def run(src, test, entry):
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "t.py")
        open(f, "w").write(src + "\n\n" + test + f"\n\ncheck({entry})\n")
        try:
            r = subprocess.run([sys.executable, "-I", f], cwd=d, capture_output=True, timeout=15)
            return r.returncode == 0, r.stderr.decode()[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
ok = 0; tot_tok = 0; tot_t = 0.0
with open(out, "w") as fo:
    for i, r in enumerate(rows):
        text, dt, usage = ask(r["prompt"])
        passed, err = run(program(r["prompt"], text, r["entry_point"]), r["test"], r["entry_point"])
        ok += passed; tot_t += dt; tot_tok += usage.get("completion_tokens", 0)
        fo.write(json.dumps({"task": r["task_id"], "pass": passed, "err": err, "s": round(dt, 2), "usage": usage, "answer": text}) + "\n"); fo.flush()
        print(f"\r{i+1}/{len(rows)} pass {ok} ({100*ok/(i+1):.1f}%) {tot_tok/max(tot_t,1e-9):.1f} tok/s end-to-end", end="", flush=True)
print(f"\nHUMANEVAL {model}: pass@1 {ok}/{len(rows)} = {100*ok/len(rows):.2f}% | {tot_tok} tokens in {tot_t:.0f} s = {tot_tok/max(tot_t,1e-9):.1f} tok/s")
