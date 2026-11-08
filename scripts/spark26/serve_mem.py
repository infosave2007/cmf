#!/usr/bin/env python3
"""Memory and disk writes of `cortiq serve` over an agent-like dialogue on macOS.
serve_mem.py CORTIQ MODEL TEXT CHARS [TURNS] [ENV=V ...]
Starts the server, sends TURNS chat requests that grow one prompt (each turn adds the previous answer
and a new user message, as an agent does), samples phys_footprint, resident size and disk bytes written
(proc_pid_rusage) every 0.25 s, and prints one line per turn plus the server's timing lines."""
import ctypes, json, os, subprocess, sys, threading, time, urllib.request

cortiq, model, text_path, chars = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
turns = int(sys.argv[5]) if len(sys.argv) > 5 else 3
extra = dict(a.split("=", 1) for a in sys.argv[6:])
port = 8199
lib = ctypes.CDLL("/usr/lib/libproc.dylib")

class RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(18)]

def usage(pid):
    r = RU()
    if lib.proc_pid_rusage(pid, 2, ctypes.byref(r)) != 0:
        return None
    return {"resident": r.f6, "footprint": r.f7, "read": r.f16, "written": r.f17}

env = dict(os.environ, **extra)
log = open(os.environ.get("SERVE_LOG", "serve_mem.log"), "w")
srv = subprocess.Popen([cortiq, "serve", model, "--port", str(port), "--host", "127.0.0.1"], stdout=log, stderr=log, env=env)
peak = {"footprint": 0, "resident": 0}
stop = False
def sampler():
    while not stop:
        u = usage(srv.pid)
        if u:
            peak["footprint"] = max(peak["footprint"], u["footprint"])
            peak["resident"] = max(peak["resident"], u["resident"])
        time.sleep(0.25)
threading.Thread(target=sampler, daemon=True).start()
for _ in range(600):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2)
        break
    except Exception:
        time.sleep(0.5)
gb = lambda b: b / 2**30
u0 = usage(srv.pid)
print(f"loaded: footprint {gb(u0['footprint']):.2f} GB resident {gb(u0['resident']):.2f} GB written {u0['written']/2**20:.1f} MB", flush=True)
text = open(text_path, encoding="utf-8").read().replace("\r", "")
msgs = [{"role": "system", "content": "You are a coding agent. Answer briefly."},
        {"role": "user", "content": text[:chars] + "\n\nSummarize the text above in three sentences."}]
for t in range(turns):
    peak["footprint"] = peak["resident"] = 0
    b = usage(srv.pid)
    t0 = time.time()
    body = json.dumps({"model": "m", "messages": msgs, "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=3600))
    a = usage(srv.pid)
    tm = resp.get("timings", {})
    print(f"turn {t}: prompt {resp['usage']['prompt_tokens']} tok, {time.time()-t0:.1f} s | peak footprint {gb(peak['footprint']):.2f} GB "
          f"resident {gb(peak['resident']):.2f} GB | after {gb(a['footprint']):.2f} GB | written {(a['written']-b['written'])/2**20:.1f} MB "
          f"read {(a['read']-b['read'])/2**20:.1f} MB | timings {json.dumps(tm)}", flush=True)
    msgs.append({"role": "assistant", "content": resp["choices"][0]["message"].get("content") or ""})
    msgs.append({"role": "user", "content": text[chars + t * 2000: chars + (t + 1) * 2000] + "\n\nAnd this part, in one sentence?"})
stop = True
if os.environ.get("VMMAP"):
    r = subprocess.run(["footprint", str(srv.pid)], capture_output=True, text=True)
    print(r.stdout[-2500:] or r.stderr[-500:])
srv.terminate(); srv.wait()
log.close()
print("".join(l for l in open(os.environ.get("SERVE_LOG", "serve_mem.log")) if "time =" in l or "WARN" in l or "ERROR" in l)[-3000:])
