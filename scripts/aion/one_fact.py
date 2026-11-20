import json, subprocess, sys, os
cortiq, model, facts, idx = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
it = json.load(open(facts))[idx]
open("/tmp/one_fact.txt", "w").write(it["msg"])
env = dict(os.environ, XDG_RUNTIME_DIR="/tmp", CMF_GPU="cuda")
r = subprocess.run([cortiq, "run", model, "--prompt-file", "/tmp/one_fact.txt", "--greedy", "--no-think", "--max-tokens", "40"], capture_output=True, text=True, env=env)
print("rc", r.returncode)
print("STDOUT tail:", r.stdout[-600:])
print("STDERR tail:", r.stderr[-2500:])
