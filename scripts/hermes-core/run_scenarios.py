#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the Hermes scenarios against an OpenAI-compatible endpoint and record traces.

Hermes itself stays root; its terminal and file tools go through the ssh backend to an
unprivileged user, so nothing a model runs can reach the pod's data. Each worker owns one
sandbox user; a scenario starts from a wiped home with that user's processes killed.

Writes <out>/results.jsonl (verdict, timings, usage) and <out>/traces/<id>.json (full messages,
tool schemas included in the first record). Rerun with the same --out to skip finished cases.
"""
import argparse
import ctypes
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scenarios import SCENARIOS  # noqa: E402


def as_user(user, script, timeout=60):
    """Run bash in the sandbox user's home; returns (exit code, combined output)."""
    try:
        p = subprocess.run(
            ["su", "-", user, "-s", "/bin/bash", "-c", script],
            capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def reset_home(user):
    subprocess.run(["pkill", "-9", "-u", user], capture_output=True)
    home = f"/home/{user}"
    keep = {".ssh"}
    for name in os.listdir(home):
        if name in keep:
            continue
        subprocess.run(["rm", "-rf", "--", os.path.join(home, name)])
    as_user(user, "cp -a /etc/skel/. ~/ 2>/dev/null; true")


def run_case(case, user, args, out):
    reset_home(user)
    setup_rc, setup_out = 0, ""
    if case["setup"]:
        setup_rc, setup_out = as_user(user, case["setup"], timeout=60)
    # A server started by setup needs a moment before the model can reach it.
    time.sleep(0.5)

    os.environ.update({
        "TERMINAL_ENV": "ssh",
        "TERMINAL_SSH_HOST": "127.0.0.1",
        "TERMINAL_SSH_USER": user,
        "TERMINAL_SSH_KEY": args.ssh_key,
        "TERMINAL_CWD": f"/home/{user}",
        "TERMINAL_TIMEOUT": "60",
    })
    from run_agent import AIAgent  # imported late: Hermes reads the environment on import

    started = time.time()
    result, error, agent = None, None, None
    try:
        agent = AIAgent(
            base_url=args.base_url, api_key="local", provider="custom", model=args.model,
            max_iterations=args.max_turns, enabled_toolsets=args.toolsets.split(","),
            max_tokens=args.max_tokens,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            skip_background_review=True, platform="cli", run_budget_seconds=args.case_seconds)
        result = agent.run_conversation(case["prompt"], task_id=f"{case['id']}-{user}")
    except Exception as exc:  # the run is data: a crash is recorded, the batch goes on
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.time() - started

    messages = (result or {}).get("messages") or []
    reply = (result or {}).get("final_response") or ""
    tool_calls = [c["function"]["name"] for m in messages for c in (m.get("tool_calls") or [])]

    state_ok = True
    check_out = ""
    if case["check"]:
        code, check_out = as_user(user, case["check"], timeout=30)
        state_ok = code == 0
    answer_ok = True
    if case["answer"]:
        answer_ok = re.search(case["answer"], reply) is not None
    tool_ok = not (case["no_tool"] and tool_calls)

    rec = dict(
        id=case["id"], kind=case["kind"], passed=bool(state_ok and answer_ok and error is None),
        state_ok=state_ok, answer_ok=answer_ok, tool_ok=tool_ok, seconds=round(seconds, 1),
        api_calls=(result or {}).get("api_calls"), tool_calls=tool_calls,
        input_tokens=(result or {}).get("input_tokens"), output_tokens=(result or {}).get("output_tokens"),
        reply=reply[:2000], check=check_out[:400], error=error, setup_rc=setup_rc, setup_out=setup_out[:400])
    (out / "traces").mkdir(parents=True, exist_ok=True)
    trace = dict(rec, prompt=case["prompt"], messages=messages,
                 system_prompt=getattr(agent, "_cached_system_prompt", None) if agent else None,
                 tools=getattr(agent, "tools", None) if agent else None)
    (out / "traces" / f"{case['id']}.json").write_text(json.dumps(trace, ensure_ascii=False, default=str))
    if agent is not None:
        try:
            agent.close()
        except Exception:
            pass
    reset_home(user)
    return rec


def worker(idx, queue, lock, args, out):
    # A worker left behind by a killed parent would keep wiping this user's home under a new run.
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    user = f"scen{idx + args.first_user}"
    os.environ["HERMES_HOME"] = str(out / "hermes" / user)
    Path(os.environ["HERMES_HOME"]).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, args.hermes)
    os.chdir(args.hermes)
    while True:
        case = queue.get()
        if case is None:
            return
        rec = run_case(case, user, args, out)
        with lock:
            with open(out / "results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{'PASS' if rec['passed'] else 'FAIL'} {rec['id']:<24} {rec['seconds']:>7.1f}s "
              f"calls={rec['api_calls']} tools={len(rec['tool_calls'])} "
              f"{(rec['error'] or '')[:90]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--first-user", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--case-seconds", type=float, default=600)
    ap.add_argument("--toolsets", default="hermes-cli")
    ap.add_argument("--hermes", default="/root/hermes-agent")
    ap.add_argument("--ssh-key", default="/root/.ssh/scen_ed25519")
    ap.add_argument("--only", default="", help="regex over scenario ids")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    results = out / "results.jsonl"
    if results.exists():
        done = {json.loads(line)["id"] for line in results.read_text(encoding="utf-8").splitlines() if line.strip()}
    cases = [c for c in SCENARIOS if c["id"] not in done and (not args.only or re.search(args.only, c["id"]))]
    print(f"{len(cases)} cases, {len(done)} done already, {args.workers} workers", flush=True)

    ctx = mp.get_context("spawn")
    queue, lock = ctx.Queue(), ctx.Lock()
    for c in cases:
        queue.put(c)
    for _ in range(args.workers):
        queue.put(None)
    procs = [ctx.Process(target=worker, args=(i, queue, lock, args, out)) for i in range(args.workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = {}
    for r in rows:
        k = kinds.setdefault(r["kind"], [0, 0])
        k[0] += int(r["passed"])
        k[1] += 1
    print("TOTAL", sum(r["passed"] for r in rows), "/", len(rows))
    for k, (a, b) in sorted(kinds.items()):
        print(f"  {k:<10} {a}/{b}")
    print("SCENARIOS-DONE", flush=True)


if __name__ == "__main__":
    main()
