#!/usr/bin/env python3
"""Jev request compatibility gate of spec decision-v4 §6.3 (local only, no paid call).

    python3 tools/decision_jev_compat.py --model MODEL --decide-rows DIR --out OUT.json
        [--cortiq BIN] [--datasets banking77,clinc150,massive] [--work DIR]

Per dataset (BANKING77 3080, CLINC150 4500, MASSIVE 2974 test rows):

* **Rebuilt bodies.** For every row of the stored Jev ledger the request body is
  rebuilt from the test text and the rubric in the ledger's run record exactly as
  `openrouter_bench.request_for` sent it:
  `json.dumps({"model": "typesafe/jev-1.13", "state": text, "questions": {"task":
  {"type": "choice", **question}}}, ensure_ascii=False).encode()`; its sha256 must
  equal the row's `request_sha256` (100 %).
* **Sent to the local server.** One replacement `"model": "typesafe/jev-1.13"` ->
  `"model": "cortiq/decision"`, POST /api/alpha/decisions to `cortiq serve MODEL`
  (127.0.0.1, fresh state directory, open mode, oracle disabled), one keep-alive
  connection, in ledger order. Required on 100 % of the rows: HTTP 200; the
  response passes the port of `openrouter_bench.validate_oracle_response`
  (`:155-199`, the validator Jev's answers passed; only the model pattern is
  ours: `cortiq/decision@<12 hex>`); `probabilities` has exactly the criteria
  keys in request order and sums to 1 within 1e-5; `confidence` equals
  `(N·p_max − 1)/(N − 1)` (f32, clamped at 0) within 1e-6; `choice` is the
  argmax; the answer is bit-equal to the `cortiq decide --input` row of the same
  text (choice, p_top, confidence, novelty, margin, accepted, certified, the five
  smallest errors) from `DIR/{ds}.test.rows.jsonl` (tools/decision_speed.sh).
  The server's accuracy must equal the decide batch's (which decision_gates.py
  holds against the §6.2 table).
* Jev on the same rows (the stored answers, `choice == truth`) is reported.

Multi-type request `reports/decision-system-20260925/protocol-request.json`
(choice `team` with labels no skill has, score `urgency`, noul `refund`). The
file carries the 0.7.x-era names `"model": "cortiq/decision-v1"` and
`cmf: {allow_oracle, approved_for_external}`, which the v4 protocol refuses
(404 / 400), so they are mapped to `"model": "cortiq/decision"` and
`cmf: {"oracle": true}`; state and questions are sent verbatim.
* Oracle disabled (the server above): 422 `UNSUPPORTED_QUESTION` with a reason
  for each of the three questions.
* Oracle enabled against a local mock of OpenRouter (a thread in this process,
  a dummy key in a dedicated environment variable; nothing leaves the machine):
  200, one oracle call, all three answers valid (choice in the options, score an
  integer level with its legend, noul 0/1), and the body the mock received is
  canonical JSON with the schema of the three questions.

Every read of a test split or of a Jev test ledger is appended to
artifacts/decision-v4-20260926/test-access.log first. The server child gets an
environment without any variable whose name mentions a key, token or
OpenRouter, so no real key can reach it.
"""

import argparse
import datetime
import hashlib
import http.client
import http.server
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

CMFPUBLIC = os.environ.get("CMFPUBLIC", "/Users/oleg/dev/cmfpublic")
ACCESS_LOG = os.path.join(CMFPUBLIC, "artifacts", "decision-v4-20260926", "test-access.log")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = {
    "banking77": {
        "split_dir": os.path.join(CMFPUBLIC, "artifacts/decision-v2-20260926/splits/banking77"),
        "jev": os.path.join(CMFPUBLIC, "reports/decision-banking77-20260925/jev-primary-combined.jsonl"),
    },
    "clinc150": {
        "split_dir": os.path.join(CMFPUBLIC, "artifacts/decision-clinc150-20260925/data"),
        "jev": os.path.join(CMFPUBLIC, "reports/decision-clinc150-20260925/jev-test.jsonl"),
    },
    "massive": {
        "split_dir": os.path.join(CMFPUBLIC, "artifacts/decision-massive-20260926/data"),
        "jev": os.path.join(CMFPUBLIC, "reports/decision-massive-20260926/test-run/jev-test.jsonl"),
    },
}
PROTOCOL_REQUEST = os.path.join(CMFPUBLIC, "reports/decision-system-20260925/protocol-request.json")
JEV_MODEL_FIELD = b'"model": "typesafe/jev-1.13"'
OUR_MODEL_FIELD = b'"model": "cortiq/decision"'
OUR_MODEL_RE = r"cortiq/decision@[0-9a-f]{12}"
MOCK_KEY_ENV = "CORTIQ_WP9_MOCK_ORACLE_KEY"
MOCK_KEY = "mock-oracle-key-wp9-not-a-secret"
ORACLE_MODEL = "deepseek/deepseek-v4.1-flash"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log_access(path, purpose):
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "file": path, "purpose": purpose}, ensure_ascii=False) + "\n")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def body_for(model, text, question):
    """tools/cortiq-decision/openrouter_bench.py request_for (and evaluate_v3/cost.py body_for)."""
    return json.dumps({"model": model, "state": text, "questions": {"task": {"type": "choice", **question}}},
                      ensure_ascii=False).encode()


def percentile(v, q):
    v = sorted(v)
    if not v:
        return None
    r = q / 100 * (len(v) - 1)
    lo = int(math.floor(r))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (r - lo)


# ---------------------------------------------------------------------- validator port

def _probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_decisions_response(result, questions, model_re=OUR_MODEL_RE):
    """Port of openrouter_bench.validate_oracle_response (:155-199): the same checks,
    with the model pattern of this server instead of Jev's. Answers the server got
    from its oracle or cache carry no probabilities (they are optional in the
    OpenRouter schema, spec §4.7); for those the verdict itself is checked: choice
    in the options, score an integer level with the legend, noul 0 or 1."""
    if not isinstance(result, dict) or not re.fullmatch(model_re, str(result.get("model", ""))):
        raise ValueError("unexpected model")
    answers = result.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("question IDs mismatch")
    actions = {q: ((result.get("cmf") or {}).get("questions") or {}).get(q, {}).get("action")
               for q in questions}
    for qid, q in questions.items():
        a = answers[qid]
        if not isinstance(a, dict) or a.get("type") != q["type"]:
            raise ValueError("answer type mismatch")
        verdict_only = actions[qid] in ("oracle", "cache") and "probabilities" not in a
        if q["type"] == "noul":
            if not _probability(a.get("noul")):
                raise ValueError("invalid noul")
            if verdict_only and a["noul"] not in (0, 1):
                raise ValueError("oracle noul is not 0/1")
            continue
        expected = set(q["criteria"]) if q["type"] == "choice" else {str(i) for i in range(len(q["criteria"]))}
        if verdict_only:
            if q["type"] == "choice":
                if a.get("choice") not in expected:
                    raise ValueError("invalid oracle choice")
            else:
                s = a.get("score")
                if type(s) is not int or not 0 <= s <= len(q["criteria"]) - 1:
                    raise ValueError("invalid oracle score")
                legend = a.get("legend")
                if not isinstance(legend, dict) or set(legend) != expected:
                    raise ValueError("invalid legend")
            continue
        p = a.get("probabilities")
        if not isinstance(p, dict) or set(p) != expected or not all(_probability(v) for v in p.values()):
            raise ValueError("invalid probabilities")
        quantized = all(abs(v * 100 - round(v * 100)) < 1e-7 for v in p.values())
        tolerance = .005 * len(p) + 1e-8 if quantized else .01000001
        if sum(p.values()) <= 0 or abs(sum(p.values()) - 1) > tolerance or not _probability(a.get("confidence")):
            raise ValueError("invalid distribution/confidence")
        if q["type"] == "choice":
            if a.get("choice") not in p or p[a["choice"]] + 1e-8 < max(p.values()):
                raise ValueError("invalid choice")
        else:
            score = a.get("score")
            n = len(p)
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= n - 1:
                raise ValueError("invalid score")
            legend = a.get("legend")
            if not isinstance(legend, dict) or set(legend) != expected:
                raise ValueError("invalid legend")
            weighted = sum(int(k) * v for k, v in p.items())
            if abs(score - weighted) > .005 * sum(range(n)) + .005 + 1e-7:
                raise ValueError("score inconsistent with distribution")
    usage = result.get("usage", {})
    cost = usage.get("cost")
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("invalid cost")
    if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ["input_tokens", "output_tokens"]):
        raise ValueError("invalid token usage")
    return result


def jev_confidence_f32(p_max, n):
    p = np.float32(p_max)
    if n <= 1:
        return float(p)
    c = (np.float32(n) * p - np.float32(1.0)) / (np.float32(n) - np.float32(1.0))
    return float(max(np.float32(0.0), c))


def strict_checks(answer, criteria):
    """§6.3: keys = criteria (request order), sum within 1e-5 of 1, confidence formula
    within 1e-6, choice = argmax. Returns a list of failed check names."""
    bad = []
    p = answer.get("probabilities") or {}
    if list(p) != list(criteria):
        bad.append("probabilities_keys")
    if abs(sum(p.values()) - 1.0) > 1e-5:
        bad.append("probability_sum")
    ch = answer.get("choice")
    if ch not in p or p[ch] != max(p.values()):
        bad.append("choice_not_argmax")
    else:
        if abs(answer.get("confidence", -1) - jev_confidence_f32(p[ch], len(criteria))) > 1e-6:
            bad.append("confidence_formula")
    return bad


def compare_with_decide(resp, row):
    """Bit-equality of the server's answer with the `cortiq decide --input` row (both
    print f32 values in their shortest form, so equal floats are equal bits)."""
    a = resp["answers"]["task"]
    q = resp["cmf"]["questions"]["task"]
    g = q.get("gate") or {}
    diffs = []
    choice = a.get("choice")
    if choice != row["choice"]:
        diffs.append("choice")
    p = a.get("probabilities") or {}
    if p.get(choice) != row["p_top"] or g.get("p_top") != row["p_top"]:
        diffs.append("p_top")
    want_conf = max(0.0, row["confidence"])
    if a.get("confidence") != want_conf:
        diffs.append("confidence")
    for k in ("novelty", "margin", "accepted"):
        if g.get(k) != row[k]:
            diffs.append(k)
    if q.get("certified") != row["certified"]:
        diffs.append("certified")
    if q.get("action") != ("local" if row["accepted"] else "abstain"):
        diffs.append("action")
    errs = q.get("errors") or {}
    if list(errs.items()) != list(row["errors_top5"].items()):
        diffs.append("errors_top5")
    return diffs


# ---------------------------------------------------------------------- server

def free_port():
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        if p not in (8080, 8788, 8791, 8792):
            return p


def child_env(extra=None):
    """The environment of the server child: no variable whose name suggests a key or
    token (values are never read), single-thread numerics as in §6.6, no GPU."""
    env = {k: v for k, v in os.environ.items()
           if not re.search(r"KEY|TOKEN|SECRET|OPENROUTER|PASSWORD", k, re.I)}
    env.update({"CMF_GPU": "0"})
    env.update(extra or {})
    return env


class Server:
    def __init__(self, cortiq, model, workdir, name, config=None, extra_env=None):
        self.port = free_port()
        self.state = os.path.join(workdir, f"state-{name}")
        self.log_path = os.path.join(workdir, f"server-{name}.log")
        args = [cortiq, "serve", model, "--host", "127.0.0.1", "--port", str(self.port), "--state", self.state]
        if config is not None:
            cfg_path = os.path.join(workdir, f"config-{name}.json")
            with open(cfg_path, "w") as f:
                json.dump(config, f, indent=1)
            args += ["--decision-config", cfg_path]
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, env=child_env(extra_env))
        deadline = time.time() + 90
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {name} exited: " + open(self.log_path).read()[-2000:])
            try:
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
                c.request("GET", "/healthz")
                r = c.getresponse()
                r.read()
                c.close()
                if r.status == 200:
                    break
            except OSError:
                pass
            time.sleep(0.3)
        else:
            raise RuntimeError(f"server {name} not healthy")
        self.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)

    def post(self, path, body):
        t0 = time.perf_counter_ns()
        self.conn.request("POST", path, body, {"Content-Type": "application/json"})
        r = self.conn.getresponse()
        raw = r.read()
        return r.status, raw, (time.perf_counter_ns() - t0) / 1e6

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


# ---------------------------------------------------------------------- mock oracle

class MockOracle:
    """OpenRouter chat/completions on 127.0.0.1: answers the json_schema of the request
    (enum -> "billing" when offered else the first option, integer -> 1 when allowed,
    boolean -> true) with a finite cost; records every request."""

    def __init__(self):
        self.calls = []
        mock = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                rec = {"path": self.path, "auth_ok": self.headers.get("Authorization") == "Bearer " + MOCK_KEY,
                       "title": self.headers.get("X-Title"), "sha256": sha256_bytes(raw), "bytes": len(raw)}
                try:
                    body = json.loads(raw)
                    rec["canonical"] = canonical(body).encode() == raw
                    schema = body["response_format"]["json_schema"]["schema"]
                    rec["model"] = body.get("model")
                    rec["qids"] = list(schema["required"])
                    rec["schema"] = schema["properties"]
                    verdict = {}
                    for q in schema["required"]:
                        t = schema["properties"][q]
                        if "enum" in t:
                            verdict[q] = "billing" if "billing" in t["enum"] else t["enum"][0]
                        elif t.get("type") == "integer":
                            verdict[q] = min(1, t.get("maximum", 1))
                        else:
                            verdict[q] = True
                    rec["verdict"] = verdict
                    out = {"id": "gen-mock-wp9", "object": "chat.completion", "model": body.get("model"),
                           "provider": "Mock",
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": json.dumps(verdict)}}],
                           "usage": {"prompt_tokens": 321, "completion_tokens": 9, "total_tokens": 330,
                                     "cost": 1.5e-06}}
                    status, payload = 200, json.dumps(out).encode()
                except Exception as e:  # pragma: no cover - reported, never silent
                    rec["error"] = repr(e)
                    status, payload = 400, b'{"error":{"code":400,"message":"mock could not parse"}}'
                mock.calls.append(rec)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------------- per dataset

def read_ledger(path):
    run, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("record_type") == "run" and run is None:
                run = r
            elif r.get("record_type") == "result" or "text_sha256" in r:
                rows.append(r)
    return run, rows


def run_dataset(ds, srv, rows_dir):
    info = DATASETS[ds]
    test_file = os.path.join(info["split_dir"], "test.jsonl")
    log_access(test_file, f"decision-v4 WP9 §6.3 Jev compatibility: rebuild the Jev bodies from the test texts and send them to the local server ({ds})")
    log_access(info["jev"], f"decision-v4 WP9 §6.3 Jev compatibility: rubric, request_sha256 and stored answers of the Jev test ledger ({ds})")
    test_rows = [json.loads(l) for l in open(test_file, encoding="utf-8") if l.strip()]
    by_sha = {}
    for r in test_rows:
        by_sha.setdefault(sha256_bytes(r["text"].encode()), r)
    run, ledger = read_ledger(info["jev"])
    q = run["question"]
    if isinstance(q, dict) and "task" in q:
        q = {k: v for k, v in q["task"].items() if k != "type"}
    model = run["requested_model"]
    criteria = q["criteria"]
    decided = {}
    for l in open(os.path.join(rows_dir, f"{ds}.test.rows.jsonl"), encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            decided.setdefault(r["text_sha256"], r)
    res = {"test_file": test_file, "test_sha256": sha256_file(test_file), "test_rows": len(test_rows),
           "jev_ledger": info["jev"], "jev_ledger_sha256": sha256_file(info["jev"]), "ledger_rows": len(ledger),
           "labels": len(criteria), "requested_model": model,
           "sha_reproduced": 0, "missing_text": 0, "http_200": 0, "status": {}, "validator_pass": 0,
           "strict_pass": 0, "strict_failures": {}, "equal_to_decide": 0, "decide_diffs": {},
           "server_correct": 0, "decide_correct": 0, "jev_correct": 0, "jev_answered": 0,
           "local": 0, "abstain": 0, "examples_failed": []}
    lat = []
    for r in ledger:
        t = by_sha.get(r["text_sha256"])
        if t is None:
            res["missing_text"] += 1
            continue
        body = body_for(model, t["text"], q)
        res["sha_reproduced"] += sha256_bytes(body) == r.get("request_sha256")
        if not body.startswith(b"{" + JEV_MODEL_FIELD + b", "):
            raise SystemExit(f"{ds}: the rebuilt body does not start with the Jev model field")
        sent = body.replace(JEV_MODEL_FIELD, OUR_MODEL_FIELD, 1)
        status, raw, ms = srv.post("/api/alpha/decisions", sent)
        lat.append(ms)
        res["status"][str(status)] = res["status"].get(str(status), 0) + 1
        truth = r.get("truth", t.get("label"))
        a = r.get("answer") or {}
        if a.get("choice") is not None:
            res["jev_answered"] += 1
        res["jev_correct"] += a.get("choice") == truth
        row = decided.get(r["text_sha256"])
        if row is None:
            raise SystemExit(f"{ds}: no decide row for text {r['text_sha256'][:12]}")
        res["decide_correct"] += row["choice"] == truth
        if status != 200:
            if len(res["examples_failed"]) < 5:
                res["examples_failed"].append({"text_sha256": r["text_sha256"], "status": status, "body": raw[:300].decode("utf-8", "replace")})
            continue
        res["http_200"] += 1
        resp = json.loads(raw)
        try:
            validate_decisions_response(resp, {"task": {"type": "choice", "criteria": criteria}})
            res["validator_pass"] += 1
        except ValueError as e:
            if len(res["examples_failed"]) < 5:
                res["examples_failed"].append({"text_sha256": r["text_sha256"], "validator": str(e)})
        bad = strict_checks(resp["answers"]["task"], criteria)
        if not bad:
            res["strict_pass"] += 1
        for b in bad:
            res["strict_failures"][b] = res["strict_failures"].get(b, 0) + 1
        diffs = compare_with_decide(resp, row)
        if not diffs:
            res["equal_to_decide"] += 1
        for d in diffs:
            res["decide_diffs"][d] = res["decide_diffs"].get(d, 0) + 1
        res["server_correct"] += resp["answers"]["task"].get("choice") == truth
        act = resp["cmf"]["questions"]["task"].get("action")
        res[act if act in ("local", "abstain") else "other_action"] = res.get(act if act in ("local", "abstain") else "other_action", 0) + 1
    n = len(ledger)
    res["n"] = n
    res["round_trip_ms"] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95), "p99": percentile(lat, 99)}
    res["checks"] = {
        "request_sha256_100pct": res["sha_reproduced"] == n and res["missing_text"] == 0,
        "http_200_100pct": res["http_200"] == n,
        "validator_100pct": res["validator_pass"] == n,
        "strict_100pct": res["strict_pass"] == n,
        "equal_to_decide_100pct": res["equal_to_decide"] == n,
        "accuracy_equal_to_decide": res["server_correct"] == res["decide_correct"],
        "rows_equal_test_split": n == len(test_rows),
    }
    res["pass"] = all(res["checks"].values())
    return res


def run_multitype(srv_disabled, cortiq, model, workdir):
    raw = open(PROTOCOL_REQUEST, "rb").read()
    req = json.loads(raw)
    mapped = dict(req)
    mapped["model"] = "cortiq/decision"
    mapped["cmf"] = {"oracle": True}
    body = json.dumps(mapped, ensure_ascii=False).encode()
    qs = req["questions"]
    out = {"file": PROTOCOL_REQUEST, "sha256": sha256_bytes(raw),
           "mapping": {"model": [req.get("model"), "cortiq/decision"], "cmf": [req.get("cmf"), {"oracle": True}]}}
    # 1. oracle disabled -> 422 with a reason per question
    st, rb, _ = srv_disabled.post("/v1/decisions", body)
    r1 = json.loads(rb)
    meta = (r1.get("error") or {}).get("metadata") or {}
    details = ((meta.get("details") or {}).get("questions")) or {}
    out["without_oracle"] = {"status": st, "reason": meta.get("reason"),
                             "questions": {q: details.get(q) for q in qs}}
    ok1 = st == 422 and meta.get("reason") == "UNSUPPORTED_QUESTION" and all(
        isinstance(details.get(q), dict) and details[q].get("reason") for q in qs)
    out["without_oracle"]["pass"] = ok1
    # 2. oracle enabled against the local mock -> 200, three valid answers
    mock = MockOracle()
    srv = None
    try:
        cfg = {"oracle": {"enabled": True, "base_url": f"http://127.0.0.1:{mock.port}", "api_key_env": MOCK_KEY_ENV,
                          "model": ORACLE_MODEL, "budget_usd": 0.01, "max_calls": 5},
               "learning": {"enabled": False}}
        srv = Server(cortiq, model, workdir, "mock-oracle", config=cfg, extra_env={MOCK_KEY_ENV: MOCK_KEY})
        st2, rb2, _ = srv.post("/v1/decisions", body)
    finally:
        if srv is not None:
            srv.close()
        mock.close()
    r2 = json.loads(rb2)
    val_err = None
    try:
        validate_decisions_response(r2, qs)
    except ValueError as e:
        val_err = str(e)
    ans = r2.get("answers") or {}
    actions = {q: ((r2.get("cmf") or {}).get("questions") or {}).get(q, {}).get("action") for q in qs}
    call = mock.calls[0] if mock.calls else {}
    out["with_mock_oracle"] = {
        "status": st2, "answers": ans, "actions": actions, "usage": r2.get("usage"), "validator_error": val_err,
        "mock_calls": len(mock.calls),
        "mock_request": {k: call.get(k) for k in ("path", "auth_ok", "title", "canonical", "model", "qids", "sha256", "bytes")},
    }
    ok2 = (st2 == 200 and val_err is None and len(mock.calls) == 1 and call.get("auth_ok") and call.get("canonical")
           and call.get("path") == "/chat/completions" and call.get("qids") == list(qs)
           and all(a == "oracle" for a in actions.values())
           and ans.get("team", {}).get("choice") in qs["team"]["criteria"]
           and ans.get("urgency", {}).get("score") in range(len(qs["urgency"]["criteria"]))
           and ans.get("refund", {}).get("noul") in (0, 1))
    out["with_mock_oracle"]["pass"] = bool(ok2)
    out["pass"] = bool(ok1 and ok2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--decide-rows", required=True, help="directory with {ds}.test.rows.jsonl (tools/decision_speed.sh)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cortiq", default=os.path.join(REPO, "target", "release", "cortiq"))
    ap.add_argument("--datasets", default="banking77,clinc150,massive")
    ap.add_argument("--work", default=None, help="state directories and server logs (default: a temp dir, removed)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    work = a.work or tempfile.mkdtemp(prefix="cortiq-jev-compat.")
    os.makedirs(work, exist_ok=True)
    try:
        return run_all(a, work)
    finally:
        # The temp dir goes whatever happens (a failure included); a --work
        # directory is the caller's and stays.
        if not a.work:
            shutil.rmtree(work, ignore_errors=True)


def run_all(a, work):
    t0 = time.time()
    doc = {"schema": "cortiq-decision-v4-jev-compat/1", "utc": utc_now(),
           "model": {"path": a.model, "sha256": sha256_file(a.model)},
           "binary": {"path": a.cortiq, "sha256": sha256_file(a.cortiq)},
           "server": "cortiq serve MODEL --host 127.0.0.1 (open mode, oracle disabled, fresh state)",
           "replacement": [JEV_MODEL_FIELD.decode(), OUR_MODEL_FIELD.decode()],
           "datasets": {}}
    srv = Server(a.cortiq, a.model, work, "jev")
    try:
        for ds in a.datasets.split(","):
            print(f"[jev-compat] {ds}", file=sys.stderr, flush=True)
            doc["datasets"][ds] = run_dataset(ds, srv, a.decide_rows)
            d = doc["datasets"][ds]
            print(json.dumps({ds: {k: d[k] for k in ("n", "sha_reproduced", "http_200", "validator_pass", "strict_pass",
                                                     "equal_to_decide", "server_correct", "decide_correct", "pass")}}),
                  file=sys.stderr, flush=True)
        doc["multi_type"] = run_multitype(srv, a.cortiq, a.model, work)
    finally:
        srv.close()
    doc["seconds"] = round(time.time() - t0, 3)
    doc["pass"] = all(d["pass"] for d in doc["datasets"].values()) and doc["multi_type"]["pass"]
    tmp = a.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
    os.replace(tmp, a.out)
    print(json.dumps({"jev_compat": a.out, "pass": doc["pass"]}))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
