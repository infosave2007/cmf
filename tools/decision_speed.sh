#!/usr/bin/env bash
# Speed gate of spec decision-v4 §6.6 on the test split of every skill.
#
#   tools/decision_speed.sh MODEL OUTDIR [--no-http] [--skills banking77,clinc150,massive]
#
# For every skill:
#   CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1 cortiq decide MODEL --input test.jsonl \
#       --skill S --bench --out OUTDIR/S.test.rows.jsonl
# (50 warm-up texts, then the whole test split, one text at a time; p50/p95/p99
# of tokenize, encode, hash, resonance and total). The result rows (one JSON
# object per test row, never the text) are the `cortiq decide --input` batch
# that tools/decision_gates.py scores and tools/decision_jev_compat.py compares
# the server with.
#
# Then (unless --no-http) HTTP on loopback with one client: `cortiq serve MODEL`
# (same single-thread environment, fresh state directory, open mode on
# 127.0.0.1, oracle disabled) and one keep-alive connection that POSTs to
# /api/alpha/decisions the Jev request of every test text in order —
# openrouter_bench.request_for(text, question.json) with model
# "cortiq/decision" — after the same 50 warm-up requests; round-trip
# p50/p95/p99 per skill.
#
# Thresholds (§6.6): total p50 <= 25 ms on every skill; resonance p50 <= 1.2 x
# v3 (2.59 / 3.78 / 2.09 ms). The machine (CPU, memory, OS), the load average
# and the busiest processes before and after the run are recorded, because the
# numbers are only meaningful with them (§6.6: publish only with the machine).
#
# Writes OUTDIR/speed.json (and the rows, the decide summaries and the server
# log). Every read of a test split is appended to
# $CMFPUBLIC/artifacts/decision-v4-20260926/test-access.log first.
#
# Environment: CORTIQ (default <repo>/target/release/cortiq), CMFPUBLIC
# (/Users/oleg/dev/cmfpublic), PYTHON (python3), B/C/M (split directories).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMFPUBLIC="${CMFPUBLIC:-/Users/oleg/dev/cmfpublic}"
CORTIQ="${CORTIQ:-$REPO/target/release/cortiq}"
PYTHON="${PYTHON:-python3}"
B="${B:-$CMFPUBLIC/artifacts/decision-v2-20260926/splits/banking77}"
C="${C:-$CMFPUBLIC/artifacts/decision-clinc150-20260925/data}"
M="${M:-$CMFPUBLIC/artifacts/decision-massive-20260926/data}"
ACCESS_LOG="${ACCESS_LOG:-$CMFPUBLIC/artifacts/decision-v4-20260926/test-access.log}"
export CMF_GPU=0

usage() { sed -n '2,36p' "$0"; }
[ $# -ge 2 ] || { usage >&2; exit 2; }
MODEL="$1"; OUTDIR="$2"; shift 2
HTTP=1
SKILLS_CSV="banking77,clinc150,massive"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-http) HTTP=0; shift ;;
        --skills) SKILLS_CSV="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
IFS=, read -r -a SKILLS <<<"$SKILLS_CSV"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
skill_dir() {
    case "$1" in
        banking77) echo "$B" ;;
        clinc150) echo "$C" ;;
        massive) echo "$M" ;;
        *) die "unknown skill $1" ;;
    esac
}
log_access() {
    "$PYTHON" - "$ACCESS_LOG" "$1" "$2" <<'PY'
import datetime, json, sys
log, path, purpose = sys.argv[1:]
line = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "file": path, "purpose": purpose}
with open(log, "a", encoding="utf-8") as f:
    f.write(json.dumps(line, ensure_ascii=False) + "\n")
PY
}
snapshot() {
    # load average and the five busiest processes (command names only)
    "$PYTHON" - <<'PY'
import json, os, subprocess
ps = subprocess.run(["ps", "-A", "-o", "%cpu=,comm="], capture_output=True, text=True).stdout.splitlines()
rows = []
for l in ps:
    l = l.strip()
    if not l:
        continue
    cpu, _, comm = l.partition(" ")
    try:
        rows.append((float(cpu), os.path.basename(comm.strip())))
    except ValueError:
        pass
rows.sort(reverse=True)
print(json.dumps({"loadavg": list(os.getloadavg()), "top_cpu": [{"cpu_pct": c, "command": n} for c, n in rows[:5]]}))
PY
}

[ -f "$MODEL" ] || die "no model $MODEL"
[ -x "$CORTIQ" ] || die "no cortiq binary at $CORTIQ"
mkdir -p "$OUTDIR"
for s in "${SKILLS[@]}"; do
    [ ! -e "$OUTDIR/$s.test.rows.jsonl" ] || die "$OUTDIR/$s.test.rows.jsonl exists (use a new OUTDIR)"
done
snapshot >"$OUTDIR/machine-before.json"

# ------------------------------------------------------------------ cortiq decide --bench
for s in "${SKILLS[@]}"; do
    test_file="$(skill_dir "$s")/test.jsonl"
    log_access "$test_file" "decision-v4 WP9 §6.6 speed + §6.2 test gates: CMF_THREADS=1 cortiq decide --input --bench (skill $s; rows reused by decision_gates.py and decision_jev_compat.py)"
    log "decide --bench $s"
    env CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "$CORTIQ" decide "$MODEL" --input "$test_file" --skill "$s" \
        --bench --out "$OUTDIR/$s.test.rows.jsonl" 2>"$OUTDIR/$s.test.stderr" \
        || { tail -20 "$OUTDIR/$s.test.stderr" >&2; die "cortiq decide failed on $s"; }
    grep '^{"summary"' "$OUTDIR/$s.test.stderr" >"$OUTDIR/$s.test.summary.json" \
        || die "no summary line in $OUTDIR/$s.test.stderr"
done

# ------------------------------------------------------------------ HTTP, one client
if [ "$HTTP" = 1 ]; then
    PORT="$("$PYTHON" -c '
import socket
while True:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    if p not in (8788, 8791, 8792, 8080):
        print(p); break')"
    STATE="$OUTDIR/http-state"
    [ ! -e "$STATE" ] || die "$STATE exists"
    log "cortiq serve on 127.0.0.1:$PORT (CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1)"
    env CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "$CORTIQ" serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --state "$STATE" >"$OUTDIR/http-server.log" 2>&1 &
    SERVER_PID=$!
    trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then break; fi
        kill -0 "$SERVER_PID" 2>/dev/null || { tail -20 "$OUTDIR/http-server.log" >&2; die "server exited"; }
        sleep 0.5
    done
    curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || die "server not healthy on port $PORT"
    for s in "${SKILLS[@]}"; do
        d="$(skill_dir "$s")"
        log_access "$d/test.jsonl" "decision-v4 WP9 §6.6 speed: HTTP loopback latency, one client, Jev request of every test text (skill $s)"
        log "HTTP latency $s"
        "$PYTHON" - "$PORT" "$d/test.jsonl" "$d/question.json" "$OUTDIR/$s.test.rows.jsonl" "$OUTDIR/$s.http.json" <<'PY'
import hashlib, http.client, json, sys, time
port, test_file, question_file, rows_file, out = sys.argv[1:]
WARMUP = 50
question = json.load(open(question_file, encoding="utf-8"))
texts = [json.loads(l)["text"] for l in open(test_file, encoding="utf-8") if l.strip()]
decided = [json.loads(l) for l in open(rows_file, encoding="utf-8") if l.strip()]
assert len(decided) == len(texts)
def body(text):
    # openrouter_bench.request_for(text, question) with the model renamed
    return json.dumps({"model": "cortiq/decision", "state": text,
                       "questions": {"task": {"type": "choice", **question}}}, ensure_ascii=False).encode()
conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=60)
def post(b):
    t0 = time.perf_counter_ns()
    conn.request("POST", "/api/alpha/decisions", b, {"Content-Type": "application/json"})
    r = conn.getresponse()
    raw = r.read()
    return r.status, raw, (time.perf_counter_ns() - t0) / 1000.0
for t in texts[:WARMUP]:
    st, raw, _ = post(body(t))
    assert st == 200, (st, raw[:300])
lat, statuses, same_choice = [], {}, 0
for t, row in zip(texts, decided):
    st, raw, us = post(body(t))
    statuses[st] = statuses.get(st, 0) + 1
    lat.append(us)
    if st == 200:
        a = json.loads(raw)["answers"]["task"]
        same_choice += a.get("choice") == row["choice"]
conn.close()
def pct(v, q):
    v = sorted(v); r = q / 100 * (len(v) - 1); lo = int(r); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (r - lo)
res = {"n": len(lat), "warmup": WARMUP, "status": {str(k): v for k, v in sorted(statuses.items())},
       "choice_equal_to_decide": same_choice,
       "round_trip_ms": {"p50": pct(lat, 50) / 1000, "p95": pct(lat, 95) / 1000, "p99": pct(lat, 99) / 1000,
                         "mean": sum(lat) / len(lat) / 1000},
       "request": "POST /api/alpha/decisions, openrouter_bench.request_for(text, question.json) with model cortiq/decision, one keep-alive connection, sequential",
       "body_bytes_mean": sum(len(body(t)) for t in texts) / len(texts)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({"skill_http": out, "p50_ms": res["round_trip_ms"]["p50"], "status": res["status"]}))
PY
    done
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    trap - EXIT
    rm -rf "$STATE"
fi
snapshot >"$OUTDIR/machine-after.json"

# ------------------------------------------------------------------ speed.json
"$PYTHON" - "$OUTDIR" "$MODEL" "$CORTIQ" "$HTTP" "${SKILLS[@]}" <<'PY'
import datetime, hashlib, json, os, platform, subprocess, sys
outdir, model, cortiq, http = sys.argv[1:5]
skills = sys.argv[5:]
V3_RESONANCE_P50_MS = {"banking77": 2.59, "clinc150": 3.78, "massive": 2.09}   # spec §6.6
TOTAL_P50_MAX_MS = 25.0
RESONANCE_RATIO_MAX = 1.2
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()
def sysctl(k):
    try:
        return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None
res = {"schema": "cortiq-decision-v4-speed/1",
       "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
       "command": "CMF_THREADS=1 VECLIB_MAXIMUM_THREADS=1 cortiq decide MODEL --input test.jsonl --skill S --bench",
       "model": {"path": model, "sha256": sha(model), "bytes": os.path.getsize(model)},
       "binary": {"path": cortiq, "sha256": sha(cortiq),
                  "version": subprocess.run([cortiq, "--version"], capture_output=True, text=True).stdout.strip()},
       "machine": {"cpu": sysctl("machdep.cpu.brand_string"), "memsize_bytes": int(sysctl("hw.memsize") or 0),
                   "ncpu": int(sysctl("hw.ncpu") or 0), "os": platform.platform()},
       "env": {"CMF_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "CMF_GPU": "0"},
       "before": json.load(open(os.path.join(outdir, "machine-before.json"))),
       "after": json.load(open(os.path.join(outdir, "machine-after.json"))),
       "thresholds": {"total_p50_ms_max": TOTAL_P50_MAX_MS, "resonance_p50_max_ratio_to_v3": RESONANCE_RATIO_MAX,
                      "v3_resonance_p50_ms": V3_RESONANCE_P50_MS},
       "skills": {}}
ok = True
for s in skills:
    summ = json.loads(open(os.path.join(outdir, f"{s}.test.summary.json")).read())["summary"]
    bench = summ["bench"]
    stages = {k: {q: bench[k][q + "_us"] / 1000.0 for q in ("p50", "p95", "p99", "mean")}
              for k in ("tokenize", "encode", "hash", "resonance", "total")}
    v3 = V3_RESONANCE_P50_MS[s]
    e = {"n": summ["n"], "warmup": bench["warmup"], "model_sha": summ["model_sha"], "stages_ms": stages,
         "total_p50_ms": stages["total"]["p50"], "resonance_p50_ms": stages["resonance"]["p50"],
         "resonance_v3_p50_ms": v3, "resonance_ratio_to_v3": stages["resonance"]["p50"] / v3,
         "pass_total": stages["total"]["p50"] <= TOTAL_P50_MAX_MS,
         "pass_resonance": stages["resonance"]["p50"] <= RESONANCE_RATIO_MAX * v3}
    hp = os.path.join(outdir, f"{s}.http.json")
    if http == "1" and os.path.exists(hp):
        e["http"] = json.load(open(hp))
        e["pass_http_status"] = e["http"]["status"] == {"200": e["http"]["n"]}
    e["pass"] = e["pass_total"] and e["pass_resonance"] and e.get("pass_http_status", True)
    ok &= e["pass"]
    res["skills"][s] = e
res["pass"] = ok
json.dump(res, open(os.path.join(outdir, "speed.json"), "w"), indent=1)
print(json.dumps({"speed": os.path.join(outdir, "speed.json"), "pass": ok,
                  "total_p50_ms": {s: round(res["skills"][s]["total_p50_ms"], 3) for s in skills},
                  "resonance_p50_ms": {s: round(res["skills"][s]["resonance_p50_ms"], 3) for s in skills}}))
PY
log "done"
