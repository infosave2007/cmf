#!/usr/bin/env python3
"""Local acceptance gates of spec decision-v4 §6 for the release artifacts
(tools/decision_build_release.sh), written to gates.json and gates-release.json.

    python3 tools/decision_gates.py [--base BASE.cmf] [--release RELEASE.cmf]
        [--out-dir DIR] [--work DIR] [--cortiq BIN] [--no-release] [--no-http]

Defaults: BASE = $ART/cortiq-decision-base.cmf, RELEASE = $ART/release/cortiq-decision.cmf,
DIR = $ART, with ART = $CMFPUBLIC/artifacts/decision-v4-20260926 (CMFPUBLIC: the
directory holding artifacts/ and reports/, default the repository root). ENC_SRC (the
directory with encoder.onnx and encoder_tokenizer/, required for E1–E5) has no default.
Nothing touches the network: the oracle answers come from the stored ledgers only.

gates.json — the reproduction model (§3.8: train only, K = 16):

* **F1–F3** (§6.2, the fitter and the certifier on the stored v3 Python features):
  `cargo test --release -p cortiq-decision --test parity_v3 -- --ignored`; F1 dev
  1396/2888/1755 with 0 flips; F2 principal angle <= 1e-6 rad and relative E <= 1e-6;
  F3 T within 1 f32 ulp, θ within 2 ulp, τ equal, odd half 649/635, 1414/1406, 727/710,
  and the Clopper–Pearson bound of every 3×14 grid row against scipy within 1e-9.
* **E1–E5** (§1.6): a fresh encoder export, HF `tokenizers` 0.22.2 ids and the Unicode
  probe (tools/decision_encoder_parity.py), then
  `cargo test --release -p cortiq-decision --test encoder_real -- --ignored --test-threads=1`.
* **The §6.2 table** on the built file, row by row with its tolerance: dev (count and
  winner flips against the v3 PH runtime, recomputed here from the shipped v3
  topologies and the stored v3 signal), calibration, T, θ, τ, odd half and its bound,
  then test (all rows, flips against the stored v3 test decisions), the certified gate,
  CLINC150 OOS and the static cascade (the certified gate, then the stored DeepSeek
  answers of `reports/decision-v4-20260926/oracle/{ds}.part*.jsonl` on the rows it
  rejects; a rejected row without a stored answer keeps its local top-1 and is counted
  as `missing`, no call is made). Jev on the same rows, Wilson intervals and exact
  McNemar tests are recorded next to every accuracy.
* **§6.3** tools/decision_jev_compat.py, **§6.6** tools/decision_speed.sh, **§6.7** size
  <= SIZE_LIMIT (320,000,000 B, see below), the two-build sha256 equality of
  build/build-release.json.

Order and stop conditions (§6.10; they halt the run and are reported, never worked
around: τ ≠ v3, any §6.2 tolerance violated, E1 < 100 %): F1–F3 (the stored v3
train/dev/calibration features, no test split), then the part of the §6.2 table that
needs no test split (dev, calibration, T, θ, τ, odd half) — a stop there opens no test
data —, then E1–E5 (whose reference ids, E2 and E4 read the test splits), then the
test rows. A gate passes only when its cargo test exited 0 and its JSON passed.
Isolation violations and the oracle budget belong to the cascade evaluation (§5c),
which spends nothing here.

Provenance: gates.json records the commit, whether the worktree had tracked changes,
the untracked files under tools/ and crates/ and the sha256 of every tool of this
chain, so the numbers can be tied to the exact scripts that produced them.

gates-release.json — the published model (§3.9: train ∪ dev, K per skill from
max-recipe/cv.json): its gate and calibration, test (all rows, certified gate), CLINC150
OOS and the static cascade from the stored ledgers, each next to Jev on the same rows,
size and sha256. The v3 tolerances do not apply to it (other data, other K); its
numbers are reported as measured. It is evaluated only when gates.json passed.

Every read of a test split, a test-derived v3 file or a stored test ledger is appended
to test-access.log first (the tools called here log their own reads).
"""

import argparse
import datetime
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMFPUBLIC = os.environ.get("CMFPUBLIC") or REPO
ART = os.path.join(CMFPUBLIC, "artifacts", "decision-v4-20260926")
ACCESS_LOG = os.path.join(ART, "test-access.log")
V3_DIR = os.path.join(CMFPUBLIC, "artifacts", "decision-v3-20260926")
V3_EVAL = os.path.join(CMFPUBLIC, "reports", "decision-v3-20260926", "evaluate", "batch")
ORACLE_DIR = os.path.join(CMFPUBLIC, "reports", "decision-v4-20260926", "oracle")
CV_JSON = os.path.join(CMFPUBLIC, "reports", "decision-v4-20260926", "max-recipe", "cv.json")
ENC_SRC = os.environ.get("ENC_SRC") or None
SPLITS = {
    "banking77": os.path.join(CMFPUBLIC, "artifacts/decision-v2-20260926/splits/banking77"),
    "clinc150": os.path.join(CMFPUBLIC, "artifacts/decision-clinc150-20260925/data"),
    "massive": os.path.join(CMFPUBLIC, "artifacts/decision-massive-20260926/data"),
}
DS = ("banking77", "clinc150", "massive")
# §6.7 size limit. Raised from the spec's 300,000,000 B to 320,000,000 B (decision
# of the release, package C4): the published model (§3.9) carries the rows blob of
# every skill — the training rows exact self-learning refits from (§5.7, §5.14) —
# and measured 304,520,292 B (gates-release.json of 2026-09-26); the file is not
# shrunk to fit the old number.
SIZE_LIMIT = 320_000_000
# E1 (§1.6): every file of the reference ids and the total number of texts.
E1_FILES = ("banking77.train.json", "banking77.dev.json", "banking77.calibration.json", "banking77.test.json",
            "clinc150.train.json", "clinc150.dev.json", "clinc150.calibration.json", "clinc150.test.json",
            "clinc150.oos.json", "clinc150.latency.json",
            "massive.train.json", "massive.dev.json", "massive.calibration.json", "massive.test.json",
            "massive.latency.json")
E1_TEXTS = 53396
# The scripts of the gate chain (their sha256 goes into gates.json).
TOOLS = ("tools/decision_gates.py", "tools/decision_speed.sh", "tools/decision_jev_compat.py",
         "tools/decision_encoder_parity.py", "tools/decision_export_encoder.py", "tools/decision_build_release.sh",
         "tools/mk_decision_toy.py")
ALPHA = 0.05 / 14
JEV_USD_PER_1M = {"banking77": 183.69, "clinc150": 271.61, "massive": 110.86}   # evaluate/cost-scale.json

# spec §6.2 — the v3 reference and the tolerances of the full v4 build
V3 = {
    "banking77": {"dev": (1396, 1498), "cal": (1388, 1498), "T": 0.0245571, "theta": 0.788527, "tau": 0.8,
                  "odd": (649, 635), "test": (2882, 3080), "gate": (2666, 2615), "cascade": 2880},
    "clinc150": {"dev": (2888, 2998), "cal": (2924, 3000), "T": 0.0328242, "theta": 0.725459, "tau": 0.0,
                 "odd": (1414, 1406), "test": (4301, 4500), "gate": (4105, 4049), "cascade": 4382, "oos": (860, 1000)},
    "massive": {"dev": (1755, 2025), "cal": (2000, 2288), "T": 0.0303290, "theta": 0.827191, "tau": 0.9,
                "odd": (727, 710), "test": (2535, 2974), "gate": (1856, 1795), "cascade": 2625},
}
TOL = {
    "banking77": {"dev": 2, "dev_flips": 3, "cal": 2, "odd_acc": 15, "test": 4, "test_flips": 7, "gate_acc": 31,
                  "gate_prec": 0.9779, "cascade": 10},
    "clinc150": {"dev": 3, "dev_flips": 6, "cal": 3, "odd_acc": 30, "test": 5, "test_flips": 9, "gate_acc": 45,
                 "gate_prec": 0.9834, "cascade": 14, "oos": 20},
    "massive": {"dev": 3, "dev_flips": 5, "cal": 3, "odd_acc": 23, "test": 3, "test_flips": 6, "gate_acc": 30,
                "gate_prec": 0.9641, "cascade": 9},
}
# the shipped v3 gate values (exact f64 of the f32 numbers; reference_numbers of the scout report)
V3_SHIPPED = {"banking77": (0.024557100608944893, 0.7885268330574036), "clinc150": (0.03282419219613075, 0.7254585027694702),
              "massive": (0.03032897412776947, 0.8271908164024353)}
# F1/F3 expectations (spec §6.2)
F_EXPECT = {"banking77": (1396, 649, 635), "clinc150": (2888, 1414, 1406), "massive": (1755, 727, 710)}
E4_RATE = 0.002


class Stop(Exception):
    """A §6.10 stop condition."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log(msg):
    print(f"[gates {datetime.datetime.now(datetime.timezone.utc).strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def log_access(path, purpose):
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": utc_now(), "file": path, "purpose": purpose}, ensure_ascii=False) + "\n")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(cmd, env=None, cwd=None, log_path=None, check=True):
    log(" ".join(cmd) if len(" ".join(cmd)) < 300 else " ".join(cmd)[:300] + " …")
    full = dict(os.environ)
    full["CMF_GPU"] = "0"
    full.update(env or {})
    t0 = time.time()
    with open(log_path, "w") if log_path else open(os.devnull, "w") as out:
        p = subprocess.run(cmd, env=full, cwd=cwd, stdout=out if log_path else subprocess.PIPE,
                           stderr=subprocess.STDOUT if log_path else subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        tail = open(log_path).read()[-3000:] if log_path else (p.stdout or "")[-2000:] + (p.stderr or "")[-2000:]
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{tail}")
    return p, time.time() - t0


# ---------------------------------------------------------------------- statistics

def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 2), round(100 * (c + h), 2)]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(ours, jev):
    """Accuracy of two paired correctness vectors, Δ in pp and the exact McNemar test."""
    n = len(ours)
    k_o, k_j = sum(ours), sum(jev)
    b = sum(1 for x, y in zip(ours, jev) if x and not y)
    c = sum(1 for x, y in zip(ours, jev) if y and not x)
    return {"n": n, "correct": k_o, "acc_pct": round(100 * k_o / n, 2) if n else None, "wilson95": wilson(k_o, n),
            "jev_correct": k_j, "jev_acc_pct": round(100 * k_j / n, 2) if n else None, "jev_wilson95": wilson(k_j, n),
            "delta_pp": round(100 * (k_o - k_j) / n, 2) if n else None,
            "mcnemar": {"ours_only_right": b, "jev_only_right": c, "p_exact": float("%.3g" % mcnemar_exact(b, c))}}


def cp_lower(k, n, alpha=ALPHA):
    from scipy.stats import beta
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def f32_ulps(a, b):
    ia = np.array([a], dtype=np.float32).view(np.int32)[0]
    ib = np.array([b], dtype=np.float32).view(np.int32)[0]
    return abs(int(ia) - int(ib))


# ---------------------------------------------------------------------- v3 reference (dev winners)

def read_cmf_tensors(path):
    """Minimal reader of a CMF v2 container (cortiq-core format.rs): name -> (dtype, shape, bytes)."""
    b = open(path, "rb").read()
    if b[:4] != b"CMF\x01":
        raise ValueError(f"{path}: not a CMF file")
    u64 = lambda o: int.from_bytes(b[o:o + 8], "little")
    dir_off, dir_len, data_off = u64(0x20), u64(0x28), u64(0x30)
    d = b[dir_off:dir_off + dir_len]
    count = int.from_bytes(d[0:8], "little")
    pool = d[int.from_bytes(d[8:16], "little"):]
    out = {}
    for i in range(count):
        r = d[16 + i * 56:16 + (i + 1) * 56]
        name_off, name_len = int.from_bytes(r[0:4], "little"), int.from_bytes(r[4:6], "little")
        ndim = r[7]
        shape = [int.from_bytes(r[8 + 4 * k:12 + 4 * k], "little") for k in range(ndim)]
        off, nbytes = int.from_bytes(r[32:40], "little"), int.from_bytes(r[40:48], "little")
        out[pool[name_off:name_off + name_len].decode()] = (r[6], shape, b[data_off + off:data_off + off + nbytes])
    return out


def l2f32(v):
    """evaluate_v3/common.py l2f32."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True).astype(np.float32)
    n[n == 0] = 1
    return (v / n).astype(np.float32)


def v3_signal(ds, split):
    """The v3 PH signal of a selection split exactly as the shipped v3 fit and runtime saw it:
    [l2f32(stored φ_P) ; 0.5·stored φ_H] in f32, with the split file and feature sha256 of the v3 build."""
    build = json.load(open(os.path.join(V3_DIR, f"{ds}-product", "build.json")))
    inp = build["inputs"][split]
    if sha256_file(inp["path"]) != inp["sha256"]:
        raise ValueError(f"{inp['path']} changed since the v3 build")
    rows = read_jsonl(inp["path"])
    npy = os.path.join(V3_DIR, "features-product", ds, f"{split}.npy")
    if sha256_file(npy) != build["product_features"][split]["sha256"]:
        raise ValueError(f"{npy} changed")
    P = np.load(npy)
    hdir = os.path.join(V3_DIR, "hash-features", ds)
    man = json.load(open(os.path.join(hdir, f"manifest-{split}.json")))
    npz = os.path.join(hdir, f"{split}.npz")
    if man["input_sha256"] != inp["sha256"] or sha256_file(npz) != man["npz_sha256"]:
        raise ValueError(f"{npz} does not belong to {inp['path']}")
    z = np.load(npz)
    dim = int(z["dim"])
    H = np.zeros((len(rows), dim), dtype=np.float32)
    indptr, indices, values = z["indptr"], z["indices"], z["values"].astype(np.float32)
    for i in range(len(rows)):
        H[i, indices[indptr[i]:indptr[i + 1]]] = values[indptr[i]:indptr[i + 1]]
    if P.shape != (len(rows), 384):
        raise ValueError(f"{npy}: shape {P.shape}")
    X = np.hstack([l2f32(P), (np.float32(0.5) * H).astype(np.float32)]).astype(np.float32)
    return rows, X


def v3_topologies(ds):
    t = read_cmf_tensors(os.path.join(V3_DIR, f"{ds}-product", "cortiq.cmf"))
    m = json.loads(t["decision.manifest"][2])
    meta = m["metadata"]
    dim = int(meta["input_dim"])
    tasks = []
    for i, rec in enumerate(meta["tasks"]):
        mean = np.frombuffer(t[f"decision.task.{i}.mean"][2], dtype="<f4").copy()
        bt = t.get(f"decision.task.{i}.basis")
        basis = np.frombuffer(bt[2], dtype="<f4").reshape(-1, dim).copy() if bt else np.zeros((0, dim), np.float32)
        if rec["active"] and len(basis):
            tasks.append({"label": rec["label"], "mean": mean, "basis": basis})
    return tasks, dim


def v3_errors_f64(tasks, X):
    """Sequential projection in f64 (c_j = b_j·r_{j-1}, r_j = r_{j-1} − c_j b_j), in closed form
    through the raw projections and the basis Gram matrix: E = |r|² − 2Σc_j p_j + Σ c_i c_j G_ij."""
    X64 = X.astype(np.float64)
    E = np.empty((len(X), len(tasks)))
    for t, task in enumerate(tasks):
        R = X64 - task["mean"].astype(np.float64)
        B = task["basis"].astype(np.float64)
        Pj = R @ B.T
        G = B @ B.T
        C = np.zeros_like(Pj)
        for j in range(B.shape[0]):
            C[:, j] = Pj[:, j] - C[:, :j] @ G[j, :j]
        E[:, t] = np.einsum("ij,ij->i", R, R) - 2 * np.einsum("ij,ij->i", C, Pj) + np.einsum("ij,jk,ik->i", C, G, C)
    return E


def v3_errors_f32_exact(tasks, x):
    """The v3 runtime's f32 arithmetic for one row, bit for bit: r = x − μ; per basis row
    c = Σ r_i b_i sequentially (cumsum is a sequential f32 sum), r −= c·b (no FMA);
    E = Σ r² sequentially."""
    out = np.empty(len(tasks), dtype=np.float32)
    for t, task in enumerate(tasks):
        r = (x - task["mean"]).astype(np.float32)
        for b in task["basis"]:
            c = np.cumsum(r * b, dtype=np.float32)[-1]
            r = (r - (np.float32(c) * b).astype(np.float32)).astype(np.float32)
        out[t] = np.cumsum(r * r, dtype=np.float32)[-1]
    return out


def v3_winners(ds, split):
    """Winner labels of the shipped v3 PH runtime on a selection split. f64 errors decide
    every row whose two smallest errors are more than 0.2 % apart; the other rows are
    recomputed with the exact f32 runtime arithmetic (score 1/(1+E) in f32, stable order)."""
    rows, X = v3_signal(ds, split)
    tasks, dim = v3_topologies(ds)
    if X.shape[1] != dim:
        raise ValueError(f"v3 {ds}: signal {X.shape[1]} vs topologies {dim}")
    E = v3_errors_f64(tasks, X)
    order = np.argsort(E, axis=1, kind="stable")
    e1 = E[np.arange(len(E)), order[:, 0]]
    e2 = E[np.arange(len(E)), order[:, 1]]
    close = np.nonzero((e2 - e1) <= 2e-3 * np.maximum(e1, 1e-12))[0]
    win = order[:, 0].copy()
    for i in close:
        e = v3_errors_f32_exact(tasks, X[i])
        s = (np.float32(1.0) / (np.float32(1.0) + e)).astype(np.float32)
        win[i] = int(np.argmax(s))   # first maximum = stable order of the tasks
    labels = [tasks[w]["label"] for w in win]
    correct = sum(l == r["label"] for l, r in zip(labels, rows))
    return {"rows": rows, "labels": labels, "correct": correct, "exact_f32_rows": int(len(close))}


# ---------------------------------------------------------------------- cortiq helpers

def decide_rows(cortiq, model, skill, input_path, out_path, env=None):
    if os.path.exists(out_path):
        os.remove(out_path)   # an output of an earlier gates run in the work directory
    run([cortiq, "decide", model, "--input", input_path, "--skill", skill, "--out", out_path],
        env=env, log_path=out_path + ".log")
    summ = None
    for l in open(out_path + ".log", encoding="utf-8"):
        if l.startswith('{"summary"'):
            summ = json.loads(l)["summary"]
    return read_jsonl(out_path), summ


def model_info(cortiq, model):
    p, _ = run([cortiq, "decision", "info", model, "--json"])
    return json.loads(p.stdout.strip().splitlines()[-1])


def skill_manifest(info, ds):
    for s in info["skills"]:
        if s["id"] == ds:
            return s
    raise KeyError(ds)


def gate_of(m):
    g = m["gate"]
    grid = g["evidence"]["odd"]["grid"]
    chosen = next((r for r in grid if g["certified"] and np.float32(r["t"]) == np.float32(g["tau"])), None)
    return g, grid, chosen


def stop_on_table(table, doc):
    """§6.10: τ ≠ v3 or any violated §6.2 tolerance halts the gates."""
    for r in table:
        if not r["pass"]:
            what = "tau != v3" if r["metric"] == "tau" else "§6.2 tolerance violated"
            doc["stop"].append(f"{r['dataset']}: {r['metric']} = {r['value']} (v3 {r['v3']}, {r['tolerance']}): {what} (§6.10)")
    if doc["stop"]:
        doc["gates"]["table_6_2"]["pass"] = False
        raise Stop()


def check(rows, name, value, ref, ok, tol, ds=None):
    rows.append({"dataset": ds, "metric": name, "value": value, "v3": ref, "tolerance": tol, "pass": bool(ok)})
    return bool(ok)


# ---------------------------------------------------------------------- F1–F3, E1–E5

def gates_f(work):
    out = os.path.join(work, "parity_v3.json")
    if os.path.exists(out):
        os.remove(out)
    p, secs = run(["cargo", "test", "--offline", "--release", "-p", "cortiq-decision", "--test", "parity_v3", "--",
                   "--ignored", "--nocapture"],
                  env={"CORTIQ_DECISION_V3_DIR": V3_DIR, "CORTIQ_DECISION_PARITY_OUT": out, "CORTIQ_TEST_THREADS": "4"},
                  cwd=REPO, log_path=os.path.join(work, "parity_v3.log"), check=False)
    if not os.path.exists(out):
        raise RuntimeError("parity_v3 wrote no JSON; see " + os.path.join(work, "parity_v3.log"))
    pj = json.load(open(out))
    res = {"command": "CMF_GPU=0 CORTIQ_DECISION_V3_DIR=… cargo test --release -p cortiq-decision --test parity_v3 -- --ignored",
           "exit_code": p.returncode, "seconds": round(secs, 1), "test_pass": pj["pass"], "test_failures": pj["failures"]}
    f1, f2, f3 = {}, {}, {}
    for d in pj["datasets"]:
        ds = d["dataset"]
        dev, odd_a, odd_c = F_EXPECT[ds]
        f1[ds] = {"dev_correct": d["f1"]["dev_correct"], "dev_n": d["f1"]["dev_n"], "expected": dev,
                  "flips_vs_v3": d["f1"]["flips_vs_v3"],
                  "pass": d["f1"]["dev_correct"] == dev and d["f1"]["flips_vs_v3"] == 0}
        f2[ds] = {"max_principal_angle_rad": d["f2"]["max_principal_angle_rad"],
                  "max_relative_e_dev": d["f2"]["max_relative_e_dev"],
                  "pass": d["f2"]["max_principal_angle_rad"] <= 1e-6 and d["f2"]["max_relative_e_dev"] <= 1e-6}
        g = d["f3"]
        grid_rows = []
        for r in g["grid"]:
            lb_s = cp_lower(r["correct"], r["accepted"])
            grid_rows.append({"t": r["t"], "accepted": r["accepted"], "correct": r["correct"], "lb": r["lb"],
                              "lb_scipy": lb_s, "abs_diff": abs(r["lb"] - lb_s)})
        worst = max(x["abs_diff"] for x in grid_rows)
        f3[ds] = {"temperature": g["temperature"], "temperature_ulps_vs_v3": g["temperature_ulps_vs_v3"],
                  "novelty_theta": g["novelty_theta"], "theta_ulps_vs_v3": g["theta_ulps_vs_v3"],
                  "tau": g["tau"], "tau_v3": g["tau_v3"], "odd_half": g["odd_half"],
                  "cp_lb_vs_scipy_max_abs": worst, "grid_rows": len(grid_rows), "grid": grid_rows,
                  "pass": (g["temperature_ulps_vs_v3"] <= 1 and g["theta_ulps_vs_v3"] <= 2 and g["tau"] == g["tau_v3"]
                           and g["odd_half"]["accepted"] == odd_a and g["odd_half"]["correct"] == odd_c
                           and len(grid_rows) == 14 and worst <= 1e-9)}
    # A gate passes only when the test run itself passed (exit 0, its own JSON verdict).
    run_ok = p.returncode == 0 and pj["pass"] is True
    res["F1"] = {"datasets": f1, "pass": run_ok and all(v["pass"] for v in f1.values()) and len(f1) == 3}
    res["F2"] = {"datasets": f2, "pass": run_ok and all(v["pass"] for v in f2.values()) and len(f2) == 3}
    res["F3"] = {"datasets": f3, "pass": run_ok and all(v["pass"] for v in f3.values()) and len(f3) == 3}
    res["resonance_dev_single_thread"] = {d["dataset"]: d.get("resonance_dev_single_thread") for d in pj["datasets"]}
    return res


def gates_e(work):
    enc = os.path.join(work, "encoder-export")
    ids = os.path.join(work, "hf-ids")
    probe = os.path.join(work, "unicode-probe.tsv")
    gates_out = os.path.join(work, "encoder-gates")
    for p in (enc, ids, gates_out):
        if os.path.exists(p):
            shutil.rmtree(p)
    if os.path.exists(probe):
        os.remove(probe)
    run([sys.executable, os.path.join(REPO, "tools", "decision_export_encoder.py"), "--onnx",
         os.path.join(ENC_SRC, "encoder.onnx"), "--tokenizer-dir", os.path.join(ENC_SRC, "encoder_tokenizer"),
         "--out", enc], log_path=os.path.join(work, "export.log"))
    art = ["--artifacts-dir", os.path.join(CMFPUBLIC, "artifacts"), "--access-log", ACCESS_LOG]
    run([sys.executable, os.path.join(REPO, "tools", "decision_encoder_parity.py"), "hf-ids", "--tokenizer-dir",
         os.path.join(ENC_SRC, "encoder_tokenizer"), "--out", ids, *art], log_path=os.path.join(work, "hf-ids.log"))
    run([sys.executable, os.path.join(REPO, "tools", "decision_encoder_parity.py"), "unicode-probe", "--out", probe,
         *art], log_path=os.path.join(work, "unicode-probe.log"))
    p, secs = run(["cargo", "test", "--offline", "--release", "-p", "cortiq-decision", "--test", "encoder_real", "--",
                   "--ignored", "--nocapture", "--test-threads=1"],
                  env={"CORTIQ_DECISION_ENCODER_DIR": enc, "CORTIQ_DECISION_HF_IDS_DIR": ids,
                       "CORTIQ_DECISION_UNICODE_PROBE": probe, "CORTIQ_DECISION_V3_DIR": V3_DIR,
                       "CORTIQ_DECISION_ENCODER_OUT": gates_out, "CORTIQ_DECISION_THREADS": "4",
                       "CORTIQ_DECISION_TEST_ACCESS_LOG": ACCESS_LOG},
                  cwd=REPO, log_path=os.path.join(work, "encoder_real.log"), check=False)
    export_sha = sha256_file(os.path.join(enc, "encoder.json"))
    shutil.rmtree(enc, ignore_errors=True)   # 91 MB, only the test needed it
    J = lambda n: json.load(open(os.path.join(gates_out, n + ".json"))) if os.path.exists(os.path.join(gates_out, n + ".json")) else None
    e1, e234, e5 = J("e1"), J("e2_e3_e4"), J("e5")
    res = {"command": "CMF_GPU=0 … cargo test --release -p cortiq-decision --test encoder_real -- --ignored --test-threads=1",
           "exit_code": p.returncode, "seconds": round(secs, 1),
           "encoder_export_json_sha256": export_sha,
           "speed_one_thread": J("speed"), "unicode_exhaustive": J("unicode"), "gpu": J("gpu"), "init_file": J("init_file")}
    run_ok = p.returncode == 0
    e1_files = sorted(x["file"] for x in (e1 or {}).get("files") or [] if x["file"] != "stress.json")
    e1_complete = (bool(e1) and e1_files == sorted(E1_FILES) and e1["texts"] == E1_TEXTS
                   and all(x.get("equal") == x["n"] for x in e1["files"] if x["file"] != "stress.json"))
    res["E1"] = {"texts": e1 and e1["texts"], "equal": e1 and e1["equal"], "files": e1 and e1["files"],
                 "stress_differences": e1 and e1["stress_differences"],
                 "required_files": len(E1_FILES), "required_texts": E1_TEXTS, "complete": e1_complete,
                 "pass": run_ok and e1_complete and e1["equal"] == e1["texts"]}
    e2, e3, e4 = {}, {}, {}
    ok2 = ok3 = ok4 = bool(e234)
    for ds in DS if e234 else ():
        d = e234[ds]
        t = d["e3_train"]
        e3[f"{ds}/train"] = {k: t[k] for k in ("n", "index_set_differs", "rows", "max_abs")}
        ok3 &= t["max_abs"] <= 1e-7
        for split in ("dev", "calibration", "test"):
            s = d.get(split) or {}
            if "e2" in s:
                x = s["e2"]
                e2[f"{ds}/{split}"] = {k: x[k] for k in ("n", "max_abs", "min_cos", "bit_equal_rows")}
                ok2 &= x["max_abs"] <= 1e-6 and x["min_cos"] >= 0.999999
            if "e3" in s:
                x = s["e3"]
                e3[f"{ds}/{split}"] = {k: x[k] for k in ("n", "index_set_differs", "rows", "max_abs")}
                ok3 &= x["max_abs"] <= 1e-7
            if "e4" in s:
                x = s["e4"]
                lim = math.ceil(E4_RATE * x["n"])
                e4[f"{ds}/{split}"] = {"n": x["n"], "flips": x["flips"], "limit": lim,
                                       "correct_v3_signal": x["correct_v3_signal"], "correct_v4_signal": x["correct_v4_signal"]}
                ok4 &= x["flips"] <= lim
    ok4 &= sorted(e4) == sorted(f"{ds}/{s}" for ds in DS for s in ("dev", "test"))
    res["E2"] = {"splits": e2, "tolerance": "max |Δ| <= 1e-6, min cos >= 0.999999",
                 "pass": run_ok and bool(ok2) and len(e2) == 9}
    res["E3"] = {"splits": e3, "tolerance": "max |Δ| <= 1e-7; index-set differences listed",
                 "pass": run_ok and bool(ok3) and len(e3) == 9}
    res["E4"] = {"splits": e4, "tolerance": "flips <= ceil(0.2 %·n) on dev and test", "pass": run_ok and bool(ok4)}
    res["E5"] = dict(e5 or {})
    res["E5"]["pass"] = run_ok and bool(e5 and e5.get("pass"))
    return res


# ---------------------------------------------------------------------- per-model evaluation

def load_ledger_answers(ds, purpose):
    """Stored DeepSeek answers on the test rows the v3 model rejected: text_sha256 -> oracle record."""
    ans, files = {}, []
    for p in sorted(glob.glob(os.path.join(ORACLE_DIR, f"{ds}.part*.jsonl"))):
        log_access(p, purpose)
        files.append({"path": p, "sha256": sha256_file(p)})
        for r in read_jsonl(p):
            if r.get("record_type") == "oracle_call":
                ans[r["text_sha256"]] = r["oracle"]
    return ans, files


def load_v3_test_rows(ds, purpose):
    p = os.path.join(V3_EVAL, f"{ds}-PH", "rows.jsonl")
    log_access(p, purpose)
    rows = [r for r in read_jsonl(p) if r["set"] == "test-train-start"]
    rows.sort(key=lambda r: r["index"])
    return rows, {"path": p, "sha256": sha256_file(p)}


def test_metrics(ds, rows, v3rows, ledger):
    """Test rows of `cortiq decide` (in test-file order) → all rows, certified gate, static
    cascade, each next to Jev on the same rows (stored per-row verdicts)."""
    if len(rows) != len(v3rows):
        raise ValueError(f"{ds}: {len(rows)} decided rows vs {len(v3rows)} v3 rows")
    for r, v in zip(rows, v3rows):
        if r["i"] != v["index"] or r["text_sha256"] != v["text_sha256"]:
            raise ValueError(f"{ds}: row {r['i']} is not aligned with the v3 test rows")
    truth = [v["truth"] for v in v3rows]
    jev_ok = [bool(v["jev_correct"]) for v in v3rows]
    ok = [r["choice"] == t for r, t in zip(rows, truth)]
    if any(("correct" in r) and r["correct"] != o for r, o in zip(rows, ok)):
        raise ValueError(f"{ds}: decide's correct flag disagrees with the truth")
    flips = sum(r["choice"] != v["pred"] for r, v in zip(rows, v3rows))
    acc_idx = [i for i, r in enumerate(rows) if r["accepted"]]
    not_certified = sum(1 for i in acc_idx if not rows[i]["certified"])
    casc, called, missing, failed, orc_ok, cost = [], [], [], [], 0, 0.0
    for i, r in enumerate(rows):
        if r["accepted"]:
            casc.append(ok[i])
            continue
        o = ledger.get(r["text_sha256"])
        if o is None:
            missing.append(i)
            casc.append(ok[i])
        elif not o.get("choice"):
            called.append(i)
            failed.append(i)
            casc.append(ok[i])
            cost += (o.get("usage") or {}).get("cost") or 0
        else:
            called.append(i)
            c = o["choice"] == truth[i]
            orc_ok += c
            casc.append(c)
            cost += (o.get("usage") or {}).get("cost") or 0
    n = len(rows)
    gate_ok = sum(ok[i] for i in acc_idx)
    return {
        "n": n,
        "all_rows": dict(paired(ok, jev_ok), flips_vs_v3=flips),
        "certified_gate": {"accepted": len(acc_idx), "correct": gate_ok,
                           "precision_pct": round(100 * gate_ok / len(acc_idx), 2) if acc_idx else None,
                           "precision_wilson95": wilson(gate_ok, len(acc_idx)),
                           "coverage_pct": round(100 * len(acc_idx) / n, 2), "silent_errors": len(acc_idx) - gate_ok,
                           "accepted_not_certified": not_certified,
                           "jev_on_accepted_rows": paired([ok[i] for i in acc_idx], [jev_ok[i] for i in acc_idx])},
        "static_cascade": dict(paired(casc, jev_ok),
                               oracle={"rejected_rows": n - len(acc_idx), "answers_from_ledger": len(called),
                                       "failed_in_ledger": len(failed), "missing_no_stored_answer": len(missing),
                                       "oracle_correct": orc_ok,
                                       "jev_correct_on_ledger_rows": sum(jev_ok[i] for i in called),
                                       "usd": round(cost, 6),
                                       "usd_per_1m_decisions": round(cost / n * 1e6, 2),
                                       "jev_usd_per_1m_decisions": JEV_USD_PER_1M[ds]},
                               rule="accepted by the certified gate → local answer; rejected → the stored DeepSeek answer "
                                    "(reports/decision-v4-20260926/oracle), no stored answer or a failed call → the local top-1"),
    }


def oos_metrics(cortiq, model, work, tag):
    oos = os.path.join(SPLITS["clinc150"], "oos.jsonl")
    log_access(oos, f"decision-v4 WP9 §6.2 CLINC150 OOS rejections ({tag} model, cortiq decide --input)")
    rows, summ = decide_rows(cortiq, model, "clinc150", oos, os.path.join(work, f"{tag}.clinc150.oos.rows.jsonl"))
    rejected = sum(1 for r in rows if not r["accepted"])
    return {"n": len(rows), "rejected": rejected, "rejected_pct": round(100 * rejected / len(rows), 2),
            "jev": "Jev always picks one of the 150 labels (0/1000 correct on OOS, spec §6.2 has no Jev row)"}


def model_record(path):
    return {"path": path, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def pre_test_rows(cortiq, model, info, work, tag, table, v3_dev):
    """dev, calibration, T, θ, τ, odd half of the reproduction model (no test data)."""
    per = {}
    for ds in DS:
        m = skill_manifest(info, ds)
        g, grid, chosen = gate_of(m)
        d = SPLITS[ds]
        dev_rows, _ = decide_rows(cortiq, model, ds, os.path.join(d, "dev.jsonl"), os.path.join(work, f"{tag}.{ds}.dev.rows.jsonl"))
        cal_rows, _ = decide_rows(cortiq, model, ds, os.path.join(d, "calibration.jsonl"), os.path.join(work, f"{tag}.{ds}.calibration.rows.jsonl"))
        dev_ok = sum(bool(r.get("correct")) for r in dev_rows)
        cal_ok = sum(bool(r.get("correct")) for r in cal_rows)
        v3w = v3_dev[ds]
        if [sha256_text(r["text"]) for r in v3w["rows"]] != [r["text_sha256"] for r in dev_rows]:
            raise ValueError(f"{ds}: dev rows of the v3 reference and of the decide batch differ")
        flips = sum(a != r["choice"] for a, r in zip(v3w["labels"], dev_rows))
        ref, tol = V3[ds], TOL[ds]
        T, theta, tau = g["temperature"], g["novelty_theta"], g["tau"]
        odd = g["evidence"]["odd"]
        lb = chosen["lb"] if chosen else None
        lb_scipy = cp_lower(chosen["correct"], chosen["accepted"]) if chosen else None
        check(table, "dev all rows (correct)", [dev_ok, len(dev_rows)], list(ref["dev"]),
              abs(dev_ok - ref["dev"][0]) <= tol["dev"] and len(dev_rows) == ref["dev"][1], f"|Δ| <= {tol['dev']}", ds)
        check(table, "dev winner flips vs v3", flips, 0, flips <= tol["dev_flips"], f"<= {tol['dev_flips']}", ds)
        check(table, "calibration all rows (correct)", [cal_ok, len(cal_rows)], list(ref["cal"]),
              abs(cal_ok - ref["cal"][0]) <= tol["cal"] and len(cal_rows) == ref["cal"][1], f"|Δ| <= {tol['cal']}", ds)
        check(table, "T", T, ref["T"], abs(T - ref["T"]) / ref["T"] <= 0.01, "relative <= 1 %", ds)
        check(table, "theta", theta, ref["theta"], abs(theta - ref["theta"]) <= 0.005, "|Δ| <= 0.005", ds)
        tau_ok = np.float32(tau) == np.float32(ref["tau"]) and g["certified"]
        check(table, "tau", tau, ref["tau"], tau_ok, "equal, else STOP", ds)
        check(table, "odd half accepted/correct", [odd["accepted"], odd["correct"]], list(ref["odd"]),
              abs(odd["accepted"] - ref["odd"][0]) <= tol["odd_acc"] and lb is not None and lb >= 0.95,
              f"accepted ±{tol['odd_acc']}; lb >= 0.95", ds)
        per[ds] = {"gate": {"temperature": T, "temperature_ulps_vs_v3_shipped": f32_ulps(T, V3_SHIPPED[ds][0]),
                            "novelty_theta": theta, "theta_ulps_vs_v3_shipped": f32_ulps(theta, V3_SHIPPED[ds][1]), "tau": tau,
                            "certified": g["certified"], "odd_half": {"n": odd["n"], "accepted": odd["accepted"],
                                                                     "correct": odd["correct"], "lb": lb,
                                                                     "lb_scipy": lb_scipy},
                            "grid_cp_lb_vs_scipy_max_abs": max(abs(r["lb"] - cp_lower(r["correct"], r["accepted"])) for r in grid)},
                   "recipe": {"K": m["recipe"].get("K"), "k": sorted({t["k"] for t in m["tasks"]}),
                              "tasks_active": sum(t["state"] == "active" for t in m["tasks"]), "labels": len(m["labels"])},
                   "data": {"train": m["data"]["train"]["n"], "calibration": m["data"]["calibration"]["n"]},
                   "dev": {"correct": dev_ok, "n": len(dev_rows), "flips_vs_v3": flips, "v3_reference_correct": v3w["correct"],
                           "v3_reference_exact_f32_rows": v3w["exact_f32_rows"],
                           "accepted": sum(r["accepted"] for r in dev_rows),
                           "accepted_correct": sum(bool(r["accepted"] and r.get("correct")) for r in dev_rows)},
                   "calibration": {"correct": cal_ok, "n": len(cal_rows)}}
    return per


def release_rows(cortiq, model, info, work, tag):
    per = {}
    for ds in DS:
        m = skill_manifest(info, ds)
        g, grid, chosen = gate_of(m)
        d = SPLITS[ds]
        cal_rows, _ = decide_rows(cortiq, model, ds, os.path.join(d, "calibration.jsonl"), os.path.join(work, f"{tag}.{ds}.calibration.rows.jsonl"))
        odd = g["evidence"]["odd"]
        per[ds] = {"gate": {"temperature": g["temperature"], "novelty_theta": g["novelty_theta"], "tau": g["tau"],
                            "certified": g["certified"],
                            "odd_half": {"n": odd["n"], "accepted": odd["accepted"], "correct": odd["correct"],
                                         "lb": chosen["lb"] if chosen else None,
                                         "lb_scipy": cp_lower(chosen["correct"], chosen["accepted"]) if chosen else None},
                            "grid_cp_lb_vs_scipy_max_abs": max(abs(r["lb"] - cp_lower(r["correct"], r["accepted"])) for r in grid)},
                   "recipe": {"K": m["recipe"].get("K"), "k_source": m["recipe"].get("k_source"),
                              "k": sorted({t["k"] for t in m["tasks"]}),
                              "tasks_active": sum(t["state"] == "active" for t in m["tasks"]), "labels": len(m["labels"])},
                   "data": {"train": m["data"]["train"]["n"], "train_parts": len(m["data"]["train"].get("parts") or []) or 1,
                            "calibration": m["data"]["calibration"]["n"], "dev_in_training": True},
                   "calibration": {"correct": sum(bool(r.get("correct")) for r in cal_rows), "n": len(cal_rows)}}
    return per


def test_rows_via_speed(model, work, tag, http):
    sp = os.path.join(work, f"speed-{tag}")
    if os.path.exists(sp):
        shutil.rmtree(sp)
    cmd = [os.path.join(REPO, "tools", "decision_speed.sh"), model, sp]
    if not http:
        cmd.append("--no-http")
    run(cmd, log_path=os.path.join(work, f"speed-{tag}.log"))
    return sp, json.load(open(os.path.join(sp, "speed.json")))


# ---------------------------------------------------------------------- main

def machine():
    def sysctl(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    return {"cpu": sysctl("machdep.cpu.brand_string"), "memsize_bytes": int(sysctl("hw.memsize") or 0),
            "os": platform.platform(), "python": platform.python_version(), "numpy": np.__version__}


def git(*a):
    try:
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="decision-v4 local acceptance gates (spec §6)")
    ap.add_argument("--base", default=os.path.join(ART, "cortiq-decision-base.cmf"))
    ap.add_argument("--release", default=os.path.join(ART, "release", "cortiq-decision.cmf"))
    ap.add_argument("--out-dir", default=ART)
    ap.add_argument("--work", default=None, help="per-row outputs, logs, temporary exports (default: <out-dir>/gates-work)")
    ap.add_argument("--cortiq", default=os.path.join(REPO, "target", "release", "cortiq"))
    ap.add_argument("--build-report", default=os.path.join(ART, "build", "build-release.json"))
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--no-http", action="store_true")
    a = ap.parse_args()
    if not ENC_SRC:
        sys.exit("decision_gates: set ENC_SRC to the directory with encoder.onnx and encoder_tokenizer/ "
                 "(the encoder source of gates E1–E5)")
    work = a.work or os.path.join(a.out_dir, "gates-work")
    os.makedirs(work, exist_ok=True)
    cortiq = a.cortiq
    t_start = time.time()
    access_before = sum(1 for _ in open(ACCESS_LOG)) if os.path.exists(ACCESS_LOG) else 0
    doc = {"schema": "cortiq-decision-v4-gates/1", "utc_start": utc_now(), "spec": "reports/decision-v4-20260926/SPEC_RU.md §6",
           "commit": git("rev-parse", "HEAD"), "worktree_clean": git("status", "--porcelain", "--untracked-files=no") == "",
           "untracked": [l[3:] for l in (git("status", "--porcelain", "--untracked-files=all", "--", "tools", "crates") or "").splitlines()
                         if l.startswith("?? ")],
           "tools": {t: {"sha256": sha256_file(os.path.join(REPO, t)),
                         "tracked": bool(git("ls-files", "--error-unmatch", t))}
                     for t in TOOLS if os.path.exists(os.path.join(REPO, t))},
           "machine": machine(),
           "binary": {"path": cortiq, "sha256": sha256_file(cortiq),
                      "version": subprocess.run([cortiq, "--version"], capture_output=True, text=True).stdout.strip()},
           "model": model_record(a.base), "stop": [], "gates": {}}
    gates = doc["gates"]
    out_path = os.path.join(a.out_dir, "gates.json")
    try:
        # --- reproducibility of the build
        br = json.load(open(a.build_report)) if os.path.exists(a.build_report) else None
        doc["build"] = None if br is None else {
            "report": a.build_report, "reproducible": br["reproducible"],
            "runs": {k: {"threads": v["threads"], "sha256_base": v["sha256"]["base"], "sha256_release": v["sha256"]["release"]}
                     for k, v in br["runs"].items()},
            "source_date_epoch": br["source_date_epoch"], "commit": br["commit"]}
        gates["reproducible_build"] = {"pass": bool(br and br["reproducible"]
                                                    and all(v["sha256"]["base"] == doc["model"]["sha256"] for v in br["runs"].values()))}
        # --- F1–F3
        log("F1–F3 (parity_v3)")
        f = gates_f(work)
        gates["F1"], gates["F2"], gates["F3"] = f.pop("F1"), f.pop("F2"), f.pop("F3")
        gates["F_run"] = f
        # --- §6.2 table, part without test data
        log("§6.2: dev, calibration, gate of the reproduction model")
        info = model_info(cortiq, a.base)
        doc["model"]["model_sha"] = info["model_sha"]
        v3_dev = {}
        for ds in DS:
            v3_dev[ds] = v3_winners(ds, "dev")
            if v3_dev[ds]["correct"] != V3[ds]["dev"][0]:
                raise RuntimeError(f"the v3 dev reference of {ds} gives {v3_dev[ds]['correct']}, not {V3[ds]['dev'][0]}")
        table = []
        per = pre_test_rows(cortiq, a.base, info, work, "base", table, v3_dev)
        for ds in DS:
            rc = per[ds]["recipe"]
            if rc["K"] != 16:
                raise RuntimeError(f"{ds}: the reproduction model has K={rc['K']}, not 16")
        doc["datasets"] = per
        gates["table_6_2"] = {"rows": table}
        stop_on_table(table, doc)
        # --- E1–E5: after the pre-test table, because the reference ids (E1) and
        # E2/E4 read the test splits.
        log("E1–E5 (encoder_real)")
        e = gates_e(work)
        for k in ("E1", "E2", "E3", "E4", "E5"):
            gates[k] = e.pop(k)
        gates["E_run"] = e
        if not gates["E1"]["pass"]:
            doc["stop"].append(f"E1 {gates['E1']['equal']}/{gates['E1']['texts']} < 100 %"
                               f" or incomplete ({gates['E1']['complete']}) (§6.10)")
            raise Stop()
        # --- test: decide --bench (speed) + HTTP
        log("§6.6 speed and the test rows (tools/decision_speed.sh)")
        sp_dir, speed = test_rows_via_speed(a.base, work, "base", not a.no_http)
        gates["speed_6_6"] = speed
        for ds in DS:
            v3rows, v3src = load_v3_test_rows(ds, "decision-v4 WP9 §6.2: v3 per-row test decisions (flips) and the stored Jev verdicts on the same rows")
            ledger, lfiles = load_ledger_answers(ds, "decision-v4 WP9 §6.2 static cascade: stored DeepSeek answers on rejected test rows (no network)")
            rows = read_jsonl(os.path.join(sp_dir, f"{ds}.test.rows.jsonl"))
            tm = test_metrics(ds, rows, v3rows, ledger)
            tm["sources"] = {"v3_rows": v3src, "oracle_ledgers": lfiles}
            per[ds]["test"] = tm
            ref, tol = V3[ds], TOL[ds]
            k = tm["all_rows"]["correct"]
            check(table, "test all rows (correct)", [k, tm["n"]], list(ref["test"]),
                  abs(k - ref["test"][0]) <= tol["test"] and tm["n"] == ref["test"][1], f"|Δ| <= {tol['test']}", ds)
            check(table, "test winner flips vs v3", tm["all_rows"]["flips_vs_v3"], 0,
                  tm["all_rows"]["flips_vs_v3"] <= tol["test_flips"], f"<= {tol['test_flips']}", ds)
            cg = tm["certified_gate"]
            check(table, "test certified gate accepted/correct", [cg["accepted"], cg["correct"]], list(ref["gate"]),
                  abs(cg["accepted"] - ref["gate"][0]) <= tol["gate_acc"] and cg["accepted"] > 0
                  and cg["correct"] / cg["accepted"] >= tol["gate_prec"] and cg["accepted_not_certified"] == 0,
                  f"accepted ±{tol['gate_acc']}; precision >= {100 * tol['gate_prec']:.2f} %", ds)
            if ds == "clinc150":
                o = oos_metrics(cortiq, a.base, work, "base")
                per[ds]["oos"] = o
                check(table, "CLINC OOS rejected", [o["rejected"], o["n"]], list(ref["oos"]),
                      abs(o["rejected"] - ref["oos"][0]) <= tol["oos"] and o["n"] == ref["oos"][1], f"±{tol['oos']}", ds)
            sc = tm["static_cascade"]["correct"]
            check(table, "static cascade correct", sc, ref["cascade"], abs(sc - ref["cascade"]) <= tol["cascade"],
                  f"|Δ| <= {tol['cascade']}", ds)
        stop_on_table(table, doc)
        gates["table_6_2"]["pass"] = True
        # --- §6.3 Jev compatibility
        log("§6.3 Jev compatibility (tools/decision_jev_compat.py)")
        jc = os.path.join(work, "jev-compat-base.json")
        if os.path.exists(jc):
            os.remove(jc)
        run([sys.executable, os.path.join(REPO, "tools", "decision_jev_compat.py"), "--model", a.base, "--decide-rows", sp_dir,
             "--out", jc, "--cortiq", cortiq, "--work", os.path.join(work, "jev-compat-work")],
            log_path=os.path.join(work, "jev-compat.log"), check=False)
        j = json.load(open(jc))
        for ds in DS:
            d = j["datasets"][ds]
            d["accuracy_equal_to_6_2"] = d["server_correct"] == per[ds]["test"]["all_rows"]["correct"]
            d["pass"] = d["pass"] and d["accuracy_equal_to_6_2"]
        j["pass"] = all(j["datasets"][ds]["pass"] for ds in DS) and j["multi_type"]["pass"]
        gates["jev_compat_6_3"] = j
        # --- §6.7 size
        gates["size_6_7"] = {"bytes": doc["model"]["bytes"], "limit": SIZE_LIMIT, "sha256": doc["model"]["sha256"],
                             "pass": doc["model"]["bytes"] <= SIZE_LIMIT}
    except Stop:
        log("STOP: " + "; ".join(doc["stop"]))
    except Exception as exc:  # recorded, then re-raised after gates.json is written
        doc["error"] = f"{type(exc).__name__}: {exc}"
        doc["pass"] = False
        doc["utc_end"] = utc_now()
        write_json(out_path, doc)
        raise
    doc["isolation_and_budget"] = {"isolation_violations": None, "spent_usd": 0.0,
                                   "note": "no self-learning and no oracle call happen in these gates; both belong to the §5c cascade evaluation"}
    required = ["reproducible_build", "F1", "F2", "F3", "E1", "E2", "E3", "E4", "E5", "table_6_2", "jev_compat_6_3",
                "speed_6_6", "size_6_7"]
    doc["summary"] = {k: (gates.get(k) or {}).get("pass") for k in required}
    doc["pass"] = not doc["stop"] and all(doc["summary"][k] is True for k in required)
    doc["seconds"] = round(time.time() - t_start, 1)
    doc["utc_end"] = utc_now()
    doc["test_access_log_lines_added"] = (sum(1 for _ in open(ACCESS_LOG)) - access_before) if os.path.exists(ACCESS_LOG) else None
    write_json(out_path, doc)
    log(f"wrote {out_path}: pass={doc['pass']} " + json.dumps(doc["summary"]))
    if not doc["pass"] or a.no_release:
        if not doc["pass"]:
            log("the published model is not evaluated: gates.json did not pass")
        return 0 if doc["pass"] else 1

    # ------------------------------------------------------------------ published model (§3.9)
    log("published model: calibration, test, OOS, static cascade")
    rdoc = {"schema": "cortiq-decision-v4-gates-release/1", "utc_start": utc_now(),
            "spec": "§3.9 published model; numbers as measured (the v3 tolerances of §6.2 apply to the reproduction model)",
            "commit": doc["commit"], "machine": doc["machine"], "binary": doc["binary"], "model": model_record(a.release)}
    t1 = time.time()
    access_before = sum(1 for _ in open(ACCESS_LOG))
    info = model_info(cortiq, a.release)
    rdoc["model"]["model_sha"] = info["model_sha"]
    cv = json.load(open(CV_JSON)) if os.path.exists(CV_JSON) else None
    rdoc["recipe"] = {"cv_json": {"path": CV_JSON, "sha256": sha256_file(CV_JSON)} if cv else None,
                      "chosen_K": cv["chosen_K"] if cv else {ds: 16 for ds in DS},
                      "note": None if cv else "cv.json missing: K = 16"}
    per = release_rows(cortiq, a.release, info, work, "release")
    for ds in DS:
        if per[ds]["recipe"]["K"] != rdoc["recipe"]["chosen_K"][ds]:
            raise RuntimeError(f"{ds}: published K {per[ds]['recipe']['K']} != chosen_K {rdoc['recipe']['chosen_K'][ds]}")
    sp_dir, speed = test_rows_via_speed(a.release, work, "release", False)
    rdoc["speed_6_6"] = speed
    for ds in DS:
        v3rows, _ = load_v3_test_rows(ds, "decision-v4 WP9 gates-release: stored Jev verdicts on the same test rows (published model)")
        ledger, lfiles = load_ledger_answers(ds, "decision-v4 WP9 gates-release static cascade: stored DeepSeek answers (no network)")
        rows = read_jsonl(os.path.join(sp_dir, f"{ds}.test.rows.jsonl"))
        tm = test_metrics(ds, rows, v3rows, ledger)
        tm["all_rows"]["flips_vs_v3_note"] = "another model (train ∪ dev, its own K): flips are differences, not a gate"
        tm["sources"] = {"oracle_ledgers": lfiles}
        per[ds]["test"] = tm
        per[ds]["reproduction_model_same_rows"] = {
            "all_rows_correct": doc["datasets"][ds]["test"]["all_rows"]["correct"],
            "certified_gate": [doc["datasets"][ds]["test"]["certified_gate"]["accepted"], doc["datasets"][ds]["test"]["certified_gate"]["correct"]],
            "static_cascade_correct": doc["datasets"][ds]["test"]["static_cascade"]["correct"]}
        if ds == "clinc150":
            per[ds]["oos"] = oos_metrics(cortiq, a.release, work, "release")
    rdoc["datasets"] = per
    rdoc["size_6_7"] = {"bytes": rdoc["model"]["bytes"], "limit": SIZE_LIMIT, "within_limit": rdoc["model"]["bytes"] <= SIZE_LIMIT}
    rdoc["checks"] = {"certified_gate_every_skill": all(per[ds]["gate"]["certified"] for ds in DS),
                      "odd_half_lb_ge_0_95": all((per[ds]["gate"]["odd_half"]["lb"] or 0) >= 0.95 for ds in DS),
                      "size_within_limit": rdoc["size_6_7"]["within_limit"],
                      "speed_thresholds": speed["pass"]}
    rdoc["seconds"] = round(time.time() - t1, 1)
    rdoc["utc_end"] = utc_now()
    rdoc["test_access_log_lines_added"] = sum(1 for _ in open(ACCESS_LOG)) - access_before
    write_json(os.path.join(a.out_dir, "gates-release.json"), rdoc)
    log("wrote gates-release.json: " + json.dumps(rdoc["checks"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
