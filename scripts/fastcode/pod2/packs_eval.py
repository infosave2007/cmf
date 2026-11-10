# packs_eval.py FITDIR HELDDIR GENERIC.profile: the expert packs measured on held-out text.
# 1) the patent-1 router (FITDIR/router.json): argmin reconstruction error of each held-out window's φ over the
#    packs' topologies, per task and window length (128 / 256 / 512 tokens);
# 2) coverage: the share of a task's held-out routing that lands on the top-K experts per layer of its own pack,
#    of the generic code+task profile the published cut was made by, and of the best other pack.
import base64, json, os, struct, subprocess, sys
try:
    import numpy as np
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy"], check=True)
    import numpy as np

fitdir, helddir, generic = sys.argv[1], sys.argv[2], sys.argv[3]
R = json.load(open(f"{fitdir}/router.json"))
L = R["phi_layers"]

def f16(b64):
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)

def prof(path):
    b = open(path, "rb").read()
    nl, ne, tok = struct.unpack_from("<QQQ", b, 8)
    return np.frombuffer(b, dtype="<u4", count=nl * ne, offset=32).reshape(nl, ne).astype(np.float64)

names = [p["name"] for p in R["packs"]]
subs = []
for p in R["packs"]:
    d = p["descriptor"]
    m = f16(d["mean"]); B = f16(d["basis"]).reshape(d["rank"], -1)
    subs.append((m, B))
ne = len(subs[0][0]) // L

def phi(rows):
    h = rows[:L].astype(np.float64)
    h = h / np.maximum(h.sum(axis=1, keepdims=True), 1.0)
    v = h.reshape(-1).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None

stats = [(p["descriptor"].get("err_mean") or 0.0, p["descriptor"].get("err_std") or 1.0) for p in R["packs"]]
ZS = os.environ.get("ZSCORE") == "1"
def decide(v):
    errs = []
    for (m, B), (em, es) in zip(subs, stats):
        r = v - m
        c = B @ r
        e = float(r @ r - c @ c)
        errs.append((e - em) / es if ZS else e)
    return int(np.argmin(errs))

print("router (held-out texts):")
tot = {1: [0, 0], 2: [0, 0], 4: [0, 0]}
for ti, nm in enumerate(names):
    path = f"{helddir}/{nm}.phi"
    if not os.path.exists(path):
        continue
    nl = 0
    raw = np.fromfile(path, dtype="<u4")
    P = prof(f"{helddir}/{nm}.profile")
    nl, ne_ = P.shape
    rows = raw.reshape(-1, nl, ne_)
    line = []
    for b in (1, 2, 4):
        ok = n = 0
        conf = {}
        for i in range(0, len(rows) - b + 1, b):
            v = phi(rows[i:i + b].sum(axis=0))
            if v is None:
                continue
            k = decide(v)
            n += 1
            if k == ti:
                ok += 1
            else:
                conf[names[k]] = conf.get(names[k], 0) + 1
        tot[b][0] += ok; tot[b][1] += n
        top = ",".join(f"{a}:{c}" for a, c in sorted(conf.items(), key=lambda x: -x[1])[:2])
        line.append(f"{128*b}tok {ok}/{n}" + (f" ({top})" if top else ""))
    print(f"  {nm}: " + "; ".join(line))
print("  total: " + "; ".join(f"{128*b}tok {o}/{n} = {100*o/max(n,1):.1f}%" for b, (o, n) in tot.items()))

G = prof(generic)
F = {nm: prof(f"{fitdir}/{nm}.profile") for nm in names if os.path.exists(f"{fitdir}/{nm}.profile")}
def cover(H, S, K):
    idx = np.argsort(-S, axis=1, kind="stable")[:, :K]
    return float(np.take_along_axis(H, idx, axis=1).sum() / max(H.sum(), 1))
print("coverage of held-out routing by the top-K experts a layer (own pack / generic cut profile / best other pack):")
for K in (16, 27, 48, 64, 128):
    parts = []
    acc = [0, 0, 0]
    for nm in names:
        hp = f"{helddir}/{nm}.profile"
        if nm not in F or not os.path.exists(hp):
            continue
        H = prof(hp)
        own = cover(H, F[nm], K); gen = cover(H, G, K)
        oth = max(cover(H, F[o], K) for o in F if o != nm)
        acc[0] += own; acc[1] += gen; acc[2] += oth
        parts.append(f"{nm} {own*100:.0f}/{gen*100:.0f}/{oth*100:.0f}")
    k = max(len(parts), 1)
    print(f"  K={K}: mean {acc[0]/k*100:.1f}/{acc[1]/k*100:.1f}/{acc[2]/k*100:.1f} | " + " ".join(parts))
