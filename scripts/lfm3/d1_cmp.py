"""d1 parity: every request of the Transformers reference (d1_ref.py) through
`cortiq system-one`, answers compared field by field: the choices, levels
and spans, and the largest probability difference.

usage: python3 -I d1_cmp.py REF.json MODEL.cmf [CORTIQ]
"""

import json
import os
import subprocess
import sys
import tempfile


def flat(v, path=""):
    """(path, value) leaves of a nested answer."""
    if isinstance(v, dict):
        for k, x in v.items():
            yield from flat(x, f"{path}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from flat(x, f"{path}[{i}]")
    else:
        yield path, v


def main():
    ref_path, model = sys.argv[1], sys.argv[2]
    cortiq = sys.argv[3] if len(sys.argv) > 3 else "/root/lfm3/bin/cortiq"
    worst = 0.0
    for rec in json.load(open(ref_path)):
        with tempfile.TemporaryDirectory() as tmp:
            qpath = os.path.join(tmp, "q.json")
            json.dump(rec["questions"], open(qpath, "w"))
            args = [cortiq, "system-one", model, "--questions", "@" + qpath]
            if rec["state"] is not None:
                spath = os.path.join(tmp, "state.json" if not isinstance(rec["state"], str) else "state.txt")
                with open(spath, "w") as f:
                    f.write(rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"]))
                args += ["--state", "@" + spath]
            if rec.get("image"):
                args += ["--image", rec["image"]]
            if rec.get("audio"):
                args += ["--audio", rec["audio"]]
            out = subprocess.run(args, capture_output=True, text=True, env={**os.environ, "XDG_RUNTIME_DIR": "/tmp"})
        if out.returncode:
            print("FAILED:", out.stderr[-1500:])
            continue
        got = json.loads(out.stdout)["answers"]
        want = rec["result"]["answers"]
        print(f"## state {str(rec['state'])[:50]!r} image={bool(rec.get('image'))} audio={bool(rec.get('audio'))}"
              f"  ({out.stderr.strip().splitlines()[-1]})")
        for name, a in want.items():
            g = got.get(name)
            diffs, same = [], True
            gl, wl = dict(flat(g)), dict(flat(a))
            for k, wv in wl.items():
                gv = gl.get(k, "MISSING")
                if isinstance(wv, float) and isinstance(gv, (int, float)) and not isinstance(gv, bool):
                    d = abs(wv - gv)
                    diffs.append(d)
                elif wv != gv:
                    same = False
                    print(f"  {name}{k}: ref {wv!r} cortiq {gv!r}")
            extra = set(gl) - set(wl)
            if extra:
                print(f"  {name}: extra fields {sorted(extra)}")
            m = max(diffs, default=0.0)
            worst = max(worst, m)
            print(f"  {name}: {'same' if same else 'DIFFERENT'} values, max |dp| {m:.4f}")
    print(f"worst |dp| {worst:.4f}")


if __name__ == "__main__":
    main()
