"""Build the R set: expand families.py, run every reference in the sandbox,
write r.jsonl.

    python -I build_r.py --tldr <dir>/tldr-main --out <run dir>/r.jsonl

For an exec-checked task the reference command runs twice in fresh
containers: the first run fixes the expectation, the second must pass the
check against it (a reference that fails its own check, or a nondeterministic
one, stops the build). Match-checked tasks must accept their own reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import families as F  # noqa: E402
import sandbox as S  # noqa: E402
from cards import card  # noqa: E402

INSTR = {
    "command": "Answer with one shell command.",
    "command_or_none": "Answer with one shell command, or with NONE if no card can do the task.",
    "label": "Answer with one word.",
    "number": "Answer with a number.",
    "name": "Answer with the file name.",
}


def fill(tpl: str, params: dict) -> str:
    for k, v in params.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def instances(params: list[dict], asks: list[str], limit: int) -> list[tuple[dict, str]]:
    out, seen = [], set()
    for shift in range(len(asks)):
        for i, p in enumerate(params):
            key = (i, (i + shift) % len(asks))
            if key not in seen:
                seen.add(key)
                out.append((p, asks[key[1]]))
    return out[:limit]


def cards_text(tldr: Path, specs) -> str:
    texts = [card(tldr, page, keep, f"[{label}]" if label else None) for label, page, keep in specs]
    head = "Reference card:" if len(texts) == 1 else "Reference cards:"
    return head + "\n" + "\n\n".join(texts)


def build(tldr: Path) -> list[dict]:
    tasks: list[dict] = []

    def add(cat, fam, idx, prompt, kind, check, ref=None, setup="", probe="", first=None):
        tasks.append(dict(id=f"{cat}.{fam}.{idx}", cat=cat, family=fam, prompt=prompt, kind=kind,
                          check=check, ref=ref, setup=setup, probe=probe, copy_first=first))

    def first_example(specs) -> str:
        from cards import load_page
        label, page, keep = specs[0]
        ex = load_page(tldr, page)[2]
        return ex[keep[0] if keep else 0][1]

    for f in F.FAMILIES:
        ct = cards_text(tldr, f["cards"])
        kind = "command_or_none" if f["cat"] == "choose" else f["kind"]
        for i, (p, ask) in enumerate(instances(f["params"], f["asks"], f["limit"])):
            parts = [ct]
            if f["context"]:
                parts.append(fill(f["context"], p))
            parts.append("Task: " + fill(ask, p))
            parts.append(INSTR[kind])
            check = dict(f["check"]) if isinstance(f["check"], dict) else {"type": f["check"]}
            add(f["cat"], f["name"], i, "\n\n".join(parts), kind, check, fill(f["ref"], p),
                fill(f["setup"], p), fill(f["probe"], p), first_example(f["cards"]))

    for o in F.OS:
        specs = [(os_name, spec[1], spec[2]) for os_name, spec in o["pages"].items()]
        ct = cards_text(tldr, specs)
        if len(specs) > 1:
            ct = ct.replace("Reference cards:", "Reference card:", 1)
        i = 0
        for os_name, ask, accept, params in o["tasks"]:
            for p in params:
                prompt = "\n\n".join([ct, f"Task: On {os_name}, {fill(ask, p)}.", INSTR["command"]])
                acc = [fill(a, p) for a in accept]
                add("os", o["name"], i, prompt, "command",
                    {"type": "match", "accept": acc, "fold_case": os_name == F.WIN}, acc[0], first=first_example(specs))
                i += 1

    for i, (pages, ask) in enumerate(F.NONE_TASKS):
        specs = [(None, pg, [0, 1, 2]) for pg in pages]
        prompt = "\n\n".join([cards_text(tldr, specs), "Task: " + ask, INSTR["command_or_none"]])
        add("none", "none", i, prompt, "command_or_none", {"type": "label", "accept": ["NONE"]}, "NONE",
            first=first_example(specs))

    for i, (label, cmd, out) in enumerate(F.REACT):
        prompt = "\n\n".join([F.RULES, f"Previous command: {cmd}\nIts output:\n{out}", "Task: Decide the next action.", INSTR["label"]])
        add("react", "rules", i, prompt, "label", {"type": "label", "accept": [label]}, label, first="DONE")

    for i, (page, ask, accept) in enumerate(F.DIRECT):
        prompt = "\n\n".join([cards_text(tldr, [(None, page, None)]), "Task: " + ask, INSTR["command"]])
        add("direct", "direct", i, prompt, "command",
            {"type": "match", "accept": accept, "fold_case": page.startswith("windows/")}, accept[0],
            first=first_example([(None, page, None)]))

    for t in arith(random.Random(20261114)):
        tasks.append(t)
    return tasks


def arith(rng: random.Random) -> list[dict]:
    out = []

    def add(fam, i, cmd, table, q, value, kind="number"):
        prompt = f"Output of `{cmd}`:\n{table}\n\nQuestion: {q}\n{INSTR[kind]}"
        out.append(dict(id=f"arith.{fam}.{i}", cat="arith", family=fam, prompt=prompt, kind=kind,
                        check={"type": "value", "value": value}, ref=str(value), setup="", probe="", copy_first="0"))

    names = ["app.log", "error.log", "access.log", "debug.log", "cron.log", "auth.log", "mail.log", "db.log", "boot.log"]
    for i in range(5):
        fs = rng.sample(names, 6)
        sizes = [rng.randrange(200, 9500) for _ in fs]
        table = "\n".join(f"-rw-r--r-- 1 user user {s:>5} Jan 15 10:00 {n}" for n, s in zip(fs, sizes))
        thr = sorted(sizes)[rng.choice([1, 2, 3, 4])] - rng.randrange(1, 40)
        add("ls_count", i, "ls -l /var/log/app", table, f"How many of these files are larger than {thr} bytes?",
            sum(s > thr for s in sizes))
        a, b = rng.sample(range(6), 2)
        add("ls_sum", i, "ls -l /var/log/app", table, f"What is the combined size in bytes of {fs[a]} and {fs[b]}?",
            sizes[a] + sizes[b])
    for i in range(5):
        fs = rng.sample(names, 4)
        ls = [rng.randrange(12, 900) for _ in fs]
        table = "\n".join(f"{l:>5} {n}" for n, l in zip(fs, ls)) + f"\n{sum(ls):>5} total"
        a, b = rng.sample(range(4), 2)
        add("wc_sum", i, "wc -l *.log", table, f"How many lines do {fs[a]} and {fs[b]} have together?", ls[a] + ls[b])
        add("wc_max", i, "wc -l *.log", table, "Which file has the most lines?", fs[ls.index(max(ls))], kind="name")
    mounts = ["/", "/home", "/var", "/data", "/backup"]
    for i in range(5):
        ms = rng.sample(mounts, 4)
        rows = []
        for j, m in enumerate(ms):
            size = rng.choice([50, 100, 200, 250, 500])
            used = rng.randrange(5, size - 4)
            rows.append((f"/dev/sd{chr(97 + j)}1", size, used, size - used, m))
        table = "Filesystem      Size  Used Avail Use% Mounted on\n" + "\n".join(
            f"{d:<14} {s:>4}G {u:>4}G {a:>4}G {100 * u // s:>3}% {m}" for d, s, u, a, m in rows)
        k = rng.randrange(4)
        add("df_avail", i, "df -h", table, f"How many GB are available on {rows[k][4]}?", rows[k][3])
        a, b = rng.sample(range(4), 2)
        add("df_used_sum", i, "df -h", table, f"How many GB are used on {rows[a][4]} and {rows[b][4]} together?",
            rows[a][2] + rows[b][2])
    dirs = ["/var/cache", "/var/log", "/home/anna", "/opt/app", "/srv/www"]
    for i in range(5):
        ds = rng.sample(dirs, 3)
        ks = rng.sample(range(120, 98000), 3)
        table = "\n".join(f"{k}\t{d}" for k, d in zip(ks, ds))
        a, b = (0, 1) if ks[0] > ks[1] else (1, 0)
        add("du_diff", i, "du -sk " + " ".join(ds), table, f"How many KB larger is {ds[a]} than {ds[b]}?", ks[a] - ks[b])
    users = ["root", "anna", "postgres", "www-data", "boris"]
    procs = ["sshd", "nginx", "postgres", "python3", "bash", "cron", "node", "redis-server"]
    for i in range(5):
        rows = [(rng.choice(users), rng.randrange(300, 9000), rng.choice(procs)) for _ in range(9)]
        table = "USER       PID COMMAND\n" + "\n".join(f"{u:<9} {p:>5} {c}" for u, p, c in rows)
        u = rng.choice(sorted({r[0] for r in rows}))
        add("ps_count", i, "ps -eo user,pid,comm", table, f"How many of these processes belong to the user {u}?",
            sum(r[0] == u for r in rows))
    return out


def fix_expectations(tasks: list[dict]) -> None:
    def one(t):
        ck = t["check"]
        if ck["type"] in S.EXEC_CHECKS:
            r1 = S.run(t["ref"], t["setup"], t["probe"])
            ck["expect"] = S.expectation(ck, r1)
            r2 = S.run(t["ref"], t["setup"], t["probe"])
            ok = S.check_exec(ck, r2)
            empty = ck["type"] in ("out", "outset") and not (ck["expect"].get("lines") or ck["expect"].get("set"))
            empty |= ck["type"] == "num" and ck["expect"]["num"] is None
            empty |= ck["type"] == "post" and not ck["expect"]["probe"]
            return t["id"], ok and not empty and r1["exit"] == 0, r1
        if ck["type"] == "match":
            return t["id"], S.check_match(ck, t["ref"]), None
        if ck["type"] == "label":
            return t["id"], S.check_label(ck, t["ref"]), None
        return t["id"], S.check_value(ck, t["ref"]), None

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, tasks))
    bad = [(i, r) for i, ok, r in res if not ok]
    for i, r in bad:
        print(f"REFERENCE FAILS: {i} {r}", file=sys.stderr)
    if bad:
        raise SystemExit(f"{len(bad)} reference(s) fail their own check")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tldr", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tasks = build(Path(args.tldr))
    ids = [t["id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    fix_expectations(tasks)
    body = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    Path(args.out).write_text(body)
    by = {}
    for t in tasks:
        by[t["cat"]] = by.get(t["cat"], 0) + 1
    print(f"{len(tasks)} tasks: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    print("sha256", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
