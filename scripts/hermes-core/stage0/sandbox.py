"""Sandbox execution and answer checks of the R set.

A model-written command runs in a throwaway container of `cortiq-r-sandbox`
(see sandbox/Dockerfile): no network, read-only root, a 64 MB tmpfs at /tmp,
an unprivileged user, 256 MB of memory, 5 s for the command itself. `/data`
is the fixed fixture, `/tmp/work` the only writable place.

Checks (the `check.type` of a task):
  out     stdout equals the reference's (whitespace-normalised lines)
  outset  the set of stdout lines equals the reference's
  num     the first integer of stdout equals the reference's
  hash    the first token of the reference's stdout appears in stdout
  has     every listed token appears in stdout
  post    a probe command run afterwards prints what it prints after the
          reference (file system effects)
  match   no execution: the command equals one of the accepted forms after
          normalisation (quotes, spacing, argument order, optional sudo);
          for macOS and Windows commands, which the Linux sandbox cannot run
  label   the first word is one of the accepted labels
  value   the first number (or the token) of the answer equals the expected
"""

from __future__ import annotations

import base64
import re
import subprocess
import uuid

IMAGE = "cortiq-r-sandbox:1"
EXEC_CHECKS = ("out", "outset", "num", "hash", "has", "post")


def run(cmd: str, setup: str = "", probe: str = "") -> dict:
    """Run `cmd` in a fresh container; returns exit code, stdout, probe output."""
    b64 = base64.b64encode(cmd.encode()).decode()
    script = f"""mkdir -p /tmp/work && cd /tmp/work
{setup}
printf '%s' '{b64}' | base64 -d > /tmp/cmd.sh
timeout -k 1 5 bash /tmp/cmd.sh > /tmp/out 2> /tmp/err < /dev/null
echo "$?" > /tmp/exit
cd /tmp/work 2>/dev/null
{{ {probe or ':'} ; }} > /tmp/probe 2>&1 < /dev/null
cat /tmp/exit
echo '---8<---OUT'
head -c 20000 /tmp/out
echo
echo '---8<---PROBE'
head -c 20000 /tmp/probe
"""
    name = "r-" + uuid.uuid4().hex[:12]
    argv = [
        "docker", "run", "--rm", "-i", "--name", name, "--network", "none",
        "--memory", "256m", "--pids-limit", "128", "--cpus", "1", "--read-only",
        "--tmpfs", "/tmp:rw,exec,size=64m,uid=1000,gid=1000,mode=1777",
        "--user", "1000:1000", IMAGE, "bash", "-s",
    ]
    try:
        p = subprocess.run(argv, input=script.encode(), capture_output=True, timeout=40)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return {"exit": 124, "stdout": "", "probe": ""}
    text = p.stdout.decode("utf-8", "replace")
    head, _, rest = text.partition("---8<---OUT\n")
    out, _, pr = rest.partition("\n---8<---PROBE\n")
    try:
        code = int(head.strip().splitlines()[-1])
    except (ValueError, IndexError):
        code = 125
    return {"exit": code, "stdout": out, "probe": pr}


def norm_lines(s: str) -> list[str]:
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in s.replace("\r", "").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def first_int(s: str) -> str | None:
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else None


def expectation(check: dict, res: dict) -> dict:
    """What the reference run fixes for an exec check."""
    t = check["type"]
    if t == "out":
        return {"lines": norm_lines(res["stdout"])}
    if t == "outset":
        return {"set": sorted(set(l for l in norm_lines(res["stdout"]) if l))}
    if t == "num":
        return {"num": first_int(res["stdout"])}
    if t == "hash":
        return {"token": (res["stdout"].split() or [""])[0]}
    if t == "post":
        return {"probe": norm_lines(res["probe"])}
    return {}


def check_exec(check: dict, res: dict) -> bool:
    t, e = check["type"], check.get("expect", {})
    if t == "out":
        return norm_lines(res["stdout"]) == e["lines"]
    if t == "outset":
        return sorted(set(l for l in norm_lines(res["stdout"]) if l)) == e["set"]
    if t == "num":
        return e["num"] is not None and first_int(res["stdout"]) == e["num"]
    if t == "hash":
        return bool(e["token"]) and e["token"] in res["stdout"]
    if t == "has":
        return all(tok in res["stdout"] for tok in check["tokens"])
    if t == "post":
        return norm_lines(res["probe"]) == e["probe"]
    raise ValueError(t)


def norm_cmd(c: str, fold_case: bool = False) -> list[tuple[str, tuple[str, ...]]]:
    """A command as pipeline segments of (program, sorted arguments)."""
    c = c.strip().strip("`").strip()
    c = re.sub(r"^\$\s+", "", c)
    c = re.sub(r"^sudo\s+", "", c)
    c = c.replace('"', "'")
    c = re.sub(r"'([^' ]*)'", r"\1", c)
    c = re.sub(r"\s*([<>|])\s*", r" \1 ", c)
    c = re.sub(r"\s+", " ", c).strip()
    if fold_case:
        c = c.lower()
    segs = []
    for seg in c.split(" | "):
        toks = seg.split(" ")
        segs.append((toks[0], tuple(sorted(toks[1:]))))
    return segs


def check_match(check: dict, answer: str) -> bool:
    fold = bool(check.get("fold_case"))
    got = norm_cmd(answer, fold)
    return any(got == norm_cmd(a, fold) for a in check["accept"])


def check_label(check: dict, answer: str) -> bool:
    m = re.match(r"\W*([A-Za-z_]+)", answer)
    return bool(m) and m.group(1).upper() in check["accept"]


def check_value(check: dict, answer: str) -> bool:
    exp = str(check["value"])
    if re.fullmatch(r"-?\d+(\.\d+)?", exp):
        m = re.search(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
        return bool(m) and float(m.group(0)) == float(exp)
    return answer.strip().strip("`'\".").split("/")[-1] == exp


def grade(task: dict, answer: str) -> tuple[bool, dict]:
    """Grade one extracted answer; returns (correct, detail)."""
    ck = task["check"]
    t = ck["type"]
    answer = answer.strip()
    if not answer:
        return False, {"why": "empty"}
    if t in EXEC_CHECKS:
        if answer.upper().startswith("NONE"):
            return False, {"why": "none"}
        res = run(answer, task.get("setup", ""), task.get("probe", ""))
        return check_exec(ck, res), {"exit": res["exit"]}
    if t == "match":
        return check_match(ck, answer), {}
    if t == "label":
        return check_label(ck, answer), {}
    if t == "value":
        return check_value(ck, answer), {}
    raise ValueError(t)
