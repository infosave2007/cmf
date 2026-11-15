"""Teacher answers and code of the Qwen3.8-27B runs as sources of base shards.

    python -I q27_data.py --src DIR --out DIR

DIR holds the files as they come from the Hub, each repository in its own
directory (the paths are relative to DIR):

  Qwen3.8-27B-Code-14B-Experimental-cmf/data/q27-code-t27.tdata.answers.jsonl
        966 answers of Qwen3.8-27B to 483 coding prompts, two per prompt
  mtp/own.tdata/records.jsonl, mtp/own2.tdata/records.jsonl
        answers of Qwen3.8-27B Turbo (cmf-experiments/q27-mtp-chain/data-own-own2-prompts.tgz, unpacked)
  cmf-experiments/nvg27/chat/answers.jsonl
        teacher answers to the chat prompts of the NVG runs
  cmf-experiments/data/q27/run3/chat3.txt,
  Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf/data/chat4.txt
        the chat text of NVG runs 3 and 4: OpenAssistant oasst1 dialogues
        (en, ru) and teacher answers, ChatML. A dialogue ends at the
        assistant turn that opens with an empty think block (build_chat3.py)
  cmf-experiments/data/q27/code.calib.txt, cmf-experiments/data/q27/code.txt
        source code: Python stdlib, Rust and C crates; cmf's own sources
  cmf-experiments/nvg27/chat/chat_wrap.txt
        its code turns ("Continue this code", "Write code that starts like this")

Writes to --out, one {"text"} per line, a document each:
  q27_answers.jsonl.gz   the teacher's answers, rendered by convert.render (no think block)
  q27_chat.jsonl.gz      the oasst dialogues and the code turns, rendered the same way
  q27_code.jsonl.gz      plain code, cut at blank lines into documents of up to 12 000 characters
and the held-out part of each as *.val.jsonl.gz: for the answers every 40th
prompt with both its answers, else every 100th document. A conversation that
occurs twice (run 4 repeats part of run 3) is written once.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from convert import Skip, render  # noqa: E402

TURN = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)(?:<\|im_end\|>\n|(?=<\|im_start\|>)|\Z)", re.S)
EMPTY_THINK = "<think>\n\n</think>\n\n"
CODE_DOC = 12_000


def held(key: str, every: int) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % every == 0


def dialogues(text: str):
    """Dialogues of a chat text of the NVG runs: lists of (role, content)."""
    turns = []
    for role, body in TURN.findall(text):
        last = role == "assistant" and body.startswith(EMPTY_THINK)
        turns.append((role, body[len(EMPTY_THINK):] if last else body))
        if last:
            if turns[0][0] == "user" and all(a[0] != b[0] for a, b in zip(turns, turns[1:])):
                yield turns
            turns = []


def chat(turns) -> str | None:
    """The dialogue in the contract's frame; None when the renderer refuses it (a special token inside a turn)."""
    try:
        return render(None, None, [(r, t) if r == "user" else (r, None, t, []) for r, t in turns])
    except Skip:
        return None


def code_docs(text: str):
    doc, size = [], 0
    for block in text.split("\n\n"):
        if size + len(block) > CODE_DOC and doc:
            yield "\n\n".join(doc)
            doc, size = [], 0
        doc.append(block)
        size += len(block) + 2
    if doc:
        yield "\n\n".join(doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    stats: dict[str, list[int]] = {}

    def writer(name):
        return gzip.open(out / f"{name}.jsonl.gz", "wt", encoding="utf-8"), gzip.open(out / f"{name}.val.jsonl.gz", "wt", encoding="utf-8")

    def put(files, name, text, val, key=None):
        if text is None:
            stats.setdefault("refused by the renderer", [0, 0])[0] += 1
            return
        key = hashlib.md5((key or text).encode()).hexdigest()
        if key in seen:
            return
        seen.add(key)
        files[1 if val else 0].write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        s = stats.setdefault(name + (" (held out)" if val else ""), [0, 0])
        s[0] += 1
        s[1] += len(text)

    # the teacher's answers
    f = writer("q27_answers")
    for line in (src / "Qwen3.8-27B-Code-14B-Experimental-cmf/data/q27-code-t27.tdata.answers.jsonl").read_text().splitlines():
        r = json.loads(line)
        answer = (r.get("response") or "").strip()
        if answer and r["messages"][-1]["role"] == "user":
            turns = [(m["role"], m["content"]) for m in r["messages"]] + [("assistant", answer)]
            put(f, "27B answers to coding prompts", chat(turns), held(r["id"], 40), "\n".join(t for _, t in turns))
    for part in ("own", "own2"):
        for line in (src / f"mtp/{part}.tdata/records.jsonl").read_text().splitlines():
            r = json.loads(line)
            answer = ((r.get("response") or {}).get("content") or "").strip()
            if answer and r.get("usable", True) and r["messages"][-1]["role"] == "user":
                turns = [(m["role"], m["content"]) for m in r["messages"]] + [("assistant", answer)]
                put(f, "27B Turbo answers", chat(turns), held(r["pid"], 40), "\n".join(t for _, t in turns))
    for line in (src / "cmf-experiments/nvg27/chat/answers.jsonl").read_text().splitlines():
        r = json.loads(line)
        prompt, answer = r.get("prompt") or r.get("question"), r.get("answer") or r.get("response")
        if prompt and isinstance(answer, str) and answer.strip():
            put(f, "teacher answers to chat prompts", chat([("user", prompt), ("assistant", answer.strip())]), False,
                prompt + "\n" + answer.strip())
    for x in f:
        x.close()

    # dialogues: oasst1 and the teacher's answers inside the chat texts; the code turns of the wrapped text
    f = writer("q27_chat")
    for path in ("cmf-experiments/data/q27/run3/chat3.txt", "Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf/data/chat4.txt"):
        for turns in dialogues((src / path).read_text()):
            key = "\n".join(t for _, t in turns)
            put(f, "dialogues of the chat texts", chat(turns), held(key, 100), key)
    for turns in dialogues((src / "cmf-experiments/nvg27/chat/chat_wrap.txt").read_text()):
        if turns[0][1].startswith(("Continue this code:", "Write code that starts like this:")):
            key = "\n".join(t for _, t in turns)
            put(f, "code turns", chat(turns), held(key, 100), key)
    for x in f:
        x.close()

    # plain code
    f = writer("q27_code")
    for path in ("cmf-experiments/data/q27/code.calib.txt", "cmf-experiments/data/q27/code.txt"):
        for doc in code_docs((src / path).read_text()):
            put(f, f"code, {Path(path).name}", doc, held(doc, 100))
    for x in f:
        x.close()

    for name, (n, chars) in stats.items():
        print(f"{name}: {n} documents, {chars / 1e6:.2f} M characters")


if __name__ == "__main__":
    main()
