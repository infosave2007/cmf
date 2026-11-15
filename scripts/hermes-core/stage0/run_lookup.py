"""Run a CMF model on the R set inside the lookup loop of the training contract.

    python -I run_lookup.py --cortiq <bin> --model <file.cmf> --tasks r.jsonl --out <name>.outputs.jsonl
        [--mode lookup|context|teach] [--store pool|own] [--ranked] [--k 3] [--max-lookups 3] [--max-tokens 160] [--port 8191] [--limit N]
    python -I run_lookup.py --tasks r.jsonl --selftest
    python -I run_lookup.py --tasks r.jsonl --out mock.outputs.jsonl --mock

The model gets the system text and the tools of `lookup_gen.py` (`lookup`,
`terminal`) and the task without its card. Every turn is one raw completion
of `cortiq serve` over the transcript so far, rendered as in training
(`datasets/convert.py`). A `lookup` call is answered here, a `terminal` call
or a plain line (NONE) is the answer; grade.py grades it as for the other
runners. `react` and `arith` tasks have no card: the R prompt alone, a
plain answer, as in training.

Modes:
  lookup   the user turn has no card: the model has to fetch it (the test of
           the loop).
  context  the card is in the user turn, as the engine's `context` mode puts
           it there: no lookup needed (what the loop costs is the difference).
  teach    learning in operation, over the `slot` and `direct` tasks with the
           `note` tool listed. The first task of a family meets an empty
           store; after the model's answer (NONE is the right one) the user
           gives the reference command, and the model has to save a card
           with `note` and run the command. The other tasks of the family
           then run with a store that holds that note alone (nothing, if no
           note was saved): they pass only through what the model wrote.

The reference store stands in for the engine's router over its records:
  pool     the cards of all R tasks, one per command with the union of the
           examples the tasks show (the sections of an `os` task are cards
           with an OS tag). A query is matched by words: the command name
           first, then the words of the description and of the example
           captions, weighted by rarity. Up to --k cards come back, those
           that score at least half of the best; no common word at all
           gives "No cards found.". For the tasks whose point is that no
           card fits (reference NONE) the store holds only the task's own
           cards: with the pool behind it the task would have an answer.
  own      any query returns the task's own cards (a router that never
           misses): the procedure without the search.
With --ranked the cards come back as the traces of `lookup_gen.py --ranked`
have them, the best match last: `pool` in the reverse of its score order,
`own` with the card the answer needs after the others.

`--selftest` needs no model: over the R tasks it reports how often the store
returns the task's card for the three kinds of query the training traces
use (the command name, the caption of the example, a few words of the task).
`--mock` runs the loop with the reference policy in place of the model (one
lookup by the command's name, then the reference answer): the plumbing, and
with grade.py the grader, without a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "datasets"))

from convert import render  # noqa: E402
from lookup_gen import SYSTEM, TOOLS, TOOLS_NOTE, Skip, find, match, note_card, parse_card, split_prompt, vague, what_query  # noqa: E402

STOP = set("a an the of to in on for and or with from by is are be it its as at all into this that using use given "
           "how do i need please show print get file files".split())
WORD = re.compile(r"[a-z0-9][a-z0-9_+.-]*")
THINK = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ASSISTANT = "<|im_start|>assistant\n"


def words(text: str) -> list[str]:
    out = []
    for w in WORD.findall(text.lower()):
        w = w.strip(".-")
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and w not in STOP:
            out.append(w)
    return out


def card_text(c: dict) -> str:
    return (f"[{c['os']}] — " if c["os"] else "") + f"{c['name']}: {c['desc']}" + "".join(f"\n- {w}: {t}" for w, t in c["ex"])


class Store:
    """The reference store: one record per card; `search` is the router's stand-in."""

    def __init__(self, cards: list[dict]):
        self.cards = cards
        self.names = [set(words(c["name"])) for c in cards]
        self.text = [set(words(" ".join([c["name"], c["os"] or "", c["desc"]] + [x[0] for x in c["ex"]]))) for c in cards]
        df: dict[str, int] = {}
        for t in self.text:
            for w in t:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + len(cards) / n) for w, n in df.items()}

    def search(self, query: str, k: int) -> list[int]:
        """Up to k cards: the share of the card's name the query covers first, then common words by rarity."""
        q = set(words(query))
        scored = []
        for i in range(len(self.cards)):
            s = 10.0 * len(q & self.names[i]) / max(1, len(self.names[i])) + sum(self.idf[w] for w in q & self.text[i])
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [i for s, i in scored[:k] if s >= 0.5 * scored[0][0]]

    def body(self, ids: list[int]) -> str:
        return "\n\n".join(self.cards[i]["text"] for i in ids) or "No cards found."


def task_parts(t: dict):
    """(cards, user turn without the card, user turn with it) of an R task; cards is None for a direct task."""
    if t["cat"] in ("react", "arith"):
        return None, t["prompt"], t["prompt"]
    cards, ctx, task, _ = split_prompt(t["prompt"])
    user = ("\n\n".join(ctx) + "\n\n" if ctx else "") + task
    label = "Reference cards" if len(cards) > 1 and t["cat"] != "os" else "Reference card"
    return cards, user, f"{label}:\n" + "\n\n".join(c["text"] for c in cards) + f"\n\n{user}"


def build_pool(tasks: list[dict]) -> Store:
    """One card per command.

    The tasks show a card with different subsets of its examples, and the
    same command both alone and as a section of an `os` task: the store
    holds one card with the union of the examples, under the OS tag when
    the set has one for it.
    """
    tags: dict[str, set] = {}
    for t in tasks:
        for c in task_parts(t)[0] or []:
            tags.setdefault(c["name"], set()).add(c["os"])
    merged: dict[tuple, dict] = {}
    for t in tasks:
        for c in task_parts(t)[0] or []:
            # one name under two OS tags is two commands (`shutdown` of Linux and of Windows)
            key = (c["name"], c["os"] if len(tags[c["name"]] - {None}) > 1 else None)
            h = merged.setdefault(key, dict(c, ex=[]))
            h["os"] = h["os"] or c["os"]
            h["ex"] += [e for e in c["ex"] if e not in h["ex"]]
    for c in merged.values():
        c["text"] = card_text(c)
    return Store(list(merged.values()))


def wanted(t: dict, cards: list[dict]) -> set:
    """(name, OS tag) of the cards the answer needs: the one the reference instantiates, else all of the task's."""
    hit = find(cards, t["ref"]) if t["cat"] not in ("two", "none") else None
    return {(c["name"], c["os"]) for c in ([hit[0]] if hit else cards)}


def is_one_of(card: dict, want: set) -> bool:
    """The store's card is one of the wanted: same name, and the same OS tag when both carry one."""
    return any(card["name"] == n and (not o or not card["os"] or card["os"] == o) for n, o in want)


def opening(user: str, direct: bool, tools=TOOLS) -> str:
    text = render(None if direct else SYSTEM, None if direct else tools, [("user", user), ("assistant", None, "X", [])])
    return text[: text.rindex(ASSISTANT)] + ASSISTANT


def complete(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=json.dumps({"model": "m", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def tool_calls(text: str) -> list:
    calls = []
    for m in CALL.finditer(text):
        try:
            c = json.loads(m.group(1))
            calls.append((c["name"], c.get("arguments") or {}))
        except (ValueError, KeyError, TypeError):
            calls.append((None, {}))
    return calls


def reference_turn(t: dict, cards, prompt: str, mode: str) -> dict:
    """The reference policy, for --mock: one lookup by the command's name, then the reference answer."""
    if cards is None:
        text = t["ref"]
    elif mode in ("lookup", "teach") and "<tool_response>" not in prompt:
        q = ", ".join(sorted(f"{n} {o}" if o else n for n, o in wanted(t, task_parts(t)[0])))
        text = "<think>\nI need the card for this.\n</think>\n<tool_call>\n" + json.dumps({"name": "lookup", "arguments": {"query": q}}) + "\n</tool_call>"
    elif mode == "teach" and "No cards found." in prompt and "<|im_start|>user\nUse `" not in prompt:
        text = "<think>\nNone of the cards covers this task.\n</think>\nNONE"
    elif mode == "teach" and "No cards found." in prompt:
        hit = find(task_parts(t)[0], t["ref"])
        name, what, tpl = (hit[0]["name"], hit[1], hit[2]) if hit else (t["ref"].split()[0], "Do the task", t["ref"])
        note = {"name": "note", "arguments": {"name": name, "what": what, "command": tpl}}
        text = ("<think>\nSave it, then run it.\n</think>\n<tool_call>\n" + json.dumps(note) + "\n</tool_call>\n<tool_call>\n"
                + json.dumps({"name": "terminal", "arguments": {"command": t["ref"]}}) + "\n</tool_call>")
    elif t["ref"] == "NONE":
        text = "<think>\nNone of the cards covers this task.\n</think>\nNONE"
    else:
        text = "<think>\nThe card has it.\n</think>\n<tool_call>\n" + json.dumps({"name": "terminal", "arguments": {"command": t["ref"]}}) + "\n</tool_call>"
    return {"choices": [{"text": text}], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


def plain_answer(text: str) -> str:
    text = THINK.sub("", text.replace("<|im_end|>", "")).strip()
    line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return re.sub(r"^\$\s+", "", line.strip("`").strip())


def run_task(t: dict, pool: Store, args, given=None, tools=TOOLS) -> dict:
    """One task in the loop. `given`: the cards of the store in place of the task's own (teach mode)."""
    cards, bare, with_card = task_parts(t)
    direct = cards is None
    if given is not None:
        cards = given
    own = None if direct else Store(cards)
    store = own if (args.store == "own" or t["ref"] == "NONE") else pool
    mine = set() if direct else wanted(t, cards) if given is None else {(c["name"], c["os"]) for c in cards}
    prompt = opening(with_card if (args.mode == "context" and not direct) else bare, direct, tools)
    row = {"id": t["id"], "answer": "", "raw": [], "lookups": [], "hits": [], "prompt_tokens": 0, "new_tokens": 0, "finish": ""}
    for _ in range(args.max_lookups + 1):
        row["_prompt"] = prompt
        r = reference_turn(t, cards, prompt, args.mode) if args.mock else complete(args.port, prompt, args.max_tokens)
        text = r["choices"][0]["text"]
        row["raw"].append(text)
        row["prompt_tokens"] += r["usage"]["prompt_tokens"]
        row["new_tokens"] += r["usage"]["completion_tokens"]
        calls = tool_calls(text)
        if not calls:
            if "<tool_call>" in text:
                row["finish"] = "malformed"
            else:
                row["answer"], row["finish"] = plain_answer(text), "plain"
            return row
        term = next((a for n, a in calls if n == "terminal"), None)
        if term is not None:
            row["answer"], row["finish"] = str(term.get("command", "")).strip(), "terminal"
            return row
        if any(n != "lookup" for n, _ in calls):
            row["finish"] = "malformed"
            return row
        prompt += text.strip() + "<|im_end|>\n"
        for _, a in calls:
            q = str(a.get("query", ""))
            ids = list(range(len(own.cards))) if (args.store == "own" and q.strip()) else store.search(q, args.k)
            if args.ranked and args.store == "own":
                ids.sort(key=lambda i: is_one_of(store.cards[i], mine))
            elif args.ranked:
                ids.reverse()
            row["lookups"].append(q)
            row["hits"].append(any(is_one_of(store.cards[i], mine) for i in ids))
            prompt += "<|im_start|>tool\n<tool_response>\n" + store.body(ids) + "\n</tool_response><|im_end|>\n"
        prompt += ASSISTANT
    row["finish"], row["_prompt"] = "no_answer", prompt[: -len(ASSISTANT)]
    return row


def teach_task(t: dict, args) -> dict:
    """An empty store, the model's own answer, then the user gives the command: a note and the command are due."""
    row = run_task(t, None, args, given=[], tools=TOOLS_NOTE)
    prompt = row.pop("_prompt")
    if row["finish"] != "no_answer":
        prompt += row["raw"][-1].strip() + "<|im_end|>\n"
    prompt += f"<|im_start|>user\nUse `{t['ref']}`.<|im_end|>\n" + ASSISTANT
    r = reference_turn(t, [], prompt, "teach") if args.mock else complete(args.port, prompt, args.max_tokens)
    text = r["choices"][0]["text"]
    calls = tool_calls(text)
    note = next((a for n, a in calls if n == "note"), None)
    term = next((a for n, a in calls if n == "terminal"), None)
    card = None
    if note and all(isinstance(note.get(k), str) and note[k].strip() for k in ("name", "what", "command")):
        card = parse_card(note_card(note["name"].strip(), note["what"].strip(), note["command"].strip()))
    hit = find(task_parts(t)[0], t["ref"])
    row.update(first=row["answer"] if row["finish"] in ("plain", "terminal") else row["finish"], phase="teach",
               raw=row["raw"] + [text], note=note, card=card["text"] if card else None,
               # the note is a card of this command whose example gives the reference command back
               note_ok=bool(card and card["name"].split()[0] == t["ref"].split()[0] and match(card["ex"][0][1], t["ref"]) is not None),
               note_general=bool(card and hit and ("{{" in card["ex"][0][1]) == ("{{" in hit[2])),
               answer=str(term.get("command", "")).strip() if term else "" if calls else plain_answer(text),
               finish="terminal" if term else "malformed" if calls else "plain")
    row["prompt_tokens"] += r["usage"]["prompt_tokens"]
    row["new_tokens"] += r["usage"]["completion_tokens"]
    return row


def run_teach(tasks: list[dict], args, emit) -> None:
    fams: dict[tuple, list[dict]] = {}
    for t in tasks:
        if t["cat"] in ("slot", "direct"):
            fams.setdefault((t["cat"], t["family"]), []).append(t)
    args.store = "own"
    for group in fams.values():
        first = teach_task(group[0], args)
        emit(first)
        card = parse_card(first["card"]) if first["card"] else None
        for t in group[1:]:
            row = run_task(t, None, args, given=[card] if card else [], tools=TOOLS_NOTE)
            row.pop("_prompt", None)
            row["phase"] = "reuse"
            emit(row)


def selftest(tasks: list[dict], k: int) -> None:
    pool = build_pool(tasks)
    rng = random.Random(3)
    kinds = {"name": [0, 0], "caption": [0, 0], "task words": [0, 0]}
    sizes = []
    for t in tasks:
        cards, bare, _ = task_parts(t)
        if cards is None or t["ref"] == "NONE" or t["cat"] == "two":
            continue
        hit = find(cards, t["ref"])
        if not hit:
            continue
        target, what = hit[0], hit[1]
        task = bare.rsplit("\n\n", 1)[-1]
        for kind, q in (("name", target["name"]), ("caption", what_query(what)), ("task words", vague(task, rng))):
            ids = pool.search(q, k)
            sizes.append(len(ids))
            kinds[kind][0] += any(is_one_of(pool.cards[i], {(target["name"], target["os"])}) for i in ids)
            kinds[kind][1] += 1
    longest = max(len(c["text"]) for c in pool.cards)
    print(f"store: {len(pool.cards)} cards from {len(tasks)} tasks, the longest {longest} characters; "
          f"up to {k} per query, mean returned {sum(sizes) / len(sizes):.2f}")
    for kind, (h, n) in kinds.items():
        print(f"  query = {kind:<10} the task's card is returned {h}/{n} ({100 * h / n:.1f} %)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cortiq")
    ap.add_argument("--model")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=("lookup", "context", "teach"), default="lookup")
    ap.add_argument("--store", choices=("pool", "own"), default="pool")
    ap.add_argument("--ranked", action="store_true", help="the store returns the best match last")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-lookups", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--port", type=int, default=8191)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mock", action="store_true", help="no model: the reference policy answers (a test of the loop)")
    args = ap.parse_args()
    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if args.selftest:
        selftest(tasks, args.k)
        return
    pool = build_pool(tasks)  # the whole set's cards, also under --limit
    if args.limit:
        tasks = tasks[:: max(1, len(tasks) // args.limit)][: args.limit]
    log = open(args.out + ".serve.log", "w")
    srv = None if args.mock else subprocess.Popen(
        [args.cortiq, "serve", args.model, "--port", str(args.port), "--host", "127.0.0.1"],
        env=dict(os.environ, CMF_GPU="0", CMF_GROWTH="off"), stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(0 if args.mock else 120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=2).read()
                break
            except OSError:
                if srv.poll() is not None:
                    raise SystemExit(f"cortiq serve exited with {srv.returncode}, see {args.out}.serve.log")
                time.sleep(0.5)
        else:
            if not args.mock:
                raise SystemExit("cortiq serve did not come up")
        t0 = time.time()
        rows = []
        with open(args.out, "w") as f:
            def emit(row):
                row.pop("_prompt", None)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if len(rows) % 50 == 0:
                    print(f"{len(rows)} tasks, {time.time() - t0:.0f} s", flush=True)

            if args.mode == "teach":
                run_teach(tasks, args, emit)
            for t in tasks if args.mode != "teach" else []:
                try:
                    row = run_task(t, pool, args)
                except Skip:
                    row = {"id": t["id"], "answer": "", "raw": [], "lookups": [], "hits": [], "prompt_tokens": 0,
                           "new_tokens": 0, "finish": "unparsed_task"}
                emit(row)
    finally:
        if srv is not None:
            srv.terminate()
            try:
                srv.wait(timeout=20)
            except subprocess.TimeoutExpired:
                srv.kill()
        log.close()
    cat = {t["id"]: t["cat"] for t in tasks}
    card = [r for r in rows if cat[r["id"]] not in ("react", "arith")]
    looked = [r for r in card if r["lookups"]]
    fin: dict[str, int] = {}
    for r in rows:
        fin[r["finish"]] = fin.get(r["finish"], 0) + 1
    nl = sum(len(r["lookups"]) for r in card)
    print(f"{len(rows)} tasks in {time.time() - t0:.0f} s, mode {args.mode}, store {args.store}; "
          f"{sum(r['prompt_tokens'] for r in rows)} prompt and {sum(r['new_tokens'] for r in rows)} new tokens")
    print(f"card tasks {len(card)}: with a lookup {len(looked)}, lookups {nl}, the task's card came back in "
          f"{sum(sum(r['hits']) for r in card)} of them, in {sum(any(r['hits']) for r in card)} tasks")
    print("turn endings: " + ", ".join(f"{k} {v}" for k, v in sorted(fin.items())))
    taught = [r for r in rows if r.get("phase") == "teach"]
    if taught:
        ref = {t["id"]: t["ref"] for t in tasks}
        print(f"taught {len(taught)} families: NONE before the command was given {sum(r['first'] == 'NONE' for r in taught)}, "
              f"a note saved {sum(bool(r['card']) for r in taught)}, the note gives the command back {sum(r['note_ok'] for r in taught)}, "
              f"placeholders as on the card {sum(r['note_ok'] and r['note_general'] for r in taught)}, "
              f"the given command run {sum(r['answer'] == ref[r['id']] for r in taught)}; "
              f"{sum(r.get('phase') == 'reuse' for r in rows)} tasks then ran on the notes")


if __name__ == "__main__":
    main()
