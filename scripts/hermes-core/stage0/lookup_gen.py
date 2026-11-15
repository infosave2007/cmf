"""Card tasks as conversations in which the model fetches its reference card itself, inside the reasoning loop.

    python -I lookup_gen.py --tasks cards.messages.jsonl --out lookup [--seed 11] [--near P] [--cite]

Input: the tasks of `train_gen.py` (the card is given in the prompt). Output:

    <out>.messages.jsonl   {"messages": [{"role", "content"}], "group", "cat", "mode"} per line: the turns
                           of the contract as they are rendered (tools in the system turn, `<think>` and
                           `<tool_call>` in the assistant's, `<tool_response>` in the tool's), the input
                           of `cortiq-embryo sft-prepare --format messages --group-field group`
    <out>.jsonl.gz         the same conversations as text of the training contract ({"text": ...})
    <out>.val.jsonl.gz     held out by group (1 group in 100)

The contract is that of `datasets/convert.py`: the reference store is a tool,
`lookup`, answered by the engine's router over its records; a command goes
out through `terminal`. Modes of a task:

  lookup   the user turn has no card. The model reasons that it needs one,
           calls `lookup` with a query (the command's name or what it does),
           gets the card (alone, or with one or two unrelated cards, as a
           router returns), reasons over it and answers.
  retry    the first `lookup` returns only unrelated cards; the model says
           so and asks again with a more specific query (the loop).
  context  the card is already in the user turn (the engine's `context`
           mode put it there): no `lookup`.
  two      a pipeline: one `lookup` per command, or one for both.
  none     no card fits: after one or two lookups the answer is NONE.

`react` and `arith` tasks have no card and stay direct answers. The
reasoning is procedural and short, built by this program from the card:
which card, which pattern, which value goes where. No model wrote it.

Without options the cards that come back next to the right one (and the
cards of a missed query) are drawn at random from the pool: unrelated, easy
to tell apart. A search returns neighbours. `--near P`: with probability P
those cards are the pool's nearest to the query by words (command name,
description, example captions, weighted by rarity), the reasoning names the
chosen card whenever more than one came back, and a lone card gets
neighbours four times in five instead of every second time. `--cite`: the
reasoning quotes the caption of the chosen example before its pattern
(which line of the card, then what goes where). The default output is
byte-identical to the set the base run's shard was made from.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))

from convert import render, render_turns  # noqa: E402

SYSTEM = ("You are the command core of an agent. Work from reference cards, not from memory: if the card for a "
          "command is not in the conversation, fetch it with lookup. Run commands with terminal. If no card can do "
          "the task, answer NONE.")
TOOLS = [
    {"name": "lookup", "description": "Fetch reference cards from the local reference store. The query is a command "
     "name or a few words about what is needed.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "terminal", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]
HEAD = re.compile(r"^(?:\[(Linux|macOS|Windows)\] — )?(.+?): (.*)$")
NEED = ["I need the card for this.", "The exact syntax should come from the reference.", "No card in the conversation yet: look it up.",
        "Check the reference before writing the command.", "I should fetch the card first."]
MISS = ["None of these cards covers the task. Ask again, more specifically.", "These cards are about other commands. Try another query.",
        "Not what I need. Look up by a different description."]


class Skip(Exception):
    pass


def parse_card(block: str):
    lines = block.split("\n")
    m = HEAD.match(lines[0])
    if not m or len(lines) < 2 or any(not l.startswith("- ") or ": " not in l for l in lines[1:]):
        return None
    ex = [tuple(l[2:].split(": ", 1)) for l in lines[1:]]
    return {"os": m.group(1), "name": m.group(2), "desc": m.group(3), "ex": ex, "text": block}


def split_prompt(prompt: str):
    """(cards, context blocks, task text, instruction line) of a train_gen prompt."""
    if "\n\nTask: " not in prompt:
        raise Skip
    head, tail = prompt.split("\n\nTask: ", 1)
    task, _, instr = tail.rpartition("\n\n")
    if not task:
        task, instr = tail, ""
    cards, ctx = [], []
    if head.startswith("Reference card"):
        for b in head.split("\n", 1)[1].split("\n\n"):
            c = parse_card(b)
            (cards if c else ctx).append(c or b)
    else:
        ctx = [head]
    return cards, ctx, task, instr


def match(template: str, cmd: str):
    """The placeholders of a card example and their values in the command, or None."""
    parts = re.split(r"\{\{(.*?)\}\}", template)
    rx = "".join(re.escape(p) if i % 2 == 0 else "(.+?)" for i, p in enumerate(parts))
    m = re.fullmatch(rx, cmd)
    return list(zip(parts[1::2], m.groups())) if m else None


def find(cards, cmd):
    """(card, what, template, [(placeholder, value)]) of the example the command instantiates.

    Several examples can match as regular expressions (`base64 {{file}}`
    swallows `-d /x` as a value): the one with the most literal text is the
    one the command was made from. A value that still starts with a dash
    means no example fits.
    """
    best = None
    for c in cards:
        for what, tpl in c["ex"]:
            pairs = match(tpl, cmd)
            if pairs is None:
                continue
            literal = len(re.sub(r"\{\{.*?\}\}", "", tpl))
            if best is None or literal > best[0]:
                best = (literal, c, what, tpl, pairs)
    if best is None:
        return None
    if any(v.startswith("-") and not re.search(r"option|flag|arg", ph) for ph, v in best[4]):
        return None
    return best[1:]


LEAD = re.compile(r"^(How do I|I need to|Please|On (Linux|macOS|Windows)[:,])\s+", re.I)


def vague(task: str, rng, tail=False) -> str:
    """A loose query: a few words of the task, without its values."""
    words = [w for w in LEAD.sub("", LEAD.sub("", task)).rstrip(".?").split() if not re.search(r"[/\\=:@]|\d", w)]
    k = rng.randint(3, 5)
    return " ".join(words[-k:] if tail else words[:k]).lower().strip(" ,;—-")


def what_query(what: str) -> str:
    q = re.sub(r"\[([A-Za-z])\]", r"\1", what).replace("`", "")
    q = re.sub(r"\s*\(.*?\)", "", q).strip().rstrip(".")
    return q[0].lower() + q[1:] if q else q


def call(name, **args):
    return {"name": name, "arguments": args}


WORD = re.compile(r"[a-z0-9]+")


class Near:
    """The pool's cards nearest to a text by words: what a search puts next to the right card."""

    def __init__(self, pool):
        self.pool = pool
        bags = [set(WORD.findall(" ".join([c["name"], c["desc"]] + [w for w, _ in c["ex"]]).lower())) for c in pool]
        df = collections.Counter(w for b in bags for w in b)
        self.idf = {w: math.log(len(pool) / d) for w, d in df.items()}
        self.inv = collections.defaultdict(list)
        for i, b in enumerate(bags):
            for w in b:
                self.inv[w].append(i)

    def cards(self, text, k, avoid, rng):
        score = collections.Counter()
        for w in set(WORD.findall(text.lower())):
            if self.idf.get(w, 0.0) > 1.0:  # a word of a third of the cards says nothing
                for i in self.inv[w]:
                    score[i] += self.idf[w]
        top = [self.pool[i] for i, _ in score.most_common(k + 12) if self.pool[i]["name"] not in avoid][: k + 3]
        return rng.sample(top, k) if len(top) >= k else None


def pattern_think(card, what, tpl, pairs, lead="", cite=False):
    head = f"{lead}Card {card['name']}" + (f", section {card['os']}" if card["os"] else "")
    head += (f', "{what_query(what)}"' if cite and what else "") + f": `{tpl}`."
    if not pairs:
        return head + " No values to fill in."
    return head + " " + "; ".join(f"{ph} → {v}" for ph, v in pairs) + "."


def build(rec, pool, rng, near=None, p_near=0.0, cite=False):
    prompt, answer = rec["messages"][0]["content"], rec["messages"][1]["content"]
    cat = rec["cat"]
    if cat in ("react", "arith"):
        return "direct", [("user", prompt), ("assistant", None, answer, [])]
    cards, ctx, task, _ = split_prompt(prompt)
    if not cards:
        raise Skip
    user = ("\n\n".join(ctx) + "\n\n" if ctx else "") + task
    def others(k, avoid, text=""):
        if near is not None and rng.random() < p_near:
            got = near.cards(text, k, avoid, rng)
            if got:
                return got
        return [c for c in rng.sample(pool, k + 6) if c["name"] not in avoid][:k]

    store = lambda cs: "\n\n".join(c["text"] for c in cs)

    if cat == "none":
        turns = [("user", user)]
        q1 = vague(task, rng)
        turns += [("assistant", rng.choice(NEED), None, [call("lookup", query=q1)]), ("tool", store(cards))]
        if rng.random() < 0.4:
            q2 = vague(task, rng, tail=True)
            more = others(rng.choice([1, 2]), {c["name"] for c in cards}, q2)
            body = store(more) if rng.random() < 0.7 else "No cards found."
            turns += [("assistant", rng.choice(MISS), None, [call("lookup", query=q2)]), ("tool", body)]
        turns.append(("assistant", "None of the cards covers this task.", "NONE", []))
        return "none", turns

    if cat == "two":
        a, _, b = answer.partition(" | ")
        src = [c for c in cards if any(tpl == a for _, tpl in c["ex"])]
        if len(cards) != 2 or len(src) != 1 or not b:
            raise Skip
        ca = src[0]
        cb = cards[1] if cards[0] is ca else cards[0]
        if not b.startswith(cb["name"].split(" ")[0]):
            raise Skip
        final = f"First `{a}` (card {ca['name']}). Its output goes through `{b}` (card {cb['name']}). Join them with a pipe."
        turns = [("user", user)]
        if rng.random() < 0.5:
            turns += [("assistant", "Two commands are needed: get the cards for both.", None,
                       [call("lookup", query=f"{ca['name']}, {cb['name']}")]), ("tool", store(rng.sample([ca, cb], 2)))]
        else:
            turns += [("assistant", "Two steps. First the command that produces the output.", None,
                       [call("lookup", query=ca["name"])]), ("tool", ca["text"]),
                      ("assistant", "Now the filter for that output.", None,
                       [call("lookup", query=rng.choice([cb["name"], what_query(cb["desc"].split(".")[0])]))]), ("tool", cb["text"])]
        turns.append(("assistant", final, None, [call("terminal", command=answer)]))
        return "two", turns

    hit = find(cards, answer)
    if cat == "fix":
        target = hit[0] if hit else cards[0]
        failed = re.search(r"The command `(.*?)`", user)
        if "No such file or directory" in user and failed:
            bad = set(failed.group(1).split()) - set(answer.split())
            good = set(answer.split()) - set(failed.group(1).split())
            if len(bad) != 1 or len(good) != 1:
                raise Skip
            think = (f"`{bad.pop()}` does not exist; the listing has `{good.copy().pop().rsplit('/', 1)[-1]}`. "
                     f"Same command with `{good.pop()}`.")
        elif hit:
            think = pattern_think(*hit, lead="The form for this case is on the card. ", cite=cite)
        elif answer.startswith("sudo "):
            think = "The error is about privileges: run the same command with sudo."
        else:
            raise Skip
        name = target["name"]
    else:
        if not hit:
            raise Skip
        target, what, tpl, pairs = hit
        lead = ""
        if cat == "os":
            lead = f"The task is for {target['os']}. "
        elif len(cards) > 1:
            lead = f"Of these cards, {target['name']} does it. "
        think = pattern_think(target, what, tpl, pairs, lead, cite)
        name = target["name"]

    final = ("assistant", think, None, [call("terminal", command=answer)])
    mode = rng.random()
    if mode < 0.3:  # the card is already in the user turn
        label = "Reference cards" if len(cards) > 1 and cat != "os" else "Reference card"
        return "context", [("user", f"{label}:\n{store(cards)}\n\n{user}"), final]
    query = name if (cat == "fix" or rng.random() < 0.4) else what_query(hit[1] if hit else target["desc"])
    if cat == "os":
        query = rng.choice([f"{query} {target['os']}", query])
    shown = list(cards)
    if len(shown) == 1 and rng.random() < (0.8 if near is not None else 0.5):  # a router returns neighbours too
        shown += others(rng.choice([1, 2]), {name}, f"{query} {task}")
        rng.shuffle(shown)
        if near is not None and cat not in ("fix", "os") and hit:  # more than one card came back: say which one
            final = ("assistant", pattern_think(target, what, tpl, pairs, f"Of these cards, {name} does it. ", cite), None,
                     [call("terminal", command=answer)])
    turns = [("user", user)]
    if mode < 0.45:  # the first query misses
        k, avoid = rng.choice([2, 3]), {c["name"] for c in cards}
        if near is None:  # the order of the draws of the first set
            wrong, need, loose = others(k, avoid), rng.choice(NEED), vague(task, rng)
        else:
            need, loose = rng.choice(NEED), vague(task, rng)
            wrong = others(k, avoid, loose)
        turns += [("assistant", need, None, [call("lookup", query=loose)]), ("tool", store(wrong)),
                  ("assistant", rng.choice(MISS), None, [call("lookup", query=query)]), ("tool", store(shown)), final]
        return "retry", turns
    turns += [("assistant", rng.choice(NEED), None, [call("lookup", query=query)]), ("tool", store(shown)), final]
    return "lookup", turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--near", type=float, default=0.0, help="share of the extra cards that are the nearest by words, not random")
    ap.add_argument("--cite", action="store_true", help="the reasoning quotes the caption of the chosen example")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recs = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    pool, seen = [], set()
    for r in recs:
        if r["cat"] in ("react", "arith"):
            continue
        try:
            for c in split_prompt(r["messages"][0]["content"])[0]:
                if c["name"] not in seen and not c["os"]:
                    seen.add(c["name"])
                    pool.append(c)
        except Skip:
            pass
    near = Near(pool) if args.near > 0 else None
    counts: dict[str, int] = {}
    skipped = 0
    with open(args.out + ".messages.jsonl", "w") as fm, gzip.open(args.out + ".jsonl.gz", "wt", encoding="utf-8") as ft, \
            gzip.open(args.out + ".val.jsonl.gz", "wt", encoding="utf-8") as fv:
        for r in recs:
            try:
                mode, turns = build(r, pool, rng, near, args.near, args.cite)
                direct = mode == "direct"
                text = render(None if direct else SYSTEM, None if direct else TOOLS, turns)
            except Exception as e:
                if not isinstance(e, Skip) and type(e).__name__ != "Skip":
                    raise
                skipped += 1
                continue
            counts[mode] = counts.get(mode, 0) + 1
            msgs = [{"role": role, "content": content}
                    for role, content in render_turns(None if direct else SYSTEM, None if direct else TOOLS, turns)]
            held = int(hashlib.md5(r["group"].encode()).hexdigest()[:8], 16) % 100 == 0
            (fv if held else ft).write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if not held:
                fm.write(json.dumps({"messages": msgs, "group": r["group"], "cat": r["cat"], "mode": mode}, ensure_ascii=False) + "\n")
    print(f"{sum(counts.values())} conversations, {skipped} tasks skipped; modes: {counts}; cards in the pool: {len(pool)}")


if __name__ == "__main__":
    main()
