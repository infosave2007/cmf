"""Card tasks as conversations in which the model fetches its reference card itself, inside the reasoning loop.

    python -I lookup_gen.py --tasks cards.messages.jsonl --out lookup [--seed 11] [--near P] [--cite] [--weigh]
                            [--ranked] [--need] [--teach P] [--pipes] [--reason]

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
(which line of the card, then what goes where). `--weigh`: whenever more
than one card came back, and before every NONE, the reasoning first says
what each card is for in the card's own words (name and the first sentence
of its description) and only then which one does the task, or that none
does: the choice is written next to what it is made from. It draws nothing
from the generator, so a set with it holds the same conversations as the
set without. The default output is byte-identical to the set the base run's
shard was made from.

`--pipes`: a pipeline is fetched and built like any other answer. Without it
the model announces "two commands" before any card and asks the store for
both BY NAME: it has to know the names already, and on commands it has not
seen it writes names from the training set and never reads what came back
(at half of the base: 0 of R's 27 pipelines, the queries were "l, fmt",
"data, fold"). With it the first query says what is needed, like every other
lookup; the two cards come back together (sometimes with a neighbour), or one
comes back and the model asks for the other by what it does; in one
conversation in four both cards are already in the user turn. The reasoning
says that one card does not do all of it, which card gives the output and
which works on it, then each card's example and its values; the second
example is used without its file, or it is the part after the pipe of the
card's own example.

`--reason`: `react` and `arith` answers come after a line of reasoning that
puts what the answer is read from next to it: the quoted string the output
contains and its rule's label; the rows taken from the table and the
operation on them. Without it they are bare answers, and at half of the base
the model gave one label of its training set to every R rule task (0 of 25)
and a number near the table's (6 of 40).
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
NOTE = {"name": "note", "description": "Save a new card to the local reference store, so that lookup finds it next time: the "
        "command's name, what the example does, and the command with {{placeholders}} for its values.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "what": {"type": "string"},
                                                           "command": {"type": "string"}},
                       "required": ["name", "what", "command"]}}
TOOLS_NOTE = [TOOLS[0], NOTE, TOOLS[1]]
TOLD = ["Use `{c}`.", "`{c}`", "Do it with `{c}`.", "The command is `{c}`.", "No card for it yet. This works: `{c}`."]
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


FILE_END = re.compile(r" \{\{[^{}]*(?:file|path)[^{}]*\}\}$")


def find_piped(card, cmd):
    """(what, template, [(placeholder, value)], how) of the card example the command is the `stdin` form of.

    The form is the example without the file at its end (how = "file") or,
    for an example that is itself a pipeline, the part after its pipe
    ("pipe"). Of several matching examples the one with the most literal text.
    """
    best = None
    for what, tpl in card["ex"]:
        if " | " in tpl:
            form, how = tpl.rsplit(" | ", 1)[1], "pipe"
        elif FILE_END.search(tpl):
            form, how = FILE_END.sub("", tpl), "file"
        else:
            continue
        pairs = match(form, cmd)
        if pairs is None:
            continue
        literal = len(re.sub(r"\{\{.*?\}\}", "", form))
        if best is None or literal > best[0]:
            best = (literal, what, tpl, pairs, how)
    return best[1:] if best else None


LEAD = re.compile(r"^(How do I|I need to|Please|On (Linux|macOS|Windows)[:,])\s+", re.I)


def vague(task: str, rng, tail=False, k=None) -> str:
    """A loose query: a few words of the task, without its values."""
    words = [w for w in LEAD.sub("", LEAD.sub("", task)).rstrip(".?").split() if not re.search(r"[/\\=:@]|\d", w)]
    k = k or rng.randint(3, 5)
    return " ".join(words[-k:] if tail else words[:k]).lower().strip(" ,;—-")


def need_of(task: str) -> str:
    """What the task asks for, without its values: the task's sentence up to where the values start."""
    t = LEAD.sub("", LEAD.sub("", task.split("\n")[0])).strip()
    t = re.split(r": | — |, namely | \(|\. The | for /|\?", t, maxsplit=1)[0].rstrip(".")
    return t[0].lower() + t[1:] if t else t


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
        for w in sorted(set(WORD.findall(text.lower()))):  # sorted: a set's order changes from run to run
            if self.idf.get(w, 0.0) > 1.0:  # a word of a third of the cards says nothing
                for i in self.inv[w]:
                    score[i] += self.idf[w]
        ranked = sorted(score.items(), key=lambda s: (-s[1], s[0]))[: k + 12]  # equal scores: by the pool's order
        top = [self.pool[i] for i, _ in ranked if self.pool[i]["name"] not in avoid][: k + 3]
        return rng.sample(top, k) if len(top) >= k else None


def said(cards, verdict):
    """What each card that came back is for, in the card's words, then the verdict."""
    return " ".join(f"{c['name']}: {c['desc'].split('. ')[0].rstrip('.')}." for c in cards) + " " + verdict


def pattern_think(card, what, tpl, pairs, lead="", cite=False):
    head = f"{lead}Card {card['name']}" + (f", section {card['os']}" if card["os"] else "")
    head += (f', "{what_query(what)}"' if cite and what else "") + f": `{tpl}`."
    if not pairs:
        return head + " No values to fill in."
    return head + " " + "; ".join(f"{ph} → {v}" for ph, v in pairs) + "."


RULE = re.compile(r"^- If the output contains (.+), answer (\S+)\.$")
LS_ROW = re.compile(r"^\S+ +\d+ \S+ \S+ +(\d+) \w+ +\d+ [\d:]+ (\S+)$")
ASKS = [
    ("count", re.compile(r"How many of these files are larger than (\d+) bytes\?")),
    ("sum", re.compile(r"What is the combined size in bytes of (\S+) and (\S+)\?")),
    ("sum", re.compile(r"How many lines do (\S+) and (\S+) have together\?")),
    ("max", re.compile(r"Which file has the most lines\?")),
    ("avail", re.compile(r"How many GB are available on (\S+)\?")),
    ("used", re.compile(r"How many GB are used on (\S+) and (\S+) together\?")),
    ("diff", re.compile(r"How many KB larger is (\S+) than (\S+)\?")),
    ("user", re.compile(r"How many of these processes belong to the user (\S+)\?")),
]


def react_think(prompt: str, answer: str) -> str:
    """Which quoted string of the rules the output contains, and the label of that rule."""
    head, rest = prompt.split("\n\nPrevious command: ", 1)
    out = rest.split("Its output:\n", 1)[1].split("\n\nTask:", 1)[0]
    for line in head.split("\n"):
        m = RULE.match(line)
        for trig in re.findall(r'"([^"]+)"', m.group(1)) if m else ():
            if trig in out:
                if m.group(2) != answer:
                    raise Skip
                return f'The output contains "{trig}": the rule for it says {answer}.'
    other = re.search(r"^- Otherwise answer (\S+)\.$", head, re.M)
    if not other or other.group(1) != answer:
        raise Skip
    return f"The output contains none of the quoted strings: the last rule says {answer}."


def arith_think(prompt: str, answer: str) -> str:
    """The rows the question is about, read from the table, and the operation on them."""
    table, _, question = prompt.partition("\n\nQuestion: ")
    rows = table.split("\n")[1:]
    kind, m = next(((k, rx.search(question)) for k, rx in ASKS if rx.search(question)), (None, None))
    if kind is None:
        raise Skip
    if kind in ("count", "sum", "max"):  # name -> number: `ls -l` sizes or `wc -l` counts
        val = {}
        for r in rows:
            ls = LS_ROW.match(r)
            if ls:
                val[ls.group(2)] = int(ls.group(1))
            elif len(r.split()) == 2 and r.split()[0].isdigit() and r.split()[1] != "total":
                val[r.split()[1]] = int(r.split()[0])
        if kind == "count":
            over = [v for v in val.values() if v > int(m.group(1))]
            think, result = f"Larger than {m.group(1)}: {', '.join(map(str, over))}. That is {len(over)}.", str(len(over))
        elif kind == "sum":
            a, b = val[m.group(1)], val[m.group(2)]
            think, result = f"{m.group(1)}: {a}. {m.group(2)}: {b}. {a} + {b} = {a + b}.", str(a + b)
        else:
            top = max(val, key=val.get)
            think, result = f"The largest count is {val[top]}: {top}.", top
    elif kind in ("avail", "used"):  # df -h: Filesystem Size Used Avail Use% Mounted on
        col = {r.split()[5]: r.split() for r in rows[1:] if len(r.split()) == 6}
        if kind == "avail":
            a = col[m.group(1)][3]
            think, result = f"Row {m.group(1)}: Avail {a}.", a.rstrip("G")
        else:
            a, b = int(col[m.group(1)][2].rstrip("G")), int(col[m.group(2)][2].rstrip("G"))
            think, result = f"{m.group(1)}: Used {a}G. {m.group(2)}: Used {b}G. {a} + {b} = {a + b}.", str(a + b)
    elif kind == "diff":  # du -sk: KB<TAB>directory
        kb = {r.split("\t")[1]: int(r.split("\t")[0]) for r in rows if "\t" in r}
        a, b = kb[m.group(1)], kb[m.group(2)]
        think, result = f"{m.group(1)}: {a}. {m.group(2)}: {b}. {a} - {b} = {a - b}.", str(a - b)
    else:  # ps -eo user,pid,comm
        pids = [r.split()[1] for r in rows[1:] if r.split()[0] == m.group(1)]
        think, result = f"Rows of {m.group(1)}: PID {', '.join(pids)}. That is {len(pids)}.", str(len(pids))
    if result != answer:
        raise Skip
    return think


def build(rec, pool, rng, near=None, p_near=0.0, cite=False, weigh=False, ranked=False, need=False, pipes=False,
          reason=False):
    prompt, answer = rec["messages"][0]["content"], rec["messages"][1]["content"]
    cat = rec["cat"]
    if cat in ("react", "arith"):
        think = (react_think if cat == "react" else arith_think)(prompt, answer) if reason else None
        return "direct", [("user", prompt), ("assistant", think, answer, [])]
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
    which = lambda cs, n: said(cs, f"{n} does it. ") if weigh else f"Of these cards, {n} does it. "
    missed = lambda cs, m: said(cs, m) if weigh else m

    def best_last(cs, best):  # a ranked router: the best match is the last card, the nearest to the answer
        return [c for c in cs if c is not best] + [best] if rng.random() < 0.7 else cs

    if cat == "none":
        if ranked and rng.random() < 0.3:  # the cards are already in the user turn, and none of them fits
            return "none", [("user", f"Reference cards:\n{store(cards)}\n\n{user}"),
                            ("assistant", "None of the cards covers this task.", "NONE", [])]
        turns = [("user", user)]
        pre = f"Need: {need_of(task)}. " if need else ""  # the task is far behind the cards: say what is needed first
        q1 = vague(task, rng)
        turns += [("assistant", rng.choice(NEED), None, [call("lookup", query=q1)]), ("tool", store(cards))]
        last = cards
        if rng.random() < 0.4:
            q2 = vague(task, rng, tail=True)
            more = others(rng.choice([1, 2]), {c["name"] for c in cards}, q2)
            last = more if rng.random() < 0.7 else []
            turns += [("assistant", pre + missed(cards, rng.choice(MISS)), None, [call("lookup", query=q2)]),
                      ("tool", store(last) if last else "No cards found.")]
        end = said(last, "None of them covers the task.") if weigh and last else "None of the cards covers this task."
        turns.append(("assistant", pre + end, "NONE", []))
        return "none", turns

    if cat == "two" and pipes:
        a, _, b = answer.partition(" | ")
        got = None
        for ca, cb in ((cards[0], cards[1]), (cards[1], cards[0])) if len(cards) == 2 else ():
            ha, hb = find([ca], a), find_piped(cb, b)
            if ha and hb and not ca["os"] and not cb["os"]:
                got = (ca, cb, ha, hb)
                break
        if not got:
            raise Skip
        ca, cb, (_, wa, ta, pa), (wb, tb, pb, how) = got
        fill = lambda pairs: " " + "; ".join(f"{ph} → {v}" for ph, v in pairs) + "." if pairs else " No values to fill in."
        reads = (f"`{tb}`, without the file: it reads the output of {ca['name']}." if how == "file"
                 else f"`{tb}`: the part after the pipe, on the output of {ca['name']}.")
        think = (f"One card does not do all of it: {ca['name']} gives the output, {cb['name']} works on it. "
                 f"Card {ca['name']}: `{ta}`.{fill(pa)} Card {cb['name']}: {reads}{fill(pb)} Join them with a pipe.")
        final = lambda pre: ("assistant", pre + think, None, [call("terminal", command=answer)])
        mode = rng.random()
        if mode < 0.25:  # both cards are already in the user turn
            return "two", [("user", f"Reference cards:\n{store(cards)}\n\n{user}"), final("")]
        pre = f"Need: {need_of(task)}. " if need else ""
        q = vague(task, rng, k=rng.randint(5, 9))  # what is needed, as for any lookup: no command names
        turns = [("user", user), ("assistant", rng.choice(NEED), None, [call("lookup", query=q)])]
        both = {ca["name"], cb["name"]}
        if mode < 0.75:  # the router returns both for the need, sometimes with a neighbour
            shown = [ca, cb]
            rng.shuffle(shown)
            if rng.random() < 0.4:
                extra = others(1, both, f"{q} {task}")
                shown = extra + shown if ranked and rng.random() < 0.7 else rng.sample(extra + shown, 3)
            return "two", turns + [("tool", store(shown)), final(pre)]
        if rng.random() < 0.6:  # one of the two came back: ask for the other by what it does
            have, miss, q2 = ca, cb, what_query(wb)
            why = f"Card {ca['name']} gives the output; no card here works on it. Look that up."
        else:
            have, miss, q2 = cb, ca, what_query(wa)
            why = f"Card {cb['name']} works on an output; the command that gives it is not here. Look that up."
        first = [have] + (others(rng.choice([1, 2]), both, f"{q} {task}") if rng.random() < 0.5 else [])
        rng.shuffle(first)
        return "two", turns + [("tool", store(first)), ("assistant", pre + why, None, [call("lookup", query=q2)]),
                               ("tool", miss["text"]), final(pre)]

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
        if ranked and cat != "os" and len(cards) > 1:
            cards = best_last(cards, target)
        lead = ""
        if cat == "os":
            lead = f"The task is for {target['os']}. "
        elif len(cards) > 1:
            lead = which(cards, target["name"])
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
        if ranked and hit:
            shown = best_last(shown, target)
        if (near is not None or ranked) and cat not in ("fix", "os") and hit:  # more than one card came back: say which one
            final = ("assistant", pattern_think(target, what, tpl, pairs, which(shown, name), cite), None,
                     [call("terminal", command=answer)])
    turns = [("user", user)]
    pre = f"Need: {what_query(hit[1])}. " if need and hit else ""
    final = (final[0], pre + final[1], final[2], final[3])
    if mode < (0.36 if ranked else 0.45):  # the first query misses
        k, avoid = rng.choice([1, 2, 3] if ranked else [2, 3]), {c["name"] for c in cards}
        if near is None:  # the order of the draws of the first set
            wrong, need, loose = others(k, avoid), rng.choice(NEED), vague(task, rng)
        else:
            need, loose = rng.choice(NEED), vague(task, rng)
            wrong = others(k, avoid, loose)
        turns += [("assistant", need, None, [call("lookup", query=loose)]), ("tool", store(wrong)),
                  ("assistant", pre + missed(wrong, rng.choice(MISS)), None, [call("lookup", query=query)]), ("tool", store(shown)), final]
        return "retry", turns
    turns += [("assistant", rng.choice(NEED), None, [call("lookup", query=query)]), ("tool", store(shown)), final]
    return "lookup", turns


def note_card(name: str, what: str, tpl: str) -> str:
    """A saved note as the store returns it: a card with one example."""
    return f"{name}: Saved note.\n- {what}: {tpl}"


def build_teach(rec, pool, rng, near=None, need=False):
    """A command the store does not have: NONE, the user gives the command, the model saves a card and runs it."""
    prompt, answer = rec["messages"][0]["content"], rec["messages"][1]["content"]
    cards, ctx, task, _ = split_prompt(prompt)
    hit = find(cards, answer) if rec["cat"] in ("slot", "direct", "choose") and not ctx else None
    if not hit or hit[0]["os"]:
        raise Skip
    target, what, tpl, pairs = hit
    q = what_query(what) if rng.random() < 0.6 else vague(task, rng)
    k = 0 if rng.random() < 0.6 else rng.choice([1, 2, 3])  # mostly the store has nothing near: a plain "No cards found."
    wrong = []
    if k:
        wrong = (near.cards(f"{q} {task}", k, {target["name"]}, rng) if near is not None else None) or \
                [c for c in rng.sample(pool, k + 6) if c["name"] != target["name"]][:k]
    pre = f"Need: {what_query(what)}. " if need else ""
    told = rng.choice(TOLD).format(c=answer if rng.random() < 0.7 else tpl)
    fill = "; ".join(f"{ph} → {v}" for ph, v in pairs) + "." if pairs else "No values to fill in."
    think = f"{pre}A command the reference does not have: save it as a card, then run it. Card {target['name']}: `{tpl}`. {fill}"
    return [("user", task),
            ("assistant", rng.choice(NEED), None, [call("lookup", query=q)]),
            ("tool", "\n\n".join(c["text"] for c in wrong) if wrong else "No cards found."),
            ("assistant", pre + "None of the cards covers this task.", "NONE", []),
            ("user", told),
            ("assistant", think, None, [{"name": "note", "arguments": {"name": target["name"], "what": what, "command": tpl}},
                                        call("terminal", command=answer)])]


def as_note(turns, rec):
    """The card the answer uses comes back as a saved note: one example, no description."""
    cards = split_prompt(rec["messages"][0]["content"])[0]
    hit = find(cards, rec["messages"][1]["content"]) if rec["cat"] in ("slot", "direct", "choose") else None
    if not hit or hit[0]["os"]:
        return turns
    card, what, tpl, _ = hit
    return [(t[0], t[1].replace(card["text"], note_card(card["name"], what, tpl))) if t[0] == "tool" else t for t in turns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--near", type=float, default=0.0, help="share of the extra cards that are the nearest by words, not random")
    ap.add_argument("--cite", action="store_true", help="the reasoning quotes the caption of the chosen example")
    ap.add_argument("--weigh", action="store_true", help="the reasoning says what each returned card is for before choosing")
    ap.add_argument("--ranked", action="store_true",
                    help="the cards come back as from a ranked router (the fitting one last in 7 of 10), a choice among "
                         "several always opens the same way, fewer first queries miss, NONE also with the cards in the user turn")
    ap.add_argument("--need", action="store_true",
                    help="after a tool response the reasoning first says what is needed, in the words of a card's caption")
    ap.add_argument("--pipes", action="store_true",
                    help="a pipeline is looked up by what is needed, not by command names, and built from the cards that "
                         "came back: which gives the output, which works on it, each example and its values")
    ap.add_argument("--reason", action="store_true",
                    help="react and arith answers come after the reasoning they are read from: the matching rule; the rows "
                         "of the table and the operation")
    ap.add_argument("--teach", type=float, default=0.0,
                    help="share of conversations where the store lacks the command, the user gives it after NONE and the model "
                         "saves a card with the note tool; half of the other conversations then list that tool too, and in "
                         "8 of 100 lookups the card comes back as a saved note")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rng2 = random.Random(args.seed + 1009)  # the draws of --teach: the other conversations stay as without it
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
                mode, turns = build(r, pool, rng, near, args.near, args.cite, args.weigh, args.ranked, args.need, args.pipes,
                                    args.reason)
                direct = mode == "direct"
                tools = TOOLS
                if args.teach > 0 and not direct:
                    if rng2.random() < args.teach:
                        try:
                            mode, turns = "teach", build_teach(r, pool, rng2, near, args.need)
                        except Skip:
                            pass
                    if mode == "teach" or rng2.random() < 0.5:
                        tools = TOOLS_NOTE
                    if mode in ("lookup", "retry") and rng2.random() < 0.08:
                        turns = as_note(turns, r)
                text = render(None if direct else SYSTEM, None if direct else tools, turns)
            except Exception as e:
                if not isinstance(e, Skip) and type(e).__name__ != "Skip":
                    raise
                skipped += 1
                continue
            counts[mode] = counts.get(mode, 0) + 1
            msgs = [{"role": role, "content": content}
                    for role, content in render_turns(None if direct else SYSTEM, None if direct else tools, turns)]
            held = int(hashlib.md5(r["group"].encode()).hexdigest()[:8], 16) % 100 == 0
            (fv if held else ft).write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if not held:
                fm.write(json.dumps({"messages": msgs, "group": r["group"], "cat": r["cat"], "mode": mode}, ensure_ascii=False) + "\n")
    print(f"{sum(counts.values())} conversations, {skipped} tasks skipped; modes: {counts}; cards in the pool: {len(pool)}")


if __name__ == "__main__":
    main()
