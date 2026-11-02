#!/usr/bin/env python3
"""tests.json of the Space: fixed-seed samples of the real gates, nothing picked by hand.

usage: build_tests.py <run dir> <out>
"""
import json
import os
import random
import sys

RUN, OUT = sys.argv[1], sys.argv[2]
REL = os.path.join(RUN, "final5-release")
rng = random.Random(53)


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sample(xs, n):
    xs = list(xs)
    rng.shuffle(xs)
    return xs[:n]


def by_lang(xs, n):
    out = []
    for lang in ("en", "ru"):
        out += sample([x for x in xs if x.get("lang") == lang], n // 2)
    return out


KIND = json.load(open(os.path.join(REL, "world/groups.json")))["of"]

plants = by_lang(rows(os.path.join(RUN, "popular/popular.jsonl")) + rows(os.path.join(RUN, "router-key/popular2.jsonl")), 24)
world_all = rows(os.path.join(REL, "world/world-test-held.jsonl"))
world = []
for record in sorted(set(KIND.values())):
    world += sample([x for x in world_all if KIND.get(x.get("kind")) == record], 5)
pos = sample(rows(os.path.join(REL, "fu-dialogs-pos.jsonl")), 12)
neg = sample(rows(os.path.join(REL, "fu-dialogs-neg.jsonl")), 8)
fact = sample(rows(os.path.join(REL, "kl-test-fact.jsonl")), 14)
chat = sample(rows(os.path.join(REL, "kl-test-chat.jsonl")), 14)

sets = [
    dict(id="plants", cases=[dict(turns=[x["prompt"]], expect="record", record="herbs") for x in plants]),
    dict(id="world", cases=[dict(turns=[x["prompt"]], expect="record", record=KIND[x["kind"]]) for x in world]),
    dict(id="followups", cases=[dict(turns=x["turns"], expect="followup", record="herbs") for x in pos] + [dict(turns=x["turns"], expect="not_followup", record="herbs") for x in neg]),
    dict(id="refusal", cases=[dict(turns=[x["prompt"]], expect="refusal") for x in fact] + [dict(turns=[x["prompt"]], expect="not_refusal") for x in chat]),
    dict(id="network", cases=[dict(turns=[t], expect="speed") for t in ("Tell me a story about a dragon.", "Write a poem about the sea.", "Write a Python function that adds two numbers.", "Explain how a bicycle stays upright.", "Расскажи историю о старом маяке.", "Напиши стихотворение о море.")]),
]
json.dump(dict(sets=sets), open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print({s["id"]: len(s["cases"]) for s in sets})
print(sorted({c["record"] for s in sets for c in s["cases"] if c.get("record")}))
