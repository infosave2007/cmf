#!/usr/bin/env python3
"""Embryo chat SFT mix -> one messages JSONL for
`cortiq-embryo sft-prepare --format messages --group-field group`.

Sources (texts only, no foreign weights):
  * smol-smoltalk (HuggingFaceTB/smol-smoltalk, Apache-2.0): a seeded sample of
    conversations, every conversation its own group;
  * OASST2 ready export (OpenAssistant/oasst2, Apache-2.0): en/ru branches whose
    every assistant turn passes the sft-prepare OASST rule (reviewed, not deleted,
    not synthetic, rank 0 or unranked), grouped by conversation tree;
  * Qwen3.8-27B code answers (t27.messages.jsonl), grouped by prompt;
  * optional Russian: GrandMaster-PRO-MAX (Vikhrmodels/GrandMaster-PRO-MAX, Apache-2.0),
    a seeded sample of conversations with Russian answers.

Run (parquet needs pyarrow):
  uv run --with pyarrow python scripts/embryo/build_sft_mix.py \
    --smol ~/embryo-data/smol-smoltalk/hf --smol-n 60000 \
    --oasst ~/embryo-data/oasst2/hf/2023-11-05_oasst2_ready.messages.jsonl.gz \
    --q27 ~/embryo-data/o1/sft-t27/t27.messages.jsonl --out mix.messages.jsonl
"""
import argparse
import collections
import glob
import gzip
import json
import os
import random

ROLES = {"user", "assistant", "system"}


def oasst_ok_assistant(m):
    return (
        m.get("role") == "assistant"
        and m.get("lang") in ("en", "ru")
        and bool(m.get("review_result"))
        and not m.get("deleted")
        and not m.get("synthetic")
        and (m.get("rank") or 0) == 0
    )


def oasst_records(path):
    msgs = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            msgs[m["message_id"]] = m
    out = []
    for m in msgs.values():
        if not oasst_ok_assistant(m):
            continue
        chain, cur, bad = [], m, False
        while cur is not None:
            if cur.get("deleted") or not (cur.get("text") or "").strip():
                bad = True
                break
            if cur.get("role") == "assistant" and not oasst_ok_assistant(cur):
                bad = True
                break
            chain.append(cur)
            pid = cur.get("parent_id")
            cur = msgs.get(pid) if pid else None
            if pid and cur is None:
                bad = True
                break
        if bad:
            continue
        chain.reverse()
        if chain[0].get("role") != "prompter":
            continue
        out.append(
            {
                "messages": [
                    {"role": "user" if c["role"] == "prompter" else "assistant", "content": c["text"]}
                    for c in chain
                ],
                "lang": m["lang"],
                "group": "oasst:" + m["message_tree_id"],
                "source": "oasst2",
            }
        )
    return out


def smol_records(root, n, seed):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(root, "data", "train-*.parquet")))
    if not files:
        raise SystemExit(f"no train parquet under {root}/data")
    convs = []
    for f in files:
        t = pq.read_table(f)
        cols = t.column_names
        msgs = t.column("messages").to_pylist()
        src = t.column("source").to_pylist() if "source" in cols else [""] * len(msgs)
        for mm, s in zip(msgs, src):
            convs.append((mm, s or ""))
    rng = random.Random(seed)
    idx = list(range(len(convs)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        mm, s = convs[i]
        # long-document QA does not fit a 1024-token window
        if s == "longalign":
            continue
        if not mm or any(m.get("role") not in ROLES or not (m.get("content") or "").strip() for m in mm):
            continue
        if not any(m["role"] == "assistant" for m in mm):
            continue
        out.append(
            {
                "messages": [{"role": m["role"], "content": m["content"]} for m in mm],
                "lang": "en",
                "group": f"smol:{i}",
                "source": "smol:" + s,
            }
        )
    return out, len(convs)


def grandmaster_records(root, n, seed, max_answer_tokens):
    """Russian answers of GrandMaster-PRO-MAX (Vikhrmodels, Apache-2.0; answers written by
    GPT-4-Turbo): conversations whose answers are Russian and not longer than
    `max_answer_tokens` (the dataset's own count), a seeded sample of `n`."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(root, "data", "train-*.parquet")))
    if not files:
        raise SystemExit(f"no train parquet under {root}/data")
    convs = []
    for f in files:
        t = pq.read_table(f, columns=["conversation", "source", "answer_lang", "answer_tokens"])
        for conv, s, lang, at in zip(
            t.column("conversation").to_pylist(),
            t.column("source").to_pylist(),
            t.column("answer_lang").to_pylist(),
            t.column("answer_tokens").to_pylist(),
        ):
            if lang == "ru" and (at or 0) <= max_answer_tokens:
                convs.append((conv, s or ""))
    rng = random.Random(seed)
    idx = list(range(len(convs)))
    rng.shuffle(idx)
    out = []
    for i in idx:
        if len(out) >= n:
            break
        mm, s = convs[i]
        if not mm or any(m.get("role") not in ROLES or not (m.get("content") or "").strip() for m in mm):
            continue
        if not any(m["role"] == "assistant" for m in mm):
            continue
        out.append(
            {
                "messages": [{"role": m["role"], "content": m["content"]} for m in mm],
                "lang": "ru",
                "group": f"gm:{i}",
                "source": "grandmaster:" + s,
            }
        )
    return out, len(convs)


def q27_records(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            v = json.loads(line)
            out.append(
                {
                    "messages": v["messages"],
                    "lang": "code",
                    "group": "q27:" + str(v.get("group", len(out))),
                    "source": "q27",
                }
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smol", required=True)
    ap.add_argument("--smol-n", type=int, default=60000)
    ap.add_argument("--oasst", required=True)
    ap.add_argument("--q27", required=True)
    ap.add_argument("--ru", help="GrandMaster-PRO-MAX dir (Russian answers)")
    ap.add_argument("--ru-n", type=int, default=25000)
    ap.add_argument("--ru-max-answer-tokens", type=int, default=800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    smol, smol_total = smol_records(a.smol, a.smol_n, a.seed)
    oasst = oasst_records(a.oasst)
    q27 = q27_records(a.q27)
    ru, ru_total = grandmaster_records(a.ru, a.ru_n, a.seed, a.ru_max_answer_tokens) if a.ru else ([], 0)
    recs = smol + oasst + q27 + ru
    random.Random(a.seed + 1).shuffle(recs)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_lang = collections.Counter(r["lang"] for r in recs)
    turns = collections.Counter(r["source"].split(":")[0] for r in recs for m in r["messages"] if m["role"] == "assistant")
    print(f"smol-smoltalk: {len(smol)} of {smol_total} conversations")
    print(f"oasst2 en/ru branches: {len(oasst)} ({collections.Counter(r['lang'] for r in oasst)})")
    print(f"q27 code answers: {len(q27)}")
    print(f"grandmaster ru: {len(ru)} of {ru_total} eligible conversations")
    print(f"records {len(recs)}, by lang {dict(by_lang)}, assistant turns by source {dict(turns)} -> {a.out}")


if __name__ == "__main__":
    main()
