#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stack Exchange dump -> question and answer JSONL for the Hermes core corpus (CC BY-SA).

Streams Posts.xml, keeps questions with an answer of score >= --min-score (the accepted one when it
qualifies, else the best), strips HTML while keeping code as plain text, and records the source link
that the licence asks for.

usage: corpus_se.py <site> <Posts.xml or -> <out.jsonl> [--tags a,b,c]
--tags keeps only questions with one of the tags, which is how Stack Overflow fits: it streams from
7z and holds in memory only what matches.
"""
import argparse
import html
import json
import re
import sys
import xml.etree.ElementTree as ET

CODE = re.compile(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.S)
INLINE = re.compile(r"<code>(.*?)</code>", re.S)
BLOCK = re.compile(r"</?(p|li|ul|ol|blockquote|h[1-6]|hr|br)[^>]*>")
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\n{3,}")


def clean(body):
    # Code goes through placeholders: its text must not lose lines or look like tags.
    keep = []

    def stash(m):
        keep.append(m.group(1))
        return f"\x00{len(keep) - 1}\x00"

    body = CODE.sub(stash, body)
    body = INLINE.sub(lambda m: "`" + m.group(1) + "`", body)
    body = BLOCK.sub("\n", body)
    body = TAG.sub("", body)
    body = re.sub("\x00(\\d+)\x00", lambda m: "\n```\n" + keep[int(m.group(1))].strip("\n") + "\n```\n", body)
    return SPACE.sub("\n\n", html.unescape(body)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("posts")
    ap.add_argument("out")
    ap.add_argument("--min-score", type=int, default=1)
    ap.add_argument("--tags", default="")
    args = ap.parse_args()

    want = set(args.tags.split(",")) if args.tags else None
    questions, best = {}, {}
    src = sys.stdin.buffer if args.posts == "-" else args.posts
    for _, el in ET.iterparse(src, events=("end",)):
        if el.tag != "row":
            continue
        a = el.attrib
        kind = a.get("PostTypeId")
        if kind == "1" and "ClosedDate" not in a:
            if want is not None and not want.intersection(a.get("Tags", "").strip("|<>").replace("><", "|").split("|")):
                el.clear()
                continue
            questions[a["Id"]] = dict(
                title=a.get("Title", ""), body=a.get("Body", ""), tags=a.get("Tags", ""),
                score=int(a.get("Score", 0)), accepted=a.get("AcceptedAnswerId"))
        elif kind == "2":
            s = int(a.get("Score", 0))
            pid = a.get("ParentId")
            # Dumps go by Id, so a question always comes before its answers.
            if s >= args.min_score and pid in questions:
                cur = best.setdefault(pid, {})
                cur[a["Id"]] = (s, a.get("Body", ""))
                # Only the accepted answer and the top one can ever be used.
                if len(cur) > 2:
                    low = min(cur, key=lambda i: cur[i][0])
                    q = questions.get(pid)
                    if not (q and q["accepted"] == low):
                        del cur[low]
        el.clear()

    n, chars = 0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for qid, q in questions.items():
            answers = best.get(qid)
            if not answers:
                continue
            aid = q["accepted"] if q["accepted"] in answers else max(answers, key=lambda i: answers[i][0])
            rec = dict(
                site=args.site, id=int(qid), url=f"https://{args.site}/q/{qid}",
                title=html.unescape(q["title"]), tags=q["tags"].strip("|<>").replace("><", " ").replace("|", " "),
                score=q["score"], answer_score=answers[aid][0],
                question=clean(q["body"]), answer=clean(answers[aid][1]))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            chars += len(rec["title"]) + len(rec["question"]) + len(rec["answer"])
    print(f"{args.site}: {n} pairs, {chars / 1e9:.2f} G chars, about {chars / 4e9:.2f} G tokens", flush=True)


if __name__ == "__main__":
    main()
