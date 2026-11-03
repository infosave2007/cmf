#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw sources -> documents for the tokenizer and the shards: <out>/<source>.jsonl.gz with a "text"
field, every 100th document going to <source>.val.jsonl.gz. Prints documents and characters per
source; the shard step gives the real token count."""
import glob
import gzip
import html
import json
import os
import re
import sys

R = "/root/corpus/raw"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/corpus/docs"
os.makedirs(OUT, exist_ok=True)

FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)
SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.S)
SCRIPT = re.compile(r"<(script|style|nav|footer|header)\b.*?</\1>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")
BLANK = re.compile(r"\n{3,}")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def md(text):
    return BLANK.sub("\n\n", SHORTCODE.sub("", FRONT.sub("", text))).strip()


def strip_html(text):
    text = SCRIPT.sub("", text)
    text = re.sub(r"</(p|div|li|tr|h[1-6]|pre|table)>|<br\s*/?>", "\n", text, flags=re.I)
    return BLANK.sub("\n\n", html.unescape(TAG.sub("", text))).strip()


def files(pattern):
    return sorted(glob.glob(pattern, recursive=True))


def tldr(pages):
    for p in files(f"{R}/tldr/{pages}/*/*.md"):
        yield f"Platform: {os.path.basename(os.path.dirname(p))}\n" + read(p).strip()


def man():
    for p in files("/root/corpus/man/txt/*.txt"):
        t = read(p).strip()
        if len(t) > 200:
            yield t


def se(site):
    with open(f"/root/corpus/jsonl/{site}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield f"Question: {r['title']}\n{r['question']}\n\nAnswer:\n{r['answer']}"


def docs(*patterns, conv=md, min_len=300):
    for pat in patterns:
        for p in files(f"{R}/{pat}"):
            if os.path.isfile(p):
                t = conv(read(p))
                if len(t) >= min_len:
                    yield t


SOURCES = {
    "tldr_en": lambda: tldr("pages"),
    "tldr_ru": lambda: tldr("pages.ru"),
    "man": man,
    "se_unix": lambda: se("unix.stackexchange.com"),
    "se_askubuntu": lambda: se("askubuntu.com"),
    "se_serverfault": lambda: se("serverfault.com"),
    "se_superuser": lambda: se("superuser.com"),
    "se_ru_so": lambda: se("ru.stackoverflow.com"),
    "se_so_ops": lambda: se("stackoverflow.com"),
    "powershell": lambda: docs("powershell-docs/reference/7.5/**/*.md", "powershell-docs/reference/docs-conceptual/**/*.md"),
    "windows_ps": lambda: docs("windows-powershell-docs/docset/winserver2025-ps/**/*.md"),
    "freebsd": lambda: docs("freebsd-doc/documentation/content/en/books/**/*.adoc", "freebsd-doc/documentation/content/ru/books/**/*.adoc"),
    "debian_handbook": lambda: docs("debian-handbook/en-US/*.xml", conv=strip_html),
    "git": lambda: docs("git-docs/Documentation/*.adoc", "git-docs/Documentation/*.txt"),
    "arch_wiki": lambda: docs("arch-wiki/usr/share/doc/arch-wiki/html/en/*.html", conv=strip_html),
    "docker": lambda: docs("docker-docs/content/**/*.md"),
    "kubernetes": lambda: docs("kubernetes/content/en/docs/**/*.md"),
    "python": lambda: docs("cpython/Doc/library/*.rst", "cpython/Doc/using/*.rst", "cpython/Doc/tutorial/*.rst"),
    "hermes": lambda: docs("../../hermes-agent/website/docs/**/*.md", "../../hermes-agent/README.md"),
}

want = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SOURCES)
for name in want:
    n = chars = 0
    with gzip.open(f"{OUT}/{name}.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as tr, \
            gzip.open(f"{OUT}/{name}.val.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as va:
        try:
            for i, t in enumerate(SOURCES[name]()):
                (va if i % 100 == 99 else tr).write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                n += 1
                chars += len(t)
        except FileNotFoundError as exc:
            print(f"{name}: missing {exc.filename}", flush=True)
    print(f"{name}\t{n}\t{chars}", flush=True)
print("BUILD-DONE", flush=True)
