#!/usr/bin/env python3
"""Word and character error rates of transcript files against the
references of a manifest, with the reference package's own text
normalization (`normalize_raw_text`: lower case, ё → е, punctuation to
spaces), applied to both sides.

    wer.py MANIFEST.tsv NAME=HYP.jsonl [NAME=HYP.jsonl ...] [--field e2e]
           [--against NAME] [--json OUT]

A hypothesis file is JSON lines with `file` and the transcript in `--field`
(`e2e`, `raw`, or `text`); `ref_transcribe.py` and `cmf_transcribe.py` both
write it. `--against NAME` also reports each file's error rate against
that file's transcripts (how far a CMF file is from the reference
implementation, clip for clip) and the share of clips with an identical
normalized transcript.
"""
import argparse, json, os, sys, unicodedata


def normalize(text):
    text = text.replace("ё", "е").replace("Ё", "Е").lower()
    for q in ("’", "‘", "ʻ", "ʼ"):
        text = text.replace(q, "'")
    out = []
    for i, c in enumerate(text):
        if c == "'":
            ok = 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum()
            out.append(c if ok else " ")
        elif c.isalnum() or c.isspace():
            out.append(c)
        elif unicodedata.category(c).startswith("P") and c not in "՛՜՞":
            out.append(" ")
    return " ".join("".join(out).split())


def edits(ref, hyp):
    """Levenshtein distance between two sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def rate(pairs, unit):
    err = tot = 0
    for ref, hyp in pairs:
        r, h = (ref.split(), hyp.split()) if unit == "word" else (list(ref), list(hyp))
        err += edits(r, h)
        tot += len(r)
    return 100.0 * err / max(tot, 1), err, tot


ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("hyps", nargs="+")
ap.add_argument("--field", default="e2e")
ap.add_argument("--against")
ap.add_argument("--json")
a = ap.parse_args()

refs = {}
for line in open(a.manifest, encoding="utf-8"):
    name, _secs, ref = line.rstrip("\n").split("\t")
    refs[name] = normalize(ref)

hyps = {}
for spec in a.hyps:
    name, path = spec.split("=", 1)
    rows = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        text = row.get(a.field)
        if isinstance(text, dict):
            text = text.get("text", "")
        rows[os.path.basename(row["file"])] = normalize(text or "")
    hyps[name] = rows

common = [f for f in refs if all(f in h for h in hyps.values())]
print(f"{len(common)} clips common to every file (manifest {len(refs)}), field '{a.field}'")
report = {"clips": len(common), "field": a.field, "files": {}}
for name, rows in hyps.items():
    w, we, wt = rate([(refs[f], rows[f]) for f in common], "word")
    c, _, _ = rate([(refs[f], rows[f]) for f in common], "char")
    line = f"{name:14s} WER {w:6.3f} % ({we}/{wt})  CER {c:6.3f} %"
    entry = {"wer": round(w, 4), "cer": round(c, 4), "word_errors": we, "words": wt}
    if a.against and a.against in hyps and name != a.against:
        base = hyps[a.against]
        dw, _, _ = rate([(base[f], rows[f]) for f in common], "word")
        same = sum(base[f] == rows[f] for f in common)
        line += f"  | vs {a.against}: WER {dw:6.3f} %, identical {same}/{len(common)}"
        entry.update(vs=a.against, vs_wer=round(dw, 4), identical=same)
    print(line)
    report["files"][name] = entry
if a.json:
    json.dump(report, open(a.json, "w"), ensure_ascii=False, indent=1)
