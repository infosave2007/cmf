#!/usr/bin/env python3
"""Brief-answer chat mix: every assistant turn cut to its first sentences (a small model spends its capacity on the
direct reply and learns to stop); problem-solving and code answers stay whole when short, else leave the mix.
Run: uv run --with pyarrow python scripts/embryo/build_brief_mix.py --smol ... --oasst ... --q27 ... --ru ... --out brief.messages.jsonl"""
import argparse, collections, json, random, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_mix import grandmaster_records, oasst_records, q27_records, smol_records
SENT = re.compile(r'(?<=[.!?…])["»)]?\s+(?=[A-ZА-ЯЁ"«(0-9])')
SOLVE = re.compile(r'^(to (find|solve|determine|calculate|compute)|let\'s|let us|first,|we (can|need)|давайте|для (того|решения|нахождения)|чтобы (найти|решить)|сначала)', re.I)
FILLER = re.compile(r"^(конечно|разумеется|хорошо|отлично|давайте|рад помочь|с удовольствием|certainly|sure|of course|absolutely|great|okay|ok\\b|i'd be happy|i would be happy|i'm happy|i can help|happy to help|that's a (great|good)|what a (great|good))", re.I)
EMPTY = re.compile(r"^(here's|here is|here are|вот\\b|ниже\\b|следующ|для начала|давайте|sure|certainly|конечно)", re.I)
DROP_SOURCES = ('smol:smol-contraints', 'smol:smollm-rewrite', 'smol:smol-summarize', 'smol:explore-instruct-rewrite')
CYR = re.compile('[А-Яа-яЁё]')
def brief(t, lo=140, hi=480):
    t = t.replace('**', '').strip()
    if t.startswith('#'):
        return None
    if '```' in t:
        return t if len(t) <= 900 else None
    if SOLVE.match(t) and re.search(r'\d', t):
        return t if len(t) <= 700 else None
    paras = [p.strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
    if not paras:
        return None
    p = paras[0]
    if p.endswith(':'):
        if len(paras) < 2:
            return None
        lines, n = [], 0
        for ln in paras[1].split('\n'):
            if n + len(ln) > hi:
                break
            lines.append(ln); n += len(ln)
        return (p + '\n\n' + '\n'.join(lines)) if lines else None
    sents = SENT.split(p)
    while sents and FILLER.match(sents[0]) and len(sents[0]) < 120:
        sents = sents[1:]
    if not sents and len(paras) > 1:
        sents = SENT.split(paras[1])
    if not sents:
        return None
    if len(sents) == 1 and EMPTY.match(sents[0]):
        return None
    out, n = [], 0
    for s in sents:
        if out and n + len(s) > hi:
            break
        out.append(s); n += len(s) + 1
        if n >= lo:
            break
    r = ' '.join(out).strip()
    if r.endswith(':'):
        return None
    return r if len(r) >= 2 and len(r) <= hi + 200 else None
def transform(rec):
    ms = []
    for m in rec['messages']:
        if m['role'] == 'assistant':
            b = brief(m['content'])
            if b is None:
                break
            ms.append({'role': 'assistant', 'content': b})
        else:
            ms.append(m)
    while ms and ms[-1]['role'] != 'assistant':
        ms.pop()
    if not ms:
        return None
    return dict(rec, messages=ms)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smol', required=True); ap.add_argument('--smol-n', type=int, default=300000)
    ap.add_argument('--oasst', required=True); ap.add_argument('--q27', required=True)
    ap.add_argument('--ru', required=True); ap.add_argument('--ru-n', type=int, default=70000)
    ap.add_argument('--out', required=True); ap.add_argument('--seed', type=int, default=19)
    a = ap.parse_args()
    smol, _ = smol_records(a.smol, a.smol_n, a.seed)
    ru, _ = grandmaster_records(a.ru, a.ru_n, a.seed, 2000)
    recs = smol + ru + oasst_records(a.oasst) + q27_records(a.q27)
    def same_lang(r):
        u = [m for m in r['messages'] if m['role'] == 'user'][-1]['content']
        a = r['messages'][-1]['content']
        return (len(CYR.findall(u)) / max(1, len(u)) > .3) == (len(CYR.findall(a)) / max(1, len(a)) > .3)
    recs = [r for r in recs if not r['source'].startswith(DROP_SOURCES)]
    out = [r for r in map(transform, recs) if r and same_lang(r)]
    random.Random(a.seed + 1).shuffle(out)
    with open(a.out, 'w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('records', len(recs), '→', len(out), collections.Counter(r['lang'] for r in out), 'assistant turns', sum(1 for r in out for m in r['messages'] if m['role'] == 'assistant'))
    L = sorted(len(m['content']) for r in out for m in r['messages'] if m['role'] == 'assistant')
    print('answer chars pct', {p: L[int(len(L) * p / 100)] for p in (10, 25, 50, 75, 90)})
main()
