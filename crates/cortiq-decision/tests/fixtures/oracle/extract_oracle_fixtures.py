#!/usr/bin/env python3
"""Extract the WP6 oracle body fixtures: 3 answered calls per dataset from the v4
DeepSeek *dev* ledgers (reports/decision-v4-20260926/oracle/{ds}-dev.part*.jsonl).

For each call the dev text is looked up by its sha256 in the dev split (dev is not
a test split: nothing is appended to test-access.log). The body is rebuilt with the
driver's own `request_body` and must hash to the ledger's `request_sha256` before it
is written. Kept per row: text, text_sha256, request_sha256, request_bytes,
reserved_usd, ledger file and line, and the recorded answer (status, usage, returned
model/provider, choice) so a mock server can replay it. Per dataset: the rubric
(question.json, criteria in file order). Selection (deterministic): among the
answered calls in file order, the first, the middle one and the last.
"""
import hashlib, importlib.util, json, pathlib

ROOT = pathlib.Path('/Users/oleg/dev/cmfpublic')
LEDGERS = ROOT / 'reports/decision-v4-20260926/oracle'
OUT = pathlib.Path(__file__).resolve().parent / 'deepseek_bodies.json'
DATA = {
    'banking77': ROOT / 'artifacts/decision-v2-20260926/splits/banking77',
    'clinc150': ROOT / 'artifacts/decision-clinc150-20260925/data',
    'massive': ROOT / 'artifacts/decision-massive-20260926/data',
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def driver():
    spec = importlib.util.spec_from_file_location('deepseek_oracle', LEDGERS / 'deepseek_oracle.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    d = driver()
    out = {'source': {'driver': 'reports/decision-v4-20260926/oracle/deepseek_oracle.py',
                      'driver_sha256': sha((LEDGERS / 'deepseek_oracle.py').read_bytes()),
                      'selection': 'answered oracle_call records of {ds}-dev.part*.jsonl in file order: first, middle, last'},
           'datasets': {}}
    for ds, base in DATA.items():
        qpath = base / 'question.json'
        question = json.loads(qpath.read_text())
        assert set(question) == {'instructions', 'criteria'}
        texts = {}
        for line in (base / 'dev.jsonl').read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                texts[sha(r['text'].encode())] = r['text']
        calls = []
        for p in sorted(LEDGERS.glob('%s-dev.part*.jsonl' % ds)):
            for n, line in enumerate(p.read_text().splitlines(), 1):
                r = json.loads(line)
                if r.get('record_type') == 'oracle_call' and r['oracle'].get('choice'):
                    calls.append((p.name, n, r))
        pick = [calls[0], calls[len(calls) // 2], calls[-1]]
        rows = []
        for name, n, r in pick:
            text = texts[r['text_sha256']]
            body = d.request_body(question, text)
            assert sha(body) == r['request_sha256'], (ds, name, n)
            assert len(body) == r['request_bytes']
            assert d.reservation(body) == r['reserved_usd']
            o = r['oracle']
            rows.append({'ledger': name, 'line': n, 'text': text, 'text_sha256': r['text_sha256'],
                         'request_sha256': r['request_sha256'], 'request_bytes': r['request_bytes'],
                         'reserved_usd': r['reserved_usd'],
                         'oracle': {k: o.get(k) for k in ('status', 'usage', 'returned_model', 'returned_provider', 'id', 'choice')}})
        out['datasets'][ds] = {'question_file': str(qpath.relative_to(ROOT)),
                               'question_sha256': sha(qpath.read_bytes()),
                               'question': question, 'rows': rows}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    print(OUT, OUT.stat().st_size)


if __name__ == '__main__':
    main()
