#!/usr/bin/env python3
"""Extract WP5 Jev-shape fixtures: the rubric and 20 stored Jev answers per dataset.

Only the answer objects, usage, model and provider are copied; no text, no label,
no text hash. Selection (deterministic): the first 10 successful result records in
file order, then the first 10 further records whose max probability is below 1.
"""
import datetime, hashlib, json, pathlib

ROOT = pathlib.Path('/Users/oleg/dev/cmfpublic')
OUT = pathlib.Path('/Users/oleg/dev/cmf-decision/crates/cortiq-decision/tests/fixtures/jev')
LOG = ROOT / 'artifacts/decision-v4-20260926/test-access.log'
SOURCES = {
    'banking77': ('reports/decision-banking77-20260925/jev-test.jsonl', True),
    'clinc150': ('reports/decision-clinc150-20260925/jev-calibration.jsonl', False),
    'massive': ('reports/decision-massive-20260926/test-run/latency-jev.jsonl', True),
}


def log_access(path, purpose):
    rec = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z'),
           'file': str(path), 'purpose': purpose}
    with LOG.open('a') as f:
        f.write(json.dumps(rec) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ds, (rel, log) in SOURCES.items():
        path = ROOT / rel
        if log:
            log_access(path, 'decision-v4 WP5 fixtures: 20 stored Jev answer objects (answer/usage/model only; '
                             'no text, label or text hash copied) for the validator and round=2 tests')
        raw = path.read_bytes()
        lines = raw.decode().splitlines()
        run = json.loads(lines[0])
        assert run['record_type'] == 'run'
        q = run['question']
        q = q.get('task', q)
        question = {'type': 'choice', 'instructions': q['instructions'], 'criteria': q['criteria']}
        first, low = [], []
        for line in lines[1:]:
            r = json.loads(line)
            if r.get('record_type') != 'result' or r.get('status') != 200 or 'answer' not in r:
                continue
            if 'error' in r:
                continue
            entry = {'index': r['index'],
                     'response': {'model': r['model'], 'provider': r.get('provider'),
                                  'answers': {'task': r['answer']}, 'usage': r['usage']}}
            if len(first) < 10:
                first.append(entry)
            elif len(low) < 10 and max(r['answer']['probabilities'].values()) < 1:
                low.append(entry)
            if len(first) == 10 and len(low) == 10:
                break
        chosen = first + low
        assert len(chosen) == 20, (ds, len(chosen))
        fx = {
            'dataset': ds,
            'source': {'ledger': rel, 'ledger_sha256': hashlib.sha256(raw).hexdigest(),
                       'selection': 'first 10 successful result records in file order, then the first 10 further '
                                    'records whose max probability is below 1; answer objects copied verbatim'},
            'question': question,
            'responses': chosen,
        }
        (OUT / f'{ds}.json').write_text(json.dumps(fx, ensure_ascii=False, indent=1) + '\n')
        print(ds, len(chosen), sum(1 for c in chosen if max(c['response']['answers']['task']['probabilities'].values()) < 1))
    # The stored multi-type Jev call (reports/decision-system-20260925).
    req = json.loads((ROOT / 'reports/decision-system-20260925/protocol-request.json').read_text())
    res = json.loads((ROOT / 'reports/decision-system-20260925/protocol-jev-native.json').read_text())['result']
    jev = {'model': 'typesafe/jev-1.13-20260917', 'provider': 'TypeSafe',
           'answers': res['answers'], 'usage': res['usage']}
    mt = {'source': {'request': 'reports/decision-system-20260925/protocol-request.json',
                     'response': 'reports/decision-system-20260925/protocol-jev-native.json',
                     'note': 'request: model set to cortiq/decision and the legacy cmf block dropped; response: '
                             'answers and usage as stored (model/provider as every stored Jev row)'},
          'request': {'model': 'cortiq/decision', 'state': req['state'], 'questions': req['questions']},
          'jev_response': jev}
    (OUT / 'multitype.json').write_text(json.dumps(mt, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
