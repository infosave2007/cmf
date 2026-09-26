#!/usr/bin/env python3
"""Extract the WP6 oracle body fixtures of
crates/cortiq-decision/tests/fixtures/oracle/deepseek_bodies.json: 3 answered calls
per dataset from the v4 DeepSeek *dev* ledgers
(reports/decision-v4-20260926/oracle/{ds}-dev.part*.jsonl).

    python3 tools/decision_extract_oracle_fixtures.py [--cmfpublic DIR] [--out FILE]

The ledgers, the driver and the dev splits are read from --cmfpublic (default:
$CMFPUBLIC, else the main checkout of this repository, which holds the local
reports/ and artifacts/).

For each call the dev text is looked up by its sha256 in the dev split (dev is not
a test split: nothing is appended to test-access.log). The body is rebuilt with the
driver's own `request_body` and must hash to the ledger's `request_sha256` before it
is written. Kept per row: text, text_sha256, request_sha256, request_bytes,
reserved_usd, ledger file and line, and the recorded answer (status, usage, returned
model/provider, choice) so a mock server can replay it. Per dataset: the rubric
(question.json, criteria in file order). Selection (deterministic): among the
answered calls in file order, the first, the middle one and the last.
"""
import argparse, hashlib, importlib.util, json, os, pathlib, subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / 'crates/cortiq-decision/tests/fixtures/oracle/deepseek_bodies.json'
ROOT = LEDGERS = DATA = None


def default_cmfpublic():
    """$CMFPUBLIC, else the main checkout of this repository (a worktree's common git dir)."""
    if os.environ.get('CMFPUBLIC'):
        return pathlib.Path(os.environ['CMFPUBLIC'])
    try:
        common = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                capture_output=True, text=True, check=True).stdout.strip()
        return pathlib.Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return REPO


def sha(b):
    return hashlib.sha256(b).hexdigest()


def driver():
    spec = importlib.util.spec_from_file_location('deepseek_oracle', LEDGERS / 'deepseek_oracle.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    global ROOT, LEDGERS, DATA, OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument('--cmfpublic', type=pathlib.Path, default=None)
    ap.add_argument('--out', type=pathlib.Path, default=OUT)
    a = ap.parse_args()
    ROOT = a.cmfpublic or default_cmfpublic()
    OUT = a.out
    LEDGERS = ROOT / 'reports/decision-v4-20260926/oracle'
    DATA = {
        'banking77': ROOT / 'artifacts/decision-v2-20260926/splits/banking77',
        'clinc150': ROOT / 'artifacts/decision-clinc150-20260925/data',
        'massive': ROOT / 'artifacts/decision-massive-20260926/data',
    }
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
