#!/usr/bin/env python3
# Sets of the follow-up decision. `prompts`: the messages and key prompts for phi-pool. `rows`: φ rows in the follow-up form
# unit(unit(key tokens) + alpha·unit(message)) for route-fit-followup. `dialogs`: plain two-turn dialogs of the gates.
# usage: mk_followup.py prompts <dir> | rows <dir> <alpha> | dialogs <dir> <popular.jsonl>
import json, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'followup_phrases.json')))
R = os.path.expanduser('~/embryo-data/o1/chat-sft/')
H = 384
mode, D = sys.argv[1], sys.argv[2]
def write(name, rows):
    with open(os.path.join(D, name), 'w') as f:
        for p in rows:
            f.write(json.dumps({'prompt': p}, ensure_ascii=False) + '\n')
if mode == 'prompts':
    rng = random.Random(5)
    herbs = [json.loads(l)['prompt'] for l in open(os.path.join(D, 'fit-herbs.prompts.jsonl'))]
    rng.shuffle(herbs)
    write('fu-keys.jsonl', herbs[:1500])
    write('fu-pos.jsonl', P['fit'])
    write('fu-neg2.jsonl', P['neg_extra'])
    gen = [json.loads(l)['prompt'] for l in open(R + 'router-fit2/general-balanced.jsonl')]
    gen = [g for g in gen if len(g) < 400]
    rng.shuffle(gen)
    write('fu-neg.jsonl', gen[:3000])
elif mode == 'rows':
    alpha = float(sys.argv[3])
    L = lambda n: np.fromfile(os.path.join(D, n + '.phi.f32'), dtype=np.float32).reshape(-1, H)
    unit = lambda x: x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    keys, pos, neg, neg2 = L('fu-keys'), L('fu-pos'), L('fu-neg'), L('fu-neg2')
    rng = np.random.default_rng(3)
    pick = lambda X, n: X[rng.integers(0, len(X), n)]
    n = 16000
    unit(pick(keys, n) + alpha * pick(pos, n)).astype(np.float32).tofile(os.path.join(D, 'fu-herbs.f32'))
    n1 = n * 7 // 10
    N = np.vstack([unit(pick(keys, n1) + alpha * pick(neg, n1)), unit(pick(keys, n - n1) + alpha * pick(neg2, n - n1))])
    rng.shuffle(N)
    N.astype(np.float32).tofile(os.path.join(D, 'fu-general.f32'))
else:
    plants = {}
    for l in open(sys.argv[3]):
        r = json.loads(l)
        plants.setdefault((r.get('lang'), r.get('src') or r.get('expect_src')), r['prompt'])
    rng = random.Random(11)
    pairs = sorted(k for k in plants if k[1])
    def out(name, lang_msgs):
        with open(os.path.join(D, name), 'w') as f:
            for (lang, src) in pairs:
                msgs = lang_msgs.get(lang) or []
                for m in msgs:
                    f.write(json.dumps({'src': src, 'turns': [plants[(lang, src)], m], 'expect_src': src}, ensure_ascii=False) + '\n')
    out('fu-dialogs-pos.jsonl', {'en': P['test_en'], 'ru': P['test_ru']})
    en = [m for m in P['neg_test'] if not any('а' <= c <= 'я' for c in m.lower())]
    ru = [m for m in P['neg_test'] if m not in en]
    out('fu-dialogs-neg.jsonl', {'en': en, 'ru': ru})
