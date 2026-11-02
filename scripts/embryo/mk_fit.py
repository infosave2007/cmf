# Fit sets of the key_span router. general = chat prompts naming a key (capped per key so a word like python cannot fill the topology), no short factual questions.
import json, os, random, re, sys, collections
D = sys.argv[1]; FIT_ALL = sys.argv[2] == '1'
R = os.path.expanduser('~/embryo-data/o1/chat-sft/'); K = os.path.expanduser('~/embryo-data/herbs/skill/')
norm = lambda s: ' '.join(re.sub(r'[^\w]+', ' ', s.lower()).split())
ev = set(norm(json.loads(l)['prompt']) for l in open(K + 'general-eval.jsonl'))
FACT = re.compile(r"^\s*(what|who|when|where|how|why|explain|tell me|define|что|кто|где|когда|как|почему|расскажи|объясни|какой|какая|какое|какие)\b", re.I)
by = collections.defaultdict(list)
GR = json.load(open(os.path.join(D, 'world/groups.json')))
for stem in ['hard-herbs'] + ['hard-' + g for g in GR['groups']]:
    hp = [json.loads(l)['prompt'] for l in open(os.path.join(D, stem + '.keyed.jsonl'))]
    hk = [json.loads(l)['key'] for l in open(os.path.join(D, stem + '.jsonl'))]
    for p, k in zip(hp, hk):
        if norm(p) in ev or (len(p) < 90 and FACT.match(p)):
            continue
        by[k].append(p)
gb = [json.loads(l)['prompt'] for l in open(R + 'router-fit2/general-balanced.jsonl')]
rng = random.Random(17)
gen = []
for k in sorted(by):
    v = sorted(set(by[k])); rng.shuffle(v); gen += v[:10]
# A short question naming a table key is what the records are for: it must not teach the backbone topology.
world_keys = set()
for g in GR['groups']:
    for l in open(os.path.join(D, 'world/entries-%s.jsonl' % g)):
        for k in json.loads(l)['keys']:
            world_keys.add(norm(k))
long_keys = [k for k in world_keys if len(k.split()) >= 3]
def names_key(p):
    n = ' ' + norm(p) + ' '
    if any(' ' + k + ' ' in n for k in long_keys):
        return True
    ws = norm(p).split()
    return any(' '.join(ws[i:i + m]) in world_keys for m in (1, 2) for i in range(len(ws) - m + 1))
gen += [p for p in gb if norm(p) not in ev and not (len(p) < 90 and (FACT.match(p) or any(' ' + k + ' ' in ' ' + norm(p) + ' ' for k in long_keys)) and names_key(p))]
gen = list(dict.fromkeys(gen)); rng.shuffle(gen)
def write(name, rows):
    with open(os.path.join(D, name + '.jsonl'), 'w') as f:
        for p in rows:
            f.write(json.dumps({'prompt': p}, ensure_ascii=False) + '\n')
write('fit-general', gen)
herbs = [json.loads(l)['prompt'] for l in open(R + ('router-key/skill-final.jsonl' if FIT_ALL else 'router-fit2/skill-balanced.jsonl'))]
rng.shuffle(herbs); write('fit-herbs', herbs)
rows = [json.loads(l) for l in open(os.path.join(D, 'world/world-fit-all.jsonl' if FIT_ALL else 'world/world-fit.jsonl'))]
n = {}
for g in GR['groups']:
    v = [r['prompt'] for r in rows if GR['of'][r['kind']] == g]
    rng.shuffle(v); write('fit-' + g, v); n[g] = len(v)
print('general', len(gen), 'herbs', len(herbs), 'world', n, 'keys', len(by))
