#!/usr/bin/env python3
"""The 50 canonical-JSON fixtures of crates/cortiq-decision/tests/format.rs
(`CANONICAL_FIXTURES`): input = json.dumps with shuffled key order,
ensure_ascii=True, indent None/1/2; expected = json.dumps(sort_keys=True,
separators=(',', ':'), ensure_ascii=False). Fixed objects for numbers, escapes,
Unicode key order, decision-v4 manifests and an oracle body, then seeded random
objects; every fixture is a distinct object (a random draw equal to an earlier
fixture is drawn again).

    python3 tools/decision_canon_fixtures.py OUT.rs.txt

writes the Rust array entries (the body of `CANONICAL_FIXTURES`). The fixtures
of the test were made with CPython 3.9.6.
"""
import json, random, math, struct, sys
random.seed(20260926)

def shuffle_keys(o):
    if isinstance(o, dict):
        items = list(o.items()); random.shuffle(items)
        return {k: shuffle_keys(v) for k, v in items}
    if isinstance(o, list):
        return [shuffle_keys(v) for v in o]
    return o

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

objs = []
objs.append({"b": 1, "a": 2, "c": 3})
objs.append({"z": {"y": {"x": [3, 2, 1]}}, "a": []})
objs.append({})
objs.append([])
objs.append({"": "", "a": {}, "b": [], "c": [{}], "d": [[]]})
objs.append([None, True, False, 0, -0, 1, -1])
objs.append({"ints": [0, 1, -1, 2**31, -2**31, 2**53, 2**53 + 1, 2**63 - 1, -2**63, 2**64 - 1]})
objs.append({"floats": [0.0, -0.0, 1.0, -1.0, 0.5, 0.1, 0.2, 0.3, 1/3, 2/3, 3.141592653589793]})
objs.append({"small": [1e-4, 1e-5, 0.0001, 0.00011, 1.5e-5, 1e-7, 1e-12, 1e-8, 5e-324, 2.2250738585072014e-308]})
objs.append({"large": [1e15, 1e16, 9999999999999998.0, 1e17, 1.5e16, 1e22, 1e23, 1.7976931348623157e308, 123456789012345680000.0]})
objs.append({"ties": [1997107851181081.2, 111275153569243.12, 147117772004750.62, 11815244629156.312, 0.125, 0.375, 2.5, 1e-3]})
objs.append({"f32": [f32(0.024557100608944893), f32(0.7885268330574036), f32(0.8), f32(0.9), f32(0.1), f32(1e-12), f32(3.4028234663852886e38), f32(1.401298464324817e-45)]})
objs.append({"neg": [-1e-5, -1e16, -0.0001, -123.456, -5e-324]})
objs.append({"mixed": [1, 1.0, "1", [1], {"1": 1}]})
objs.append({"escapes": "quote \" backslash \\ slash / newline \n cr \r tab \t bs \b ff \f"})
objs.append({"controls": "".join(chr(c) for c in range(0, 32)) + "\x7f"})
objs.append({"seps": "  line sep   para sep \u0085 nel ﻿ bom"})
objs.append({"cyr": "Привет, мир! Как мне пополнить счёт?", "грек": "ΣΊΣΥΦΟΣ ς", "tr": "İstanbul ıi"})
objs.append({"cjk": "日本語のテキスト 口座の残高を教えて", "한국어": "문장", "中文": ["银行", "卡"]})
objs.append({"emoji": "😀 🎉🎉 🇬🇧 👨‍👩‍👧", "key😀": 1, "keyǅ": 2})
objs.append({"": "private use", "😀": "astral", "￿": "bmp max", "é": "e acute", "é": "combining"})
objs.append({"a": 1, "a\u0000": 2, "a ": 3, "a_": 4, "ab": 5, "A": 6, "B": 7, "_": 8, "~": 9, "0": 10, "10": 11, "9": 12})
objs.append({"Z": 1, "z": 2, "Ä": 3, "ä": 4, "ß": 5, "ẞ": 6, "ﬁ": 7, "Ω": 8, "Ω": 9})
objs.append({"devanagari": "हिन्दी भाषा", "circled": "Ⓐⓑ ①②③", "fractions": "½ ⅷ ² ³ Ⅻ"})
objs.append({"nested": {"level1": {"level2": {"level3": {"level4": {"level5": [1, {"deep": True}]}}}}}})
objs.append([[[[[[[]]]]]], {"a": [{"b": [{"c": None}]}]}])
objs.append({"unicode_escape_input": "\u0001\u001f\u007f\u0080߿ࠀ￿\U00010000\U0010ffff"})
objs.append({"long": "x" * 300 + "é" * 50 + "😀" * 20})
objs.append({"html": "<script>alert('x')</script> & &amp; > <"})
objs.append({"numbers_as_strings": ["1e5", "0.1", "-0", "NaN", "Infinity"]})
objs.append({"bool_keys": {"true": True, "false": False, "null": None}})
objs.append({"list_order": [3, 1, 2, "b", "a", {"b": 1, "a": 2}]})
# decision-v4 shaped objects
objs.append({"schema": "cortiq-decision/1", "profile": "cortiq-decision-ph-v1", "model_id": "cortiq/decision", "name": "Cortiq Decision", "created_unix": 0, "generation": 0, "skills": [{"id": "banking77", "manifest_sha256": "ab" * 32}], "representation_id": "cd" * 32})
objs.append({"signal": {"kind": "concat-v1", "parts": [{"name": "phi_P", "dim": 384, "weight": 1.0}, {"name": "phi_H", "dim": 4096, "weight": 0.5}], "dim": 4480, "renormalize": False}})
objs.append({"config": {"layers": 12, "hidden": 384, "heads": 12, "head_dim": 32, "intermediate": 384, "max_position": 512, "vocab": 30522, "type_vocab": 2, "token_type": 0, "ln_eps": 1e-12, "activation": "gelu_erf"}})
objs.append({"normalizer": {"clean_text": True, "handle_chinese_chars": True, "strip_accents": None, "lowercase": True}, "template": "[CLS] $A [SEP]"})
objs.append({"gate": {"temperature": f32(0.024557100608944893), "novelty_theta": f32(0.7885268330574036), "tau": f32(0.8), "certified": True,
    "rule": {"thresholds": [0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999], "alpha": "0.05/14", "min_accepted": 100, "target": 0.95},
    "evidence": {"even": {"n": 749, "nll": 0.20523418281301234}, "odd": {"n": 749, "grid": [{"t": 0.8, "accepted": 649, "correct": 635, "lb": 0.9580812345678901, "novelty_rejected": 0}], "grid_theta_off": []}}}})
objs.append({"task": {"i": 0, "label": "card_arrival", "state": "active", "origin": "data", "k": 16, "n_train": 91, "err_mean": 0.21000000000000002, "err_std": 0.05, "mean_sha256": "0" * 64, "basis_sha256": None}})
objs.append({"hashing": {"kind": "cortiq-hashfeat-v1", "dim": 4096, "tokens": {"bigram": "{a}_{b}", "bigram_weight": 0.7, "byte_ngrams": [3, 4, 5], "byte_ngram_weight": 0.5, "word_weight": 1.0}, "hash": "fnv1a64", "seed": 0}})
labels = ["card_arrival", "card_linking", "exchange_rate", "top_up_by_cash_or_cheque", "Refund_not_showing_up"]
crit = {l: "The customer asks about " + l.replace("_", " ") + "." for l in labels}
body = {"model": "deepseek/deepseek-v4.1-flash", "temperature": 0, "max_tokens": 64, "reasoning": {"enabled": False}, "stream": False,
        "provider": {"sort": "price", "require_parameters": True, "allow_fallbacks": True, "max_price": {"prompt": 0.1, "completion": 0.5}},
        "messages": [{"role": "system", "content": "Return one typed verdict per question.\n" + json.dumps({"questions": {"task": {"type": "choice", "instructions": "Pick one.", "criteria": crit}}}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
                     {"role": "user", "content": json.dumps({"state": "I still have not received my new card"}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "cmf_verdicts", "strict": True, "schema": {"type": "object", "properties": {"task": {"type": "string", "enum": labels}}, "required": ["task"], "additionalProperties": False}}}}
objs.append(body)
objs.append({"state": {"customer": {"name": "Анна", "tier": "gold", "balance": 1234.5}, "messages": ["hi", "где моя карта?"]}})
objs.append({"questions": {"task": {"type": "score", "instructions": "Rate urgency 0..3", "criteria": {"0": "none", "1": "low", "2": "high", "3": {"note": "critical", "examples": [1, 2]}}}}})
objs.append({"id": "cmf-dec-1790500000-abcdefghij0123456789", "created": 1790500000, "usage": {"input_tokens": 14, "output_tokens": 77, "cost": 5.88e-07}})
objs.append({"probabilities": {"a": 0.9696, "b": 0.0304, "c": 1.2e-08, "d": 0.0}, "confidence": 0.9544})
# random objects
def rnd_scalar():
    t = random.randrange(8)
    if t == 0: return random.randint(-10**12, 10**12)
    if t == 1: return random.uniform(-1e6, 1e6)
    if t == 2: return struct.unpack('<d', struct.pack('<Q', random.getrandbits(62)))[0]
    if t == 3: return random.choice([True, False, None])
    if t == 4: return f32(random.random())
    if t == 5: return random.random() * 10 ** random.randint(-20, 20)
    return "".join(chr(random.choice([random.randint(32, 126), random.randint(0, 31), random.randint(0xa0, 0x2fff), random.randint(0x1f300, 0x1f6ff)])) for _ in range(random.randint(0, 12)))
def rnd_obj(depth):
    if depth == 0 or random.random() < 0.3:
        return rnd_scalar()
    if random.random() < 0.5:
        return [rnd_obj(depth - 1) for _ in range(random.randint(0, 5))]
    return {rnd_scalar() if False else "".join(chr(random.choice([random.randint(32, 126), random.randint(0x400, 0x4ff), random.randint(0x1f600, 0x1f64f), random.randint(0, 31)])) for _ in range(random.randint(0, 6))): rnd_obj(depth - 1) for _ in range(random.randint(0, 6))}
def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
seen = {canon(o) for o in objs}
assert len(seen) == len(objs), "the fixed objects repeat"
while len(objs) < 50:
    o = rnd_obj(4)
    if canon(o) in seen:
        continue
    seen.add(canon(o))
    objs.append(o)
assert len(objs) == 50 and len(seen) == 50

def rust_lit(s):
    out = []
    for ch in s:
        c = ord(ch)
        if ch == '\\': out.append('\\\\')
        elif ch == '"': out.append('\\"')
        elif 32 <= c < 127: out.append(ch)
        else: out.append('\\u{%x}' % c)
    return '"' + ''.join(out) + '"'

lines = []
for i, o in enumerate(objs):
    o2 = shuffle_keys(o)
    indent = [None, 1, 2][i % 3]
    inp = json.dumps(o2, ensure_ascii=True, indent=indent)
    exp = json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    # the shuffled order must not matter
    assert exp == json.dumps(json.loads(inp), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    lines.append('    (\n        %s,\n        %s,\n    ),' % (rust_lit(inp), rust_lit(exp)))
out = sys.argv[1] if len(sys.argv) > 1 else 'canon_fixtures.rs.txt'
open(out, 'w').write('\n'.join(lines) + '\n')
print(len(lines), sum(len(l) for l in lines), file=sys.stderr)
