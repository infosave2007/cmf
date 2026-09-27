#!/usr/bin/env python3
"""Fixtures of the native decision encoder (spec decision-v4 §6.1), written to
crates/cortiq-decision/tests/fixtures/toy/:

* ``encoder/`` — a toy BERT export directory in the exact format of
  `tools/decision_export_encoder.py` (2 layers, hidden 32, 4 heads × 8,
  intermediate 48, 24 positions, a 142-token WordPiece vocab, truncation at 20),
  random weights from a fixed seed, a toy `tokenizer.json` built with HF
  `tokenizers`;
* ``encoder_golden.json`` — for 12 texts: the HF `tokenizers` ids of the toy
  tokenizer, the numpy float64 `last_hidden_state` (the reference forward of
  `decision_export_encoder.forward_f64`, from the f32 weights) and φ_P in float64;
* ``wordpiece_golden.json`` + ``bge-small-en-v1.5-vocab.txt`` — the release
  tokenizer (vocab.txt of the release encoder) on the golden cases below,
  ids by HF `tokenizers` (the version that made the stored features, 0.22.2),
  with the stress-set cases marked; cases the native tokenizer does not
  reproduce would be listed in ``exceptions`` with the reason.

    python3 tools/mk_decision_toy.py \\
        --tokenizer-dir .../registry_bake/encoder_tokenizer [--out DIR]

Deterministic: the same inputs and library versions give the same bytes.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_export_encoder as dx  # noqa: E402

OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates",
                           "cortiq-decision", "tests", "fixtures", "toy")
SEED = 20260926
CFG = {"layers": 2, "hidden": 32, "heads": 4, "head_dim": 8, "intermediate": 48,
       "max_position": 24, "vocab": None, "type_vocab": 2, "token_type": 0, "ln_eps": 1e-12,
       "activation": "gelu_erf"}
MAX_LENGTH = 20
TOY_SOURCE_NAME = "toy BERT (tools/mk_decision_toy.py, random weights, no ONNX source)"

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = ["the", "card", "top", "up", "my", "how", "do", "i", "money", "cafe", "creme", "un",
         "pin", "account", "transfer", "rate", "exchange", "is", "not", "working", "why", "was",
         "declined", "new", "get", "a", "still", "have", "received", "what", "s", "hello", "world",
         "野", "口", "привет", "мир", "σας", "istanbul"]
PIECES = ["##aff", "##able", "##s", "##ing", "##ed", "##e", "##er", "##ly", "##ment", "##n", "##t"]
TOY_TEXTS = [
    "",
    "How do I top up my card?",
    "Café crème naïve ÀÉÎ",
    "my card was declined!!!",
    "unaffable [MASK] transfer",
    "野口 world, привет мир",
    "ΣΑΣ İstanbul",
    "what's the exchange rate?",
    "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    "xyzzyqq unknownword",
    "top\tup\nmy card​ now",
    "[CLS] hello [SEP] world [UNK]",
]


def toy_vocab():
    ascii_punct = [chr(c) for c in range(33, 127) if not chr(c).isalnum()]
    letters = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    digits = [str(d) for d in range(10)]
    cont = ["##" + c for c in letters]
    toks = SPECIALS + ascii_punct + digits + letters + cont + PIECES + WORDS
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def toy_tokenizer(vocab):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    v = {t: i for i, t in enumerate(vocab)}
    tok = Tokenizer(models.WordPiece(vocab=v, unk_token="[UNK]", max_input_chars_per_word=100))
    tok.normalizer = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True,
                                                strip_accents=None, lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", v["[CLS]"]), ("[SEP]", v["[SEP]"])])
    tok.add_special_tokens(SPECIALS)
    tok.enable_truncation(max_length=MAX_LENGTH)
    return tok


def toy_weights(cfg, rng):
    W = {}
    for name, shape in dx.weight_shapes(cfg):
        if name.endswith("LayerNorm.weight"):
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name.endswith("LayerNorm.bias") or name.endswith(".bias"):
            a = 0.05 * rng.standard_normal(shape)
        elif name == "embeddings.word_embeddings.weight":
            a = 0.5 * rng.standard_normal(shape)
        elif name.startswith("embeddings."):
            a = 0.1 * rng.standard_normal(shape)
        else:  # [out, in]
            a = rng.standard_normal(shape) / np.sqrt(shape[1])
        W[name] = a.astype("<f4")
    return W


# ------------------------------------------------------------------ wordpiece golden cases

ENGLISH = [
    "I still have not received my new card",
    "How do I top up my account with a cheque?",
    "what's the exchange rate for EUR -> USD?",
    "PIN blocked!!!",
    "Why was my card declined at the supermarket yesterday?",
    "Can I change my PIN at an ATM?",
    "I want to cancel a transfer I made by mistake.",
    "My top-up failed, but the money left my account.",
    "Where can I find my IBAN and BIC/SWIFT code?",
    "Is there a fee for exchanging GBP to JPY?",
    "the app keeps crashing when I open 'Cards'",
    "Hi, I'm locked out of my account :(",
    "Set an alarm for 6:30am tomorrow",
    "wake me up at seven thirty",
    "what's the weather like in San Francisco this weekend",
    "play some jazz music in the living room",
    "turn off the lights in the kitchen",
    "how many calories are in a banana",
    "book a table for two at 8 pm at Pizzeria Delfina",
    "are you a bot or a real person?",
    "tell me a joke about programmers",
    "what is my credit score",
    "remind me to call mom on Sunday",
    "translate 'good morning' into Spanish",
    "How long does a SEPA transfer take (in business days)?",
    "I'd like a refund for order #12345, please.",
    "Contactless isn't working on my card; what should I do?",
    "Send $50.00 to John via Apple Pay",
    "My card number is 1234-5678-9012-3456 — is that safe to share?",
    "Email me at jane.doe+bank@example.co.uk",
    "Visit https://www.example.com/help?topic=cards&lang=en for details",
    "The rate is 1.2345 (or ~1.23) as of 2024-01-01T12:00:00Z",
    "Well... I don't know -- maybe?!",
    "C'est la vie; l'argent n'est pas tout.",
    "state-of-the-art, well-known, e-mail, re-enter",
    "UPPERCASE lowercase MiXeDcAsE CamelCaseWord",
    "numbers: 0 1 2 3 4 5 6 7 8 9 10 100 1000 3.14159 -42 +7",
    "hash#tag @mention $dollar %percent ^caret &amp *star",
    "brackets (round) [square] {curly} <angle>",
    "quotes \"double\" 'single' `back` «guillemets» „low” ‘curly’",
    "slashes / \\ | and tilde ~ underscore_word",
    "unaffable unbelievable antidisestablishmentarianism",
    "don't won't can't shouldn't y'all o'clock",
    "Mr. Smith Jr., Ph.D., U.S.A., e.g., i.e., etc.",
    "a1b2c3 x86_64 utf-8 base64 sha256 v0.7.8",
    "I'm travelling to Zürich, München and Köln next week.",
    "fiancée résumé naïve café coöperate façade",
    "¿Dónde está mi tarjeta? ¡Necesito ayuda!",
    "Ich möchte Geld überweisen, aber die Überweisung schlägt fehl.",
    "Tôi muốn chuyển tiền sang tài khoản khác.",
    "Je voudrais recharger ma carte à l'étranger.",
    "O cartão foi recusado no caixa eletrônico.",
    "Hvordan endrer jeg PIN-koden min?",
    "Jak mogę zmienić mój kod PIN?",
    "Nasıl para yatırabilirim?",
]

STRESS = [
    # Cyrillic
    "Привет, мир! Как мне пополнить счёт?",
    "ЁЛКА ёлка Йогурт йогурт",
    "Здравствуйте, карта заблокирована.",
    # Greek, final sigma
    "ΣΑΣ σας ΟΔΥΣΣΕΥΣ Ὀδυσσεύς",
    "Καλημέρα, πώς είστε;",
    "ΆΈΉΊΌΎΏ ΐΰ",
    # Turkish dotted/dotless i
    "İstanbul IĞDIR ıslak İi Iı",
    "KİTAP kitap",
    # German sharp s
    "Straße STRASSE ẞ ß",
    # Devanagari and other Indic
    "नमस्ते दुनिया, मेरा खाता",
    "क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़",
    "বাংলা தமிழ் తెలుగు ಕನ್ನಡ",
    # Arabic, Hebrew
    "مرحبا بالعالم، أين بطاقتي؟",
    "שלום עולם",
    # CJK, kana, hangul
    "日本語のテキスト 口座の残高を教えて",
    "我想查询我的银行卡余额。",
    "한국어 텍스트 계좌 잔액",
    "ｶﾀｶﾅ カタカナ ひらがな",
    "〇一二三 ㈱ ㍿",
    "𠀀𠀁 CJK extension B 𪜀 𫝀 𫠠 𬺰",
    # Thai
    "สวัสดีครับ",
    # circled / enclosed / styled letters
    "Ⓐⓑⓒ ①②③ ⑴⑵ ⒜⒝",
    "𝐀𝐁𝐂 𝑎𝑏𝑐 𝔄𝔅 𝕏𝕐 𝟘𝟙𝟚",
    "ＡＢＣ ｄｅｆ １２３ ！？",
    "ﬁnance ﬂoor ﬀ ﬃ ﬄ Ǆ ǅ ǆ Ĳ ĳ",
    "x² x³ H₂O ½ ¼ ¾ ™ ℠ № ℃ Å K",
    # emoji
    "I love it 😀😂🥲🫠!",
    "👍🏽 👩‍👩‍👧‍👦 🏳️‍🌈 🇺🇸🇬🇧",
    "❤️ ☺️ ✈️ keycap 1️⃣ #️⃣",
    "💳💰🏦 card money bank",
    # whitespace varieties
    "a b c d e f g h i j　k",
    "line sep para\u0085nel",
    "tab\there\nnewline\r\ncrlf\u000bvt\u000cff",
    "  leading and trailing spaces  ",
    " ogham space mark᠎mongolian vowel separator",
    # zero-width, format and control characters
    "zero​width‌non‍joiner﻿bom",
    "ctrl\u0001chars\u0007bell\u001bescape\u007fdel",
    "c1\u0080\u0081\u009f controls",
    "nul\u0000byte and replacement�char",
    "soft­hyphen word⁠joiner",
    "bidi ‎LRM‏RLM ‪LRE‬ ⁦LRI⁩",
    "private use \U000f0000 end",
    "tags \U000e0001\U000e0041\U000e007f end",
    "noncharacters ﷐ ￿ \U0001fffe end",
    "unassigned ͸ ͹ ⿿ \U0001fb95 end",
    # combining sequences
    "é ȩ́ å ȫ",
    "ñ ǘ ́leading mark",
    "variation️selector ⃝ enclosing circle",
    "Hangul jamo 각 vs 각",
    "Dives Akuru \U00011938 Tulu \U000113c5",
    "Garay \U00010d50\U00010d70 Latin Ɤꟍ",
    # specials in text
    "[CLS] [SEP] [MASK] [PAD] [UNK]",
    "[cls] [Sep] [MASK[MASK]]",
    "x[UNK]y[SEP]z",
    "[[CLS]]]",
    # word length
    "a" * 99 + " " + "b" * 100,
    "c" * 101,
    "pneumonoultramicroscopicsilicovolcanoconiosis" * 3,
    "é" * 101,
    # empty-ish
    "",
    " ",
    "\t\n\r",
    "!!!",
    "...?!",
    "​​",
]


def long_texts():
    base = "I would like to know why my card payment was declined at the store yesterday evening. "
    return [
        base * 60,                                        # > 512 tokens: truncated
        " ".join(str(i) for i in range(600)),             # many numbers
        "x " * 700,                                        # 700 one-token words
        "unaffable" * 60,                                  # one long word (540 chars) -> [UNK]
        ("Привет мир " * 120),                             # cyrillic, truncated
    ]


def golden_cases():
    cases = []
    for t in ENGLISH:
        cases.append(("english", t))
    # Case / punctuation / spacing variants of the first sentences.
    for t in ENGLISH[:30]:
        cases.append(("variant", t.upper()))
        cases.append(("variant", "  " + t.replace(" ", "\t", 2) + "  "))
        cases.append(("variant", t.replace(" ", "")))
    for t in STRESS:
        cases.append(("stress", t))
    for t in long_texts():
        cases.append(("long", t))
    # Every printable ASCII character alone and between letters.
    for c in range(32, 127):
        cases.append(("ascii", f"a{chr(c)}b {chr(c)}"))
    seen, out = set(), []
    for kind, t in cases:
        if t not in seen:
            seen.add(t)
            out.append((kind, t))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True,
                    help="the release encoder_tokenizer/ (tokenizer.json, vocab.txt)")
    ap.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args()
    import tokenizers
    from tokenizers import Tokenizer

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- toy encoder export
    vocab = toy_vocab()
    cfg = dict(CFG, vocab=len(vocab))
    tok = toy_tokenizer(vocab)
    tok_bytes = tok.to_str(pretty=True).encode("utf-8")
    vocab_bytes = ("\n".join(vocab) + "\n").encode("utf-8")
    tokenizer = dx.check_tokenizer(json.loads(tok_bytes), vocab_bytes)
    W = toy_weights(cfg, rng)
    source = {"name": TOY_SOURCE_NAME,
              "onnx_sha256": hashlib.sha256(TOY_SOURCE_NAME.encode()).hexdigest(),
              "tokenizer_json_sha256": dx.sha256_bytes(tok_bytes),
              "vocab_sha256": dx.sha256_bytes(vocab_bytes)}
    tools = {"numpy": np.__version__, "tokenizers": tokenizers.__version__,
             "script": "tools/mk_decision_toy.py", "seed": SEED}
    enc_dir = os.path.join(out, "encoder")
    if os.path.exists(enc_dir):
        shutil.rmtree(enc_dir)
    dx.write_export(enc_dir, W, cfg, source, tokenizer, vocab_bytes, tok_bytes,
                    {"reference": "numpy float64 forward (decision_export_encoder.forward_f64)"}, tools)
    m, W2 = dx.read_export(enc_dir)
    rows = []
    for t in TOY_TEXTS:
        e = tok.encode(t)
        hidden = dx.forward_f64(W2, cfg, e.ids)
        rows.append({"text": t, "ids": e.ids, "tokens": e.tokens,
                     "hidden": [float(x) for x in hidden.reshape(-1)],
                     "phi_p": [float(x) for x in dx.phi_p_f64(hidden)]})
    golden = {"config": cfg, "max_length": MAX_LENGTH, "tolerance": 1e-5,
              "tokenizers": tokenizers.__version__, "numpy": np.__version__, "rows": rows}
    with open(os.path.join(out, "encoder_golden.json"), "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=0)
        f.write("\n")

    # ---- wordpiece golden on the release vocab
    rel_tok_path = os.path.join(a.tokenizer_dir, "tokenizer.json")
    rel_vocab_bytes = open(os.path.join(a.tokenizer_dir, "vocab.txt"), "rb").read()
    rel_tok_bytes = open(rel_tok_path, "rb").read()
    rel = dx.check_tokenizer(json.loads(rel_tok_bytes), rel_vocab_bytes)
    hf = Tokenizer.from_file(rel_tok_path)
    hf.no_padding()
    vocab_name = "bge-small-en-v1.5-vocab.txt"
    with open(os.path.join(out, vocab_name), "wb") as f:
        f.write(rel_vocab_bytes)
    cases = []
    for kind, t in golden_cases():
        e = hf.encode(t)
        cases.append({"kind": kind, "text": t, "ids": e.ids})
    wp = {"tokenizers": tokenizers.__version__,
          "tokenizer_json_sha256": dx.sha256_bytes(rel_tok_bytes),
          "vocab_file": vocab_name, "vocab_sha256": dx.sha256_bytes(rel_vocab_bytes),
          "unk": rel["unk"], "prefix": rel["prefix"],
          "max_input_chars_per_word": rel["max_input_chars_per_word"],
          "ids": rel["ids"], "max_length": rel["max_length"],
          # Cases the native tokenizer does not reproduce (none known): text -> reason.
          "exceptions": {},
          "cases": cases}
    with open(os.path.join(out, "wordpiece_golden.json"), "w", encoding="utf-8") as f:
        json.dump(wp, f, ensure_ascii=False, indent=0)
        f.write("\n")
    kinds = {}
    for c in cases:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    total = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(out) for fn in fns)
    print(f"wrote {out}: toy vocab {len(vocab)}, {len(rows)} encoder rows, {len(cases)} wordpiece cases "
          f"{kinds}, {total} bytes")


if __name__ == "__main__":
    main()
