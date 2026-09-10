#!/usr/bin/env python3
"""Tokenizer parity fixture: adversarial texts → expected ids from the
reference HF `tokenizers` library, for the given tokenizer.json.

Usage: gen_tokenizer_cases.py <tokenizer.json> <out.json>
"""
import json
import sys

from tokenizers import Tokenizer

CASES = [
    "Hello, world!",
    "The capital of France is Paris.",
    "Столица Франции — Париж. Ёжик прыгнул через реку!",
    "привет мир",
    "日本語のテキストと中文文本",
    "emoji: 🌍🚀 → ok; family: 👨‍👩‍👧‍👦",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "SELECT COUNT(*) FROM users WHERE name != 'O''Brien';",
    "tabs\tand\r\nCRLF   multiple   spaces\n\n\nblank lines",
    "MixedРусскийEnglish词汇interleaved",
    "numbers 12345 and 3.14159 and 0xDEADBEEF",
    "don't can't won't it's I'll you're we've they'd I'M CAN'T",
    "<|im_start|>user\nQuestion?<|im_end|>\n<|im_start|>assistant\n",
    "<think>reasoning</think> and <tool_call>{}</tool_call>",
    "trailing space ",
    " leading space",
    "ĠĊ literal byte-alphabet chars",
    "á combining accent (NFC test) + ﬁ ligature",
    "",
    "\n",
    "   ",
]


def main():
    tok = Tokenizer.from_file(sys.argv[1])
    cases = []
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({
            "text": text,
            "ids": ids,
            "decoded": tok.decode(ids, skip_special_tokens=True),
        })
    with open(sys.argv[2], "w") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False)
    print(f"wrote {len(cases)} cases → {sys.argv[2]}")


if __name__ == "__main__":
    main()
