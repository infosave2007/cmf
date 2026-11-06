#!/usr/bin/env python3
"""Prompts whose rounds fit the DFlash sidecar's router: q27_dflash_fit_prompts.py OUT_DIR
Writes fNN_KIND.txt: code in several languages, edits of code that is in the prompt, work on a text that is in the
prompt, prose in English, prose in Russian. None repeats the five prompts that measure the result
(q27_spec_prompts.sh). The texts in the prompts are written here, so the set needs no corpus."""
import os, sys

out = sys.argv[1]
CODE = [
    "Write a Python class for an LRU cache with get and put in O(1), using a dict and a doubly linked list. Add a short usage example.",
    "Write a Go function that merges overlapping intervals in a slice and returns the merged slice. Include a table-driven test.",
    "Write a C function that reverses a singly linked list in place, and a main that builds a list of five nodes and prints it before and after.",
    "Write a TypeScript function that debounces another function, with proper generic types, and show how to use it on an input handler.",
    "Write a Rust struct for a ring buffer of fixed capacity with push, pop and an iterator, and unit tests for wrap-around.",
    "Write a Python script that walks a directory tree, computes the SHA-256 of every file and prints duplicates grouped by hash.",
    "Write a SQL schema for a small library (books, members, loans) and three queries: overdue loans, the most borrowed books, members with no loans.",
    "Write a Java method that finds the k most frequent words in a list of strings using a heap, with a JUnit test.",
    "Write a Bash script that rotates log files in a directory: compress those older than a day, delete those older than thirty days, and print what it did.",
    "Write a C++ template function for binary search over a sorted vector that returns an optional index, with a few assertions as tests.",
    "Write a Python function that parses an ISO 8601 duration such as P3DT4H5M into seconds, with doctests.",
    "Write a Rust function that reads a file line by line and counts word frequencies into a BTreeMap, returning an error type of your own.",
]
SNIPPET_PY = '''def moving_average(values, window):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def normalize(values):
    low = min(values)
    high = max(values)
    return [(v - low) / (high - low) for v in values]


def top_n(values, n):
    return sorted(values, reverse=True)[:n]
'''
SNIPPET_JS = '''function groupBy(items, key) {
  const groups = {};
  for (const item of items) {
    const k = item[key];
    if (!groups[k]) {
      groups[k] = [];
    }
    groups[k].push(item);
  }
  return groups;
}

function sumBy(items, key) {
  let total = 0;
  for (const item of items) {
    total += item[key];
  }
  return total;
}

function pluck(items, key) {
  return items.map((item) => item[key]);
}
'''
EDIT = [
    "Here is a Python module:\n\n" + SNIPPET_PY + "\nRewrite the whole module with type hints and docstrings, and make normalize return zeros when all values are equal. Print the full module.",
    "Here is a Python module:\n\n" + SNIPPET_PY + "\nRename moving_average to rolling_mean everywhere, add a parameter min_periods to it, and print the full module.",
    "Here is a JavaScript file:\n\n" + SNIPPET_JS + "\nConvert the whole file to TypeScript with generics and print the full file.",
    "Here is a JavaScript file:\n\n" + SNIPPET_JS + "\nAdd input validation to every function (throw a TypeError on a non-array) and print the full file.",
    "Here is a Python module:\n\n" + SNIPPET_PY + "\nAdd a function median(values) in the same style and a main block that demonstrates all four. Print the full module.",
]
TEXT = (
    "The lighthouse at Skerryvore stands on a reef twelve miles off the island of Tiree. It was built between 1838 and "
    "1844 by Alan Stevenson, who lived on the rock with his workmen in a wooden barrack that the sea carried away in the "
    "first winter. The tower is 156 feet tall and its lowest courses are of granite quarried on Mull; each stone was cut "
    "on shore to fit its neighbours and landed from boats in the short summer calms. Stevenson gave the tower a "
    "hyperbolic outline, wider at the base than any before it, and the light was first shown on 1 February 1844. A fire "
    "in 1954 gutted the interior, and the keepers were taken off by boat the next day. The light was rebuilt, made "
    "electric and has been automatic since 1994. On a clear night it can be seen for twenty-three miles."
)
QUOTE = [
    "Here is a text:\n\n" + TEXT + "\n\nQuote, word for word, every sentence of the text that mentions a year, one per line.",
    "Here is a text:\n\n" + TEXT + "\n\nCopy the text exactly, then under it list the numbers it contains with what each one measures.",
    "Here is a text:\n\n" + TEXT + "\n\nSummarize the text in three sentences, then quote the sentence about the fire exactly.",
    "Here is a text:\n\n" + TEXT + "\n\nTranslate the text into French, keeping the sentence order.",
    "Here is a text:\n\n" + TEXT + "\n\nRewrite the text as a timeline: one line per event, each starting with its year.",
]
PROSE = [
    "Write a detailed essay on why the printing press spread so quickly through fifteenth-century Europe.",
    "Explain how a suspension bridge carries its load, for a curious reader with no engineering background.",
    "Describe the life cycle of a star like the Sun, from the cloud it forms in to what remains at the end.",
    "Write a balanced comparison of living in a large city and in a small town, with concrete examples.",
    "Tell the story of the first ascent of a great mountain as a narrative, with the weather and the doubts of the climbers.",
    "Give practical advice to someone starting to learn a musical instrument as an adult, in several paragraphs.",
]
RU = [
    "Напиши подробный рассказ о том, как устроена работа маяка и зачем маяки нужны сегодня.",
    "Объясни простыми словами, почему небо голубое, а закат красный. Приведи бытовые примеры.",
    "Сравни жизнь в большом городе и в деревне: плюсы и минусы, с примерами.",
    "Напиши эссе о том, как книгопечатание изменило Европу.",
    "Дай развёрнутые советы человеку, который во взрослом возрасте начинает учить иностранный язык.",
    "Расскажи, как работает холодильник и почему он греет комнату.",
]
os.makedirs(out, exist_ok=True)
n = 0
for kind, prompts in (("code", CODE), ("edit", EDIT), ("quote", QUOTE), ("prose", PROSE), ("ru", RU)):
    for p in prompts:
        with open(os.path.join(out, f"f{n:02d}_{kind}.txt"), "w") as f:
            f.write(p + "\n")
        n += 1
print(n, "prompts in", out)
