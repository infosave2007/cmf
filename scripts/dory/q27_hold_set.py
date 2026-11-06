#!/usr/bin/env python3
"""Held-out prompts for q27_spec_hold.sh: q27_hold_set.py OUT_DIR
Twelve each of English prose, Russian prose and code, on topics no training prompt of q27_mtp_prompts.py has."""
import os, sys

EN = [
    "Write a short essay about why medieval castles were built on hills.",
    "Explain how a submarine dives and surfaces to a curious beginner in a few paragraphs.",
    "Write an essay of several paragraphs about the history of the postal service.",
    "Explain how chocolate is made from cocoa beans, step by step.",
    "Tell the story of the first transatlantic telegraph cable: what happened and why it mattered.",
    "Explain why the Moon always shows the same face to the Earth.",
    "Write a short essay about how cities get their drinking water.",
    "Explain what causes dust storms on Mars and how long they last.",
    "Describe how a violin produces sound and why old violins are prized.",
    "Write an essay about the invention of the zipper and how it spread.",
    "Explain how a lighthouse lens concentrates light.",
    "What should I consider before adopting a dog? Give practical advice.",
]
RU = [
    "Напиши короткое эссе о том, почему средневековые замки строили на холмах.",
    "Объясни простыми словами, как подводная лодка погружается и всплывает.",
    "Расскажи, как появилась почта в России.",
    "Объясни, как из какао-бобов делают шоколад.",
    "Напиши эссе в несколько абзацев о том, почему Луна всегда повёрнута к Земле одной стороной.",
    "Расскажи, как город получает питьевую воду.",
    "Объясни, как скрипка издаёт звук и почему ценятся старые скрипки.",
    "Расскажи, как изобрели застёжку-молнию.",
    "Объясни другу, который ничего об этом не знает, как работает маяк.",
    "Что нужно учесть, прежде чем завести собаку? Дай практические советы.",
    "Напиши короткое эссе о Байкале: чем он уникален.",
    "Объясни, почему в июне в Петербурге почти нет ночи, а в декабре день такой короткий.",
]
CODE = [
    "Write a Python function that computes the n-th Fibonacci number with memoization, with a few tests.",
    "Write a Rust function that checks whether a string is a palindrome ignoring case and punctuation, with tests.",
    "Write a Go function that merges two sorted slices into one sorted slice.",
    "Write a TypeScript function that throttles calls to another function.",
    "Write a C++ class for a fixed-size matrix with addition and multiplication.",
    "Write a Java method that finds the first non-repeated character in a string.",
    "Write a Python function that flattens a nested dictionary into dotted keys, with tests.",
    "Write a SQL query that returns the running total of sales per customer by date, and explain it.",
    "Write a Bash script that counts the lines of every .py file under a directory and prints the total.",
    "Write a Rust function that finds the two numbers in a slice that add up to a target, with tests.",
    "Write a Python class for a simple bank account with deposit, withdraw and a transaction log.",
    "Write a C function that converts an integer to a string in a given base.",
]
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
for name, rows in (("en", EN), ("ru", RU), ("code", CODE)):
    for i, p in enumerate(rows, 1):
        open(f"{out}/p_{name}{i:02d}.txt", "w").write(p + "\n")
print(f"{len(EN) + len(RU) + len(CODE)} prompts in {out}")
