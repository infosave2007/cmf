#!/usr/bin/env python3
"""Add the everyday names of popular plants to `lookup-entries.jsonl` keys.

The table is keyed by the names its source cards spell, so many everyday names
are missing (`мелисса`, `крапива`, `корица`, `nettle`, `elderberry`). Each alias
goes to the entry that already holds the anchor key; a missing anchor is an error.

  python3 scripts/embryo/lookup_aliases.py lookup-entries.jsonl lookup-entries-plus.jsonl
"""
import json
import re
import sys

ALIASES = {
    "valeriana officinalis": ["валериана", "валерьяна", "valerian root"],
    "urtica dioica": ["nettle", "крапива", "крапива двудомная"],
    "sambucus": ["elderberry", "elderflower"],
    "syzygium aromaticum": ["cloves", "гвоздика", "гвоздичное дерево"],
    "aloe vera": ["алоэ", "алоэ вера"],
    "thymus": ["чабрец", "чебрец"],
    "foeniculum vulgare": ["фенхель"],
    "cinnamon": ["корица"],
    "oregano": ["душица"],
    "basil": ["базилик"],
    "lemon balm": ["мелисса", "мелисса лекарственная"],
}


def norm(s):
    return " ".join(re.sub(r"[^\w]+", " ", s.lower()).split())


def main():
    src, dst = sys.argv[1], sys.argv[2]
    ents = [json.loads(line) for line in open(src, encoding="utf-8")]
    idx = {}
    for i, e in enumerate(ents):
        for k in e["keys"]:
            idx.setdefault(norm(k), []).append(i)
    for anchor, aliases in ALIASES.items():
        hits = idx.get(norm(anchor))
        if not hits:
            raise SystemExit(f"anchor key not in the table: {anchor}")
        e = ents[hits[-1]]
        card = (e.get("en") or {}).get("card") or (e.get("ru") or {}).get("card") or ""
        for a in aliases:
            other = [i for i in idx.get(norm(a), []) if i != hits[-1]]
            if other:
                print(f"note: alias {a!r} also a key of entry {other}")
            if norm(a) not in map(norm, e["keys"]):
                e["keys"].append(a)
        print(f"{anchor} -> {card[:60]!r} + {aliases}")
    with open(dst, "w", encoding="utf-8") as f:
        for e in ents:
            f.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
