"""Compare the engine's CMF_CLOVER_DUMP with clover_ref.py's record.

usage: python3 -I clover_cmp.py REF.json PROMPT_INDEX ENGINE_DUMP.json
Prints whether the prompt ids match, the per-row top-1 agreement of the
first denoising pass, and the top-1 logit gap per row.
"""

import json
import sys


def main():
    ref = json.load(open(sys.argv[1]))[int(sys.argv[2])]
    eng = json.load(open(sys.argv[3]))
    same_prompt = ref["prompt_ids"] == eng["prompt_ids"]
    print(f"prompt ids equal: {same_prompt} ({len(ref['prompt_ids'])} vs {len(eng['prompt_ids'])})")
    agree, gaps, overlap = 0, [], []
    for r, e in zip(ref["rows"], eng["rows"]):
        agree += r["ids"][0] == e["ids"][0]
        gaps.append(abs(r["logits"][0] - e["logits"][0]))
        overlap.append(len(set(r["ids"][:5]) & set(e["ids"][:5])))
    n = len(ref["rows"])
    print(f"top-1 agree {agree}/{n}; top-5 overlap mean {sum(overlap) / n:.2f}; "
          f"|top-1 logit gap| mean {sum(gaps) / n:.3f} max {max(gaps):.3f}")
    for i, (r, e) in enumerate(zip(ref["rows"], eng["rows"])):
        if i < 8 or r["ids"][0] != e["ids"][0]:
            print(f"row {i:2d}: ref {r['ids'][:3]} {[round(x, 2) for x in r['logits'][:3]]} | "
                  f"eng {e['ids'][:3]} {[round(x, 2) for x in e['logits'][:3]]}")


if __name__ == "__main__":
    main()
