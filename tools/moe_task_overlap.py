#!/usr/bin/env python3
"""How task-conditional is this model's expert routing?

    CMF_MOE_STATS=prose.json cortiq ppl model.cmf --file prose.txt --tokens 400
    CMF_MOE_STATS=code.json  cortiq ppl model.cmf --file code.txt  --tokens 400
    python3 tools/moe_task_overlap.py prose.json code.json

Two numbers decide whether a memory-budgeted expert set is worth building:

  * COVERAGE — how many experts per layer hold 95% of the routing mass. If a
    task needs most of them, there is nothing to drop.
  * JACCARD — how much the top sets of two different tasks overlap. If they
    are near-disjoint, a model serving one task is carrying the other's
    experts around; if they coincide, the whole idea is moot.

Measured on KAT-Coder, code against prose gave a Jaccard of 0.25 on the
top-64. This asks the same question of whatever produced these dumps.
"""
import json
import sys


def load(path):
    d = json.load(open(path))
    return {int(k): v for k, v in d.items() if v}


def cover(counts, frac):
    """Smallest set of experts holding `frac` of the routing mass."""
    total = sum(counts)
    if total == 0:
        return 0
    kept, acc = 0, 0
    for c in sorted(counts, reverse=True):
        acc += c
        kept += 1
        if acc >= frac * total:
            break
    return kept


def topset(counts, n):
    order = sorted(range(len(counts)), key=lambda i: -counts[i])
    return {i for i in order[:n] if counts[i] > 0}


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = load(a_path), load(b_path)
    layers = sorted(set(a) & set(b))
    if not layers:
        print("нет общих слоёв — статистика пуста")
        return 1
    n_exp = len(a[layers[0]])
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, n_exp // 4)

    print(f"слоёв со статистикой: {len(layers)}, экспертов на слой: {n_exp}, "
          f"верхний набор: {topn}\n")
    print(f"{'слой':>5} {'исп.A':>7} {'исп.B':>7} {'95%A':>6} {'95%B':>6} {'Жаккар':>8}")
    tot_c95a = tot_c95b = 0
    jacs = []
    for li in layers:
        ca, cb = a[li], b[li]
        used_a = sum(1 for c in ca if c > 0)
        used_b = sum(1 for c in cb if c > 0)
        c95a, c95b = cover(ca, 0.95), cover(cb, 0.95)
        sa, sb = topset(ca, topn), topset(cb, topn)
        j = len(sa & sb) / max(len(sa | sb), 1)
        jacs.append(j)
        tot_c95a += c95a
        tot_c95b += c95b
        print(f"{li:>5} {used_a:>7} {used_b:>7} {c95a:>6} {c95b:>6} {j:>8.3f}")

    n = len(layers)
    print(f"\nв среднем: 95% массы держат {tot_c95a / n:.1f} экспертов (A) и "
          f"{tot_c95b / n:.1f} (B) из {n_exp}")
    print(f"Жаккар по верхним {topn}: среднее {sum(jacs) / n:.3f}, "
          f"минимум {min(jacs):.3f}, максимум {max(jacs):.3f}")
    saved = 1 - (tot_c95a / n) / n_exp
    print(f"срез по 95% массы задачи A убрал бы {saved * 100:.0f}% экспертов")
    if sum(jacs) / n > 0.8:
        print("\nнаборы почти совпадают — задачно-обусловленный срез тут не даст "
              "ничего, эксперты используются одинаково независимо от задачи")
    return 0


if __name__ == "__main__":
    sys.exit(main())
