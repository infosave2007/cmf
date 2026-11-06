#!/usr/bin/env python3
"""Where a steady chunk's time goes, from a CMF_ANE_TRACE log: q27_ane_gaps.py FILE.err [CHUNK]"""
import collections, re, sys

ev = []
for line in open(sys.argv[1], errors="ignore"):
    m = re.match(r"ane-trace ([\d.]+) lane (\d) layer (\d+) (\S+) ([\d.]+)", line)
    if m:
        ev.append((float(m[1]), int(m[2]), int(m[3]), m[4], float(m[5])))
ev.sort()
starts = [e[0] for e in ev if e[1] == 0 and e[2] == 0 and e[3] == "gate"]
print("chunk ms:", [round(starts[i + 1] - starts[i]) for i in range(len(starts) - 1)])
ci = int(sys.argv[2]) if len(sys.argv) > 2 else 3
a, b = starts[ci], starts[ci + 1]
E = [e for e in ev if a <= e[0] < b]


def busy(kinds):
    tot, cur = 0, None
    for s, t in sorted((e[0], e[0] + e[4]) for e in E if e[3] in kinds):
        if cur is None:
            cur = [s, t]
        elif s <= cur[1]:
            cur[1] = max(cur[1], t)
        else:
            tot += cur[1] - cur[0]
            cur = [s, t]
    return tot + (cur[1] - cur[0] if cur else 0)


g, n = busy({"gpu"}), busy({"ane-ffn", "ane-mix"})
print("chunk %d: %.0f ms, %.1f a layer; GPU busy %.0f%%, ANE busy %.0f%%" % (ci + 1, b - a, (b - a) / 64, 100 * g / (b - a), 100 * n / (b - a)))
st = collections.defaultdict(list)
for e in E:
    st[e[3]].append(e[4])
print("  " + "; ".join("%s %.1f ms x%d" % (k, sum(v) / len(v), len(v)) for k, v in st.items() if k not in ("gate", "commit")))
by = collections.defaultdict(list)
for e in E:
    by[(e[1], e[2])].append(e)
g1, g2 = collections.defaultdict(list), collections.defaultdict(list)
for (lane, L), es in by.items():
    gp = [e for e in es if e[3] == "gpu"]
    ff = [e for e in es if e[3] == "ane-ffn"]
    if gp and ff:
        g1[lane].append(ff[0][0] - (gp[-1][0] + gp[-1][4]))
    nx = by.get((lane, L + 1))
    if nx and ff:
        m2 = [e for e in nx if e[3] == "ane-mix"]
        if m2:
            g2[lane].append(m2[0][0] - (ff[0][0] + ff[0][4]))
for lane in sorted(g1):
    print("  lane %d: its GPU's end to its FFN %.1f ms, its FFN's end to the next mixer call %.1f ms" % (lane, sum(g1[lane]) / len(g1[lane]), sum(g2[lane]) / max(1, len(g2[lane]))))
