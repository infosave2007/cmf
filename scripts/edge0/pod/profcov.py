# profcov.py PROFILE [OTHER]: how much of the routing the hottest experts carry; with OTHER, how much of PROFILE its top carries
import struct, sys
def load(p):
    b = open(p, "rb").read()
    if b[:4] == b"STRP":
        v, nl, ne, want, nr = struct.unpack("<5I", b[4:24])
        c = [0] * (nl * ne)
        for i in range(nr):
            l, e = struct.unpack("<2H", b[24 + 4 * i:28 + 4 * i])
            c[l * ne + e] = nr - i
        return nl, ne, c
    nl, ne, tok = struct.unpack("<3Q", b[8:32])
    return nl, ne, list(struct.unpack("<%dI" % (nl * ne), b[32:32 + 4 * nl * ne]))
nl, ne, c = load(sys.argv[1])
tot = sum(c)
order = sorted(range(len(c)), key=lambda i: -c[i])
print("experts", len(c), "picks", tot, "never picked", sum(1 for v in c if v == 0))
for pct in (5, 10, 20, 25, 35, 50, 75):
    k = len(c) * pct // 100
    print(f"top {pct:2d}% of experts carry {100.0 * sum(c[i] for i in order[:k]) / tot:5.1f}% of the picks")
if len(sys.argv) > 2:
    _, _, o = load(sys.argv[2])
    oo = sorted(range(len(o)), key=lambda i: -o[i])
    for pct in (10, 25, 35, 50):
        k = len(c) * pct // 100
        print(f"the other profile top {pct}% carry {100.0 * sum(c[i] for i in oo[:k]) / tot:5.1f}% of these picks")
