# ggufhdr.py FILE: minimal GGUF header dump (metadata + tensor infos), tolerant of unknown ggml types
import struct, sys, re, collections
f = open(sys.argv[1], "rb")
def rd(fmt): s = struct.calcsize(fmt); return struct.unpack("<" + fmt, f.read(s))
def rstr(): (n,) = rd("Q"); return f.read(n).decode("utf-8", "replace")
SC = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
def rval(t):
    if t in SC: return rd(SC[t])[0]
    if t == 8: return rstr()
    if t == 9:
        (et,) = rd("I"); (n,) = rd("Q")
        if et == 8: return [rstr() for _ in range(n)]
        if et in SC: return list(rd(f"{n}{SC[et]}"))
        return [rval(et) for _ in range(n)]
    raise ValueError(t)
magic = f.read(4); (ver,) = rd("I"); (nt,) = rd("Q"); (nkv,) = rd("Q")
print("magic", magic, "version", ver, "tensors", nt, "kv", nkv)
for _ in range(nkv):
    k = rstr(); (t,) = rd("I"); v = rval(t)
    if isinstance(v, list): s = f"list[{len(v)}] " + str(v[:8])[:160]
    else: s = str(v)[:300]
    if k.startswith("tokenizer.ggml.tokens") or k.startswith("tokenizer.ggml.merges") or k.startswith("tokenizer.ggml.token_type"): s = s[:80]
    print(k, "=", s)
c = collections.Counter(); ex = {}
for _ in range(nt):
    name = rstr(); (nd,) = rd("I"); dims = rd(f"{nd}Q"); (ty,) = rd("I"); (off,) = rd("Q")
    n2 = re.sub(r"blk\.\d+\.", "blk.N.", name)
    c[(n2, ty, dims if "blk" not in name else dims)] += 1
for (n, ty, d), k in sorted(c.items()):
    print(k, n, ty, d)
