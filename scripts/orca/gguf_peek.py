#!/usr/bin/env python3
"""Peek into a gated GGUF on HF by range requests: metadata, tensor type mix,
and a comparison of its F32 norm tensors with the base model's safetensors.

  gguf_peek.py REPO FILE [BASE_REPO]   (token from HF_TOKEN or the hf cache)

Norms are stored by llama.cpp as 1+w for qwen35; the base's raw w is shifted
the same way before the comparison. Identical norms across the trunk mean the
weights were not trained (a template-only "model"); different ones mean SFT.
"""
import json, struct, sys, urllib.request
from collections import Counter

try:
    from huggingface_hub import get_token
    TOKEN = get_token()
except Exception:
    TOKEN = None
import os
TOKEN = os.environ.get("HF_TOKEN") or TOKEN


def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


GT = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K",
      11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
      18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
      29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}


class R:
    def __init__(self, url):
        self.url, self.buf, self.pos = url, b"", 0
        self.more(1 << 22)

    def more(self, n):
        a = len(self.buf)
        self.buf += rng(self.url, a, a + n - 1)

    def take(self, n):
        while self.pos + n > len(self.buf):
            self.more(max(1 << 22, len(self.buf)))
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")

    def val(self, t):
        f = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in f:
            sz = struct.calcsize(f[t])
            return struct.unpack(f[t], self.take(sz))[0]
        if t == 8:
            return self.s()
        if t == 9:
            et, n = self.u32(), self.u64()
            return [self.val(et) for _ in range(n)]
        raise ValueError(t)


def parse(url):
    r = R(url)
    assert r.take(4) == b"GGUF"
    ver, nt, nkv = r.u32(), r.u64(), r.u64()
    md = {}
    for _ in range(nkv):
        k = r.s()
        md[k] = r.val(r.u32())
    ts = []
    for _ in range(nt):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ty, off = r.u32(), r.u64()
        ts.append((name, dims, ty, off))
    align = md.get("general.alignment", 32)
    data_start = (r.pos + align - 1) // align * align
    return md, ts, data_start


def main():
    repo, fname = sys.argv[1], sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else None
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    md, ts, ds = parse(url)
    arch = md.get("general.architecture")
    print("arch", arch, "tensors", len(ts), "data_start", ds)
    for k, v in md.items():
        if k.startswith("tokenizer.ggml.") and isinstance(v, list):
            print(f"  {k}: list[{len(v)}]")
        elif k == "tokenizer.chat_template":
            print(f"  {k}: {len(v)} chars")
        else:
            print(f"  {k}: {v if not isinstance(v, list) else v[:16]}")
    print("type mix (count / GB by type):")
    by = Counter()
    nb = Counter()
    for name, dims, ty, off in ts:
        by[GT.get(ty, ty)] += 1
    for i, (name, dims, ty, off) in enumerate(ts):
        end = ts[i + 1][3] if i + 1 < len(ts) else None
        if end is not None:
            nb[GT.get(ty, ty)] += end - off
    for k in by:
        print(f"  {k}: {by[k]} tensors, {nb[k]/1e9:.2f} GB")
    blocks = sorted({n.split(".")[1] for n, *_ in ts if n.startswith("blk.")}, key=int)
    last = blocks[-1]
    print("last block", last, [n for n, *_ in ts if n.startswith(f"blk.{last}.")])
    # Type per tensor kind of block 0, 3 and the last.
    for b in ["0", "3", last]:
        print(f"  blk.{b}:", [(n.split('.', 2)[2], GT.get(t, t)) for n, d, t, o in ts if n.startswith(f"blk.{b}.")])
    for n, d, t, o in ts:
        if not n.startswith("blk."):
            print("  global", n, d, GT.get(t, t))
    ct = md.get("tokenizer.chat_template", "")
    open("chat_template.jinja", "w").write(ct)
    print("chat template saved; contains 'OrcaCyber':", "OrcaCyber" in ct, "system default:",
          ct[:0] or [l for l in ct.splitlines() if "Orca" in l][:3])
    if not base:
        return
    # Norm comparison with the base safetensors.
    idx = json.loads(rng(f"https://huggingface.co/{base}/resolve/main/model.safetensors.index.json", 0, 50_000_000))
    wm = idx["weight_map"]
    hdrs = {}

    def st_tensor(name):
        f = wm[name]
        if f not in hdrs:
            u = f"https://huggingface.co/{base}/resolve/main/{f}"
            n = struct.unpack("<Q", rng(u, 0, 7))[0]
            hdrs[f] = (u, 8 + n, json.loads(rng(u, 8, 8 + n - 1)))
        u, d0, h = hdrs[f]
        e = h[name]
        a, b = e["data_offsets"]
        raw = rng(u, d0 + a, d0 + b - 1)
        assert e["dtype"] == "BF16", e["dtype"]
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]

    pfx = "model.language_model." if any(k.startswith("model.language_model.") for k in wm) else "model."
    pairs = []
    nl = int(md.get(f"{arch}.block_count")) - int(md.get(f"{arch}.nextn_predict_layers", 0) or 0)
    for b in [0, 1, 3, nl // 2, nl - 1]:
        pairs.append((f"blk.{b}.attn_norm.weight", f"{pfx}layers.{b}.input_layernorm.weight"))
        pairs.append((f"blk.{b}.post_attention_norm.weight", f"{pfx}layers.{b}.post_attention_layernorm.weight"))
    pairs.append(("output_norm.weight", f"{pfx}norm.weight"))
    pairs.append((f"blk.3.attn_q_norm.weight", f"{pfx}layers.3.self_attn.q_norm.weight"))
    pairs.append((f"blk.{nl}.nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight"))
    tmap = {n: (d, t, o) for n, d, t, o in ts}
    for g, h in pairs:
        if g not in tmap or h not in wm:
            print("skip", g, h, g in tmap, h in wm)
            continue
        d, t, o = tmap[g]
        if t != 0:
            print("not F32:", g, GT.get(t, t))
            continue
        n = 1
        for x in d:
            n *= x
        raw = rng(url, ds + o, ds + o + 4 * n - 1)
        gv = struct.unpack(f"<{n}f", raw)
        bv = st_tensor(h)

        def bf16r(x):
            # torch adds 1 in the source dtype: round w+1 to bf16 (nearest even).
            u = struct.unpack("<I", struct.pack("<f", x))[0]
            u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
            return struct.unpack("<f", struct.pack("<I", u))[0]

        diff = [abs(x - (y + 1.0)) for x, y in zip(gv, bv)]
        diffr = [abs(x - bf16r(y + 1.0)) for x, y in zip(gv, bv)]
        same = sum(1 for x in diff if x == 0.0)
        samer = sum(1 for x in diffr if x == 0.0)
        print(f"{g}: max|gguf-(1+base)| {max(diff):.3e} (bf16-rounded {max(diffr):.3e}), "
              f"identical {same}/{n} ({samer}/{n} rounded), mean gguf {sum(gv)/n:.4f}")


if __name__ == "__main__":
    main()
