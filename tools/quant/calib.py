"""Calibration corpora, window placement and the cached teacher top-K for the refit tools.

Corpus spec (repeatable `--corpus`):  PATH[,train=N][,held=OFF:CNT[+OFF:CNT...]][,name=NAME]
  PATH   token ids (.npy / .u32 / text, see cmfio.load_ids); make them with the model's own
         tokenizer (`cmfio.tokenize_with_cortiq`, i.e. CMF_PPL_IDS_OUT of `cortiq ppl`).
  train  training windows from this corpus (default: --nwin split evenly over the corpora)
  held   held-out windows: CNT evenly spaced windows over ids[OFF:] per group
         (default: 24 / #corpora windows over the last 10 % of the file)
Placement (the q2tp/q4tp studies' rule, kept exactly so their runs reproduce): the held-out
windows are placed first; train window j starts at j * ((n - WL - 1 - 4096) // N) and moves
forward in steps of 128 tokens while it overlaps a held-out window. Held-out windows choose
the checkpoint; the evaluation sets of a study must not be among the corpora.
"""
import hashlib
import json
import os
import time

import numpy as np

import cmfio


class Corpus:
    def __init__(self, path, train=None, held=None, name=None):
        self.path = path
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.train = train
        self.held_groups = held
        self.ids = cmfio.load_ids(path)
        self.train_starts, self.held_starts = [], []

    def digest(self):
        return hashlib.sha256(self.ids.astype("<i8").tobytes()).hexdigest()[:16]


def parse_corpus(spec):
    parts = spec.split(",")
    kw = {"path": parts[0]}
    for p in parts[1:]:
        k, _, v = p.partition("=")
        if k == "train":
            kw["train"] = int(v)
        elif k == "held":
            kw["held"] = [(int(o), int(c)) for o, c in (g.split(":") for g in v.split("+") if g)]
        elif k == "name":
            kw["name"] = v
        else:
            raise SystemExit(f"--corpus {spec}: unknown key {k!r} (train=, held=, name=)")
    return Corpus(**kw)


def even_starts(n, count, wl, off=0):
    """`cortiq ppl --windows` placement over ids[off:]: k * ((n - off - wl - 1) // count)"""
    if count <= 0:
        return []
    stride = (n - off - wl - 1) // count
    if stride <= 0:
        raise SystemExit(f"{count} windows of {wl} do not fit {n - off} tokens")
    return [off + k * stride for k in range(count)]


def place(corpora, wl, nwin):
    """fill train_starts / held_starts of every corpus"""
    free = [c for c in corpora if c.train is None]
    left = nwin - sum(c.train for c in corpora if c.train is not None)
    for i, c in enumerate(free):
        c.train = left // len(free) + (1 if i < left % len(free) else 0)
    for c in corpora:
        n = len(c.ids)
        groups = c.held_groups
        if groups is None:
            groups = [(int(n * 0.9), max(1, 24 // len(corpora)))]
        c.held_starts = [s for off, cnt in groups for s in even_starts(n, cnt, wl, off)]
        busy = [(s, s + wl) for s in c.held_starts]
        c.train_starts = []
        if c.train:
            stp = (n - wl - 1 - 4096) // c.train
            if stp <= 0:
                raise SystemExit(f"{c.path}: {c.train} train windows of {wl} do not fit {n} tokens")
            for j in range(c.train):
                p = j * stp
                while any(p < b1 and p + wl > b0 for b0, b1 in busy):
                    p += 128
                if p + wl + 1 > n:
                    raise SystemExit(f"{c.path}: train window {j} runs past the end after avoiding held-out")
                c.train_starts.append(p)
    return corpora


def gather(ids, starts, wl):
    if not starts:
        return np.zeros((0, wl), dtype=np.int64)
    return np.stack([ids[s:s + wl] for s in starts]).astype(np.int64)


def build_sets(corpora, wl):
    """-> train windows [N, wl], held windows [M, wl], corpus index per held window"""
    tr = [gather(c.ids, c.train_starts, wl) for c in corpora]
    he = [gather(c.ids, c.held_starts, wl) for c in corpora]
    owner = np.concatenate([np.full(len(c.held_starts), i, dtype=np.int64) for i, c in enumerate(corpora)])
    return np.concatenate(tr), np.concatenate(he), owner


def describe(corpora):
    return [{"name": c.name, "path": os.path.abspath(c.path), "n_ids": int(len(c.ids)), "sha": c.digest(),
             "train": len(c.train_starts), "held": len(c.held_starts),
             "train_starts": [int(s) for s in c.train_starts], "held_starts": [int(s) for s in c.held_starts]}
            for c in corpora]


# ----------------------------------------------------------------------------- teacher cache
def _teacher_key(teacher_path, corpora, wl, k):
    st = os.stat(teacher_path)
    return {"teacher": os.path.abspath(teacher_path), "teacher_bytes": st.st_size, "wl": wl, "topk": k,
            "corpora": [{key: v for key, v in d.items()} for d in describe(corpora)]}


def teacher_cache(cache_dir, teacher_path, corpora, wl, k, make_model, block=256, mb=8, log=print):
    """Top-K log-probs of the teacher on every train and held-out window, computed once
    (exact f32, no TF32, the teacher streamed layer by layer) and memory-mapped afterwards.
    make_model(cmf) -> qmodel.Model of the teacher. Returns dict of arrays + meta."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _teacher_key(teacher_path, corpora, wl, k)
    meta_p = os.path.join(cache_dir, "meta.json")
    names = ("train_lp", "train_ix", "held_lp", "held_ix")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        if meta.get("key") == key and all(os.path.exists(os.path.join(cache_dir, f"{n}.npy")) for n in names):
            log(f"teacher cache {cache_dir} (train CE {meta['train_ce']:.5f}, held CE {meta['held_ce']:.5f})")
            out = {n: np.load(os.path.join(cache_dir, f"{n}.npy"), mmap_mode="r") for n in names}
            out["meta"] = meta
            return out
        log(f"teacher cache {cache_dir} does not match this run; recomputing")
    import torch
    import qmodel
    train_w, held_w, _ = build_sets(corpora, wl)
    t0 = time.time()
    cmf = cmfio.CmfFile(teacher_path)
    prev = torch.backends.cuda.matmul.allow_tf32
    qmodel.set_tf32(False)
    model = make_model(cmf)
    meta = {"key": key}
    out = {}
    for tag, w in (("train", train_w), ("held", held_w)):
        lp = np.lib.format.open_memmap(os.path.join(cache_dir, f"{tag}_lp.npy.tmp"), "w+", np.float32,
                                       (len(w), wl - 1, k))
        ix = np.lib.format.open_memmap(os.path.join(cache_dir, f"{tag}_ix.npy.tmp"), "w+", np.int32,
                                       (len(w), wl - 1, k))
        nll = np.zeros(len(w))
        for b0 in range(0, len(w), block):
            r = model.evaluate(w[b0:b0 + block], mb=mb, topk=k)
            lp[b0:b0 + block], ix[b0:b0 + block], nll[b0:b0 + block] = r["lp"], r["ix"], r["nll"]
            log(f"teacher {tag} {min(b0 + block, len(w))}/{len(w)} windows ({time.time() - t0:.0f}s)")
        lp.flush()
        ix.flush()
        del lp, ix
        for n in ("lp", "ix"):
            os.replace(os.path.join(cache_dir, f"{tag}_{n}.npy.tmp"), os.path.join(cache_dir, f"{tag}_{n}.npy"))
        meta[f"{tag}_ce"] = float(nll.sum() / max(1, len(w) * (wl - 1)))
        np.save(os.path.join(cache_dir, f"{tag}_nll.npy"), nll)
    qmodel.set_tf32(prev)
    meta["seconds"] = time.time() - t0
    json.dump(meta, open(meta_p, "w"), indent=1)
    cmf.close()
    del model
    torch.cuda.empty_cache()
    log(f"teacher cached: train CE {meta['train_ce']:.5f}, held CE {meta['held_ce']:.5f} ({meta['seconds']:.0f}s)")
    out = {n: np.load(os.path.join(cache_dir, f"{n}.npy"), mmap_mode="r") for n in names}
    out["meta"] = meta
    return out
