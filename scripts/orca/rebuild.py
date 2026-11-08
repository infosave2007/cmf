#!/usr/bin/env python3
"""OrcaCyber-Nano weights = Qwen3.8-27B bf16 + the low-rank edits recovered from its GGUF.

  rebuild.py GGUF QWEN_DIR OUT_DIR [--workers N] [--tau 1.5] [--k 32] [--delete-base]

Why: the GGUF is a 4.6-bit imatrix mix (IQ4_XS/IQ3_S/Q5_K…, 8-14 % relative error per matrix), while every F32 norm
equals Qwen3.8-27B's bit for bit and the matrix differences G - B are quantization noise plus a few singular
components far above the noise edge (abliteration-like edits). So per 2-D matrix: D = G - B (G dequantized by gguf-py,
put back into the HF V-head order), keep the singular components of D above tau x the Marchenko-Pastur edge of the
rest (tau 2: flat imatrix-noise plateaus reach 1.7), W = B + those components, debiased for the noise. F32 tensors (norms, A_log, dt_bias, conv taps) are exact in the GGUF: kept from the
base when equal, taken from the GGUF otherwise. Output: the base's shard layout and names, bf16, plus a report
(rebuild_report.jsonl) of the rank and energy kept per matrix.
"""
import argparse, json, os, shutil, struct, sys, time
import numpy as np
from multiprocessing import get_context

P = "model.language_model."
NK, NV, DK, DV = 16, 48, 128, 128  # Qwen3.8-27B GDN geometry (checked against config below)


def gguf_name(hf):
    """HF tensor name -> (gguf name, kind). kind: None = no gguf counterpart (vision, lm extras)."""
    if hf == P + "embed_tokens.weight":
        return "token_embd.weight"
    if hf == "lm_head.weight":
        return "output.weight"
    if hf == P + "norm.weight":
        return "output_norm.weight"
    m = {"fc.weight": "nextn.eh_proj.weight", "pre_fc_norm_embedding.weight": "nextn.enorm.weight",
         "pre_fc_norm_hidden.weight": "nextn.hnorm.weight", "norm.weight": "nextn.shared_head_norm.weight"}
    if hf.startswith("mtp."):
        rest = hf[4:]
        if rest in m:
            return f"blk.64.{m[rest]}"
        assert rest.startswith("layers.0."), hf
        return "blk.64." + SUF[rest[len("layers.0."):]]
    if hf.startswith(P + "layers."):
        l, rest = hf[len(P + "layers."):].split(".", 1)
        return f"blk.{l}.{SUF[rest]}"
    return None


SUF = {
    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight", "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.A_log": "ssm_a", "linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.norm.weight": "ssm_norm.weight", "linear_attn.out_proj.weight": "ssm_out.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
}


def untile(x, axis, hd):
    """llama.cpp's tiled V order [r, nk, hd] back to HF's grouped [nk, r, hd] along `axis`."""
    r = NV // NK
    x = np.moveaxis(x, axis, 0)
    rest = x.shape[1:]
    x = x.reshape(r, NK, hd, *rest).transpose(1, 0, 2, *range(3, 3 + len(rest))).reshape(NV * hd, *rest)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))


def is_shifted_norm(hf):
    return ".linear_attn." not in hf and (hf.endswith("norm.weight") or ".pre_fc_norm_" in hf)


def to_hf(hf, g):
    """GGUF values (already dequantized, HF [out, in] shape) -> HF layout/convention."""
    if ".linear_attn." not in hf:
        if is_shifted_norm(hf):  # llama.cpp stores 1+w for every qwen35 norm but the GDN gated norm
            g = g - np.float32(1.0)
        return g
    if hf.endswith("in_proj_qkv.weight"):
        q = 2 * NK * DK
        g = np.concatenate([g[:q], untile(g[q:], 0, DV)], 0)
    elif hf.endswith("in_proj_z.weight"):
        g = untile(g, 0, DV)
    elif hf.endswith("in_proj_a.weight") or hf.endswith("in_proj_b.weight"):
        g = untile(g, 0, 1)
    elif hf.endswith("A_log"):
        g = np.log(-untile(g.reshape(-1, 1), 0, 1).reshape(-1))
    elif hf.endswith("dt_bias"):
        g = untile(g.reshape(-1, 1), 0, 1).reshape(-1)
    elif hf.endswith("conv1d.weight"):
        g = g.reshape(g.shape[0], -1)
        c = 2 * NK * DK
        g = np.concatenate([g[:c], untile(g[c:], 0, DV)], 0)
    elif hf.endswith("out_proj.weight"):
        g = untile(g, 1, DV)
    return g


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def st_bf16(path, d0, e):
    a, b = e["data_offsets"]
    raw = np.memmap(path, dtype=np.uint16, mode="r", offset=d0 + a, shape=((b - a) // 2,))
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])


def bf16_bytes(x, rows=4096):
    """f32 -> bf16 bytes, round to nearest even, in row chunks (the vocabulary planes are 5 GB in f32)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    flat = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    parts = []
    for i in range(0, flat.shape[0], rows):
        u = flat[i:i + rows].view(np.uint32)
        u = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16
        parts.append(u.astype(np.uint16).tobytes())
    return b"".join(parts)


def dequant_rows(t, rows=2048):
    from gguf.quants import dequantize
    raw = np.asarray(t.data)
    shape = [int(x) for x in reversed(t.shape)]
    if raw.ndim == 1 or len(shape) == 1:
        return dequantize(raw, t.tensor_type).astype(np.float32).reshape(shape)
    out = np.empty((raw.shape[0], shape[-1]), np.float32)
    for i in range(0, raw.shape[0], rows):
        out[i:i + rows] = dequantize(raw[i:i + rows], t.tensor_type).reshape(-1, shape[-1])
    return out.reshape(shape)


def lowrank(D, k, tau, seed=0):
    """Components of D above tau x the noise edge, by randomized SVD with power iterations."""
    m, n = D.shape
    rng = np.random.default_rng(seed)
    Y = D @ rng.standard_normal((n, k + 8), dtype=np.float32)
    for _ in range(3):
        Y, _ = np.linalg.qr(Y)
        Y = D @ (D.T @ Y)
    Q, _ = np.linalg.qr(Y)
    Bs = Q.T @ D
    Ub, s, Vt = np.linalg.svd(Bs, full_matrices=False)
    U = Q @ Ub
    tot = float(np.square(D, dtype=np.float64).sum())
    keep = 0
    for _ in range(4):  # the edge from what is left once the kept spikes are taken out
        rest = max(tot - float(np.square(s[:keep].astype(np.float64)).sum()), 1e-30)
        edge = np.sqrt(rest / (m * n)) * (np.sqrt(m) + np.sqrt(n))
        new = int((s > tau * edge).sum())
        if new == keep:
            break
        keep = new
    info = {"rank": keep, "sv_over_edge": [round(float(v / edge), 2) for v in s[:min(8, len(s))]],
            "energy_kept": round(float(np.square(s[:keep].astype(np.float64)).sum() / tot), 4)}
    if keep == 0:
        return None, info
    # Noise inflates a spike: s_obs^2 ~ s^2 + sig^2 (m+n) + sig^4 m n / s^2 (spiked model); invert it.
    sig2 = (rest / (m * n))
    A = s[:keep].astype(np.float64) ** 2 - sig2 * (m + n)
    disc = np.maximum(A * A - 4 * sig2 * sig2 * m * n, 0.0)
    st = np.sqrt(np.maximum((A + np.sqrt(disc)) / 2, 0.0)).astype(np.float32)
    info["shrink"] = [round(float(v), 4) for v in (st / s[:keep])]
    return (U[:, :keep] * st) @ Vt[:keep], info


G_READER = None


def init(gpath):
    global G_READER
    from gguf import GGUFReader
    G_READER = {t.name: t for t in GGUFReader(gpath).tensors}


def work(job):
    hf, spath, d0, e, tmpdir, k, tau = job
    t0 = time.time()
    B = st_bf16(spath, d0, e)
    gname = gguf_name(hf)
    rep = {"name": hf, "gguf": gname}
    if gname is None or gname not in G_READER:
        out = bf16_bytes(B)
        rep["mode"] = "base (no gguf counterpart)"
    else:
        t = G_READER[gname]
        rep["type"] = t.tensor_type.name
        Graw = dequant_rows(t)
        G = to_hf(hf, Graw).reshape(B.shape)
        if t.tensor_type.name == "F32":
            # Norms: compare in the file's own convention (1+w, added in f32: exact). Others (A_log went
            # through -exp in bf16 inside llama.cpp, conv taps and dt_bias are copied): equal within that
            # rounding means unchanged.
            if is_shifted_norm(hf):
                same = int((Graw.reshape(B.shape) == B + np.float32(1.0)).sum())
                ok = same == B.size
            else:
                same = int((G == B).sum())
                ok = bool(np.all(np.abs(G - B) <= 4e-3 * np.maximum(1.0, np.abs(B))))
            rep.update(mode="f32 unchanged", identical=same, n=int(B.size),
                       max_abs_diff=float(np.abs(G - B).max()))
            out = bf16_bytes(B if ok else G)
            if not ok:
                rep["mode"] = "f32 from gguf (differs)"
        elif B.ndim == 2:
            D = G
            D -= B
            del G, Graw
            rep["rel"] = round(float(np.linalg.norm(D) / np.linalg.norm(B)), 4)
            delta, info = lowrank(D, k, tau)
            rep.update(info)
            if info["rank"] >= k:
                rep["warn"] = "rank hit k"
            del D
            rep["mode"] = "base + low-rank edit" if delta is not None else "base (noise only)"
            if delta is not None:
                B += delta  # st_bf16 returns a fresh f32 array
                del delta
            out = bf16_bytes(B)
        else:
            out = bf16_bytes(B)
            rep["mode"] = "base (non-2D)"
    p = os.path.join(tmpdir, hf + ".bin")
    with open(p, "wb") as f:
        f.write(out)
    rep["s"] = round(time.time() - t0, 1)
    return hf, p, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf"); ap.add_argument("qwen"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--delete-base", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.qwen, "config.json")))["text_config"]
    assert (cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
            cfg["linear_value_head_dim"]) == (NK, NV, DK, DV)
    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, ".tmp"); os.makedirs(tmp, exist_ok=True)
    idx = json.load(open(os.path.join(a.qwen, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    for f in os.listdir(a.qwen):
        if f.endswith(".json") or f.endswith(".jinja") or f.endswith(".txt"):
            shutil.copy(os.path.join(a.qwen, f), a.out)
    repf = open(os.path.join(a.out, "rebuild_report.jsonl"), "a")
    ctx = get_context("fork")
    with ctx.Pool(a.workers, initializer=init, initargs=(a.gguf,)) as pool:
        for sh in shards:
            dst = os.path.join(a.out, sh)
            if os.path.exists(dst):
                print("skip", sh, flush=True); continue
            spath = os.path.join(a.qwen, sh)
            d0, hdr = st_header(spath)
            names = [n for n in hdr if n != "__metadata__"]
            # Big tensors first so the pool's tail is short; the vocabulary planes alone (memory).
            names.sort(key=lambda n: -int(np.prod(hdr[n]["shape"])))
            vocab = [n for n in names if n in (P + "embed_tokens.weight", "lm_head.weight")]
            jobs = [(n, spath, d0, hdr[n], tmp, a.k, a.tau) for n in names if n not in vocab]
            t0 = time.time()
            done = {}
            for n in vocab:
                init(a.gguf) if G_READER is None else None
                hf, p, rep = work((n, spath, d0, hdr[n], tmp, a.k, a.tau))
                done[hf] = p; repf.write(json.dumps(rep) + "\n"); repf.flush()
                print(json.dumps(rep), flush=True)
            for hf, p, rep in pool.imap_unordered(work, jobs):
                done[hf] = p
                repf.write(json.dumps(rep) + "\n"); repf.flush()
                if rep.get("mode", "").startswith("base +") or "differs" in rep.get("mode", "") or rep.get("warn"):
                    print(json.dumps(rep), flush=True)
            # Assemble the shard: same names, bf16, same order as the base header.
            order = [n for n in hdr if n != "__metadata__"]
            meta, off = {}, 0
            for n in order:
                sz = os.path.getsize(done[n])
                meta[n] = {"dtype": "BF16", "shape": hdr[n]["shape"], "data_offsets": [off, off + sz]}
                off += sz
            meta["__metadata__"] = {"format": "pt"}
            hb = json.dumps(meta, separators=(",", ":")).encode()
            hb += b" " * ((8 - len(hb) % 8) % 8)
            with open(dst + ".part", "wb") as f:
                f.write(struct.pack("<Q", len(hb))); f.write(hb)
                for n in order:
                    with open(done[n], "rb") as g:
                        shutil.copyfileobj(g, f, 1 << 24)
                    os.remove(done[n])
            os.rename(dst + ".part", dst)
            if a.delete_base:
                os.remove(spath)
            print(f"SHARD {sh} {len(order)} tensors {off/1e9:.2f} GB {time.time()-t0:.0f} s", flush=True)
    print("REBUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
