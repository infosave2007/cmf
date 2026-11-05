#!/usr/bin/env python3
"""Plain PyTorch transcription of the vendor's Dory forward (vLLM PR #60149,
vllm/model_executor/models/dory.py), no vLLM needed. The reference for cortiq.

    ref_dory.py --model DIR --out DIR --prompts prompts.json [--device cuda]

Writes for each prompt: the last row's logits and greedy continuations for
loops 1, 2, 4 with per-loop caches and for loops 2 with the last-loop cache,
the hidden state after the input block, each loop and the output block. Scores
windows of ids (NLL) when given --nll.
"""
import argparse, json, math, os, time
import numpy as np
import torch
from safetensors import safe_open

EPS = 1e-6


def justnorm(x, dim=-1):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)).to(x.dtype)


def slerp(a, b, lr):
    ya, yb = justnorm(a).float(), justnorm(b).float()
    return justnorm(ya + lr.float() * (yb - ya)).to(a.dtype)


def rope_tables(hd, theta, n, device):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64, device=device) / hd))
    t = torch.arange(n, dtype=torch.float64, device=device)
    ang = torch.outer(t, inv)
    return torch.cos(ang).float(), torch.sin(ang).float()


def rotate(x, cos, sin):
    # neox style: x [t, heads, hd]; cos/sin [t, 1, hd/2]
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class Dory:
    def __init__(self, path, device, dtype):
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        c = self.cfg
        self.dev, self.dtype = device, dtype
        self.w = {}
        with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k).to(dtype=dtype, device=device)
        self.h, self.ffn = c["hidden_size"], c["intermediate_size"]
        self.nh, self.nk, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.std = c["initializer_range"]
        self.alpha_scale = c["ngpt_alpha_init"] / self.std
        self.sqk_f = self.hd ** 0.5 / self.std
        g0 = c["attention_output_gate_logit_scale_init"] or self.h ** 0.5
        self.gate_f = g0 ** 0.5 / self.std
        self.suv_f = self.h ** 0.5
        self.loops = c["n_recurrent_loops"]
        self.kv = c["recurrent_kv_cache_mode"]
        self.types = c["layer_types"]
        self.window = c["sliding_window"]
        self.n_in, self.n_rec = c["n_input_layers"], c["n_recurrent_layers"]
        self.n = c["num_hidden_layers"]
        self.theta = {t: c["rope_parameters"][t]["rope_theta"] for t in ("sliding_attention", "full_attention")}
        self.rope = {}

    def attn(self, li, x, pos0, cache):
        """x [t, h]; cache: dict with k,v [T, nk, hd] of earlier rows of this layer + loop."""
        p = f"backbone.layers.{li}.mixer"
        w = self.w
        t = x.shape[0]
        q = x @ w[p + ".q_proj.weight"].T
        k = x @ w[p + ".k_proj.weight"].T
        v = x @ w[p + ".v_proj.weight"].T
        g = x @ w[p + ".g_proj.weight"].T
        kind = self.types[li]
        cos, sin = self.rope_for(kind, pos0 + t)
        c = cos[pos0:pos0 + t][:, None, :]
        s = sin[pos0:pos0 + t][:, None, :]
        q = rotate(q.view(t, self.nh, self.hd).float(), c, s)
        k = rotate(k.view(t, self.nk, self.hd).float(), c, s)
        v = v.view(t, self.nk, self.hd).float()
        sqk = (w[p + ".sqk"].float() * self.sqk_f).view(1, self.nh, self.hd)
        q = sqk.square() * justnorm(q)
        if getattr(self, "raw", None) is not None:
            self.raw(li, kind, k, q)
        k = justnorm(k)
        if cache is not None and cache.get("k") is not None:
            K = torch.cat((cache["k"], k), 0)
            V = torch.cat((cache["v"], v), 0)
        else:
            K, V = k, v
        T = K.shape[0]
        rep = self.nh // self.nk
        qp = q.permute(1, 0, 2)                                # [nh, t, hd]
        kp = K.permute(1, 2, 0).repeat_interleave(rep, dim=0)  # [nh, hd, T]
        vp = V.permute(1, 0, 2).repeat_interleave(rep, dim=0)  # [nh, T, hd]
        scores = torch.bmm(qp, kp) * (1.0 / math.sqrt(self.hd))  # [nh, t, T]
        qpos = torch.arange(T - t, T, device=x.device)[:, None]
        kpos = torch.arange(T, device=x.device)[None, :]
        mask = kpos <= qpos
        if kind == "sliding_attention":
            mask = mask & (kpos > qpos - self.window)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        a = torch.softmax(scores, dim=-1)
        if getattr(self, "probe", None) is not None:
            self.probe(li, kind, a)
        out = torch.bmm(a, vp).permute(1, 0, 2).reshape(t, self.nh * self.hd)  # [t, q]
        sg = (w[p + ".s_gate"].float() * self.gate_f).square()
        out = out * torch.sigmoid(g.float() * sg)
        out = justnorm(out)
        out = out.to(self.dtype) @ w[p + ".o_proj.weight"].T
        if cache is not None:
            cache["k"], cache["v"] = K, V
            if kind == "sliding_attention" and K.shape[0] > self.window:
                cache["k"], cache["v"] = K[-self.window:], V[-self.window:]
        return out

    def rope_for(self, kind, n):
        tab = self.rope.get(kind)
        if tab is None or tab[0].shape[0] < n:
            tab = rope_tables(self.hd, float(self.theta[kind]), max(n, 8192), self.dev)
            self.rope[kind] = tab
        return tab

    def mlp(self, li, x):
        p = f"backbone.layers.{li}.mixer"
        w = self.w
        g = (x @ w[p + ".gate_proj.weight"].T).float() * (w[p + ".suv_gate"].float() * self.suv_f)
        u = (x @ w[p + ".up_proj.weight"].T).float() * (w[p + ".suv_up"].float() * self.suv_f)
        m = justnorm(torch.nn.functional.silu(g) * u)
        return m.to(self.dtype) @ w[p + ".down_proj.weight"].T

    def layer(self, li, x, pos0, caches, key):
        p = f"backbone.layers.{li}"
        if self.types[li] != "mlp":
            cache = caches.setdefault(key, {}) if caches is not None else None
            out = self.attn(li, x, pos0, cache)
            al = self.w[p + ".attn_alpha"]
        else:
            out = self.mlp(li, x)
            al = self.w[p + ".mlp_alpha"]
        return slerp(x, out, (al.float() * self.alpha_scale).abs())

    @torch.no_grad()
    def forward(self, ids, pos0=0, caches=None, loops=None, kv=None, dump=None):
        loops = loops or self.loops
        kv = kv or self.kv
        h = self.w["backbone.embeddings.weight"][ids]
        n_in, n_rec = self.n_in, self.n_rec
        for li in range(n_in):
            h = self.layer(li, h, pos0, caches, (li, 0))
        input_h = h
        if dump is not None:
            dump["input"] = h.float().clone()
        state = input_h
        for i in range(loops):
            h = justnorm(state + input_h)
            for li in range(n_in, n_in + n_rec):
                # per_loop: an own cache per loop. last_loop: one cache, each
                # loop overwrites this chunk's rows (loop i reads loop-i rows of
                # this chunk and last-loop rows of the earlier ones).
                key = (li, i) if kv == "per_loop" else (li, 0)
                if kv == "last_loop" and caches is not None and i > 0 and self.types[li] != "mlp":
                    cc = caches.get(key)
                    if cc is not None and cc.get("k") is not None:
                        cc["k"] = cc["k"][:-h.shape[0]]
                        cc["v"] = cc["v"][:-h.shape[0]]
                h = self.layer(li, h, pos0, caches, key)
            state = justnorm(h)
            if dump is not None:
                dump[f"loop{i}"] = state.float().clone()
        h = state
        for li in range(n_in + n_rec, self.n):
            h = self.layer(li, h, pos0, caches, (li, 0))
        if dump is not None:
            dump["output"] = h.float().clone()
        return (h.to(self.dtype) @ self.w["lm_head.weight"].T).float() * (self.w["lm_head.logit_scale"].float() / self.std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True, help="JSON: [{name, ids}]")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--greedy", type=int, default=64)
    ap.add_argument("--nll", default="", help="JSON [{name, ids}] of windows to score")
    ap.add_argument("--chunk", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dt = {"f32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    t0 = time.time()
    m = Dory(a.model, a.device, dt)
    print(f"loaded in {time.time() - t0:.1f}s, loops {m.loops}, kv {m.kv}", flush=True)
    res = {}
    for p in json.load(open(a.prompts)):
        name, ids = p["name"], p["ids"]
        ids_t = torch.tensor(ids, device=a.device)
        dump = {}
        res[name] = {"ids": ids}
        for cfg_loops, cfg_kv in ((m.loops, "per_loop"), (1, "per_loop"), (4, "per_loop"), (m.loops, "last_loop")):
            tag = f"l{cfg_loops}_{cfg_kv}"
            caches = {}
            pos = 0
            last = None
            t1 = time.time()
            main_run = tag == f"l{m.loops}_per_loop"
            for s in range(0, len(ids), a.chunk):
                e = min(s + a.chunk, len(ids))
                lg = m.forward(ids_t[s:e], s, caches, loops=cfg_loops, kv=cfg_kv, dump=dump if main_run else None)
                last = lg[-1]
                pos = e
            t2 = time.time()
            np.save(os.path.join(a.out, f"{name}.{tag}.logits.npy"), last.cpu().numpy())
            if main_run:
                for k, v in dump.items():
                    np.save(os.path.join(a.out, f"{name}.{k}.npy"), v[-1].cpu().numpy())
            out = []
            nxt = int(last.argmax())
            top5 = torch.topk(last, 5)
            for _ in range(a.greedy):
                out.append(nxt)
                if nxt in (2, 11):
                    break
                lg = m.forward(torch.tensor([nxt], device=a.device), pos, caches, loops=cfg_loops, kv=cfg_kv)
                pos += 1
                nxt = int(lg[-1].argmax())
            t3 = time.time()
            res[name][tag] = {"top5": top5.indices.tolist(), "top5v": top5.values.tolist(), "greedy": out,
                              "prefill_s": t2 - t1, "decode_s": t3 - t2}
            print(name, tag, "top5", top5.indices.tolist(), [round(x, 3) for x in top5.values.tolist()],
                  f"prefill {t2-t1:.2f}s decode {t3-t2:.2f}s/{len(out)}", flush=True)
            del caches
        json.dump(res, open(os.path.join(a.out, "ref.json"), "w"), indent=0)
    if a.nll:
        scores = {}
        for w in json.load(open(a.nll)):
            ids = torch.tensor(w["ids"], device=a.device)
            caches = {}
            s = 0.0
            n = 0
            for b in range(0, len(ids) - 1, a.chunk):
                e = min(b + a.chunk, len(ids) - 1)
                lg = m.forward(ids[b:e], b, caches)
                s += torch.nn.functional.cross_entropy(lg, ids[b + 1:e + 1], reduction="sum").item()
                n += e - b
            scores[w["name"]] = {"nll": s / n, "ppl": math.exp(s / n), "tokens": n}
            print(w["name"], scores[w["name"]], flush=True)
        json.dump(scores, open(os.path.join(a.out, "nll.json"), "w"), indent=0)
    res["_done"] = True
    json.dump(res, open(os.path.join(a.out, "ref.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
