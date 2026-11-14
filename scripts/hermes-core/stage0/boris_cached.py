"""Incremental (KV-cached) forward for Boris-2, for greedy generation.

The model's own `modeling_boris2.py` recomputes the whole prefix at every
step. This module runs the same arithmetic over the model's modules for a
chunk of new tokens on top of a cache, for ONE document (no padding, no EOS
inside the text):

  * per layer: the raw Q|K|V projection of the last two positions (the Canon
    convolution reads t-1 and t-2 of the RAW projection), and the rotated
    keys and the values of every position;
  * the last two token ids (the n-gram lookup reads t-1 and t-2).

`self_check` compares it with the model's own forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rms(h, w, eps):
    f = h.float()
    return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * w.float()


def _lin(x, layer):
    return F.linear(x.to(layer.weight.dtype), layer.weight).float()


class BorisCache:
    def __init__(self, model):
        self.m = model.model
        self.head = model.lm_head
        self.cfg = model.config
        self.reset()

    def reset(self):
        n = self.cfg.num_hidden_layers
        self.t = 0
        self.ids: list[int] = []
        self.raw = [None] * n   # (1, <=2, C): raw qkv of the last two positions
        self.k = [None] * n     # (1, heads, T, hd), rotated
        self.v = [None] * n

    @torch.inference_mode()
    def forward(self, new_ids: list[int]) -> torch.Tensor:
        """Feed `new_ids`; returns the logits after the last of them, (vocab,)."""
        m, cfg = self.m, self.cfg
        dev = m.embed_tokens.weight.device
        n, t0 = len(new_ids), self.t
        idx = torch.tensor([new_ids], device=dev)
        hd = cfg.hidden_size // cfg.num_attention_heads
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        pos = torch.arange(t0, t0 + n, device=dev).float()
        f = pos[None, :, None] * inv
        cos, sin = f.cos()[:, :, None], f.sin()[:, :, None]              # (1, n, 1, hd/2)

        # n-gram vectors of the new positions: lookup over the last two ids too.
        ctx = self.ids[-2:]
        win = torch.tensor([ctx + new_ids], device=dev)
        boundary = torch.zeros_like(win, dtype=torch.bool)
        if t0 - len(ctx) == 0:
            boundary[0, 0] = True
        ng = m.ngram(win, boundary)[:, len(ctx):]

        allowed = None
        if n > 1:
            j = torch.arange(t0 + n, device=dev)[None, :]
            i = torch.arange(t0, t0 + n, device=dev)[:, None]
            allowed = (j <= i)[None, None]                               # (1, 1, n, T)

        h = m.embed_tokens(idx).float()
        for li, blk in enumerate(m.layers):
            if li == cfg.ngram_insert_after:
                h = h + ng
            a = blk.self_attn
            x = _rms(h, blk.input_layernorm.weight, blk.input_layernorm.eps)
            raw = _lin(x, a.qkv_proj)                                    # (1, n, C)
            prev = self.raw[li]
            hist = raw if prev is None else torch.cat([prev, raw], 1)
            p = hist.shape[1] - n                                        # rows of history in front
            zero = torch.zeros_like(raw[:, :1])
            s1 = torch.cat([hist[:, p - 1:p] if p >= 1 else zero, raw[:, :-1]], 1)
            two = torch.cat([hist[:, p - 2:p] if p >= 2 else torch.cat([zero] * (2 - p) + [hist[:, :p]], 1), raw], 1)
            s2 = two[:, :n]
            w = a.canon.float()
            qkv = raw + raw * w[:, 0] + s1 * w[:, 1] + s2 * w[:, 2]
            self.raw[li] = hist[:, -2:]
            qd, kd = a.n_head * a.hd, a.n_kv * a.hd
            q = qkv[..., :qd].view(1, n, a.n_head, a.hd)
            k = qkv[..., qd:qd + kd].view(1, n, a.n_kv, a.hd)
            v = qkv[..., qd + kd:].view(1, n, a.n_kv, a.hd)

            def rope(z):
                x1, x2 = z.chunk(2, dim=-1)
                return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)

            dt = a.qkv_proj.weight.dtype
            q, k, v = (z.transpose(1, 2).to(dt) for z in (rope(q), rope(k), v))
            if a.n_kv != a.n_head:
                k = k.repeat_interleave(a.n_head // a.n_kv, 1)
                v = v.repeat_interleave(a.n_head // a.n_kv, 1)
            self.k[li] = k if self.k[li] is None else torch.cat([self.k[li], k], 2)
            self.v[li] = v if self.v[li] is None else torch.cat([self.v[li], v], 2)
            o = F.scaled_dot_product_attention(q, self.k[li], self.v[li], attn_mask=allowed)
            h = h + _lin(o.transpose(1, 2).reshape(1, n, a.n_head * a.hd), a.o_proj)
            h = h + blk.mlp(_rms(h, blk.post_attention_layernorm.weight, blk.post_attention_layernorm.eps))
        self.t += n
        self.ids = (self.ids + new_ids)[-2:]
        last = _rms(h[:, -1], m.norm.weight, m.norm.eps)
        return F.linear(last, self.head.weight.float())[0]


@torch.inference_mode()
def self_check(model, ids: list[int], steps: int = 12) -> tuple[float, int]:
    """Max |logit| difference to the model's own forward after the prompt and
    after each of `steps` greedy tokens, and the number of argmax flips."""
    dev = model.lm_head.weight.device
    c = BorisCache(model)
    lg = c.forward(ids)
    seq, worst, flips = list(ids), 0.0, 0
    for _ in range(steps + 1):
        full = model(input_ids=torch.tensor([seq], device=dev), logits_to_keep=1).logits[0, -1]
        worst = max(worst, float((full - lg).abs().max()))
        flips += int(full.argmax() != lg.argmax())
        nxt = int(lg.argmax())
        seq.append(nxt)
        lg = c.forward([nxt])
    return worst, flips
