#!/usr/bin/env python3
"""Hy4-flash reference on one card: the transformers PR #50903 modules, the dense part resident
in bf16, the routed experts and the Engram tables read from the checkpoint on demand.

The released FP8 checkpoint is used for the experts (e4m3, 128x128 block scales) and every tensor
present in the bf16 shards of `--bf16` is taken from there instead (the attention projections the
FP8 release quantized).

usage:
  hy4_ref.py gen   OUT.json  --prompts prompts.json [--new 64]
  hy4_ref.py score OUT.json  --text file.txt [--tokens 1024]       (teacher-forced logits/ppl)
  hy4_ref.py check OUT.json  --ids ids.json                         (argmax at every position)
"""
import argparse, json, math, os, struct, sys, time

import numpy as np
import torch
import torch.nn.functional as F

from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as M

FP8 = os.environ.get("HY4_FP8", "/root/hy4/dl/Hy4-flash-FP8")
BF16 = os.environ.get("HY4_BF16", "/root/hy4/dl/Hy4-flash-bf16part")
DEV = "cuda"
# HY4_FP32=1: the whole forward in f32 (weights widened from their bf16 / e4m3
# values), so a comparison is not dominated by bf16 rounding amplified through
# near-tie expert choices
DT = torch.float32 if os.environ.get("HY4_FP32") == "1" else torch.bfloat16


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


class Store:
    """Tensor name -> (file, meta), the bf16 shards winning over the FP8 release."""

    def __init__(self):
        self.where = {}
        for root in (FP8, BF16):
            if not os.path.isdir(root):
                continue
            for fn in sorted(os.listdir(root)):
                if not fn.endswith(".safetensors"):
                    continue
                p = os.path.join(root, fn)
                base, h = header(p)
                for k, v in h.items():
                    if k == "__metadata__":
                        continue
                    self.where[k] = (p, base, v)
        self.handles = {}

    def handle(self, p):
        if p not in self.handles:
            self.handles[p] = safe_open(p, "pt", device="cpu")
        return self.handles[p]

    def has(self, k):
        return k in self.where

    def get(self, k):
        p, _, v = self.where[k]
        t = self.handle(p).get_tensor(k)
        if v["dtype"] == "F8_E4M3":
            s = self.get(k + "_scale_inv") if k + "_scale_inv" in self.where else self.get(k.replace(".weight", ".weight_scale_inv"))
            return dequant(t, s)
        return t

    def slice0(self, k, i):
        p, _, v = self.where[k]
        return self.handle(p).get_slice(k)[i]

    def memmap(self, k):
        p, base, v = self.where[k]
        assert v["dtype"] == "BF16", v["dtype"]
        a, b = v["data_offsets"]
        return np.memmap(p, dtype=np.uint16, mode="r", offset=base + a, shape=tuple(v["shape"]))


def dequant(w8, s):
    """e4m3 with one scale per 128x128 block (scale_inv = the multiplier)."""
    w = w8.to(DEV).to(torch.float32)
    s = s.to(DEV).to(torch.float32)
    rows, cols = w.shape[-2], w.shape[-1]
    s = s.repeat_interleave(128, dim=-2)[..., :rows, :].repeat_interleave(128, dim=-1)[..., :cols]
    return (w * s).to(DT)


class LazyExperts(torch.nn.Module):
    """HYV4FlashExperts with the bank read from the checkpoint per hit expert."""

    def __init__(self, store, li, cfg):
        super().__init__()
        self.store, self.li = store, li
        self.num_experts = cfg.n_routed_experts
        self.swiglu_limit = cfg.swiglu_limit
        self.gu = f"model.layers.{li}.mlp.experts.gate_up_proj"
        self.dn = f"model.layers.{li}.mlp.experts.down_proj"

    def expert(self, name, e):
        w = self.store.slice0(name, e)
        sname = name + "_scale_inv"
        if self.store.where[name][2]["dtype"] == "F8_E4M3":
            return dequant(w, self.store.slice0(sname, e))
        return w.to(DEV).to(DT)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit.flatten().tolist():
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            gu = self.expert(self.gu, expert_idx)
            dn = self.expert(self.dn, expert_idx)
            cur = F.linear(hidden_states[token_idx], gu)
            gate, up = cur.chunk(2, dim=-1)
            gate = gate.clamp(min=None, max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            cur = F.silu(gate) * up
            cur = F.linear(cur, dn) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final


class LazyEngramTable(torch.nn.Module):
    """HYV4FlashEngramEmbedding over a memory map of the bf16 table."""

    def __init__(self, store, li, head_vocab_sizes):
        super().__init__()
        self.table = store.memmap(f"model.layers.{li}.engram_layer.multi_head_embedding.embedding.weight")
        self.offsets = torch.tensor([0] + head_vocab_sizes[:-1]).cumsum(0)

    def forward(self, hash_ids):
        ids = (hash_ids.clamp_min(0).cpu() + self.offsets).numpy()
        rows = torch.from_numpy(np.ascontiguousarray(self.table[ids.reshape(-1)])).view(torch.bfloat16)
        emb = rows.reshape(*ids.shape, -1).to(DEV).to(DT)
        return emb.masked_fill(hash_ids.unsqueeze(-1) < 0, 0.0)


RENAMES = [
    (".self_attn.learnable_sink_param", ".self_attn.sinks"),
    (".self_attn.linear_gate.weight", ".self_attn.gate_proj.weight"),
    (".hc_attn_layer.hc_pre.hc_fn", ".attn_hc.fn"),
    (".hc_attn_layer.hc_pre.hc_base", ".attn_hc.base"),
    (".hc_attn_layer.hc_pre.hc_scale", ".attn_hc.scale"),
    (".hc_mlp_layer.hc_pre.hc_fn", ".ffn_hc.fn"),
    (".hc_mlp_layer.hc_pre.hc_base", ".ffn_hc.base"),
    (".hc_mlp_layer.hc_pre.hc_scale", ".ffn_hc.scale"),
    (".hc_head.hc_head_fn", ".hc_head.hc_fn"),
    (".hc_head.hc_head_base", ".hc_head.hc_base"),
    (".hc_head.hc_head_scale", ".hc_head.hc_scale"),
]

FP32_KEEP = ["e_score_correction_bias", "attn_hc.", "ffn_hc.", "hc_head.", "sinks", "key_value_projection",
             "key_gate_projection", "intra_group_ape"]


def build(store, cfg):
    tc = cfg.text_config
    tc._attn_implementation = "eager"
    torch.set_default_dtype(DT)
    with torch.device("meta"):
        model = M.HYV4FlashTextModel(tc)
    torch.set_default_dtype(torch.float32)
    for li, layer in enumerate(model.layers):
        if isinstance(layer.mlp, M.HYV4FlashMoE):
            layer.mlp.experts = LazyExperts(store, li, tc)
        if layer.engram_layer is not None:
            eng = layer.engram_layer
            eng.multi_head_embedding = LazyEngramTable(store, li, eng.config_head_vocab_sizes)
    model.to_empty(device=DEV)
    for name, p in model.named_parameters():
        if any(k in name for k in FP32_KEEP):
            p.data = p.data.float()
    want = dict(model.named_parameters())
    want.update(dict(model.named_buffers()))
    loaded = set()
    for k in list(store.where):
        if not k.startswith("model.") or "dfly" in k or k.endswith("_scale_inv"):
            continue
        if ".mlp.experts." in k or "multi_head_embedding" in k:
            continue
        n = "." + k[len("model."):]
        for a, b in RENAMES:
            n = n.replace(a, b)
        n = n[1:]
        if n not in want:
            print("unused", k, file=sys.stderr)
            continue
        t = store.get(k).to(DEV)
        dst = want[n]
        assert tuple(dst.shape) == tuple(t.shape), (n, dst.shape, t.shape)
        dst.data.copy_(t.to(dst.dtype))
        loaded.add(n)
    missing = [n for n in want if n not in loaded and "inv_freq" not in n]
    print("not loaded:", missing, file=sys.stderr)
    # buffers the checkpoint does not carry
    for mod in model.modules():
        if isinstance(mod, M.HYV4FlashRotaryEmbedding):
            for lt in mod.layer_types:
                inv, _ = mod.compute_default_rope_parameters(tc, layer_type=lt)
                getattr(mod, f"{lt}_inv_freq").copy_(inv.to(DEV))
                getattr(mod, f"{lt}_original_inv_freq").copy_(inv.to(DEV))
        if isinstance(mod, M.HYV4FlashEngramLayer):
            mod.head_vocab_sizes.copy_(torch.tensor(mod.config_head_vocab_sizes))
            mod.hash_multipliers.copy_(mod.draw_hash_multipliers(int(mod.lookup_table.max()) + 1).to(DEV))
    lm = store.get("lm_head.weight").to(DEV).float()
    model.eval()
    return model, lm


def ngram_inputs(ids, tc, image_id=120120, video_id=120124):
    pad = tc.engram_pad_id
    shifted = [ids]
    for off in range(1, tc.engram_max_ngram_size):
        shifted.append(F.pad(ids, (off, 0), value=pad)[:, : ids.shape[1]])
    ng = torch.stack(shifted, dim=-1)
    mask = (ids != image_id) & (ids != video_id) & (ids != pad)
    return ng, mask


# HY4_DSA_WINDOW=1: the sparse layers given the sliding mask too (a diagnostic)
if os.environ.get("HY4_DSA_WINDOW") == "1":
    _orig_ccm = M.create_causal_mask
    M.create_causal_mask = lambda **kw: M.create_sliding_window_causal_mask(**kw)


@torch.no_grad()
def forward(model, lm, tc, ids, cache=None, start=0):
    full = ids
    ng, mask = ngram_inputs(full, tc)
    step = full[:, start:]
    out = model(input_ids=step, cache=cache, use_cache=cache is not None,
                ngram_tokens=ng[:, start:], engram_mask=mask[:, start:])
    h = out.last_hidden_state.float()
    return h @ lm.t()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("out")
    ap.add_argument("--prompts")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--text")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--ids")
    a = ap.parse_args()
    cfg = AutoConfig.from_pretrained(FP8)
    tc = cfg.text_config
    tok = AutoTokenizer.from_pretrained(FP8)
    store = Store()
    t0 = time.time()
    model, lm = build(store, cfg)
    print(f"built in {time.time() - t0:.1f} s, cuda {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    res = []
    if a.mode == "gen":
        for p in json.load(open(a.prompts)):
            if isinstance(p, str):
                text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True, reasoning_effort="no_think")
            else:
                text = p["raw"]
            ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEV)
            cache = DynamicCache(config=tc)
            prompt_ids = ids[0].tolist()
            t1 = time.time()
            logits = forward(model, lm, tc, ids, cache, 0)
            first = logits[0, -1]
            gen, tops = [], []
            nxt = int(first.argmax())
            for i in range(a.new):
                gen.append(nxt)
                lg = logits[0, -1]
                top = torch.topk(lg, 5)
                tops.append([[int(x) for x in top.indices], [float(x) for x in top.values]])
                if nxt == tc.eos_token_id:
                    break
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
                logits = forward(model, lm, tc, ids, cache, ids.shape[1] - 1)
                nxt = int(logits[0, -1].argmax())
            dt = time.time() - t1
            r = {"prompt": p, "text": text, "prompt_ids": prompt_ids,
                 "gen_ids": gen, "gen_text": tok.decode(gen), "top5": tops, "seconds": dt}
            print(json.dumps({"prompt": str(p)[:60], "gen": r["gen_text"], "s": round(dt, 1)}, ensure_ascii=False), flush=True)
            res.append(r)
    elif a.mode == "score":
        text = open(a.text).read()
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids[:, : a.tokens].to(DEV)
        # HY4_DUMP=<dir>,<pos>,...: every layer's attention output and streams at those positions
        if os.environ.get("HY4_DUMP"):
            parts = os.environ["HY4_DUMP"].split(",")
            ddir, dpos = parts[0], [int(x) for x in parts[1:]]
            os.makedirs(ddir, exist_ok=True)
            def hook_layer(li):
                def f(mod, inp, out):
                    for p in dpos:
                        out[0, p].float().cpu().numpy().tofile(f"{ddir}/streams_l{li}_p{p}.f32")
                return f
            def hook_attn(li):
                def f(mod, inp, out):
                    for p in dpos:
                        out[0][0, p].float().cpu().numpy().tofile(f"{ddir}/attn_l{li}_p{p}.f32")
                return f
            def hook_eng(li):
                def f(mod, inp, out):
                    for p in dpos:
                        out[0, p].float().cpu().numpy().tofile(f"{ddir}/engout_l{li}_p{p}.f32")
                return f
            def hook_mem(li):
                # the looked-up rows after the fp8 round trip and the per-head norms
                def f(mod, inp, out):
                    eng = model.layers[li].engram_layer
                    emb = out.to(inp[0].dtype if False else out.dtype)
                    if eng.fp8_embedding:
                        emb = M._fp8_round_trip(emb.to(torch.bfloat16 if DT == torch.bfloat16 else DT))
                    emb = torch.stack([nrm(emb[:, :, h]) for h, nrm in enumerate(eng.lookup_per_head_norms)], dim=2).flatten(-2)
                    for p in dpos:
                        emb[0, p].float().cpu().numpy().tofile(f"{ddir}/engmem_l{li}_p{p}.f32")
                return f
            for li, layer in enumerate(model.layers):
                layer.register_forward_hook(hook_layer(li))
                layer.self_attn.register_forward_hook(hook_attn(li))
                if layer.engram_layer is not None:
                    layer.engram_layer.register_forward_hook(hook_eng(li))
                    layer.engram_layer.multi_head_embedding.register_forward_hook(hook_mem(li))
        logits = forward(model, lm, tc, ids)[0]
        lp = torch.log_softmax(logits[:-1], -1)
        nll = -lp.gather(1, ids[0, 1:, None])[:, 0]
        res = {"ids": ids[0].tolist(), "nll": nll.tolist(), "ppl": math.exp(float(nll.mean())),
               "argmax": logits.argmax(-1).tolist()}
        print("ppl", res["ppl"], flush=True)
    elif a.mode == "check":
        for item in json.load(open(a.ids)):
            ids = torch.tensor([item["ids"]], device=DEV)
            logits = forward(model, lm, tc, ids)[0]
            am = logits.argmax(-1).tolist()
            p0 = item["prompt_len"]
            agree = [am[i - 1] == item["ids"][i] for i in range(p0, len(item["ids"]))]
            top2 = torch.topk(logits, 2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).tolist()
            res.append({"name": item.get("name"), "agree": agree, "argmax": am, "margin": margin})
            print(item.get("name"), sum(agree), "/", len(agree), flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False)
    print("REF_DONE", flush=True)


if __name__ == "__main__":
    main()
