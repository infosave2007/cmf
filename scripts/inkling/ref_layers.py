"""Reference forward of Inkling-2-Mini, one layer application on the card at a time.

The decoder layers are the reference's own modules (transformers PR #51050,
`modeling_inkling2_loop.py`); the routed experts are decoded from NVFP4 / MXFP8 with
the PR's `Inkling2LoopDequantizeExperts` for each application. The checkpoint's
experts take 49 GB in bf16, more than the VM's RAM: no layer is resident between
applications. Everything runs in float32 (the NVFP4 values exactly, no bf16 rounding).
The model's loop is transcribed from `Inkling2LoopTextModel.forward` (no cache).

python3 -I ref_layers.py MODEL_DIR JOBS.json OUT.json

JOBS.json: a list of
  {"name": "wiki", "kind": "ppl", "text": "wiki.test.raw", "windows": 12, "len": 512}
      perplexity over the windows `cortiq ppl --windows N --window-len L` scores;
  {"name": "q1", "kind": "greedy", "prompt": "<rendered prompt>", "answer": "<cortiq's text>"}
      teacher-forced check of a greedy answer;
  {"name": "d", "kind": "dump", "prompt": "raw text" | "ids": [...], "positions": [0, -1], "dir": "ref_dump"}
      every application's output at those positions (`h_s<slot>_p<pos>.f32`, the layout
      of CMF_INKLING_DUMP) and the final normed hidden (`final_s48_p<pos>.f32`).
"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.core_model_loading import Inkling2LoopDequantizeExperts

md, jobs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = AutoConfig.from_pretrained(md)
tc = cfg.text_config
tc._attn_implementation = "eager"
tok = AutoTokenizer.from_pretrained(md)
dev = torch.device("cuda")
DT = torch.float32
st = safe_open(os.path.join(md, "model-00001-of-00001.safetensors"), framework="pt", device="cpu")
keys = set(st.keys())
L = "model.llm."
dq = Inkling2LoopDequantizeExperts()


class _F32Target:
    """The dequantizer casts to the target parameter's dtype: float32 here."""

    def get_parameter(self, name):
        return torch.empty(0, dtype=torch.float32)


def g(name, dtype=DT):
    return st.get_tensor(name).to(dev).to(dtype)


def experts(li):
    """gate_up_proj [E, 2I, H] (gate rows first) and down_proj [E, H, I], float32."""
    e = f"{L}layers.{li}.mlp.experts."
    out = []
    for w in ("w13", "w2"):
        srcs = [f"{e}{w}_weight", f"{e}{w}_weight_scale", f"{e}{w}_weight_amax", f"{e}{w}_weight_outer_scale_denominator"]
        d = {s: st.get_tensor(s).to(dev) for s in srcs if s in keys}
        pats = [s for s in srcs]
        r = dq.convert(d, pats, ["x"], model=_F32Target(), full_layer_name="x")["x"]
        out.append(r)
    gu, dn = out
    # w13 rows interleave gate and up (transformers' Interleave(dim=1))
    E, R, C = gu.shape
    gu = gu.reshape(E, R // 2, 2, C).transpose(1, 2).reshape(E, R, C)
    return gu.to(DT), dn.to(DT)


def layer_state(li):
    s = f"{L}layers.{li}."
    sliding = tc.layer_types[li] == "hybrid_sliding"
    sd = {}
    a = f"{s}attn."
    ren = {
        "wq_du.weight": "self_attn.q_proj.linear.weight",
        "wk_dv.weight": "self_attn.k_proj.linear.weight",
        "wv_dv.weight": "self_attn.v_proj.linear.weight",
        "wo_ud.weight": "self_attn.o_proj.linear.weight",
        "gate_attn.weight": "self_attn.o_proj.gate_proj.weight",
        "attn_sink_proj.weight": "self_attn.sink_proj.weight",
        "wr_du.weight": "self_attn.relative_position_bias.linear.weight",
        "rel_logits_proj.proj": "self_attn.relative_position_bias.proj",
        "q_l2_clip.scale": "self_attn.q_proj.norm.scale",
        "rel_l2_clip.scale": "self_attn.relative_position_bias.norm.scale",
    }
    if sliding:
        ren.update({"k_sconv.weight": "self_attn.k_proj.sconv.conv1d.weight", "v_sconv.weight": "self_attn.v_proj.sconv.conv1d.weight"})
    else:
        ren.update({
            "k_sconv.static_w": "self_attn.k_proj.sconv.static_weight", "k_sconv.weight_proj": "self_attn.k_proj.sconv.weight_proj.weight",
            "v_sconv.static_w": "self_attn.v_proj.sconv.static_weight", "v_sconv.weight_proj": "self_attn.v_proj.sconv.weight_proj.weight",
        })
    for k, v in ren.items():
        sd[v] = g(a + k)
    sd["post_attention_convolution.conv1d.weight"] = g(f"{s}attn_sconv.weight")
    sd["post_mlp_convolution.conv1d.weight"] = g(f"{s}mlp_sconv.weight")
    sd["post_attention_layernorm.weight"] = g(f"{s}mlp_norm.weight")
    if sliding:
        sd["input_layernorm.weight"] = g(f"{s}attn_norm.weight")
    else:
        for x in "qkv":
            sd[f"{x}_input_layernorm.weight"] = g(f"{s}{x}_input_norm.weight")
    sd["mlp.gate.weight"] = g(f"{s}mlp.gate.weight")
    sd["mlp.gate.e_score_correction_bias"] = g(f"{s}mlp.gate.bias")
    sd["mlp.experts.gate_up_proj"], sd["mlp.experts.down_proj"] = experts(li)
    if li > 0:
        q = g(f"{L}attn_res.attn_queries.{li - 1}")
        sd["attention_input.query"] = q
    sd["mlp_input.query"] = g(f"{L}attn_res.mlp_queries.{li}")
    if tc.loop_layer_index_start <= li < tc.loop_layer_index_end:
        for src, dst in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            c = f"{s}controllers.{src}."
            sd[f"{dst}_message_gate.down_proj.weight"] = g(c + "down_weight")
            sd[f"{dst}_message_gate.forget_gate_proj.weight"] = g(c + "forget_weight")
            sd[f"{dst}_message_gate.message_gate_proj.weight"] = g(c + "gain_weight")
    return sd


def build(li):
    sliding = tc.layer_types[li] == "hybrid_sliding"
    loop = tc.loop_layer_index_start <= li < tc.loop_layer_index_end
    cls = {(False, False): M.Inkling2LoopFullDecoderLayer, (True, False): M.Inkling2LoopSlidingDecoderLayer,
           (False, True): M.Inkling2LoopFullLoopDecoderLayer, (True, True): M.Inkling2LoopSlidingLoopDecoderLayer}[(sliding, loop)]
    with torch.device("meta"):
        m = cls(tc, li)
    m.load_state_dict(layer_state(li), strict=True, assign=True)
    return m.eval()


jobs = json.load(open(jobs_path))
batches = []
for j in jobs:
    if j["kind"] == "ppl":
        ids = tok(open(j["text"], encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
        T, N, Ln = len(ids), j["windows"], j["len"]
        stride = (T - Ln - 1) // N  # cortiq PplWindows::offsets
        x = torch.tensor([ids[k * stride:k * stride + Ln] for k in range(N)])
        j["tokens"] = T
    elif j["kind"] == "greedy":
        p = tok(j["prompt"], add_special_tokens=False)["input_ids"]
        a = tok(j["prompt"] + j["answer"], add_special_tokens=False)["input_ids"]
        assert a[:len(p)] == p, f"{j['name']}: the answer changes the prompt's tokens"
        j["prompt_len"], x = len(p), torch.tensor([a])
    else:
        if "prompt" in j:
            j["ids"] = tok(j["prompt"], add_special_tokens=False)["input_ids"]
        j["positions"] = [p % len(j["ids"]) for p in j["positions"]]
        x = torch.tensor([j["ids"]])
    batches.append((j, x.to(dev)))


def masks(S):
    i = torch.arange(S, device=dev)
    d = i[:, None] - i[None, :]
    neg = torch.finfo(DT).min
    full = torch.where(d >= 0, 0.0, neg).to(DT)[None, None]
    slid = torch.where((d >= 0) & (d < tc.sliding_window_size), 0.0, neg).to(DT)[None, None]
    return {"full_attention": full, "sliding_attention": slid}, {"full_attention": d, "sliding_attention": d}


t0 = time.time()
emb = g(f"{L}embed.weight")
en = M.Inkling2LoopRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
en.weight.data = g(f"{L}embed_norm.weight")
state = []  # per batch: h, blocks
for j, x in batches:
    h = en(emb[x])
    state.append({"h": h, "blocks": h.new_zeros(*h.shape[:-1], 0, h.shape[-1])})
del emb
torch.cuda.empty_cache()
mi = M.Inkling2LoopMessageAttentionResidual(tc).to(dev)
q, kv = g(f"{L}middle_cycle_cross_pass_queries"), g(f"{L}middle_cycle_cross_pass_kv_queries")
q, kv = q.unflatten(0, (-1, 2)), kv.unflatten(0, (-1, 2))
mi.query.data = torch.cat([q[:, :1], kv, q[:, 1:]], dim=1)

start, end, passes, nl = tc.loop_layer_index_start, tc.loop_layer_index_end, tc.loop_num_passes, tc.num_hidden_layers
nloop = end - start
plan = [(li, 0) for li in range(start)] + [(li, p) for p in range(passes) for li in range(start, end)] + [(li, 0) for li in range(end, nl)]


def slot(li, p):
    return li if p == 0 else nl + (p - 1) * nloop + (li - start)


def run(layer, li, p, bi, rows):
    j, x = batches[bi]
    s = state[bi]
    B, S = x.shape
    am, rd = masks(S)
    pos = torch.arange(S, device=dev)[None]
    typ = layer.attention_type
    outs = []
    for r0 in range(0, B, rows):
        sl = slice(r0, r0 + rows)
        kw = dict(attention_mask=am[typ], relative_distance=rd[typ], conv_mask=None, cache=None, position_ids=pos)
        with torch.no_grad():
            if p > 0:
                msg = mi(s["bank"][sl], s["bank_out"][sl], li - start)
                outs.append(layer(s["h"][sl], s["blocks"][sl], msg, **kw))
            else:
                outs.append(layer(s["h"][sl], s["blocks"][sl], **kw))
    return torch.cat(outs)


cur = None
for ai, (li, p) in enumerate(plan):
    if cur is None or cur[0] != li:
        cur = (li, build(li))
    layer = cur[1]
    for bi, (j, x) in enumerate(batches):
        s = state[bi]
        if li == start:
            if p == 0:
                s["pre_h"], s["pre_blocks"], s["npre"] = s["h"], s["blocks"], s["blocks"].shape[-2]
            else:
                s["bank"] = s["blocks"][..., s["npre"] + 1:, :]
                s["bank_out"] = s["h"]
                s["h"], s["blocks"] = s["pre_h"], s["pre_blocks"]
        rows = max(1, 4096 // x.shape[1])
        new = run(layer, li, p, bi, rows)
        if li % tc.n_layers_per_attn_res_block == 0:
            s["blocks"] = torch.cat([s["blocks"], s["h"].unsqueeze(-2)], dim=-2)
        s["h"] = new
        if j["kind"] == "dump":
            os.makedirs(j["dir"], exist_ok=True)
            for q_ in j["positions"]:
                new[0, q_].float().cpu().numpy().tofile(os.path.join(j["dir"], f"h_s{slot(li, p)}_p{q_}.f32"))
    if ai + 1 == len(plan) or plan[ai + 1][0] != li:
        cur = None
        torch.cuda.empty_cache()
    print(f"application {ai} (layer {li} pass {p}) done, {time.time() - t0:.0f} s", flush=True)

fin = M.Inkling2LoopBlockAttentionResidual(tc).to(dev)
fin.query.data = g(f"{L}attn_res.output_query")
norm = M.Inkling2LoopRMSNorm(tc.hidden_size, eps=tc.rms_norm_eps).to(dev)
norm.weight.data = g(f"{L}norm.weight")
lm = g(f"{L}unembed.weight")[: tc.unpadded_vocab_size]
cap = tc.final_logit_softcapping


def logits_of(s, sl):
    with torch.no_grad():
        xn = norm(fin(s["blocks"][sl], s["h"][sl])[0])
        lg = F.linear(xn, lm)
        return xn, torch.tanh(lg / cap) * cap


res = []
for (j, x), s in zip(batches, state):
    out = {"name": j["name"], "kind": j["kind"]}
    if j["kind"] == "ppl":
        nll, cnt = 0.0, 0
        for r in range(x.shape[0]):
            _, lg = logits_of(s, slice(r, r + 1))
            for c0 in range(0, x.shape[1] - 1, 1024):
                c1 = min(c0 + 1024, x.shape[1] - 1)
                lp = torch.log_softmax(lg[0, c0:c1].double(), -1).gather(1, x[r, c0 + 1:c1 + 1, None])[:, 0]
                nll -= lp.sum().item()
            del lg
            cnt += x.shape[1] - 1
        out.update(tokens=j["tokens"], windows=x.shape[0], len=x.shape[1], ppl=math.exp(nll / cnt))
    else:
        xn, lg = logits_of(s, slice(0, 1))
        xn, lg = xn[0], lg[0]
        am = lg.argmax(-1)
        if j["kind"] == "greedy":
            pl = j["prompt_len"]
            want, got = x[0, pl:], am[pl - 1:-1]
            bad = (want != got).nonzero()[:, 0].tolist()
            top2 = lg[pl - 1:-1].topk(2, -1).values
            out.update(answer_tokens=int(want.numel()), mismatches=len(bad), first_mismatch=(bad[0] if bad else None),
                       min_margin=float((top2[:, 0] - top2[:, 1]).min()) if want.numel() else None,
                       next=tok.decode(am[-1:].tolist()))
            if bad:
                b = bad[0]
                out["at"] = {"want": tok.decode(want[b:b + 1].tolist()), "got": tok.decode(got[b:b + 1].tolist())}
        else:
            for p in j["positions"]:
                xn[p].float().cpu().numpy().tofile(os.path.join(j["dir"], f"final_s{len(plan)}_p{p}.f32"))
            top = lg[-1].topk(5)
            out.update(ids=j["ids"], positions=j["positions"], argmax=[int(am[p]) for p in j["positions"]], argmax_text=[tok.decode([int(am[p])]) for p in j["positions"]],
                       top5=[[tok.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)])
    res.append(out)
    print(json.dumps(out, ensure_ascii=False), flush=True)
json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"REF_DONE {time.time() - t0:.0f} s")
