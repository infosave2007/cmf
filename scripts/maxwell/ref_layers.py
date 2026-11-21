"""Reference forward of Maxwell-V1-Flash, one decoder layer on the card at a time.

The bf16 checkpoint is 73.6 GB: it fits neither a 24 GB card nor 52 GB of RAM, so the
layers of the reference `modeling_maxwell_moe.py` are built and loaded one by one, every
job's hidden states carried from layer to layer. One pass over the checkpoint serves all
jobs.

python3 -I ref_layers.py MODEL_DIR JOBS.json OUT.json

JOBS.json: a list of
  {"name": "wiki", "kind": "ppl", "text": "wiki.test.raw", "windows": 12, "len": 512}
      perplexity over the windows `cortiq ppl --windows N --window-len L` scores:
      the text tokenized whole (no special tokens), window k at k*stride,
      stride = (T - L - 1) // N, every prediction inside a window scored;
  {"name": "capital", "kind": "greedy", "prompt": "<rendered prompt>", "answer": "<cortiq's text>"}
      teacher-forced check of a greedy answer: the reference's argmax at every answer
      position must be the answer's next token (then its own greedy answer is the same);
  {"name": "d", "kind": "dump", "prompt": "raw text" | "ids": [...], "positions": [0, -1], "dir": "ref_dump"}
      every layer's streams at those positions (-1: the last) as f32 files
      (`streams_l<layer>_p<pos>.f32`, the layout of CMF_MAXWELL_DUMP).
"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

md, jobs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = AutoConfig.from_pretrained(md, trust_remote_code=True)
cls = get_class_from_dynamic_module("modeling_maxwell_moe.MaxwellMoeForCausalLM", md)
M = sys.modules[cls.__module__]
tok = AutoTokenizer.from_pretrained(md, trust_remote_code=True)
index = json.load(open(os.path.join(md, "model.safetensors.index.json")))["weight_map"]
dev = torch.device("cuda")
n = cfg.num_residual_streams


def state(prefix):
    """The tensors under `prefix` on the card, in the checkpoint's own dtypes."""
    by = {}
    for k, shard in index.items():
        if k.startswith(prefix):
            by.setdefault(shard, []).append(k)
    sd = {}
    for shard, keys in by.items():
        with safe_open(os.path.join(md, shard), framework="pt", device="cuda") as f:
            for k in keys:
                sd[k[len(prefix):]] = f.get_tensor(k)
    return sd


def build(make, prefix):
    with torch.device("meta"):
        m = make()
    # assign: the weights stay bf16 and the routers' selection bias stays f32
    # (a bf16 bias collapses the routing, see `_fix_expert_bias_precision`)
    m.load_state_dict(state(prefix), strict=True, assign=True)
    return m.eval()


jobs = json.load(open(jobs_path))
batches = []  # (job, ids [B, L])
for j in jobs:
    if j["kind"] == "ppl":
        ids = tok(open(j["text"], encoding="utf-8").read(), add_special_tokens=False)["input_ids"]
        T, N, L = len(ids), j["windows"], j["len"]
        stride = (T - L - 1) // N  # cortiq PplWindows::offsets
        x = torch.tensor([ids[k * stride:k * stride + L] for k in range(N)])
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

t0 = time.time()
emb = state("model.embed_tokens.")["weight"]
rot = M.MaxwellMoeRotaryEmbedding(config=cfg).to(dev)
hs, rope = [], []
for j, x in batches:
    h = M.HyperConnectionModule.input_expand(emb[x], n)
    pos = torch.arange(x.shape[1], device=dev)[None]
    hs.append(h)
    rope.append((pos, rot(h, pos)))
del emb

for li in range(cfg.num_hidden_layers):
    layer = build(lambda: M.MaxwellMoeDecoderLayer(cfg, li), f"model.layers.{li}.")
    for bi, (j, x) in enumerate(batches):
        pos, pe = rope[bi]
        # long batches a few rows at a time: the eager sparse attention gathers
        # every row's 4096 keys
        rows = max(1, 12 * 512 // x.shape[1] // 2)
        outs = []
        for r0 in range(0, x.shape[0], rows):
            with torch.no_grad():
                outs.append(layer(hs[bi][r0:r0 + rows], attention_mask=None, position_ids=pos, past_key_value=None, use_cache=False, cache_position=None, position_embeddings=pe)[0])
        hs[bi] = torch.cat(outs)
        if j["kind"] == "dump":
            os.makedirs(j["dir"], exist_ok=True)
            for p in j["positions"]:
                hs[bi][0, p].float().cpu().numpy().tofile(os.path.join(j["dir"], f"streams_l{li}_p{p}.f32"))
    del layer
    torch.cuda.empty_cache()
    print(f"layer {li} done, {time.time() - t0:.0f} s", flush=True)

head = build(lambda: M.HCHead(cfg), "model.hc_head.")
norm = build(lambda: M.MaxwellMoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), "model.norm.")
lm = state("lm_head.")["weight"].float()
res = []
for (j, x), h in zip(batches, hs):
    out = {"name": j["name"], "kind": j["kind"]}
    if j["kind"] == "ppl":
        nll, cnt = 0.0, 0
        for r in range(x.shape[0]):
            with torch.no_grad():
                lg = F.linear(norm(head(h[r:r + 1]))[0].float(), lm)
            lp = torch.log_softmax(lg[:-1], -1).gather(1, x[r, 1:, None])[:, 0]
            nll -= lp.sum().item()
            cnt += x.shape[1] - 1
        out.update(tokens=j["tokens"], windows=x.shape[0], len=x.shape[1], ppl=math.exp(nll / cnt))
    else:
        with torch.no_grad():
            xn = norm(head(h))[0].float()
            lg = F.linear(xn, lm)
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
                xn[p].cpu().numpy().tofile(os.path.join(j["dir"], f"final_l{cfg.num_hidden_layers}_p{p}.f32"))
            top = lg[-1].topk(5)
            out.update(ids=j["ids"], positions=j["positions"], argmax=[int(am[p]) for p in j["positions"]], argmax_text=[tok.decode([int(am[p])]) for p in j["positions"]],
                       top5=[[tok.decode([int(i)]), float(v)] for v, i in zip(top.values, top.indices)])
    res.append(out)
    print(json.dumps(out, ensure_ascii=False), flush=True)
json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
print(f"REF_DONE {time.time() - t0:.0f} s")
