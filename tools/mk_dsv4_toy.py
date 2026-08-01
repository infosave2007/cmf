#!/usr/bin/env python3
"""Emit a tiny DeepSeek-V4 checkpoint with the release's real tensor names.

    python3 tools/mk_dsv4_toy.py /tmp/dsv4toy
    cortiq convert --model /tmp/dsv4toy --quant q2tp --output /tmp/dsv4toy.cmf
    cortiq run /tmp/dsv4toy.cmf --prompt "abc" --max-tokens 5

The weights are nonsense, so the text is too — what this proves is that the
whole path holds together: the converter's name map, the loader, the
hyper-connection stack, the hash and score routers, both compressor
variants and the indexer. It found three blockers the unit tests could not,
each of which stopped the real 100 GB model dead: the generic weight loader
demanding `q_proj`, the batched prefill walking an empty layer stack, and
the pair-decode path doing the same.

Proportions follow the release (head_dim > rope tail, index_head_dim >=
rope tail, ratio-4 layers overlapping) — deviate from those and you will
debug the toy instead of the engine.

Pure stdlib: safetensors is an 8-byte little-endian header length, a JSON
header, then the raw tensor bytes. Nothing here needs torch, which matters
because the point is to exercise OUR converter and loader, not a framework.
"""
import json, struct, math, sys, os

# Sizes are env-overridable so the same generator serves both roles: a
# few-second smoke checkpoint, and one big enough that a conversion can be
# interrupted mid-flight to test --resume.
def _env(name, default):
    return int(os.environ.get(name, default))

D          = _env("TOY_D", 64)        # hidden_size
NH         = 2                        # num_attention_heads
HD         = 32                       # head_dim (NOT hidden/heads — the point)
RD         = 8                        # qk_rope_head_dim
QLORA      = 16
OLORA      = 16
OGROUPS    = 2
NLAYERS    = _env("TOY_LAYERS", 4)
NHASH      = 2                        # layers 0..1 route by table
NEXP       = _env("TOY_EXPERTS", 8)
TOPK       = 2
MOE_INTER  = _env("TOY_INTER", 32)
VOCAB      = 128
HC         = 4
RATIO_MAP  = {2: 4, 3: 8}      # layer -> compress_ratio (4 => overlapping)
INDEXER_ON = {2}
IDX_HEADS  = 2
IDX_HD     = 64   # >= the rope tail, as in the release (128 vs 64)

def vals(n, seed):
    return [math.sin((i * 7 + seed * 13) * 0.017) * 0.35 for i in range(n)]

tensors = {}   # name -> (shape, list[float])

def put(name, shape, seed):
    n = 1
    for s in shape:
        n *= s
    tensors[name] = (shape, vals(n, seed))

seed = 1
def nx():
    global seed
    seed += 1
    return seed

put("embed.weight", [VOCAB, D], nx())
put("head.weight", [VOCAB, D], nx())
put("norm.weight", [D], nx())
put("hc_head_fn", [HC, HC * D], nx())
put("hc_head_base", [HC], nx())
put("hc_head_scale", [1], nx())

for li in range(NLAYERS):
    p = f"layers.{li}"
    put(f"{p}.attn_norm.weight", [D], nx())
    put(f"{p}.ffn_norm.weight", [D], nx())
    put(f"{p}.attn.wq_a.weight", [QLORA, D], nx())
    put(f"{p}.attn.q_norm.weight", [QLORA], nx())
    put(f"{p}.attn.wq_b.weight", [NH * HD, QLORA], nx())
    put(f"{p}.attn.wkv.weight", [HD, D], nx())
    put(f"{p}.attn.kv_norm.weight", [HD], nx())
    put(f"{p}.attn.wo_a.weight", [OGROUPS * OLORA, NH * HD // OGROUPS], nx())
    put(f"{p}.attn.wo_b.weight", [D, OGROUPS * OLORA], nx())
    put(f"{p}.attn.attn_sink", [NH], nx())
    # hyper-connections: [mix_hc, hc*dim] and [mix_hc]
    mix = (2 + HC) * HC
    for half in ("attn", "ffn"):
        put(f"{p}.hc_{half}_fn", [mix, HC * D], nx())
        put(f"{p}.hc_{half}_base", [mix], nx())
        put(f"{p}.hc_{half}_scale", [3], nx())
    # compressor: wkv/wgate are coff*head_dim wide, ape is [ratio, coff*hd]
    if li in RATIO_MAP:
        r = RATIO_MAP[li]
        coff = 2 if r == 4 else 1
        put(f"{p}.attn.compressor.wkv.weight", [coff * HD, D], nx())
        put(f"{p}.attn.compressor.wgate.weight", [coff * HD, D], nx())
        put(f"{p}.attn.compressor.norm.weight", [HD], nx())
        put(f"{p}.attn.compressor.ape", [r, coff * HD], nx())
    if li in INDEXER_ON:
        put(f"{p}.attn.indexer.wq_b.weight", [IDX_HEADS * IDX_HD, QLORA], nx())
        put(f"{p}.attn.indexer.weights_proj.weight", [IDX_HEADS, D], nx())
        put(f"{p}.attn.indexer.compressor.wkv.weight", [2 * IDX_HD, D], nx())
        put(f"{p}.attn.indexer.compressor.wgate.weight", [2 * IDX_HD, D], nx())
        put(f"{p}.attn.indexer.compressor.norm.weight", [IDX_HD], nx())
        put(f"{p}.attn.indexer.compressor.ape", [4, 2 * IDX_HD], nx())
    # router: hash layers carry a table, the rest a bias
    put(f"{p}.ffn.gate.weight", [NEXP, D], nx())
    if li < NHASH:
        tbl = [float(((t * TOPK + k) * 3) % NEXP) for t in range(VOCAB) for k in range(TOPK)]
        tensors[f"{p}.ffn.gate.tid2eid"] = ([VOCAB, TOPK], tbl)
    else:
        put(f"{p}.ffn.gate.bias", [NEXP], nx())
    for e in range(NEXP):
        put(f"{p}.ffn.experts.{e}.w1.weight", [MOE_INTER, D], nx())
        put(f"{p}.ffn.experts.{e}.w3.weight", [MOE_INTER, D], nx())
        put(f"{p}.ffn.experts.{e}.w2.weight", [D, MOE_INTER], nx())
    put(f"{p}.ffn.shared_experts.w1.weight", [MOE_INTER, D], nx())
    put(f"{p}.ffn.shared_experts.w3.weight", [MOE_INTER, D], nx())
    put(f"{p}.ffn.shared_experts.w2.weight", [D, MOE_INTER], nx())

out = sys.argv[1]
# A second argument shards the checkpoint, which is what exercises the
# converter's per-shard resume — a single file has nothing to resume from.
nshards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
os.makedirs(out, exist_ok=True)

def write_shard(path, items):
    header, blob, off = {}, bytearray(), 0
    for name, (shape, data) in items:
        raw = struct.pack(f"<{len(data)}f", *data)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    hj = json.dumps(header).encode()
    hj += b" " * ((-len(hj)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)

items = list(tensors.items())
if nshards <= 1:
    write_shard(os.path.join(out, "model.safetensors"), items)
else:
    per = -(-len(items) // nshards)
    wmap = {}
    for i in range(nshards):
        chunk = items[i * per:(i + 1) * per]
        if not chunk:
            continue
        fn = f"model-{i + 1:05d}-of-{nshards:05d}.safetensors"
        write_shard(os.path.join(out, fn), chunk)
        for name, _ in chunk:
            wmap[name] = fn
    json.dump({"metadata": {"total_size": sum(
        4 * len(d) for _, (_, d) in items)}, "weight_map": wmap},
        open(os.path.join(out, "model.safetensors.index.json"), "w"))

json.dump({
    "model_type": "deepseek_v4",
    "hidden_size": D, "num_attention_heads": NH, "num_key_value_heads": 1,
    "head_dim": HD, "qk_rope_head_dim": RD, "num_hidden_layers": NLAYERS,
    "vocab_size": VOCAB, "rms_norm_eps": 1e-6, "rope_theta": 10000,
    "q_lora_rank": QLORA, "o_lora_rank": OLORA, "o_groups": OGROUPS,
    "moe_intermediate_size": MOE_INTER, "n_routed_experts": NEXP,
    "num_experts_per_tok": TOPK, "n_shared_experts": 1,
    # The release announces a next-token predictor whose weights the
    # converter does not map. Declaring it here keeps the toy honest: a
    # loader that demands those weights must still load this model.
    "num_nextn_predict_layers": 1,
    "num_hash_layers": NHASH, "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
    "norm_topk_prob": True, "swiglu_limit": 10.0, "sliding_window": 8,
    "index_topk": 4, "index_n_heads": IDX_HEADS, "index_head_dim": IDX_HD,
    "hc_mult": HC, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
    "max_position_embeddings": 4096, "tie_word_embeddings": False,
    "torch_dtype": "float32", "hidden_act": "silu",
    "rope_scaling": {"type": "yarn", "factor": 16, "beta_fast": 32,
                     "beta_slow": 1, "original_max_position_embeddings": 2048},
}, open(os.path.join(out, "config.json"), "w"), indent=1)

# A minimal byte-level tokenizer so `run` has a vocab to work with.
vocab = {chr(i): i for i in range(VOCAB)}
json.dump({
    "version": "1.0", "truncation": None, "padding": None,
    "added_tokens": [], "normalizer": None,
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                      "trim_offsets": True, "use_regex": True},
    "post_processor": None,
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
    "model": {"type": "BPE", "dropout": None, "unk_token": None,
              "continuing_subword_prefix": None, "end_of_word_suffix": None,
              "fuse_unk": False, "vocab": vocab, "merges": []},
}, open(os.path.join(out, "tokenizer.json"), "w"))
json.dump({"bos_token": None, "eos_token": None, "model_max_length": 4096},
          open(os.path.join(out, "tokenizer_config.json"), "w"))

print(f"{len(tensors)} тензоров → {out}")
