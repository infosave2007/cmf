#!/usr/bin/env python3
"""Export the decision encoder from ONNX to CMF-ready tensors (spec decision-v4 §1.2).

Reads the cortiq-router encoder (`registry_bake/encoder.onnx` + `encoder_tokenizer/`)
once, offline, and writes an export directory that `cortiq decision init
--encoder-dir DIR` packs into a decision CMF file:

    encoder.json                 format, config, tokenizer settings, sha256 of every source
                                 and tensor, the checks below, tool versions
    <name>.npy                   one little-endian f32 array per weight, HF role names
                                 without the `encoder.` prefix (`layer.0.attention.self.query.weight`),
                                 linear weights transposed to [out, in]; the pooler is dropped
    vocab.txt, tokenizer.json    copies of the tokenizer files

Checks (the export is refused when one fails):
* graph: inputs `input_ids`, `attention_mask`; output 0 `last_hidden_state`; post-LN
  BERT; the 72 anonymous `onnx::MatMul_NNNN` weights are assigned to their roles by
  node name (`/encoder/layer.N/attention/self/query/MatMul`, …) and, when the ids
  follow the release layout, id = base + 13·L (base 1494 q, 1495 k, 1498 v,
  1504 attention.output, 1505 intermediate, 1506 output); the architecture comes
  from the graph and the tensors, not from config.json (intermediate 384 there
  says 1536); attention divisor = f32(√head_dim); GELU = (x·(erf(x/f32(√2)) + 1))·0.5;
  token types gathered from an all-zero constant (type 0); positions arange;
* tokenizer: WordPiece, `BertNormalizer{clean_text, handle_chinese_chars,
  strip_accents: null, lowercase: true}`, `BertPreTokenizer`, template
  `[CLS] $A [SEP]`, right truncation, added tokens = exactly the five special ids
  (normalized false, no strip, not single-word); vocab.txt read as HF
  `WordPiece::read_file` does equals the `tokenizer.json` vocab;
* numerics: a numpy float64 re-implementation *from the exported .npy files*
  matches ONNX Runtime `last_hidden_state` to max |Δ| <= 1e-5 on three texts.

    python3 tools/decision_export_encoder.py \\
        --onnx .../registry_bake/encoder.onnx --tokenizer-dir .../encoder_tokenizer --out DIR

The output directory must not exist (it is written to DIR.tmp-<pid> and renamed).
`forward_f64`, `phi_p_f64`, `write_export` and `read_export` are imported by
`tools/mk_decision_toy.py` and `tools/decision_encoder_parity.py`.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys

import numpy as np

FORMAT = "cortiq-decision-encoder-export/1"
KIND = "bert-wordpiece-v1"
RELEASE_NAME = ("cortiq-router registry_bake/encoder.onnx (bge-small-en-v1.5, MLP pruned 75% "
                "by Cortiq NVG)")
PARITY_TOL = 1e-5
CHECK_TEXTS = [
    "How do I top up my card?",
    "Café crème naïve ÀÉÎ façade",
    "I still have not received my new card, I ordered over a week ago. What should I do?",
]
BERT_NORMALIZER = {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True,
                   "strip_accents": None, "lowercase": True}
ROLE_BASE = {"attention/self/query": 1494, "attention/self/key": 1495,
             "attention/self/value": 1498, "attention/output/dense": 1504,
             "intermediate/dense": 1505, "output/dense": 1506}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def f32_shortest(x):
    """The shortest decimal whose f32 is x (e.g. 9.99999996e-13 -> 1e-12)."""
    return float(repr(np.float32(x)))


# ------------------------------------------------------------------ reference forward

def weight_shapes(cfg):
    """(name, shape) of every weight, the order of EncoderConfig::weight_shapes."""
    h, f = cfg["hidden"], cfg["intermediate"]
    out = [("embeddings.word_embeddings.weight", [cfg["vocab"], h]),
           ("embeddings.position_embeddings.weight", [cfg["max_position"], h]),
           ("embeddings.token_type_embeddings.weight", [cfg["type_vocab"], h]),
           ("embeddings.LayerNorm.weight", [h]), ("embeddings.LayerNorm.bias", [h])]
    for L in range(cfg["layers"]):
        p = f"layer.{L}."
        for m in ("query", "key", "value"):
            out += [(f"{p}attention.self.{m}.weight", [h, h]), (f"{p}attention.self.{m}.bias", [h])]
        out += [(f"{p}attention.output.dense.weight", [h, h]), (f"{p}attention.output.dense.bias", [h]),
                (f"{p}attention.output.LayerNorm.weight", [h]), (f"{p}attention.output.LayerNorm.bias", [h]),
                (f"{p}intermediate.dense.weight", [f, h]), (f"{p}intermediate.dense.bias", [f]),
                (f"{p}output.dense.weight", [h, f]), (f"{p}output.dense.bias", [h]),
                (f"{p}output.LayerNorm.weight", [h]), (f"{p}output.LayerNorm.bias", [h])]
    return out


def forward_f64(W, cfg, ids):
    """last_hidden_state [n, hidden] in float64 of one unpadded sequence.

    W: name -> array with the export layout (linear weights [out, in])."""
    g = lambda n: np.asarray(W[n], dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    n, h = len(ids), cfg["hidden"]
    nh, dh, eps = cfg["heads"], cfg["head_dim"], cfg["ln_eps"]

    def ln(x, p):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g(p + ".weight") + g(p + ".bias")

    def lin(x, p):
        return x @ g(p + ".weight").T + g(p + ".bias")

    erf = np.vectorize(math.erf)
    x = (g("embeddings.word_embeddings.weight")[ids]
         + g("embeddings.token_type_embeddings.weight")[cfg["token_type"]]
         + g("embeddings.position_embeddings.weight")[:n])
    x = ln(x, "embeddings.LayerNorm")
    for L in range(cfg["layers"]):
        p = f"layer.{L}."
        q = lin(x, p + "attention.self.query").reshape(n, nh, dh).transpose(1, 0, 2)
        k = lin(x, p + "attention.self.key").reshape(n, nh, dh).transpose(1, 0, 2)
        v = lin(x, p + "attention.self.value").reshape(n, nh, dh).transpose(1, 0, 2)
        a = q @ k.transpose(0, 2, 1) / math.sqrt(dh)
        a = np.exp(a - a.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        c = (a @ v).transpose(1, 0, 2).reshape(n, h)
        x = ln(lin(c, p + "attention.output.dense") + x, p + "attention.output.LayerNorm")
        t = lin(x, p + "intermediate.dense")
        t = 0.5 * t * (1.0 + erf(t / math.sqrt(2.0)))
        x = ln(lin(t, p + "output.dense") + x, p + "output.LayerNorm")
    return x


def phi_p_f64(hidden):
    """Mean over all tokens, v/(|v|+1e-12), then v/|v| — in float64."""
    v = hidden.mean(0)
    v = v / (np.sqrt((v * v).sum()) + 1e-12)
    n = np.sqrt((v * v).sum())
    return v / n if n > 1e-12 else v


# ------------------------------------------------------------------ export files

def write_export(out_dir, weights, cfg, source, tokenizer, vocab_bytes, tokenizer_json_bytes,
                 checks, tools):
    """Write an export directory (out_dir must not exist yet). weights: name -> f32 array."""
    os.makedirs(out_dir)
    shapes = dict((n, s) for n, s in weight_shapes(cfg))
    assert set(shapes) == set(weights), sorted(set(shapes) ^ set(weights))
    tensors = []
    for name, shape in weight_shapes(cfg):
        arr = np.ascontiguousarray(np.asarray(weights[name], dtype="<f4"))
        assert list(arr.shape) == shape, (name, arr.shape, shape)
        assert np.isfinite(arr).all(), name
        fname = name + ".npy"
        np.save(os.path.join(out_dir, fname), arr, allow_pickle=False)
        tensors.append({"name": name, "file": fname, "shape": shape, "dtype": "float32",
                        "sha256": sha256_bytes(arr.tobytes())})
    with open(os.path.join(out_dir, "vocab.txt"), "wb") as f:
        f.write(vocab_bytes)
    with open(os.path.join(out_dir, "tokenizer.json"), "wb") as f:
        f.write(tokenizer_json_bytes)
    manifest = {
        "format": FORMAT,
        "kind": KIND,
        "source": source,
        "config": cfg,
        "tokenizer": dict(tokenizer, file="tokenizer.json", vocab_file="vocab.txt"),
        "tensors": tensors,
        "checks": checks,
        "tools": tools,
    }
    with open(os.path.join(out_dir, "encoder.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    return manifest


def read_export(export_dir):
    """(manifest, weights) of an export directory, every sha256 checked."""
    m = json.load(open(os.path.join(export_dir, "encoder.json"), encoding="utf-8"))
    assert m["format"] == FORMAT, m["format"]
    W = {}
    for t in m["tensors"]:
        arr = np.load(os.path.join(export_dir, t["file"]), allow_pickle=False)
        assert arr.dtype == np.dtype("<f4") and list(arr.shape) == t["shape"], t["name"]
        assert sha256_bytes(np.ascontiguousarray(arr).tobytes()) == t["sha256"], t["name"]
        W[t["name"]] = arr
    vocab = open(os.path.join(export_dir, m["tokenizer"]["vocab_file"]), "rb").read()
    assert sha256_bytes(vocab) == m["source"]["vocab_sha256"]
    tok = open(os.path.join(export_dir, m["tokenizer"]["file"]), "rb").read()
    assert sha256_bytes(tok) == m["source"]["tokenizer_json_sha256"]
    return m, W


# ------------------------------------------------------------------ tokenizer checks

# Unicode White_Space: what Rust's str::trim_end strips.
WHITE_SPACE = "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006" \
              "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"


def hf_read_vocab(vocab_bytes):
    """HF WordPiece::read_file: BufRead::lines (\\n, a trailing \\r dropped), trim_end,
    later duplicates win."""
    text = vocab_bytes.decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    vocab = {}
    for i, line in enumerate(lines):
        if line.endswith("\r"):
            line = line[:-1]
        vocab[line.rstrip(WHITE_SPACE)] = i
    return vocab, len(lines)


def check_tokenizer(tok_json, vocab_bytes):
    """The tokenizer settings of the export, or SystemExit on anything unsupported."""
    t = tok_json
    model = t["model"]

    def need(cond, msg):
        if not cond:
            sys.exit(f"tokenizer.json: {msg}")

    need(model.get("type") == "WordPiece", f"model type {model.get('type')} is not WordPiece")
    need(t.get("normalizer") == BERT_NORMALIZER, f"normalizer {t.get('normalizer')}")
    need(t.get("pre_tokenizer") == {"type": "BertPreTokenizer"}, f"pre_tokenizer {t.get('pre_tokenizer')}")
    vocab, n_lines = hf_read_vocab(vocab_bytes)
    need(len(vocab) == n_lines, "vocab.txt has duplicate tokens")
    need(vocab == model["vocab"], "vocab.txt differs from the tokenizer.json vocab")
    pp = t.get("post_processor") or {}
    need(pp.get("type") == "TemplateProcessing", "post_processor is not TemplateProcessing")
    single = pp.get("single")
    need(isinstance(single, list) and len(single) == 3
         and single[1] == {"Sequence": {"id": "A", "type_id": 0}}
         and "SpecialToken" in single[0] and "SpecialToken" in single[2]
         and single[0]["SpecialToken"]["type_id"] == 0 and single[2]["SpecialToken"]["type_id"] == 0,
         f"template {single} is not [CLS] $A [SEP]")
    cls_tok, sep_tok = single[0]["SpecialToken"]["id"], single[2]["SpecialToken"]["id"]
    st = pp["special_tokens"]
    need(st[cls_tok]["ids"] == [vocab[cls_tok]] and st[sep_tok]["ids"] == [vocab[sep_tok]],
         "template special ids differ from the vocab")
    unk = model["unk_token"]
    need(unk in vocab, "unk token not in the vocab")
    tr = t.get("truncation")
    need(tr is not None and tr.get("direction") == "Right", f"truncation {tr}")
    added = t.get("added_tokens", [])
    pad_tok = (t.get("padding") or {}).get("pad_token", "[PAD]")
    mask_candidates = [a["content"] for a in added if a["content"] not in (pad_tok, unk, cls_tok, sep_tok)]
    need(len(mask_candidates) == 1, f"added tokens {[a['content'] for a in added]} are not the five specials")
    mask_tok = mask_candidates[0]
    ids = {"pad": vocab[pad_tok], "unk": vocab[unk], "cls": vocab[cls_tok], "sep": vocab[sep_tok],
           "mask": vocab[mask_tok]}
    for a in added:
        need(a["id"] == vocab.get(a["content"]) and a["special"] and not a["normalized"]
             and not a["lstrip"] and not a["rstrip"] and not a["single_word"],
             f"added token {a} is not a plain special token of the vocab")
    need(sorted(a["id"] for a in added) == sorted(ids.values()), "added tokens differ from the special ids")
    return {
        "unk": unk,
        "prefix": model["continuing_subword_prefix"],
        "max_input_chars_per_word": model["max_input_chars_per_word"],
        "normalizer": {k: v for k, v in BERT_NORMALIZER.items() if k != "type"},
        "pre_tokenizer": "bert",
        "template": "[CLS] $A [SEP]",
        "ids": ids,
        "max_length": tr["max_length"],
        "truncation_direction": "right",
    }


# ------------------------------------------------------------------ ONNX graph

def export_onnx(onnx_path):
    """(config, weights) from the ONNX graph; SystemExit on an unexpected graph."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(onnx_path)
    g = m.graph
    ins = [i.name for i in g.input]
    outs = [o.name for o in g.output]
    if ins != ["input_ids", "attention_mask"]:
        sys.exit(f"graph inputs {ins} (expected input_ids, attention_mask)")
    if outs[0] != "last_hidden_state":
        sys.exit(f"graph output 0 is {outs[0]}, not last_hidden_state")
    init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consts = {}
    for n in g.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)

    def value(name):
        if name in consts:
            return consts[name]
        return init[name]

    weights = {}
    matmul_ids = {}
    pat = re.compile(r"^/encoder/layer\.(\d+)/(attention/self/(?:query|key|value)|attention/output/dense|"
                     r"intermediate/dense|output/dense)/MatMul$")
    for n in g.node:
        mm = pat.match(n.name) if n.op_type == "MatMul" else None
        if not mm:
            continue
        L, role = int(mm.group(1)), mm.group(2)
        wname = n.input[1]
        if wname not in init:
            sys.exit(f"{n.name}: weight {wname} is not an initializer")
        name = f"layer.{L}.{role.replace('/', '.')}.weight"
        if name in weights:
            sys.exit(f"{name} assigned twice")
        weights[name] = init[wname].T.copy()  # [in, out] -> [out, in]
        idm = re.fullmatch(r"onnx::MatMul_(\d+)", wname)
        matmul_ids[name] = (int(idm.group(1)) if idm else None, ROLE_BASE[role] + 13 * L)
    for name, arr in init.items():
        if name.startswith("embeddings."):
            weights[name] = arr
        elif name.startswith("encoder.layer."):
            weights[name[len("encoder."):]] = arr
    unassigned = sorted(k for k in init if k.startswith("onnx::MatMul_") and
                        not any(n.input[1] == k for n in g.node if n.op_type == "MatMul" and pat.match(n.name)))
    if unassigned:
        sys.exit(f"MatMul weights without a role: {unassigned}")
    layers = 1 + max(int(k.split(".")[1]) for k in weights if k.startswith("layer."))
    ids_ok = all(a == b for a, b in matmul_ids.values())
    ids_named = all(a is not None for a, _ in matmul_ids.values())
    if ids_named and not ids_ok:
        sys.exit("MatMul initializer ids do not follow id = base + 13·L")

    word = weights["embeddings.word_embeddings.weight"]
    hidden = word.shape[1]
    inter = weights["layer.0.intermediate.dense.weight"].shape[0]
    # Attention divisor and head size.
    divs = [n for n in g.node if n.op_type == "Div" and re.match(r"^/encoder/layer\.\d+/attention/self/Div$", n.name)]
    if len(divs) != layers:
        sys.exit(f"{len(divs)} attention Div nodes for {layers} layers")
    dvals = {float(np.asarray(value(n.input[1])).reshape(())) for n in divs}
    if len(dvals) != 1:
        sys.exit(f"attention divisors differ: {dvals}")
    dv = dvals.pop()
    head_dim = int(round(dv * dv))
    if np.float32(math.sqrt(head_dim)) != np.float32(dv):
        sys.exit(f"attention divisor {dv} is not f32(sqrt({head_dim}))")
    if hidden % head_dim:
        sys.exit(f"hidden {hidden} is not a multiple of head_dim {head_dim}")
    heads = hidden // head_dim
    # LayerNorms.
    lns = [n for n in g.node if n.op_type == "LayerNormalization"]
    if len(lns) != 1 + 2 * layers:
        sys.exit(f"{len(lns)} LayerNormalization nodes for {layers} layers")
    eps = {next(a.f for a in n.attribute if a.name == "epsilon") for n in lns}
    if len(eps) != 1:
        sys.exit(f"LayerNorm epsilons differ: {eps}")
    ln_eps = f32_shortest(eps.pop())
    # GELU: (x * (erf(x / c) + 1)) * 0.5 with c = f32(sqrt 2).
    erfs = [n for n in g.node if n.op_type == "Erf"]
    if len(erfs) != layers:
        sys.exit(f"{len(erfs)} Erf nodes for {layers} layers")
    for L in range(layers):
        p = f"/encoder/layer.{L}/intermediate/intermediate_act_fn/"
        byname = {n.name: n for n in g.node if n.name.startswith(p)}
        try:
            c_div = float(np.asarray(value(byname[p + "Div"].input[1])).reshape(()))
            c_add = float(np.asarray(value(byname[p + "Add"].input[1])).reshape(()))
            c_mul = float(np.asarray(value(byname[p + "Mul_1"].input[1])).reshape(()))
        except KeyError as e:
            sys.exit(f"layer {L}: GELU node {e} not found")
        if np.float32(c_div) != np.float32(math.sqrt(2.0)) or c_add != 1.0 or c_mul != 0.5:
            sys.exit(f"layer {L}: GELU constants {c_div}, {c_add}, {c_mul}")
    if any(n.op_type == "Tanh" and not n.name.startswith("/pooler") for n in g.node):
        sys.exit("a Tanh outside the pooler (tanh GELU?)")
    # Token types gathered from an all-zero buffer, positions from arange: trace
    # each Gather's index input back through Expand/Slice to its constant source.
    producer = {o: n for n in g.node for o in n.output}

    def source_buffer(name):
        seen = 0
        while name not in init and name not in consts:
            n = producer.get(name)
            if n is None or n.op_type not in ("Expand", "Slice") or seen > 8:
                return None
            name, seen = n.input[0], seen + 1
        return name

    def gather_source(table):
        gs = [n for n in g.node if n.op_type == "Gather" and n.input[0] == table]
        if len(gs) != 1:
            sys.exit(f"{table}: {len(gs)} gathers")
        src = source_buffer(gs[0].input[1])
        if src is None:
            sys.exit(f"{table}: the gather index is not a constant buffer")
        return src, np.asarray(value(src))

    tt_src, tt_buf = gather_source("embeddings.token_type_embeddings.weight")
    pos_src, pos_buf = gather_source("embeddings.position_embeddings.weight")
    if tt_buf.any():
        sys.exit(f"token types come from {tt_src}, which is not all zeros")
    npos = weights["embeddings.position_embeddings.weight"].shape[0]
    if pos_buf.size != npos or (pos_buf.reshape(-1) != np.arange(npos)).any():
        sys.exit(f"positions come from {pos_src}, which is not arange({npos})")
    cfg = {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "head_dim": head_dim,
        "intermediate": inter,
        "max_position": int(weights["embeddings.position_embeddings.weight"].shape[0]),
        "vocab": int(word.shape[0]),
        "type_vocab": int(weights["embeddings.token_type_embeddings.weight"].shape[0]),
        "token_type": 0,
        "ln_eps": ln_eps,
        "activation": "gelu_erf",
    }
    expected = dict(weight_shapes(cfg))
    extra = sorted(set(weights) - set(expected))
    unused = sorted(k for k in init if not (k.startswith("embeddings.") or k.startswith("encoder.layer.")
                                            or k.startswith("pooler.") or k.startswith("onnx::")))
    if extra or unused:
        sys.exit(f"unexpected initializers: {extra + unused}")
    for name, shape in expected.items():
        if name not in weights:
            sys.exit(f"missing weight {name}")
        if list(weights[name].shape) != shape:
            sys.exit(f"{name}: shape {list(weights[name].shape)}, expected {shape}")
    weights = {k: np.ascontiguousarray(v.astype("<f4")) for k, v in weights.items()}
    graph_facts = {"matmul_roles_by_node_name": len(matmul_ids),
                   "matmul_ids_follow_base_plus_13L": bool(ids_named and ids_ok),
                   "attention_divisor_f32": float(np.float32(dv)), "ln_epsilon_attr_f32": float(np.float32(ln_eps)),
                   "gelu": "(x*(erf(x/f32(sqrt 2))+1))*0.5", "pooler": "dropped",
                   "token_type_source": f"{tt_src} (all zeros)", "position_source": f"{pos_src} (arange)"}
    return cfg, weights, graph_facts


def ort_parity(onnx_path, export_dir, texts, threads):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    m, W = read_export(export_dir)
    cfg = m["config"]
    tok = Tokenizer.from_file(os.path.join(export_dir, "tokenizer.json"))
    tok.no_padding()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    rows = []
    for t in texts:
        ids = tok.encode(t).ids
        out = sess.run(["last_hidden_state"], {"input_ids": np.array([ids], dtype=np.int64),
                                               "attention_mask": np.ones((1, len(ids)), dtype=np.int64)})[0][0]
        nat = forward_f64(W, cfg, ids)
        rows.append({"text": t, "tokens": len(ids), "max_abs": float(np.abs(nat - out.astype(np.float64)).max())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer-dir", required=True, help="directory with tokenizer.json and vocab.txt")
    ap.add_argument("--out", required=True, help="export directory (must not exist)")
    ap.add_argument("--name", default=RELEASE_NAME, help="source name recorded in the manifest")
    ap.add_argument("--threads", type=int, default=1, help="ONNX Runtime threads for the parity check")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        sys.exit(f"{out} exists; the export never overwrites")
    import onnx
    import onnxruntime
    import tokenizers

    onnx_bytes_sha = sha256_file(a.onnx)
    tok_path = os.path.join(a.tokenizer_dir, "tokenizer.json")
    vocab_path = os.path.join(a.tokenizer_dir, "vocab.txt")
    tok_bytes = open(tok_path, "rb").read()
    vocab_bytes = open(vocab_path, "rb").read()
    tokenizer = check_tokenizer(json.loads(tok_bytes), vocab_bytes)
    cfg, weights, facts = export_onnx(a.onnx)
    if cfg["vocab"] != len(hf_read_vocab(vocab_bytes)[0]):
        sys.exit("vocab size differs between the graph and vocab.txt")
    if tokenizer["max_length"] > cfg["max_position"]:
        sys.exit("truncation max_length exceeds max_position")
    source = {"name": a.name, "onnx_sha256": onnx_bytes_sha,
              "tokenizer_json_sha256": sha256_bytes(tok_bytes), "vocab_sha256": sha256_bytes(vocab_bytes)}
    tools = {"python": sys.version.split()[0], "numpy": np.__version__, "onnx": onnx.__version__,
             "onnxruntime": onnxruntime.__version__, "tokenizers": tokenizers.__version__,
             "script": "tools/decision_export_encoder.py"}
    tmp = f"{out}.tmp-{os.getpid()}"
    checks = {"graph": facts, "ort_parity_tolerance": PARITY_TOL}
    write_export(tmp, weights, cfg, source, tokenizer, vocab_bytes, tok_bytes, checks, tools)
    try:
        rows = ort_parity(a.onnx, tmp, CHECK_TEXTS, a.threads)
        worst = max(r["max_abs"] for r in rows)
        for r in rows:
            print(f"parity {r['tokens']:3d} tokens max|f64 export - ORT| {r['max_abs']:.3e}  {r['text'][:40]!r}")
        if worst > PARITY_TOL:
            sys.exit(f"export does not reproduce ORT: max |Δ| {worst:.3e} > {PARITY_TOL:g}")
        # Record the parity in the manifest (the tensor files are unchanged).
        mpath = os.path.join(tmp, "encoder.json")
        m = json.load(open(mpath, encoding="utf-8"))
        m["checks"]["ort_parity"] = rows
        m["checks"]["ort_parity_max_abs"] = worst
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"wrote {out}: {cfg}")
    print(f"encoder.json sha256 {sha256_file(os.path.join(out, 'encoder.json'))}")


if __name__ == "__main__":
    main()
