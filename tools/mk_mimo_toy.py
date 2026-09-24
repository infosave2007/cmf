#!/usr/bin/env python3
"""Emit a tiny MiMo-V2 checkpoint in the EXACT release storage format, plus
golden references computed by the standalone oracle (tools/mimo_ref.py).

    python3 tools/mk_mimo_toy.py --src /root/mimo/src --out /root/mimo/toy/pos
    python3 tools/mk_mimo_toy.py --src /root/mimo/src --out /root/mimo/toy/neg_contig --qkv-layout contig
    cortiq convert --model /root/mimo/toy/pos --quant f16 --output /root/mimo/toy/pos-f16.cmf
    CMF_GPU=0 CMF_SDOT=0 CMF_GOLDEN_FILE=/root/mimo/toy/pos-f16.cmf \
      CMF_GOLDEN_REF=/root/mimo/toy/pos/reference.json \
      cargo test --release -p cortiq-engine --test golden_parity -- --nocapture

What is exercised (same names, dtypes and packing as the 172 GB release):
  * fused `self_attn.qkv_proj.weight` F8_E4M3 stored as ckpt_tp =
    num_key_value_heads = 2 chunks [Q_c | K_c | V_c], with the F32
    `weight_scale_inv` tiled PER CHUNK. Full layers: 2 x 272 rows -> 6 scale
    rows where a contiguous grid would have 5 (the release's 108-vs-106).
    SWA layers: 2 x 352 rows -> 6 scale rows, the same count as a contiguous
    grid (the release's silent 116 = 116), but here a chunk is 2.75 blocks, so
    only per-chunk tiling dequantizes correctly (vLLM precedence).
  * per-layer KV heads (2 full / 4 SWA), head_dim 48 with rope_dim
    int(48 * 0.334) = 16, v_head_dim 32, attention_value_scale 0.707,
    window 8 (the 24-token prompt + 8 greedy tokens slide it), non-zero
    BF16 `attention_sink_bias` on the SWA layers, thetas 1e7 / 1e4.
  * dense FP8 layer-0 MLP (intermediate 320 -> partial 128-blocks), BF16
    o_proj / embed / lm_head / norms / router, F32 e_score_correction_bias,
    8 MXFP4 experts top-2 (U8 .weight + U8 E8M0 .weight_scale) over two
    `model_pp0_ep{0,1}_shard0.safetensors` files, 3 MTP layers in
    model_mtp.safetensors and a few visual.* / audio_encoder.* /
    speech_embeddings.* tensors that a text-only conversion must DROP.
  * the real tokenizer and chat template (vocab 152576, eos 151645).

Every stored value is exact in f16: FP8 codes x power-of-two block scales,
E2M1 x E8M0 in [2^-7, 6 * 2^-4], BF16 with |x| < 2^-14 flushed to zero. So an
`--quant f16` CMF holds the dequantized weights bit-exactly -- except the
V rows once the converter folds 0.707 into them (f16 rounding of V*0.707,
worth ~1.6e-3 in the first logits, above golden_parity's 1e-3). Hence two
references:
  reference.json        V rows pre-multiplied by 0.707 in f32 and rounded to
                        f16 (the planned converter: fold at convert time)
  reference_exact.json  0.707 applied to V at runtime (a converter that keeps
                        V as stored and scales in the engine)
The seed search requires both to give the same greedy tokens with margins
(top1-top2 >= 1e-2, router k-th vs (k+1)-th >= 1e-3).

--qkv-layout contig writes the NEGATIVE control: the same model, but qkv rows
stored as one global [Q | K | V] (the HF modeling file's reading). Its qkv block
scales are uniform per tensor and keep the per-chunk shape, so every shape
check passes and only the row order is wrong: a converter that reads the
release layout must FAIL parity against this toy's reference.json (which is
the true model). The generator proves that with the oracle itself:
`negative_check` in reference.json is what a chunked reader gets from the
file.

Outputs in --out: config.json, *.safetensors + index, tokenizer files,
reference.json (golden_parity.rs format + margins), ref/ (raw f32 per-layer
dumps for all 32 positions, logits, moe_trace.txt, picks.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mimo_ref as R  # noqa: E402

torch.set_grad_enabled(False)

TOY = dict(
    hidden_size=256, intermediate_size=320, num_hidden_layers=4,
    num_attention_heads=8, num_key_value_heads=2, head_dim=48, v_head_dim=32,
    swa_num_attention_heads=8, swa_num_key_value_heads=4, swa_head_dim=48, swa_v_head_dim=32,
    sliding_window=8, sliding_window_size=8, hybrid_layer_pattern=[0, 1, 1, 0],
    moe_layer_freq=[0, 1, 1, 1], n_routed_experts=8, num_experts_per_tok=2,
    moe_intermediate_size=64,
)
PROMPT_LEN = 24
GREEDY = 8
MIN_MARGIN = 1e-2        # greedy top1 - top2 (golden tolerance is 1e-3)
ROUTE_MARGIN = 1e-3      # k-th minus (k+1)-th biased router score, in both semantics
NORMAL_VOCAB = 151643    # ids >= this are special / padding rows
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "chat_template.jinja", "vocab.json", "merges.txt")


PROMPT_TEXT = ("The quick brown fox jumps over the lazy dog while the committee "
               "reviews forty-two proposals about sliding windows, sinks and experts.")


def make_prompt(src):
    """24 ids of a real sentence, cut where decode -> encode round-trips, so
    `cortiq run --raw --prompt "<prompt_text>"` feeds exactly these ids."""
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(os.path.join(src, "tokenizer.json"))
    ids = tk.encode(PROMPT_TEXT, add_special_tokens=False).ids[:PROMPT_LEN]
    text = tk.decode(ids)
    if len(ids) != PROMPT_LEN or tk.encode(text, add_special_tokens=False).ids != ids:
        raise SystemExit(f"prompt does not round-trip: {ids} -> {text!r}")
    return ids, text


def make_config(src_cfg: dict) -> dict:
    c = json.loads(json.dumps(src_cfg))
    c.update(TOY)
    q = c.get("quantization_config") or {}
    q["ignored_layers"] = [f"model.layers.{i}.self_attn.o_proj"
                           for i in range(TOY["num_hidden_layers"])] + ["model.decoder.self_attn.o_proj"]
    c["quantization_config"] = q
    return c


class Gen:
    def __init__(self, seed):
        self.g = torch.Generator().manual_seed(seed)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.g, dtype=torch.float64)

    def randint(self, lo, hi, shape):
        return torch.randint(lo, hi, shape, generator=self.g)

    def bf16(self, t):
        t = t.to(torch.bfloat16)
        t[t.abs() < 2.0 ** -14] = 0
        return t

    def fp8(self, rows, cols, std, row_index=None, n_scale_rows=None, uniform=False):
        """FP8 codes [rows, cols] + F32 scale_inv whose entries are powers
        of two. row_index maps rows to scale rows (default rows // 128)."""
        if row_index is None:
            row_index = torch.arange(rows) // 128
            n_scale_rows = R.cdiv(rows, 128)
        cb = R.cdiv(cols, 128)
        if uniform:
            exp = torch.full((n_scale_rows, cb), -8)
        else:
            exp = self.randint(-9, -6, (n_scale_rows, cb))        # 2^-9 .. 2^-7
        sinv = torch.pow(2.0, exp.double()).float()
        per_elem = sinv.double()[row_index][:, torch.arange(cols) // 128]
        codes = (self.randn(rows, cols) * std / per_elem).clamp(-448, 448).to(torch.float8_e4m3fn)
        return codes, sinv

    def mxfp4(self, rows, cols, exp_lo, exp_hi):
        packed = self.randint(0, 256, (rows, cols // 2)).to(torch.uint8)
        scale = (127 + self.randint(exp_lo, exp_hi + 1, (rows, cols // 32))).to(torch.uint8)
        return packed, scale


def build(cfg_dict: dict, seed: int, layout: str, mm_extras: bool):
    """Returns (stored tensors as written to disk, true tensors in the
    release (chunked) layout for the reference)."""
    cfg = R.MimoCfg(cfg_dict)
    g = Gen(seed)
    H, V, I, MI = cfg.H, cfg.vocab, cfg_dict["intermediate_size"], cfg_dict["moe_intermediate_size"]
    t = {}
    t["model.embed_tokens.weight"] = g.bf16(g.randn(V, H))
    t["lm_head.weight"] = g.bf16(g.randn(V, H) * 3.0 / math.sqrt(H))
    t["model.norm.weight"] = g.bf16(1.0 + 0.2 * g.randn(H))
    stored_qkv = {}
    for li in range(cfg.NL):
        p = f"model.layers.{li}."
        geo = cfg.geom(li)
        q, k, v = R.qkv_rows(cfg, li)
        rows, tp = q + k + v, cfg.ckpt_tp
        n_sr = tp * R.cdiv(rows // tp, 128)
        ri = R.qkv_scale_rows(cfg, li, n_sr)
        codes, sinv = g.fp8(rows, H, 1.0 / math.sqrt(H), ri, n_sr, uniform=(layout == "contig"))
        t[p + "self_attn.qkv_proj.weight"] = codes
        t[p + "self_attn.qkv_proj.weight_scale_inv"] = sinv
        if layout == "contig":
            # Same model, rows written as one global [Q | K | V]. The scales are
            # uniform, so their per-chunk shape is still a valid description.
            Qc, Kc, Vc = R.split_fused_qkv(cfg, li, codes.view(torch.uint8), "chunked")
            stored_qkv[p + "self_attn.qkv_proj.weight"] = torch.cat([Qc, Kc, Vc]).view(torch.float8_e4m3fn)
        t[p + "self_attn.o_proj.weight"] = g.bf16(g.randn(H, geo.nq * geo.vd) / math.sqrt(geo.nq * geo.vd))
        if geo.sink:
            t[p + "self_attn.attention_sink_bias"] = g.bf16(0.5 + g.randn(geo.nq))
        t[p + "input_layernorm.weight"] = g.bf16(1.0 + 0.2 * g.randn(H))
        t[p + "post_attention_layernorm.weight"] = g.bf16(1.0 + 0.2 * g.randn(H))
        if cfg.moe[li]:
            t[p + "mlp.gate.weight"] = g.bf16(g.randn(cfg.ne, H) * 2.0 / math.sqrt(H))
            t[p + "mlp.gate.e_score_correction_bias"] = (0.05 * g.randn(cfg.ne)).float()
            for e in range(cfg.ne):
                q_ = f"{p}mlp.experts.{e}."
                for n, (r_, c_, lo, hi) in {"gate": (MI, H, -6, -5), "up": (MI, H, -6, -5),
                                            "down": (H, MI, -5, -4)}.items():
                    pk, sc = g.mxfp4(r_, c_, lo, hi)
                    t[q_ + f"{n}_proj.weight"] = pk
                    t[q_ + f"{n}_proj.weight_scale"] = sc
        else:
            for n, (r_, c_) in {"gate": (I, H), "up": (I, H), "down": (H, I)}.items():
                codes, sinv = g.fp8(r_, c_, 1.0 / math.sqrt(c_))
                t[p + f"mlp.{n}_proj.weight"] = codes
                t[p + f"mlp.{n}_proj.weight_scale_inv"] = sinv
    mtp = {}
    geo = cfg.geom(1)  # MTP layers carry sinks: SWA geometry
    for m in range(cfg_dict.get("num_nextn_predict_layers") or 0):
        p = f"model.mtp.layers.{m}."
        rows = geo.nq * geo.hd + geo.nkv * (geo.hd + geo.vd)
        n_sr = cfg.ckpt_tp * R.cdiv(rows // cfg.ckpt_tp, 128)
        codes, sinv = g.fp8(rows, H, 1.0 / math.sqrt(H), R.qkv_scale_rows(cfg, 1, n_sr), n_sr)
        mtp[p + "self_attn.qkv_proj.weight"] = codes
        mtp[p + "self_attn.qkv_proj.weight_scale_inv"] = sinv
        mtp[p + "self_attn.o_proj.weight"] = g.bf16(g.randn(H, geo.nq * geo.vd) * 0.05)
        mtp[p + "self_attn.attention_sink_bias"] = g.bf16(g.randn(geo.nq))
        mtp[p + "eh_proj.weight"] = g.bf16(g.randn(H, 2 * H) * 0.05)
        for n in ("enorm", "hnorm", "final_layernorm", "input_layernorm", "pre_mlp_layernorm"):
            mtp[p + f"{n}.weight"] = g.bf16(1.0 + 0.1 * g.randn(H))
        for n, (r_, c_) in {"gate": (I, H), "up": (I, H), "down": (H, I)}.items():
            codes, sinv = g.fp8(r_, c_, 0.05)
            mtp[p + f"mlp.{n}_proj.weight"] = codes
            mtp[p + f"mlp.{n}_proj.weight_scale_inv"] = sinv
    extras = {}
    if mm_extras:
        extras = {
            "visual.patch_embed.proj.weight": g.bf16(g.randn(64, 48) * 0.1),
            "visual.blocks.0.attn.qkv.weight": g.bf16(g.randn(96, 32) * 0.1),
            "visual.blocks.0.attn.qkv.bias": g.bf16(g.randn(96) * 0.1),
            "visual.merger.ln_q.weight": g.bf16(1.0 + 0.1 * g.randn(32)),
            "audio_encoder.input_local_transformer.layers.0.input_layernorm.weight": g.bf16(1.0 + 0.1 * g.randn(32)),
            "audio_encoder.input_local_transformer.layers.0.mlp.down_proj.weight": g.bf16(g.randn(32, 64) * 0.1),
            "speech_embeddings.0.weight": g.bf16(g.randn(40, 32) * 0.1),
            "speech_embeddings.1.weight": g.bf16(g.randn(40, 32) * 0.1),
        }
    true = dict(t)
    stored = dict(t)
    stored.update(stored_qkv)
    return cfg, true, stored, mtp, extras


def shard_of(name, n_experts):
    import re

    m = re.match(r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.", name)
    if m:
        return f"model_pp0_ep{int(m.group(1)) * 2 // n_experts}_shard0.safetensors"
    return "model_pp0_ep0_shard0.safetensors"


def write_checkpoint(out, cfg_dict, stored, mtp, extras, src):
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    files = {}
    for name, ten in list(stored.items()) + list(extras.items()):
        files.setdefault(shard_of(name, cfg_dict["n_routed_experts"]), {})[name] = ten
    if mtp:
        files["model_mtp.safetensors"] = mtp
    weight_map, total = {}, 0
    for fn, ts in sorted(files.items()):
        save_file({k: v.contiguous() for k, v in ts.items()}, os.path.join(out, fn))
        for k, v in ts.items():
            weight_map[k] = fn
            total += v.numel() * v.element_size()
    index = {"metadata": {"save_format": "mxfp4", "total_size": total, "tp_size": cfg_dict["num_key_value_heads"]},
             "weight_map": dict(sorted(weight_map.items()))}
    json.dump(index, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    json.dump(cfg_dict, open(os.path.join(out, "config.json"), "w"), indent=2)
    copied = []
    for fn in TOKENIZER_FILES:
        if os.path.exists(os.path.join(src, fn)):
            shutil.copy(os.path.join(src, fn), os.path.join(out, fn))
            copied.append(fn)
    return sorted(files), copied


class FoldOracle(R.Oracle):
    """What an f16 CMF computes when the converter folds 0.707 into V."""

    def __init__(self, src, cfg, dt):
        c2 = R.MimoCfg(dict(cfg.raw, attention_value_scale=None))
        super().__init__(src, c2, dt, verbose=False)
        self.vs = cfg.vscale

    def layer(self, li):
        L = super().layer(li)
        L.wv = (L.wv.float() * torch.tensor(self.vs, dtype=torch.float32)).half().to(self.dt)
        return L


def reference(orc: R.Oracle, prompt, greedy_n):
    head = orc.head()
    ids = list(prompt)
    margins, first = [], None
    for step in range(greedy_n):
        h = orc.forward(ids)
        lg = orc.logits(h[0, -1:], head)[0]
        if first is None:
            first = lg
        top2 = lg.topk(2)
        margins.append(float(top2.values[0] - top2.values[1]))
        ids.append(int(top2.indices[0]))
    return first, ids[len(prompt):], margins


def route_margin(orc: R.Oracle, ids):
    ms = []
    orc.forward(ids, on_layer=lambda li, h, s, p: ms.append(float(p[2].min())) if p is not None else None)
    return min(ms) if ms else float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=R.DEFAULT_SRC, help="real checkpoint dir: config.json + tokenizer files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qkv-layout", default="chunked", choices=["chunked", "contig"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tries", type=int, default=64)
    ap.add_argument("--no-mm-extras", action="store_true", help="omit visual/audio/speech/MTP tensors")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg_dict = make_config(json.load(open(os.path.join(a.src, "config.json"))))
    if a.no_mm_extras:
        cfg_dict["num_nextn_predict_layers"] = 0

    dt = torch.float64
    prompt, prompt_text = make_prompt(a.src)
    for seed in range(a.seed, a.seed + a.max_tries):
        cfg, true, stored, mtp, extras = build(cfg_dict, seed, a.qkv_layout, not a.no_mm_extras)
        exact = R.Oracle(R.DictSource(true), cfg, dt, verbose=False)
        fold = FoldOracle(R.DictSource(true), cfg, dt) if cfg.vscale is not None else exact
        e_first, e_greedy, e_margins = reference(exact, prompt, GREEDY)
        first, greedy, margins = reference(fold, prompt, GREEDY)
        rmin = min(route_margin(exact, prompt + greedy), route_margin(fold, prompt + greedy))
        bad_tok = any(t >= NORMAL_VOCAB for t in greedy)
        mmin = min(margins + e_margins)
        print(f"seed {seed}: greedy {greedy} min margin {mmin:.4f} route margin {rmin:.2e}"
              f"{' special-token' if bad_tok else ''}"
              f"{' exact!=fold' if e_greedy != greedy else ''}", flush=True)
        if mmin >= MIN_MARGIN and rmin >= ROUTE_MARGIN and not bad_tok and e_greedy == greedy:
            break
    else:
        raise SystemExit("no seed met the margins")

    files, copied = write_checkpoint(a.out, cfg_dict, stored, mtp, extras, a.src)

    # Round trip: the files on disk, read back with the release layout. For
    # the negative control this is what a release-layout reader computes.
    disk_cfg = R.MimoCfg(json.load(open(os.path.join(a.out, "config.json"))))
    disk = (FoldOracle(R.DirSource(a.out), disk_cfg, dt) if cfg.vscale is not None
            else R.Oracle(R.DirSource(a.out), disk_cfg, dt, verbose=False))
    d_first, d_greedy, _ = reference(disk, prompt, GREEDY)
    disk_dlogit = float((d_first - first).abs().max())

    # Per-layer / logits dumps of the full 32-token sequence (causal: the
    # first 24 positions are the prompt's), in the primary semantics.
    seq = prompt + greedy
    d = R.Dumper(os.path.join(a.out, "ref"), list(range(len(seq))), sub=True)
    h = fold.forward(seq, on_layer=d, want_sub=True)
    head = fold.head()
    d.finish(seq, h, lambda x: fold.logits(x, head), list(range(len(seq))),
             {"dtype": "float64 math, float32 files", "layout": a.qkv_layout, "seed": seed,
              "semantics": "vfold_f16" if fold is not exact else "exact"})

    common = {
        "quant": "f16",
        "prompt_ids": prompt,
        "prompt_text": prompt_text,
        "greedy_ids": greedy,
        "seed": seed,
        "qkv_layout": a.qkv_layout,
        "expect": "pass" if a.qkv_layout == "chunked" else "FAIL (negative control)",
        "min_greedy_margin": mmin,
        "min_route_margin": rmin,
    }
    ref = dict(common, first_logits=[float(x) for x in first.float()],
               semantics=("V rows x attention_value_scale folded in f32 and rounded to f16 "
                          "(what an f16 CMF holds when the converter folds the scale into V)"
                          if fold is not exact else "exact"),
               greedy_margins=margins, max_abs_logit=float(first.abs().max()),
               roundtrip_disk_max_abs_dlogit=disk_dlogit,
               exact_vs_vfold_max_abs_dlogit=float((e_first - first).abs().max()))
    ref_exact = dict(common, first_logits=[float(x) for x in e_first.float()],
                     semantics="exact: attention_value_scale applied to V at runtime, weights as stored",
                     greedy_margins=e_margins)
    if a.qkv_layout == "contig":
        ref["negative_check"] = {
            "what": "oracle reading THIS file with the release (chunked) layout vs the true model",
            "max_abs_dlogit": disk_dlogit,
            "greedy_from_file": d_greedy,
            "greedy_equal": d_greedy == greedy,
            "top1_equal": int(d_first.argmax()) == int(first.argmax()),
        }
    json.dump(ref, open(os.path.join(a.out, "reference.json"), "w"))
    json.dump(ref_exact, open(os.path.join(a.out, "reference_exact.json"), "w"))
    summary = {k: v for k, v in ref.items() if k != "first_logits"}
    json.dump(summary, open(os.path.join(a.out, "reference_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {a.out}: {files} + {copied}")


if __name__ == "__main__":
    main()
