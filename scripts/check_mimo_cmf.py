#!/usr/bin/env python3
"""Read-only MiMo-V2 (model_type mimo_v2) CMF coverage and profile gate.

Reads only CMF envelopes, JSON headers and tensor directories (the parsing is
shared with scripts/check_dsv41_cmf.py); never touches the weight blob, so it
is safe on a 160 GB file and after the source shards are gone. Expected
coverage is derived independently from the source config.json:

  * every layer: q/k/v/o_proj with the per-layer geometry (release: q 12288
    rows; k 768 | 1536 and v 512 | 1024 rows on full | SWA layers; o_proj
    [4096, 8192]) + both RMSNorms;
  * a learned sink vector [num_heads] F32 on each sink layer (release: the 39
    SWA layers), none elsewhere;
  * dense MLP on moe_layer_freq == 0 layers (release: layer 0);
  * on MoE layers: router mlp.gate.weight F16, mlp.expert_bias F32 (renamed
    from mlp.gate.e_score_correction_bias) and n_routed_experts x
    gate/up/down (release: 47 x 256 x 3 = 36096 matrices);
  * embed_tokens, norm, untied lm_head;
  * NOTHING else: no visual.*, audio_encoder.*, speech_embeddings.*,
    model.mtp.*, no *.weight_scale / *.weight_scale_inv, no fused qkv_proj.

Profiles (--profile):
  q4tp  experts q4tp; q/k/v/o, dense MLP, embed, lm_head q8_2f; router f16;
        expert_bias and sinks f32; norms f16|f32   (the mimo_v2 default)
  f16   every matrix f16 (toys); expert_bias and sinks f32; norms f16|f32
  any   structure only; prints the dtype histogram

Header: arch_name, geometry, layer_types (SlidingAttention where
hybrid_layer_pattern == 1), kv_heads_per_layer, v_head_dim, sliding_window,
rope thetas, eps, untied head, no MTP, MoE block (sigmoid, top-k, no shared
expert). --skip-header turns these into warnings.

With --source-index every source tensor must be accounted for: mapped to a
CMF name, or one of the intentional drops (quant scales, visual / audio /
speech / MTP). An unknown source tensor is an error (it would be silently lost).

  python3 scripts/check_mimo_cmf.py --model /root/mimo/out/mimo-q4tp.cmf \
      --source-config /root/mimo/src/config.json --source-index /root/mimo/src/model.safetensors.index.json
  python3 scripts/check_mimo_cmf.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_dsv41_cmf import CheckError, Entry, artifact_paths, parse_artifact  # noqa: E402

SINK_NAMES = ("attention_sink_bias", "sinks", "attn_sink", "sink")
DROP_PREFIXES = ("visual.", "audio_encoder.", "speech_embeddings.", "model.mtp.", "mtp.")
NORM_DTYPES = ("f16", "f32")


class Geo:
    def __init__(self, cfg: dict, li: int):
        swa = (cfg.get("hybrid_layer_pattern") or [0] * cfg["num_hidden_layers"])[li] == 1
        nq = cfg["num_attention_heads"]
        nkv = cfg.get("num_key_value_heads") or nq
        hd = cfg.get("head_dim") or cfg["hidden_size"] // nq
        vd = cfg.get("v_head_dim") or hd
        if swa:
            nq = cfg.get("swa_num_attention_heads") or nq
            nkv = cfg.get("swa_num_key_value_heads") or nkv
            hd = cfg.get("swa_head_dim") or hd
            vd = cfg.get("swa_v_head_dim") or hd
        self.swa, self.nq, self.nkv, self.hd, self.vd = swa, nq, nkv, hd, vd
        self.sink = bool(cfg.get("add_swa_attention_sink_bias" if swa else "add_full_attention_sink_bias", False))


def moe_layers(cfg):
    n = cfg["num_hidden_layers"]
    f = cfg.get("moe_layer_freq")
    if isinstance(f, int):
        f = [f > 0 and i % f == 0 for i in range(n)]
    f = f or [0] * n
    return [bool(x) and bool(cfg.get("n_routed_experts")) for x in f]


def expected(cfg: dict) -> dict[str, tuple[tuple[int, ...], str]]:
    """name -> (shape, role). Roles drive the dtype profile."""
    H, V, n = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    I = cfg.get("intermediate_size")
    MI = cfg.get("moe_intermediate_size")
    E = cfg.get("n_routed_experts") or 0
    moe = moe_layers(cfg)
    out = {
        "model.embed_tokens.weight": ((V, H), "embed"),
        "model.norm.weight": ((H,), "norm"),
    }
    if not cfg.get("tie_word_embeddings", False):
        out["lm_head.weight"] = ((V, H), "lm_head")
    for li in range(n):
        g = Geo(cfg, li)
        p = f"model.layers.{li}."
        out[p + "self_attn.q_proj.weight"] = ((g.nq * g.hd, H), "attn")
        out[p + "self_attn.k_proj.weight"] = ((g.nkv * g.hd, H), "attn")
        out[p + "self_attn.v_proj.weight"] = ((g.nkv * g.vd, H), "attn")
        out[p + "self_attn.o_proj.weight"] = ((H, g.nq * g.vd), "attn")
        out[p + "input_layernorm.weight"] = ((H,), "norm")
        out[p + "post_attention_layernorm.weight"] = ((H,), "norm")
        if moe[li]:
            out[p + "mlp.gate.weight"] = ((E, H), "router")
            out[p + "mlp.expert_bias"] = ((E,), "bias")
            for e in range(E):
                q = f"{p}mlp.experts.{e}."
                out[q + "gate_proj.weight"] = ((MI, H), "expert")
                out[q + "up_proj.weight"] = ((MI, H), "expert")
                out[q + "down_proj.weight"] = ((H, MI), "expert")
        else:
            out[p + "mlp.gate_proj.weight"] = ((I, H), "dense")
            out[p + "mlp.up_proj.weight"] = ((I, H), "dense")
            out[p + "mlp.down_proj.weight"] = ((H, I), "dense")
    return out


PROFILES = {
    "q4tp": {"expert": ("q4tp",), "attn": ("q8_2f",), "dense": ("q8_2f",), "embed": ("q8_2f",),
             "lm_head": ("q8_2f",), "router": ("f16",), "bias": ("f32",), "sink": ("f32",),
             "norm": NORM_DTYPES},
    "f16": {"expert": ("f16",), "attn": ("f16",), "dense": ("f16",), "embed": ("f16",),
            "lm_head": ("f16",), "router": ("f16",), "bias": ("f32",), "sink": ("f32",),
            "norm": NORM_DTYPES},
}


def sink_layer_of(name):
    m = re.match(r"^model\.layers\.(\d+)\.self_attn\.(\w+)$", name)
    if m and m.group(2) in SINK_NAMES:
        return int(m.group(1))
    return None


def check_tensors(entries: list[Entry], cfg: dict, profile: str) -> tuple[list[str], dict]:
    errors: list[str] = []
    actual: dict[str, Entry] = {}
    for e in entries:
        if e.name in actual:
            errors.append(f"duplicate tensor across CMF shards: {e.name}")
        actual[e.name] = e
    exp = expected(cfg)
    n = cfg["num_hidden_layers"]
    sink_layers = [li for li in range(n) if Geo(cfg, li).sink]
    # Sinks: exactly one vector per sink layer, under any accepted name.
    found_sinks: dict[int, Entry] = {}
    for name, e in actual.items():
        li = sink_layer_of(name)
        if li is None:
            continue
        if li in found_sinks:
            errors.append(f"layer {li} has two sink tensors: {found_sinks[li].name}, {name}")
        found_sinks[li] = e
    for li in sink_layers:
        g = Geo(cfg, li)
        e = found_sinks.get(li)
        if e is None:
            errors.append(f"layer {li}: missing sink vector (self_attn.{{{','.join(SINK_NAMES)}}})")
            continue
        exp[e.name] = ((g.nq,), "sink")
    for li, e in found_sinks.items():
        if li not in sink_layers:
            errors.append(f"layer {li} is not a sink layer but carries {e.name}")

    forbidden = [nm for nm in actual if nm.startswith(DROP_PREFIXES)]
    if forbidden:
        errors.append(f"{len(forbidden)} tensors that a text-only mimo_v2 CMF must drop: {forbidden[:6]}")
    scales = [nm for nm in actual if nm.endswith((".weight_scale", ".weight_scale_inv"))]
    if scales:
        errors.append(f"{len(scales)} source quant-scale tensors leaked into the CMF: {scales[:4]}")
    fused = [nm for nm in actual if nm.endswith("self_attn.qkv_proj.weight")]
    if fused:
        errors.append(f"{len(fused)} fused qkv_proj tensors (must be split into q/k/v): {fused[:3]}")
    raw_bias = [nm for nm in actual if nm.endswith("mlp.gate.e_score_correction_bias")]
    if raw_bias:
        errors.append(f"{len(raw_bias)} router biases under the source name (loader reads mlp.expert_bias): {raw_bias[:3]}")

    missing = sorted(set(exp) - set(actual))
    unexpected = sorted(set(actual) - set(exp) - set(forbidden) - set(scales) - set(fused) - set(raw_bias))
    if missing:
        errors.append(f"missing {len(missing)} tensors: {missing[:8]}")
    if unexpected:
        errors.append(f"unexpected {len(unexpected)} tensors: {unexpected[:8]}")

    allowed = PROFILES.get(profile)
    shape_err, dtype_err = [], []
    hist: Counter = Counter()
    for name, (shape, role) in exp.items():
        e = actual.get(name)
        if e is None:
            continue
        hist[(role, e.dtype)] += 1
        if tuple(e.shape) != tuple(shape):
            shape_err.append(f"{name} {tuple(e.shape)} != {tuple(shape)}")
        if allowed and e.dtype not in allowed[role]:
            dtype_err.append(f"{name} ({role}) is {e.dtype}, profile {profile} wants {'|'.join(allowed[role])}")
    if shape_err:
        errors.append(f"{len(shape_err)} shape mismatches: {shape_err[:6]}")
    if dtype_err:
        errors.append(f"{len(dtype_err)} dtype mismatches: {dtype_err[:6]}")

    moe = moe_layers(cfg)
    krows = Counter(actual[f"model.layers.{li}.self_attn.k_proj.weight"].shape[0]
                    for li in range(n) if f"model.layers.{li}.self_attn.k_proj.weight" in actual)
    vrows = Counter(actual[f"model.layers.{li}.self_attn.v_proj.weight"].shape[0]
                    for li in range(n) if f"model.layers.{li}.self_attn.v_proj.weight" in actual)
    stats = {
        "tensors": len(actual),
        "expected": len(exp),
        "layers_with_qkvo": sum(all(f"model.layers.{li}.self_attn.{x}_proj.weight" in actual
                                    for x in "qkvo") for li in range(n)),
        "k_rows": dict(sorted(krows.items())),
        "v_rows": dict(sorted(vrows.items())),
        "expert_matrices": sum(1 for nm in actual if ".mlp.experts." in nm),
        "expected_expert_matrices": sum(moe) * (cfg.get("n_routed_experts") or 0) * 3,
        "expert_bias_f32": sum(1 for nm, e in actual.items() if nm.endswith(".mlp.expert_bias") and e.dtype == "f32"),
        "sinks_f32": sum(1 for e in found_sinks.values() if e.dtype == "f32"),
        "sink_layers": len(sink_layers),
        "dtype_by_role": {f"{r}:{d}": c for (r, d), c in sorted(hist.items())},
    }
    by = Counter()
    for e in actual.values():
        by[e.dtype] += e.nbytes
    stats["bytes_by_dtype"] = dict(sorted(by.items()))
    return errors, stats


def check_header(header: dict, cfg: dict, arch_name: str) -> list[str]:
    errs: list[str] = []
    arch = header.get("arch")
    if not isinstance(arch, dict):
        return ["header.arch is not an object"]
    n = cfg["num_hidden_layers"]

    def eq(field, want, rel=None):
        got = arch.get(field)
        if rel is not None and isinstance(got, (int, float)) and isinstance(want, (int, float)):
            if not math.isclose(float(got), float(want), rel_tol=rel):
                errs.append(f"arch.{field}={got!r} != {want!r}")
        elif got != want:
            errs.append(f"arch.{field}={got!r} != {want!r}")

    if arch_name:
        eq("arch_name", arch_name)
    eq("hidden_size", cfg["hidden_size"])
    eq("num_layers", n)
    eq("num_attention_heads", cfg["num_attention_heads"])
    eq("head_dim", cfg["head_dim"])
    eq("vocab_size", cfg["vocab_size"])
    eq("rms_norm_eps", cfg.get("layernorm_epsilon", 1e-6), 1e-9)
    eq("rope_theta", cfg["rope_theta"], 1e-9)
    eq("tie_word_embeddings", bool(cfg.get("tie_word_embeddings", False)))
    geos = [Geo(cfg, li) for li in range(n)]
    eq("layer_types", ["SlidingAttention" if g.swa else "FullAttention" for g in geos])
    kv = [g.nkv for g in geos]
    got_kv = arch.get("kv_heads_per_layer")
    if got_kv is None and len(set(kv)) > 1:
        errs.append("arch.kv_heads_per_layer is absent but KV heads vary by layer")
    elif got_kv is not None and got_kv != kv:
        errs.append(f"arch.kv_heads_per_layer={got_kv} != {kv}")
    # With a per-layer list, the scalar is only a default; either width is valid.
    if arch.get("num_kv_heads") not in ({cfg["num_key_value_heads"]} | (set(kv) if got_kv else set())):
        errs.append(f"arch.num_kv_heads={arch.get('num_kv_heads')!r} != {cfg['num_key_value_heads']}")
    vd = cfg.get("v_head_dim") or cfg["head_dim"]
    if vd != cfg["head_dim"]:
        eq("v_head_dim", vd)
    if any(g.swa for g in geos):
        eq("sliding_window", cfg.get("sliding_window") or cfg.get("sliding_window_size"))
        eq("rope_local_base_freq", cfg.get("swa_rope_theta") or cfg["rope_theta"], 1e-9)
    # 0.334 (source) or 64/192 (exact) both rotate int(192 * f) = 64 dims.
    prf = arch.get("partial_rotary_factor")
    hd = cfg["head_dim"]
    want_rope = int(hd * cfg.get("partial_rotary_factor", 1.0))
    if not isinstance(prf, (int, float)) or int(hd * float(prf) + 1e-4) != want_rope:
        errs.append(f"arch.partial_rotary_factor={prf!r} does not give rope dim {want_rope}")
    if arch.get("mtp") is not None:
        errs.append(f"arch.mtp={arch.get('mtp')!r}: MTP must be dropped for mimo_v2 (text-only)")
    if cfg.get("n_routed_experts"):
        moe = arch.get("moe")
        if not isinstance(moe, dict):
            errs.append("arch.moe is absent")
        else:
            for f, want in (("num_experts", cfg["n_routed_experts"]), ("top_k", cfg["num_experts_per_tok"]),
                            ("moe_intermediate_size", cfg["moe_intermediate_size"]),
                            ("norm_topk_prob", bool(cfg.get("norm_topk_prob", True)))):
                if moe.get(f) != want:
                    errs.append(f"arch.moe.{f}={moe.get(f)!r} != {want!r}")
            if not moe.get("router_sigmoid", False):
                errs.append("arch.moe.router_sigmoid is not true")
            if moe.get("shared_expert_intermediate_size"):
                errs.append("arch.moe has a shared expert; mimo_v2 has none")
            rsf = moe.get("routed_scaling_factor")
            if rsf not in (None, 1, 1.0) and cfg.get("routed_scaling_factor") in (None, 1, 1.0):
                errs.append(f"arch.moe.routed_scaling_factor={rsf!r}, source has none")
    return errs


def check_source_index(weight_map: dict, cfg: dict, cmf_names: set[str]) -> tuple[list[str], dict]:
    """Every source tensor must be mapped or intentionally dropped."""
    errs, drops, mapped = [], Counter(), 0
    unknown = []
    for raw in weight_map:
        if raw.startswith(DROP_PREFIXES):
            drops[raw.split(".")[0] if not raw.startswith("model.mtp.") else "model.mtp"] += 1
            continue
        if raw.endswith((".weight_scale", ".weight_scale_inv")):
            drops["quant_scales"] += 1
            continue
        if raw.endswith("self_attn.qkv_proj.weight"):
            p = raw[: -len("qkv_proj.weight")]
            want = [p + f"{x}_proj.weight" for x in "qkv"]
        elif raw.endswith("mlp.gate.e_score_correction_bias"):
            want = [raw.replace("mlp.gate.e_score_correction_bias", "mlp.expert_bias")]
        elif raw.endswith("self_attn.attention_sink_bias"):
            p = raw[: -len("attention_sink_bias")]
            want = [next((p + s for s in SINK_NAMES if p + s in cmf_names), p + "attention_sink_bias")]
        else:
            want = [raw]
        miss = [w for w in want if w not in cmf_names]
        if miss:
            unknown.append(f"{raw} -> {miss[0]}")
        else:
            mapped += 1
    if unknown:
        errs.append(f"{len(unknown)} source tensors have no CMF counterpart: {unknown[:6]}")
    return errs, {"source_tensors": len(weight_map), "mapped": mapped, "dropped": dict(drops)}


def run(model: str, cfg: dict, profile: str, arch_name: str, skip_header: bool, weight_map=None) -> int:
    artifacts = [parse_artifact(p) for p in artifact_paths(Path(model))]
    entries = [e for a in artifacts for e in a.entries]
    errors, stats = check_tensors(entries, cfg, profile)
    herr = check_header(artifacts[0].header, cfg, arch_name)
    warnings = []
    if skip_header:
        warnings += herr
    else:
        errors += herr
    src_stats = None
    if weight_map is not None:
        e2, src_stats = check_source_index(weight_map, cfg, {e.name for e in entries})
        errors += e2
    print(f"{'PASS' if not errors else 'FAIL'}: {model} ({len(artifacts)} file(s), "
          f"{sum(a.size for a in artifacts)} bytes, profile {profile})")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if src_stats:
        print(f"  source: {src_stats}")
    for w in warnings:
        print(f"  WARN header: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    return 0 if not errors else 1


# --------------------------------------------------------------------------
# self-test: synthetic directories, and synthetic (sparse) CMF files
# --------------------------------------------------------------------------
def expected_arch(cfg: dict) -> dict:
    """The header contract this gate checks, as a converter would write it."""
    n = cfg["num_hidden_layers"]
    geos = [Geo(cfg, li) for li in range(n)]
    return {
        "arch_name": "mimo_v2", "hidden_size": cfg["hidden_size"], "num_layers": n,
        "num_attention_heads": cfg["num_attention_heads"], "num_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["head_dim"], "vocab_size": cfg["vocab_size"],
        "layer_types": ["SlidingAttention" if g.swa else "FullAttention" for g in geos],
        "rms_norm_eps": cfg.get("layernorm_epsilon", 1e-6), "rope_theta": cfg["rope_theta"],
        "tie_word_embeddings": False, "partial_rotary_factor": cfg.get("partial_rotary_factor", 1.0),
        "kv_heads_per_layer": [g.nkv for g in geos], "v_head_dim": cfg.get("v_head_dim"),
        "sliding_window": cfg.get("sliding_window"), "rope_local_base_freq": cfg.get("swa_rope_theta"),
        "mtp": None,
        "moe": {"num_experts": cfg["n_routed_experts"], "top_k": cfg["num_experts_per_tok"],
                "moe_intermediate_size": cfg["moe_intermediate_size"], "norm_topk_prob": True,
                "router_sigmoid": True},
    }


def write_synthetic_cmf(path: str, header: dict, entries: list[Entry]) -> None:
    """A structurally valid CMF v2 whose data section is a sparse hole: the
    gate reads only envelope/header/directory, so a 160 GB q4tp directory
    costs a few MB of disk."""
    import struct

    from check_dsv41_cmf import DTYPES, expected_nbytes

    ids = {v: k for k, v in DTYPES.items()}
    hdr = json.dumps(dict(header, format="cmf", version=2)).encode()
    pool, recs, off = b"", [], 0
    for e in entries:
        nb = expected_nbytes(e.dtype, tuple(e.shape))
        name = e.name.encode()
        shape = list(e.shape) + [0] * (6 - len(e.shape))
        recs.append(struct.pack("<IHBB6IQQQ", len(pool), len(name), ids[e.dtype], len(e.shape), *shape,
                                off, nb, 0))
        pool += name
        off += (nb + 63) // 64 * 64
    directory = struct.pack("<QQ", len(recs), 16 + 56 * len(recs)) + b"".join(recs) + pool
    h_off = 128
    d_off = h_off + len(hdr)
    data_off = (d_off + len(directory) + 4095) // 4096 * 4096
    data_len = max(off, 64)
    env = b"CMF\x01" + struct.pack("<III", 2, 0, 1)
    env += struct.pack("<12Q", h_off, len(hdr), d_off, len(directory), data_off, data_len, 0, 0, 0, 0, 0, 0)
    env += struct.pack("<QQ", 0, 0)
    with open(path, "wb") as f:
        f.write(env)
        f.write(hdr)
        f.write(directory)
        f.truncate(data_off + data_len)


def self_test_files(cfg: dict, tmp: str) -> None:
    ok = os.path.join(tmp, "ok.cmf")
    write_synthetic_cmf(ok, {"arch": expected_arch(cfg)}, _entries_for(cfg, "q4tp"))
    assert run(ok, cfg, "q4tp", "mimo_v2", False) == 0
    bad_arch = expected_arch(cfg)
    bad_arch["kv_heads_per_layer"] = None
    bad_arch["mtp"] = {"num_layers": 3}
    bad = os.path.join(tmp, "bad.cmf")
    write_synthetic_cmf(bad, {"arch": bad_arch}, _entries_for(
        cfg, "q4tp", lambda es: es + [Entry("speech_embeddings.0.weight", "f16", (1280, 1024), 0, 0, 0, "s")]))
    assert run(bad, cfg, "q4tp", "mimo_v2", False) == 1
    print("  synthetic CMF files: good one PASSES, bad one (no kv_heads_per_layer, MTP kept, speech kept) FAILS")


# --------------------------------------------------------------------------
# self-test: synthetic directories, no files
# --------------------------------------------------------------------------
def _entries_for(cfg, profile, mutate=None):
    out = []
    roles = PROFILES[profile]
    exp = expected(cfg)
    for li in range(cfg["num_hidden_layers"]):
        g = Geo(cfg, li)
        if g.sink:
            exp[f"model.layers.{li}.self_attn.sinks"] = ((g.nq,), "sink")
    for name, (shape, role) in exp.items():
        out.append(Entry(name, roles[role][0], tuple(shape), 0, 0, 0, "synthetic"))
    if mutate:
        out = mutate(out)
    return out


def self_test(cfg_path=None) -> None:
    real = None
    for cand in (cfg_path, "/root/mimo/src/config.json"):
        if cand and os.path.exists(cand):
            real = json.load(open(cand))
            break
    if real is None:
        real = {
            "hidden_size": 4096, "num_hidden_layers": 48, "vocab_size": 152576, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128, "swa_num_attention_heads": 64,
            "swa_num_key_value_heads": 8, "swa_head_dim": 192, "swa_v_head_dim": 128,
            "add_swa_attention_sink_bias": True, "add_full_attention_sink_bias": False,
            # full attention at 0, 5, 11, ..., 47 (the release pattern)
            "hybrid_layer_pattern": [0 if i == 0 or (i - 5) % 6 == 0 else 1 for i in range(48)],
            "moe_layer_freq": [0] + [1] * 47, "n_routed_experts": 256, "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048, "intermediate_size": 16384, "tie_word_embeddings": False,
        }
    _, st = check_tensors(_entries_for(real, "q4tp"), real, "q4tp")
    e, _ = check_tensors(_entries_for(real, "q4tp"), real, "q4tp")
    assert not e, e
    print(f"  release geometry: {st['layers_with_qkvo']} layers q/k/v/o, k rows {st['k_rows']}, "
          f"v rows {st['v_rows']}, {st['expert_matrices']} expert matrices, {st['expert_bias_f32']} expert_bias f32, "
          f"{st['sinks_f32']} sinks f32")
    if os.path.exists("/root/mimo/src/config.json") or cfg_path:
        assert st["expert_matrices"] == 36096 and st["expert_bias_f32"] == 47 and st["sinks_f32"] == 39, st
        assert st["k_rows"] == {768: 9, 1536: 39} and st["v_rows"] == {512: 9, 1024: 39}, st
    cases = {
        "visual tensor kept": lambda es: es + [Entry("visual.blocks.0.attn.qkv.weight", "f16", (96, 32), 0, 0, 0, "s")],
        "mtp kept": lambda es: es + [Entry("model.mtp.layers.0.eh_proj.weight", "q8_2f", (4096, 8192), 0, 0, 0, "s")],
        "sink dropped": lambda es: [x for x in es if not x.name.endswith("layers.1.self_attn.sinks")],
        "sink as f16": lambda es: [x if not x.name.endswith("layers.1.self_attn.sinks")
                                   else Entry(x.name, "f16", x.shape, 0, 0, 0, "s") for x in es],
        "expert at q8_2f": lambda es: [x if not x.name.endswith("layers.3.mlp.experts.7.up_proj.weight")
                                       else Entry(x.name, "q8_2f", x.shape, 0, 0, 0, "s") for x in es],
        "k rows of an SWA layer = full": lambda es: [x if x.name != "model.layers.1.self_attn.k_proj.weight"
                                                     else Entry(x.name, x.dtype, (768, 4096), 0, 0, 0, "s") for x in es],
        "bias under source name": lambda es: [x if not x.name.endswith("layers.2.mlp.expert_bias")
                                              else Entry("model.layers.2.mlp.gate.e_score_correction_bias", "f32", x.shape, 0, 0, 0, "s")
                                              for x in es],
        "fused qkv left": lambda es: es + [Entry("model.layers.0.self_attn.qkv_proj.weight", "q8_2f", (13568, 4096), 0, 0, 0, "s")],
        "router q8_2f": lambda es: [x if not x.name.endswith("layers.5.mlp.gate.weight")
                                    else Entry(x.name, "q8_2f", x.shape, 0, 0, 0, "s") for x in es],
    }
    for title, mut in cases.items():
        e, _ = check_tensors(_entries_for(real, "q4tp", mut), real, "q4tp")
        assert e, f"negative case not caught: {title}"
        print(f"  caught: {title}: {e[0][:110]}")
    import tempfile

    for k, v in (("layernorm_epsilon", 1e-6), ("rope_theta", 1e7), ("swa_rope_theta", 1e4),
                 ("partial_rotary_factor", 0.334), ("sliding_window", 128)):
        real.setdefault(k, v)
    # tools/mk_mimo_toy.py geometry
    toy = dict(real, hidden_size=256, intermediate_size=320, num_hidden_layers=4, num_attention_heads=8,
               num_key_value_heads=2, head_dim=48, v_head_dim=32, swa_num_attention_heads=8,
               swa_num_key_value_heads=4, swa_head_dim=48, swa_v_head_dim=32, sliding_window=8,
               hybrid_layer_pattern=[0, 1, 1, 0], moe_layer_freq=[0, 1, 1, 1], n_routed_experts=8,
               num_experts_per_tok=2, moe_intermediate_size=64)
    with tempfile.TemporaryDirectory() as tmp:
        for title, c in (("release", real), ("toy", toy)):
            print(f"  [{title} geometry]")
            self_test_files(c, tmp)
    print("self-test PASS")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="CMF path (or shard 1)")
    ap.add_argument("--source-config", help="HF config.json of the source (release or toy)")
    ap.add_argument("--source-index", help="HF model.safetensors.index.json (optional, checks every source tensor)")
    ap.add_argument("--profile", default="q4tp", choices=["q4tp", "f16", "any"])
    ap.add_argument("--arch-name", default="mimo_v2", help="expected arch.arch_name ('' = do not check)")
    ap.add_argument("--skip-header", action="store_true", help="report header mismatches as warnings")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test(a.source_config)
        return 0
    if not a.model or not a.source_config:
        ap.error("--model and --source-config are required")
    try:
        cfg = json.load(open(a.source_config))
        wm = json.load(open(a.source_index))["weight_map"] if a.source_index else None
        return run(a.model, cfg, a.profile, a.arch_name, a.skip_header, wm)
    except (OSError, json.JSONDecodeError, CheckError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
