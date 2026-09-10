#!/usr/bin/env python3
"""Read-only DeepSeek-V4.1 CMF artifact gate.

The checker reads CMF envelopes, JSON headers, and binary tensor directories
only.  It never hashes or scans the weight blob, so it is safe to run after the
converter has consumed the source shards.  Coverage is derived independently
from the pinned HF index and the narrow DeepSeek-V4.1 name contract below.

Example (after conversion):
  python3 scripts/check_dsv41_cmf.py \
    --model /root/dsv41/dsv41-q4tp.cmf \
    --source-index /root/dsv41/hf/model.safetensors.index.json \
    --source-config /root/dsv41/hf/config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


MAGIC = b"CMF\x01"
ENVELOPE_LEN = 128
DIR_RECORD_LEN = 56
GROUP_SIZE = 32
# Keep this list in step with cortiq-core::format::features::SUPPORTED.  The
# high bits are legal for current CMF readers even though a plain model usually
# needs only TENSOR_DIR (and, for multi-scale quantizers, QUANT_2F).
KNOWN_FEATURES = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6)
DTYPES = {
    0: "f32",
    1: "f16",
    2: "bf16",
    3: "q8_row",
    4: "q4_block",
    5: "mix8_4",
    6: "u8",
    7: "q4_col",
    8: "vbit",
    9: "q8_2f",
    10: "vbit_ro",
    11: "q4_tiled",
    12: "q1",
    13: "q1s",
    14: "q1t",
    15: "q4tp",
    16: "q2tp",
}
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.cmf$")
LAYER_RE = re.compile(r"^layers\.(\d+)\.(.*)$")
CANON_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
SHARED_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$"
)
ENGRAM_RE = re.compile(r"^layers\.(\d+)\.engram\.(embed\.(?:weight|scale)|wkv\.weight)$")


class CheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class Entry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    off: int
    nbytes: int
    digest: int
    shard: str


@dataclass
class Artifact:
    path: Path
    size: int
    header: dict
    entries: list[Entry]
    sections: dict[str, tuple[int, int]]
    header_hash: int
    dir_hash: int


def read_at(fd: int, offset: int, size: int) -> bytes:
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise CheckError(f"short read at {offset}+{size}: got {len(data)}")
    return data


def _fmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & mask
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & mask
    x ^= x >> 33
    return x & mask


def hash64(data: bytes) -> int:
    """CMF hash64, duplicated here to keep this gate independent of Rust."""
    mask = (1 << 64) - 1
    padded = data + b"\0" * ((-len(data)) % 8)
    h = 0
    for i in range(0, len(padded), 8):
        word = struct.unpack_from("<Q", padded, i)[0]
        h ^= _fmix64(word) ^ (((i // 8) * 0x9E3779B97F4A7C15) & mask)
    return _fmix64((h ^ len(data)) & mask)


def product(shape: tuple[int, ...] | list[int]) -> int:
    n = 1
    for dim in shape:
        if dim < 0:
            raise CheckError(f"negative tensor dimension {shape}")
        n *= dim
    return n


def expected_nbytes(dtype: str, shape: tuple[int, ...]) -> int | None:
    n = product(shape)
    if dtype == "f32":
        return n * 4
    if dtype in ("f16", "bf16"):
        return n * 2
    if dtype == "u8":
        return n
    if dtype == "q8_row":
        return n + (shape[0] * 2 if shape else 0)
    if dtype in ("q4_block", "q4_tiled"):
        return ((n + GROUP_SIZE - 1) // GROUP_SIZE) * 18
    if dtype == "q1":
        return ((n + GROUP_SIZE - 1) // GROUP_SIZE) * 6
    if dtype == "q8_2f" and len(shape) == 2:
        return n + (shape[0] + shape[1]) * 2
    if dtype in ("q4tp", "q2tp") and len(shape) == 2 and shape[1] > 0:
        rows, cols = shape
        if cols % GROUP_SIZE:
            return None
        groups = cols // GROUP_SIZE
        code_stride = (groups * 5 + 7) // 8
        nibble_bytes = 16 if dtype == "q4tp" else 8
        return rows * groups * nibble_bytes + rows * 4 + rows * code_stride
    # vbit, q1s, q1t and reserved formats have self-describing or undefined
    # spans.  Bounds and the directory's own nbytes are still checked.
    return None


def _sections(envelope: bytes, file_size: int) -> dict[str, tuple[int, int]]:
    version, _flags, required = struct.unpack_from("<III", envelope, 4)
    if version != 2:
        raise CheckError(f"unsupported CMF version {version}")
    if not required & (1 << 0):
        raise CheckError("required_features does not declare TENSOR_DIR")
    unknown = required & ~KNOWN_FEATURES
    if unknown:
        raise CheckError(f"unknown required_features {unknown:#x}")
    values = struct.unpack_from("<12Q", envelope, 16)
    names = (
        "header",
        "directory",
        "data",
        "masks",
        "vocab",
        "index",
    )
    sections = {names[i]: (values[i * 2], values[i * 2 + 1]) for i in range(6)}
    for name, (offset, length) in sections.items():
        if (offset == 0) != (length == 0):
            raise CheckError(f"{name} has only one of offset/length set")
        if length and offset < ENVELOPE_LEN:
            raise CheckError(f"{name} [{offset}, {offset + length}) overlaps the envelope")
        if offset + length > file_size:
            raise CheckError(f"{name} {offset}+{length} exceeds file size {file_size}")
    if sections["header"][0] < ENVELOPE_LEN or not sections["header"][1]:
        raise CheckError("header is missing or overlaps the envelope")
    if sections["directory"][0] == 0 or sections["directory"][1] < 16:
        raise CheckError("directory is missing or shorter than its preamble")
    data_off, data_len = sections["data"]
    if data_off == 0 or data_off % 4096:
        raise CheckError(f"data offset {data_off} is not a nonzero 4096-byte boundary")
    active = [(name, off, off + length) for name, (off, length) in sections.items() if length]
    for i, (left, lo, hi) in enumerate(active):
        for right, rlo, rhi in active[i + 1 :]:
            if lo < rhi and rlo < hi:
                raise CheckError(f"CMF sections overlap: {left} and {right}")
    if data_len == 0:
        raise CheckError("CMF data section is empty")
    return sections


def parse_artifact(path: Path) -> Artifact:
    size = path.stat().st_size
    with path.open("rb") as f:
        fd = f.fileno()
        envelope = read_at(fd, 0, ENVELOPE_LEN)
        if envelope[:4] != MAGIC:
            raise CheckError(f"{path}: invalid CMF magic {envelope[:4]!r}")
        sections = _sections(envelope, size)
        header_off, header_len = sections["header"]
        header_bytes = read_at(fd, header_off, header_len)
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckError(f"{path}: invalid header JSON: {exc}") from exc
        if header.get("format") != "cmf" or header.get("version") != 2:
            raise CheckError(f"{path}: header format/version is not cmf/v2")
        directory_off, directory_len = sections["directory"]
        directory = read_at(fd, directory_off, directory_len)
        count, pool_off = struct.unpack_from("<QQ", directory, 0)
        records_end = 16 + count * DIR_RECORD_LEN
        if pool_off < records_end or pool_off > directory_len:
            raise CheckError(
                f"{path}: directory pool offset {pool_off} is outside records/end {records_end}"
            )
        entries: list[Entry] = []
        names: set[str] = set()
        data_len = sections["data"][1]
        for i in range(count):
            rec = 16 + i * DIR_RECORD_LEN
            noff, nlen, dtype_id, ndim = struct.unpack_from("<IHBB", directory, rec)
            if ndim > 6:
                raise CheckError(f"{path}: tensor record {i} has ndim {ndim} > 6")
            dtype = DTYPES.get(dtype_id)
            if dtype is None:
                raise CheckError(f"{path}: tensor record {i} has unknown dtype id {dtype_id}")
            shape = tuple(struct.unpack_from("<6I", directory, rec + 8)[:ndim])
            off, nbytes, digest = struct.unpack_from("<QQQ", directory, rec + 32)
            pool_end = pool_off + noff + nlen
            if pool_end > directory_len:
                raise CheckError(f"{path}: tensor record {i} name exceeds directory pool")
            try:
                name = directory[pool_off + noff : pool_end].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CheckError(f"{path}: tensor record {i} name is not UTF-8") from exc
            if not name:
                raise CheckError(f"{path}: tensor record {i} has an empty name")
            if name in names:
                raise CheckError(f"{path}: duplicate tensor name {name!r}")
            names.add(name)
            if off % 64:
                raise CheckError(f"{path}: tensor {name} offset {off} is not 64-byte aligned")
            if off + nbytes > data_len:
                raise CheckError(
                    f"{path}: tensor {name} payload {off}+{nbytes} exceeds data_len {data_len}"
                )
            expected = expected_nbytes(dtype, shape)
            if expected is not None and expected != nbytes:
                raise CheckError(
                    f"{path}: tensor {name} {dtype}{shape} has {nbytes} bytes, expected {expected}"
                )
            if dtype in ("q4tp", "q2tp") and (len(shape) != 2 or shape[1] % GROUP_SIZE):
                raise CheckError(f"{path}: {dtype} tensor {name} has invalid shape {shape}")
            entries.append(Entry(name, dtype, shape, off, nbytes, digest, path.name))
        header_hash, dir_hash = struct.unpack_from("<QQ", envelope, 0x70)
        if header_hash and hash64(header_bytes) != header_hash:
            raise CheckError(f"{path}: header hash mismatch")
        if dir_hash and hash64(directory) != dir_hash:
            raise CheckError(f"{path}: directory hash mismatch")
    return Artifact(path, size, header, entries, sections, header_hash, dir_hash)


def artifact_paths(path: Path) -> list[Path]:
    match = SHARD_RE.match(path.name)
    if not match:
        return [path]
    stem, number, count = match.groups()
    first = path.with_name(f"{stem}-00001-of-{count}.cmf")
    if number != "00001":
        path = first
    paths = [path.with_name(f"{stem}-{i:05}-of-{count}.cmf") for i in range(1, int(count) + 1)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise CheckError(f"sharded CMF is missing {missing[0]} (and possibly more)")
    return paths


def is_dspark_name(name: str) -> bool:
    return (
        name.startswith("dspark.")
        or ".dspark." in name
        or ".markov_head." in name
        or ".confidence_head." in name
    )


def canonical_name(raw: str) -> str | None:
    """Independent V4.1 source-to-CMF mapping (no import from the converter)."""
    if raw in ("image_start", "image_end", "image_newline"):
        return raw
    if raw == "embed.weight":
        return "model.embed_tokens.weight"
    if raw == "head.weight":
        return "lm_head.weight"
    if raw == "norm.weight":
        return "model.norm.weight"
    if raw.startswith("hc_head_"):
        return f"model.{raw}"
    match = LAYER_RE.match(raw)
    if match:
        layer, tail = match.groups()
        direct = {
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "ffn.gate.weight": "mlp.gate.weight",
            "ffn.gate.bias": "mlp.expert_bias",
            "ffn.gate.bias_vl": "mlp.expert_bias_vl",
            "ffn.gate.tid2eid": "mlp.tid2eid",
        }
        if tail in direct:
            mapped = direct[tail]
        else:
            mapped = tail.replace("ffn.shared_experts.", "mlp.shared_expert.")
            mapped = mapped.replace("ffn.experts.", "mlp.experts.")
            if mapped.startswith(("mlp.experts.", "mlp.shared_expert.")):
                for source, target in (
                    (".w1.weight", ".gate_proj.weight"),
                    (".w3.weight", ".up_proj.weight"),
                    (".w2.weight", ".down_proj.weight"),
                    (".w1.scale", ".gate_proj.scale"),
                    (".w3.scale", ".up_proj.scale"),
                    (".w2.scale", ".down_proj.scale"),
                ):
                    mapped = mapped.replace(source, target)
            mapped = mapped.replace("attn.", "self_attn.")
        return f"model.layers.{layer}.{mapped}"
    if raw.startswith("mtp."):
        rest = raw[4:]
        direct = {
            "fc.weight": "eh_proj.weight",
            "pre_fc_norm_embedding.weight": "enorm.weight",
            "pre_fc_norm_hidden.weight": "hnorm.weight",
        }
        return f"model.mtp.{direct.get(rest, rest)}"
    # V4.1 has no visual wrapper to drop. Preserve auxiliary source names
    # verbatim, matching the converter's final lfm2_canon fallback.
    return raw


def expected_names(weight_map: dict[str, str]) -> tuple[dict[str, str], set[str], int]:
    expected: dict[str, str] = {}
    optional_dspark: set[str] = set()
    omitted_scales = 0
    for raw in weight_map:
        if raw.endswith(".scale") and not raw.endswith(".engram.embed.scale"):
            omitted_scales += 1
            continue
        name = canonical_name(raw)
        if name is None:
            continue
        if name in expected:
            raise CheckError(f"source mapping collision: {raw!r} -> {name!r}, already from {expected[name]!r}")
        expected[name] = raw
        if is_dspark_name(raw):
            optional_dspark.add(name)
    return expected, optional_dspark, omitted_scales


def get_number(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CheckError(f"source config {name} is not numeric")
    return int(value)


def check_metadata(header: dict, config: dict, errors: list[str]) -> dict[str, object]:
    arch = header.get("arch")
    if not isinstance(arch, dict):
        errors.append("header.arch is not an object")
        return {}
    tc = config.get("text_config", config)
    if not isinstance(tc, dict):
        errors.append("source text_config is not an object")
        return {}
    if arch.get("arch_name") != "deepseek_v41":
        errors.append(f"arch_name {arch.get('arch_name')!r} != 'deepseek_v41'")
    if arch.get("deepseek_v41") != config:
        errors.append("arch.deepseek_v41 does not preserve the original source config")
    pairs = {
        "hidden_size": tc.get("hidden_size"),
        "num_layers": tc.get("num_hidden_layers"),
        "num_attention_heads": tc.get("num_attention_heads"),
        "num_kv_heads": tc.get("num_key_value_heads", tc.get("num_attention_heads")),
        "head_dim": tc.get("head_dim"),
        "vocab_size": tc.get("vocab_size"),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta": tc.get("rope_theta", 10000.0),
    }
    for field, source in pairs.items():
        if source is None:
            continue
        if field not in arch:
            errors.append(f"arch.{field} is absent")
        elif isinstance(source, float):
            if not math.isclose(float(arch[field]), source, rel_tol=1e-6, abs_tol=1e-12):
                errors.append(f"arch.{field}={arch[field]!r} != source {source!r}")
        elif arch[field] != source:
            errors.append(f"arch.{field}={arch[field]!r} != source {source!r}")
    n_layers = tc.get("num_hidden_layers")
    if isinstance(n_layers, int):
        raw_types = tc.get("layer_types")
        type_map = {
            "full_attention": "FullAttention",
            "sliding_attention": "SlidingAttention",
            "linear_attention": "LinearAttention",
            "conv": "ShortConv",
            "short_conv": "ShortConv",
        }
        expected_types = (
            [type_map.get(x, "FullAttention") for x in raw_types]
            if isinstance(raw_types, list)
            else ["FullAttention"] * n_layers
        )
        if arch.get("layer_types") != expected_types:
            errors.append("arch.layer_types does not match source layer/head schedule")
    heads = tc.get("num_attention_heads_per_layer")
    if heads is not None and arch.get("attention_heads_per_layer") != heads:
        errors.append("arch.attention_heads_per_layer differs from source")
    if tc.get("tie_word_embeddings", config.get("tie_word_embeddings", False)) != arch.get(
        "tie_word_embeddings", False
    ):
        errors.append("arch.tie_word_embeddings differs from source")
    rope_dim = tc.get("qk_rope_head_dim")
    head_dim = tc.get("head_dim")
    if isinstance(rope_dim, (int, float)) and isinstance(head_dim, (int, float)) and head_dim:
        expected_partial = float(rope_dim) / float(head_dim)
        if not math.isclose(float(arch.get("partial_rotary_factor", -1)), expected_partial, rel_tol=1e-6):
            errors.append("arch.partial_rotary_factor differs from qk_rope_head_dim/head_dim")
    rope_stats: dict[str, object] = {}
    rope = tc.get("rope_scaling")
    if isinstance(rope, dict):
        rope_kind = rope.get("rope_type", rope.get("type"))
        required_rope = ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow")
        if arch.get("arch_name") == "deepseek_v41":
            # V4.1 deliberately keeps the complete original HF object in
            # arch.deepseek_v41.  Its loader reads source.text_config at
            # loader.rs:1741-1899 and constructs the two effective frequency
            # tables in dsv41.rs:1091-1105; it does not consume ModelArch.yarn.
            # Requiring a flattened arch.yarn here would reject the native
            # representation and hide a real config loss behind a false error.
            if rope_kind != "yarn":
                errors.append(f"V4.1 source rope_scaling kind {rope_kind!r} != 'yarn'")
            for field in required_rope:
                value = rope.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    errors.append(f"V4.1 source rope_scaling.{field} is not numeric")
            theta = tc.get("rope_theta")
            compress_theta = tc.get("compress_rope_theta")
            rope_dim = tc.get("qk_rope_head_dim")
            model_head_dim = tc.get("head_dim")
            original = rope.get("original_max_position_embeddings")
            if isinstance(theta, (int, float)) and not isinstance(theta, bool):
                if not math.isclose(float(arch.get("rope_theta", math.nan)), float(theta), rel_tol=1e-6):
                    errors.append("V4.1 arch.rope_theta differs from nested source text_config")
            else:
                errors.append("V4.1 source text_config.rope_theta is not numeric")
            if isinstance(compress_theta, (int, float)) and not isinstance(compress_theta, bool):
                if float(compress_theta) <= 0:
                    errors.append("V4.1 compress_rope_theta must be positive")
            else:
                errors.append("V4.1 source text_config.compress_rope_theta is not numeric")
            if isinstance(original, (int, float)) and not isinstance(original, bool):
                if arch.get("max_position_embeddings") != int(original):
                    errors.append("arch.max_position_embeddings does not use the source native YaRN window")
            if isinstance(rope_dim, (int, float)) and isinstance(model_head_dim, (int, float)) and model_head_dim:
                expected_partial = float(rope_dim) / float(model_head_dim)
                if not math.isclose(float(arch.get("partial_rotary_factor", -1)), expected_partial, rel_tol=1e-6):
                    errors.append("V4.1 arch.partial_rotary_factor differs from source qk_rope_head_dim/head_dim")
            # An optional flattened copy is accepted only when it agrees with
            # the preserved nested source; the current native V4.1 header has
            # yarn=null by design.
            flattened = arch.get("yarn")
            if flattened is not None:
                if not isinstance(flattened, dict):
                    errors.append("V4.1 arch.yarn, when present, is not an object")
                else:
                    for field in required_rope:
                        if field in rope and flattened.get(field) != rope[field]:
                            errors.append(
                                f"V4.1 arch.yarn.{field}={flattened.get(field)!r} != source {rope[field]!r}"
                            )
            rope_stats = {
                "rope_contract": "dsv41_nested_text_config",
                "rope_type": rope_kind,
                "rope_theta": theta,
                "compress_rope_theta": compress_theta,
                "rope_factor": rope.get("factor"),
                "rope_original_max": original,
                "rope_beta_fast": rope.get("beta_fast"),
                "rope_beta_slow": rope.get("beta_slow"),
                "rope_head_dim": rope_dim,
                "rope_native_window": arch.get("max_position_embeddings"),
            }
        else:
            yarn = arch.get("yarn")
            if not isinstance(yarn, dict):
                errors.append("arch.yarn is absent despite source rope_scaling")
            else:
                for field in required_rope:
                    if field in rope and yarn.get(field) != rope[field]:
                        errors.append(f"arch.yarn.{field}={yarn.get(field)!r} != source {rope[field]!r}")
            original = rope.get("original_max_position_embeddings")
            if original is not None and arch.get("max_position_embeddings") != original:
                errors.append("arch.max_position_embeddings does not use the source native YaRN window")
            rope_stats = {"rope_contract": "flattened_arch_yarn", "rope_type": rope_kind}
    elif arch.get("arch_name") == "deepseek_v41":
        errors.append("V4.1 source text_config.rope_scaling is absent")
    expected_moe = tc.get("n_routed_experts", tc.get("num_experts"))
    if expected_moe:
        moe = arch.get("moe")
        if not isinstance(moe, dict):
            errors.append("arch.moe is absent despite source routed experts")
        else:
            checks = {
                "num_experts": expected_moe,
                "top_k": tc.get("num_experts_per_tok", tc.get("top_k_experts")),
                "moe_intermediate_size": tc.get("moe_intermediate_size"),
                "norm_topk_prob": tc.get("norm_topk_prob", False),
                "routed_scaling_factor": tc.get("routed_scaling_factor"),
            }
            shared = tc.get("n_shared_experts", tc.get("num_shared_experts"))
            if shared is not None and tc.get("moe_intermediate_size") is not None:
                checks["shared_expert_intermediate_size"] = shared * tc["moe_intermediate_size"]
            for field, source in checks.items():
                if source is not None and moe.get(field) != source:
                    errors.append(f"arch.moe.{field}={moe.get(field)!r} != source {source!r}")
    mtp_layers = tc.get("num_nextn_predict_layers", tc.get("mtp_num_hidden_layers"))
    if mtp_layers:
        mtp = arch.get("mtp")
        if not isinstance(mtp, dict) or mtp.get("num_layers") != mtp_layers:
            errors.append("arch.mtp does not preserve source MTP depth")
    return rope_stats


def check_coverage(
    artifacts: list[Artifact],
    source_config: dict,
    weight_map: dict[str, str],
    dspark_policy: str,
) -> dict[str, int]:
    errors: list[str] = []
    expected, optional_dspark, omitted_scales = expected_names(weight_map)
    actual_entries = [e for artifact in artifacts for e in artifact.entries]
    actual: dict[str, Entry] = {}
    for entry in actual_entries:
        if entry.name in actual:
            errors.append(f"duplicate tensor across CMF shards: {entry.name}")
        actual[entry.name] = entry
    if dspark_policy == "exclude":
        expected = {name: raw for name, raw in expected.items() if name not in optional_dspark}
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"missing {len(missing)} source-derived tensors: {', '.join(missing[:8])}")
    if unexpected:
        errors.append(f"unexpected {len(unexpected)} tensors: {', '.join(unexpected[:8])}")
    if dspark_policy in ("include", "match-source"):
        absent_dspark = sorted(optional_dspark - actual_set)
        if absent_dspark:
            errors.append(f"DSpark source tensors missing from CMF: {', '.join(absent_dspark[:8])}")
    else:
        actual_dspark = sorted(name for name in actual if is_dspark_name(name))
        if actual_dspark:
            errors.append(f"DSpark exclusion requested but CMF contains {actual_dspark[:4]}")

    tc = source_config.get("text_config", source_config)
    try:
        n_layers = get_number(tc["num_hidden_layers"], "num_hidden_layers")
        n_experts = get_number(tc["n_routed_experts"], "n_routed_experts")
    except (KeyError, CheckError) as exc:
        errors.append(str(exc))
        n_layers = n_experts = 0
    source_layers = {
        int(match.group(1))
        for raw in weight_map
        if (match := (LAYER_RE.match(raw) or CANON_LAYER_RE.match(raw)))
    }
    if source_layers != set(range(n_layers)):
        errors.append(f"source layer ids {sorted(source_layers)[:8]} do not equal 0..{n_layers - 1}")
    output_layers = {
        int(match.group(1))
        for name in actual
        if (match := CANON_LAYER_RE.match(name))
    }
    if output_layers != set(range(n_layers)):
        errors.append(f"CMF layer ids do not equal 0..{n_layers - 1}")
    # A tiny fixture may intentionally contain only a few canonical tensors.  The
    # exact source/CMF set and metadata checks still apply there, while the full
    # expert/mHC inventory is meaningful only when the source index contains the
    # expected routed matrices.  The pinned V4.1 index always takes this branch.
    source_expert_matrices = sum(
        1
        for raw in weight_map
        if (LAYER_RE.match(raw) and ".ffn.experts." in raw and raw.endswith(".weight"))
        or (EXPERT_RE.match(raw) is not None)
    )
    full_inventory = source_expert_matrices >= n_layers * n_experts * 3 if n_layers and n_experts else False
    if full_inventory:
        for layer in range(n_layers):
            for expert in range(n_experts):
                for role in ("gate_proj", "up_proj", "down_proj"):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{role}.weight"
                    if name not in actual:
                        errors.append(f"missing expert matrix {name}")
            for role in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.{layer}.mlp.shared_expert.{role}.weight"
                if name not in actual:
                    errors.append(f"missing shared expert matrix {name}")
            for control in (
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            ):
                name = f"model.layers.{layer}.{control}"
                if name not in actual:
                    errors.append(f"missing mHC control {name}")
            if not any(
                name.startswith(f"model.layers.{layer}.self_attn.") for name in actual
            ):
                errors.append(f"layer {layer} has no canonical self_attn tensors")

    engram_layers = sorted(
        int(match.group(1))
        for raw in weight_map
        if (match := ENGRAM_RE.match(raw)) and ".embed.weight" in raw
    )
    config_engram_layers = tc.get("engram_layer_ids", [])
    if sorted(config_engram_layers) != engram_layers:
        errors.append(
            f"Engram layer ids source index {engram_layers} != config {sorted(config_engram_layers)}"
        )
    for layer in engram_layers:
        weight_name = f"model.layers.{layer}.engram.embed.weight"
        scale_name = f"model.layers.{layer}.engram.embed.scale"
        for name in (weight_name, scale_name):
            entry = actual.get(name)
            if entry is None:
                errors.append(f"missing native Engram tensor {name}")
                continue
            if entry.dtype != "u8":
                errors.append(f"native Engram tensor {name} has dtype {entry.dtype}, expected u8")
            if entry.nbytes != product(entry.shape):
                errors.append(f"native Engram tensor {name} byte count does not equal its shape")
        weight = actual.get(weight_name)
        scale = actual.get(scale_name)
        if weight and scale and len(weight.shape) == 2 and len(scale.shape) == 2:
            if scale.shape[0] != weight.shape[0] or scale.shape[1] != (weight.shape[1] + 31) // 32:
                errors.append(
                    f"Engram scale shape {scale.shape} does not match weight {weight.shape}"
                )
        # The source config carries the exact native table geometry.  Check it
        # independently of the CMF self-consistency relation above, so a
        # converter cannot preserve a wrong shape in both U8 entries.
        ids = tc.get("engram_layer_ids")
        nums = tc.get("engram_num_embeddings")
        embed_dim = tc.get("engram_head_dim")
        n_heads = tc.get("engram_n_heads")
        if (
            isinstance(ids, list)
            and isinstance(nums, list)
            and len(ids) == len(nums)
            and layer in ids
            and isinstance(embed_dim, (int, float))
            and isinstance(n_heads, (int, float))
        ):
            expected_rows = int(nums[ids.index(layer)])
            if weight and weight.shape != (expected_rows, int(embed_dim)):
                errors.append(
                    f"Engram weight {weight_name} shape {weight.shape} != source geometry "
                    f"({expected_rows}, {int(embed_dim)})"
                )
            if scale and scale.shape != (expected_rows, int(n_heads)):
                errors.append(
                    f"Engram scale {scale_name} shape {scale.shape} != source geometry "
                    f"({expected_rows}, {int(n_heads)})"
                )
        for suffix in ("q_weight", "k_weight", "wkv.weight"):
            if f"model.layers.{layer}.engram.{suffix}" not in actual:
                errors.append(f"missing Engram projection model.layers.{layer}.engram.{suffix}")
    for name, entry in actual.items():
        if not name.startswith(("vision.", "aligner.")):
            continue
        if name.endswith(".weight") and len(entry.shape) >= 2 and entry.dtype != "f16":
            errors.append(f"vision/aligner matrix {name} is {entry.dtype}, expected f16")
        # Vision/aligner controls follow the converter's existing scalar
        # policy: source BF16 controls are emitted as F16, while a control
        # explicitly covered by force_f32 remains F32.  The V4.1 source
        # fixture carries these controls as BF16, so requiring every bias/norm
        # to be F32 would reject the native completed artifact.  Still reject
        # every quantized/byte codec here: controls must remain unquantized.
        if (name.endswith(".bias") or ".norm" in name and name.endswith(".weight")) and entry.dtype not in (
            "f16",
            "f32",
        ):
            errors.append(f"vision/aligner control {name} is quantized as {entry.dtype}")

    if errors:
        shown = errors[:24]
        suffix = "" if len(errors) <= len(shown) else f" (+{len(errors) - len(shown)} more)"
        raise CheckError("; ".join(shown) + suffix)
    return {
        "source_entries": len(weight_map),
        "expected_entries": len(expected_set),
        "actual_entries": len(actual_set),
        "omitted_quant_scales": omitted_scales,
        "dspark_entries": len(optional_dspark),
        "layers": n_layers,
        "experts_per_layer": n_experts,
        "full_inventory": int(full_inventory),
        "engram_layers": len(engram_layers),
        "vision_f16": sum(
            1
            for name, entry in actual.items()
            if name.startswith(("vision.", "aligner."))
            and name.endswith(".weight")
            and len(entry.shape) >= 2
            and entry.dtype == "f16"
        ),
        "vision_controls_f16": sum(
            1
            for name, entry in actual.items()
            if name.startswith(("vision.", "aligner."))
            and (name.endswith(".bias") or ".norm" in name and name.endswith(".weight"))
            and entry.dtype == "f16"
        ),
        "vision_controls_f32": sum(
            1
            for name, entry in actual.items()
            if name.startswith(("vision.", "aligner."))
            and (name.endswith(".bias") or ".norm" in name and name.endswith(".weight"))
            and entry.dtype == "f32"
        ),
    }


def self_test() -> None:
    assert canonical_name("embed.weight") == "model.embed_tokens.weight"
    assert canonical_name("layers.3.ffn.experts.7.w1.weight") == (
        "model.layers.3.mlp.experts.7.gate_proj.weight"
    )
    assert canonical_name("layers.3.attn.wkv.scale") == "model.layers.3.self_attn.wkv.scale"
    assert expected_nbytes("q4tp", (8, 64)) == 8 * 2 * 16 + 8 * 4 + 8 * 2
    assert expected_nbytes("q2tp", (8, 64)) == 8 * 2 * 8 + 8 * 4 + 8 * 2
    print("self-test PASS: independent mapping and fixed payload formulas")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="final CMF path (or shard 1)")
    parser.add_argument("--source-index", help="pinned HF model.safetensors.index.json")
    parser.add_argument("--source-config", help="pinned HF config.json")
    parser.add_argument(
        "--dspark-policy",
        choices=("match-source", "include", "exclude"),
        default="match-source",
        help="whether optional DSpark source tensors must be present (default: match-source)",
    )
    parser.add_argument("--self-test", action="store_true", help="run small pure-Python checks")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.model or not args.source_index or not args.source_config:
        parser.error("--model, --source-index, and --source-config are required")
    try:
        source_index = json.loads(Path(args.source_index).read_text())
        source_config = json.loads(Path(args.source_config).read_text())
        weight_map = source_index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckError("source index has no nonempty weight_map object")
        artifacts = [parse_artifact(p) for p in artifact_paths(Path(args.model))]
        metadata_stats = check_metadata(artifacts[0].header, source_config, errors := [])
        if errors:
            raise CheckError("; ".join(errors))
        stats = check_coverage(artifacts, source_config, weight_map, args.dspark_policy)
        stats.update(metadata_stats)
    except (OSError, json.JSONDecodeError, CheckError, struct.error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    total_bytes = sum(artifact.size for artifact in artifacts)
    print(f"PASS: {args.model}")
    print(f"  CMF files: {len(artifacts)}; file bytes: {total_bytes}")
    print(
        "  tensors: {actual_entries}; source index: {source_entries}; expected: {expected_entries}; "
        "omitted quant scales: {omitted_quant_scales}".format(**stats)
    )
    print(
        "  layers: {layers}; experts/layer: {experts_per_layer}; Engram layers: {engram_layers}; "
        "vision/aligner matrix F16: {vision_f16}; controls F16/F32: {vision_controls_f16}/{vision_controls_f32}; "
        "full expert/mHC inventory: {full_inventory}".format(**stats)
    )
    if stats.get("rope_contract") == "dsv41_nested_text_config":
        print(
            "  V4.1 RoPE: nested source text_config consumed by loader; type={rope_type}; "
            "theta={rope_theta}; compress_theta={compress_rope_theta}; head_dim={rope_head_dim}; "
            "factor={rope_factor}; original={rope_original_max}; beta={rope_beta_fast}/{rope_beta_slow}; "
            "native_window={rope_native_window}".format(**stats)
        )
    print(f"  DSpark policy: {args.dspark_policy}; source DSpark entries: {stats['dspark_entries']}")
    print("  CMF hashes/bounds, canonical coverage, metadata, Engram U8 geometry, and duplicates: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
